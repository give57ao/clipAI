# -*- coding: utf-8 -*-
"""1a 단계: '전체스코어 화면' 이진 분류기 학습.

scoreboard_frames/{scoreboard,other}/*.jpg 를 학습해
라운드 경계(스코어 화면) 감지기를 만듭니다.

- 스코어보드는 화면 중앙 고정 UI → ROI 크롭/좌우반전 안 함
- 단일 프레임 분류 (EfficientNet-B0)
- 출력: models/scoreboard_clf_best.pt, scoreboard_clf_meta.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ml_train_common import FocalLoss, build_model

CLASS_TO_IDX = {"other": 0, "scoreboard": 1}
IDX_TO_CLASS = {0: "other", 1: "scoreboard"}
IMG_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class FrameRow:
    path: str
    label_idx: int
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="스코어 화면 이진 분류기 학습")
    parser.add_argument("--dataset-root", default=r"E:\Highlights\ml_dataset")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-per-class", type=int, default=10,
        help="클래스당 최소 프레임 수 (미달 시 경고)",
    )
    return parser.parse_args()


def stable_split(path: str, seed: int) -> str:
    digest = hashlib.sha1(f"{seed}|{path}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def collect_frames(frames_root: Path, seed: int) -> list[FrameRow]:
    rows: list[FrameRow] = []
    for class_name, label_idx in CLASS_TO_IDX.items():
        class_dir = frames_root / class_name
        if not class_dir.exists():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            p = str(img_path.resolve())
            rows.append(FrameRow(p, label_idx, stable_split(p, seed)))
    return rows


def build_transform(img_size: int, train: bool) -> transforms.Compose:
    if train:
        ops = [
            transforms.ToPILImage(),
            transforms.Resize((int(img_size * 1.14), int(img_size * 1.14))),
            transforms.RandomCrop((img_size, img_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    else:
        ops = [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    return transforms.Compose(ops)


class FrameDataset(Dataset):
    def __init__(self, rows: list[FrameRow], img_size: int, train: bool):
        self.rows = rows
        self.transform = build_transform(img_size, train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        frame = cv2.imread(row.path)
        if frame is None:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(rgb), row.label_idx


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict]:
    model.eval()
    correct = 0
    total = 0
    tp = fp = fn = tn = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for pred, label in zip(preds.cpu().tolist(), labels.cpu().tolist()):
            if label == 1 and pred == 1:
                tp += 1
            elif label == 0 and pred == 1:
                fp += 1
            elif label == 1 and pred == 0:
                fn += 1
            else:
                tn += 1
    acc = correct / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return acc, {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset_root = Path(args.dataset_root)
    frames_root = dataset_root / "scoreboard_frames"
    models_dir = dataset_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_frames(frames_root, args.seed)
    if not rows:
        print(f"[sbclf] 프레임 없음: {frames_root}")
        print("  extract_scoreboard_frames.py 먼저 실행.")
        return 1

    per_class = {name: 0 for name in CLASS_TO_IDX}
    for r in rows:
        per_class[IDX_TO_CLASS[r.label_idx]] += 1
    print(f"[sbclf] frames total={len(rows)} " + " ".join(f"{k}={v}" for k, v in per_class.items()))

    for name, n in per_class.items():
        if n < args.min_per_class:
            print(f"[sbclf] 경고: '{name}' 프레임 {n}개 — 데이터 부족, 영상 추가 권장")

    train_rows = [r for r in rows if r.split == "train"]
    val_rows = [r for r in rows if r.split == "val"]
    test_rows = [r for r in rows if r.split == "test"]
    if not train_rows or not val_rows:
        print("[sbclf] train/val 분할 부족 — 프레임이 너무 적습니다.")
        return 1
    print(f"[sbclf] split train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sbclf] device={device}")

    train_loader = DataLoader(
        FrameDataset(train_rows, args.img_size, train=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        FrameDataset(val_rows, args.img_size, train=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = build_model(2).to(device)

    # 클래스 불균형 보정 가중치
    n_other = max(1, per_class["other"])
    n_score = max(1, per_class["scoreboard"])
    total = n_other + n_score
    weights = torch.tensor(
        [total / (2 * n_other), total / (2 * n_score)],
        dtype=torch.float32, device=device,
    )
    criterion = FocalLoss(gamma=2.0, weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    best_path = models_dir / "scoreboard_clf_best.pt"
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        t0 = time.time()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)
        train_loss = running / max(1, seen)

        val_acc, val_m = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(
            f"[sbclf] epoch {epoch}/{args.epochs} loss={train_loss:.4f} "
            f"val_acc={val_acc*100:.1f}% P={val_m['precision']*100:.1f}% "
            f"R={val_m['recall']*100:.1f}% F1={val_m['f1']*100:.1f}% time={elapsed:.0f}s",
            flush=True,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc, **val_m})

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_to_idx": CLASS_TO_IDX,
                    "idx_to_class": IDX_TO_CLASS,
                    "img_size": args.img_size,
                    "epoch": epoch,
                    "val_f1": best_f1,
                },
                best_path,
            )
            print(f"[sbclf] saved best -> {best_path}", flush=True)

    if best_path.exists() and test_rows:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test_loader = DataLoader(
            FrameDataset(test_rows, args.img_size, train=False),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        test_acc, test_m = evaluate(model, test_loader, device)
        print(
            f"[sbclf] test_acc={test_acc*100:.1f}% P={test_m['precision']*100:.1f}% "
            f"R={test_m['recall']*100:.1f}% F1={test_m['f1']*100:.1f}%",
            flush=True,
        )

    meta_path = models_dir / "scoreboard_clf_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "task": "scoreboard_screen_binary",
                "best_val_f1": best_f1,
                "epochs": args.epochs,
                "img_size": args.img_size,
                "per_class": per_class,
                "history": history,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[sbclf] meta -> {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
