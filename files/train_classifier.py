# -*- coding: utf-8 -*-
"""clips_index.csv 기반 EfficientNet-B0 하이라이트 분류기 학습."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from labeling_constants import ALL_CLIP_LABELS, BACKGROUND_LABEL, HIGHLIGHT_LABELS

LABEL_TO_IDX = {label: idx for idx, label in enumerate(ALL_CLIP_LABELS)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}


@dataclass
class ClipRow:
    clip_path: str
    label: str
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="하이라이트 5-class 분류기 학습")
    parser.add_argument("--dataset-root", default=r"E:\Highlights\ml_dataset")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--max-background-train",
        type=int,
        default=500,
        help="train에서 background 최대 샘플 수 (불균형 완화)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_clip_rows(index_path: Path) -> list[ClipRow]:
    rows: list[ClipRow] = []
    with index_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            label = (row.get("label") or "").strip()
            split = (row.get("split") or "train").strip()
            clip_path = (row.get("clip_path") or "").strip()
            if label not in LABEL_TO_IDX or not clip_path:
                continue
            if not Path(clip_path).exists():
                continue
            rows.append(ClipRow(clip_path=clip_path, label=label, split=split))
    return rows


def balance_train_rows(rows: list[ClipRow], max_background: int, rng: random.Random) -> list[ClipRow]:
    highlights = [r for r in rows if r.label in HIGHLIGHT_LABELS]
    backgrounds = [r for r in rows if r.label == BACKGROUND_LABEL]
    if max_background > 0 and len(backgrounds) > max_background:
        backgrounds = rng.sample(backgrounds, max_background)
    return highlights + backgrounds


class ClipFrameDataset(Dataset):
    def __init__(self, rows: list[ClipRow], train: bool):
        self.rows = rows
        self.train = train
        aug = [transforms.RandomHorizontalFlip(p=0.5)] if train else []
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                *aug,
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _read_random_frame(self, path: str) -> np.ndarray | None:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return None
        idx = random.randint(0, max(0, frame_count - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int):
        row = self.rows[index]
        label_idx = LABEL_TO_IDX[row.label]

        for _ in range(4):
            frame = self._read_random_frame(row.clip_path)
            if frame is not None:
                tensor = self.transform(frame)
                return tensor, label_idx

        # fallback: 검은 프레임
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        return self.transform(blank), label_idx


def build_model(num_classes: int) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def compute_class_weights(rows: list[ClipRow]) -> torch.Tensor:
    counts = [0] * len(ALL_CLIP_LABELS)
    for row in rows:
        counts[LABEL_TO_IDX[row.label]] += 1
    total = sum(counts)
    weights = []
    for c in counts:
        if c == 0:
            weights.append(1.0)
        else:
            weights.append(total / (len(ALL_CLIP_LABELS) * c))
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, dict[str, float]]:
    model.eval()
    correct = 0
    total = 0
    per_label_correct = {label: 0 for label in ALL_CLIP_LABELS}
    per_label_total = {label: 0 for label in ALL_CLIP_LABELS}

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for pred, label in zip(preds.cpu().tolist(), labels.cpu().tolist()):
            label_name = IDX_TO_LABEL[label]
            per_label_total[label_name] += 1
            if pred == label:
                per_label_correct[label_name] += 1

    acc = correct / max(1, total)
    per_label_acc = {
        label: per_label_correct[label] / max(1, per_label_total[label])
        for label in ALL_CLIP_LABELS
        if per_label_total[label] > 0
    }
    return acc, per_label_acc


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    return running_loss / max(1, total)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    dataset_root = Path(args.dataset_root)
    index_path = dataset_root / "manifests" / "clips_index.csv"
    models_dir = dataset_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_clip_rows(index_path)
    if not all_rows:
        print("[train] clips_index.csv에 유효한 클립이 없습니다.")
        return 1

    train_rows = balance_train_rows(
        [r for r in all_rows if r.split == "train"],
        args.max_background_train,
        rng,
    )
    val_rows = [r for r in all_rows if r.split == "val"]
    test_rows = [r for r in all_rows if r.split == "test"]

    print(f"[train] clips total={len(all_rows)} train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    train_ds = ClipFrameDataset(train_rows, train=True)
    val_ds = ClipFrameDataset(val_rows, train=False)
    test_ds = ClipFrameDataset(test_rows, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    model = build_model(len(ALL_CLIP_LABELS)).to(device)
    class_weights = compute_class_weights(train_rows).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = -1.0
    best_path = models_dir / "highlight_classifier_best.pt"
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_per_label = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        print(
            f"[train] epoch {epoch}/{args.epochs} "
            f"loss={train_loss:.4f} val_acc={val_acc*100:.1f}% time={elapsed:.0f}s"
        )
        for label in HIGHLIGHT_LABELS:
            if label in val_per_label:
                print(f"  val {label}: {val_per_label[label]*100:.1f}%")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "elapsed_sec": elapsed,
            }
        )

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label_to_idx": LABEL_TO_IDX,
                    "idx_to_label": IDX_TO_LABEL,
                    "epoch": epoch,
                    "val_acc": val_acc,
                },
                best_path,
            )
            print(f"[train] saved best -> {best_path}")

    # test with best checkpoint
    if best_path.exists() and test_rows:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test_acc, test_per_label = evaluate(model, test_loader, device)
        print(f"[train] test_acc={test_acc*100:.1f}%")
        for label in ALL_CLIP_LABELS:
            if label in test_per_label:
                print(f"  test {label}: {test_per_label[label]*100:.1f}%")
    else:
        test_acc = 0.0
        test_per_label = {}

    meta = {
        "labels": list(ALL_CLIP_LABELS),
        "epochs": args.epochs,
        "best_val_acc": best_val,
        "test_acc": test_acc,
        "test_per_label": test_per_label,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "test_size": len(test_rows),
        "history": history,
    }
    meta_path = models_dir / "highlight_classifier_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] meta -> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
