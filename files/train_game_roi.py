# -*- coding: utf-8 -*-
"""게임 ROI 모델 학습 — teacher pseudo-label → EfficientNet 회귀."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from game_roi import (
    ROI_INPUT_SIZE,
    RoiBox,
    build_roi_model,
    build_roi_transform,
    detect_game_roi_teacher,
)
from ml_train_common import load_clip_rows


class RoiFrameDataset(Dataset):
    def __init__(self, clip_paths: list[str], samples_per_clip: int, seed: int):
        self.items: list[tuple[str, int | None]] = []
        rng = random.Random(seed)
        for path in clip_paths:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                continue
            frame_count = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            cap.release()
            picks = [rng.randint(0, max(0, frame_count - 1)) for _ in range(samples_per_clip)]
            for frame_idx in picks:
                self.items.append((path, frame_idx))
        self.transform = build_roi_transform()

    def __len__(self) -> int:
        return len(self.items)

    def _read_frame(self, path: str, frame_idx: int | None) -> np.ndarray | None:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        if frame_idx is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int):
        path, frame_idx = self.items[index]
        rgb = None
        for _ in range(3):
            rgb = self._read_frame(path, frame_idx)
            if rgb is not None:
                break
        if rgb is None:
            rgb = np.zeros((ROI_INPUT_SIZE, ROI_INPUT_SIZE, 3), dtype=np.uint8)
            target = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
            return self.transform(rgb), target

        box = detect_game_roi_teacher(rgb)
        h, w = rgb.shape[:2]
        target = torch.tensor(box.to_normalized(w, h), dtype=torch.float32)
        return self.transform(rgb), target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="게임 ROI 모델 학습")
    parser.add_argument("--dataset-root", default=r"E:\Highlights\ml_dataset")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--samples-per-clip", type=int, default=1)
    parser.add_argument("--max-clips", type=int, default=600, help="ROI 학습에 쓸 최대 클립 수")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def box_iou(a: RoiBox, b: RoiBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (a.x2 - a.x1) * (a.y2 - a.y1))
    area_b = max(1, (b.x2 - b.x1) * (b.y2 - b.y1))
    return inter / max(1.0, area_a + area_b - inter)


@torch.no_grad()
def evaluate_iou(model, loader, device) -> float:
    model.eval()
    total_iou = 0.0
    count = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        preds = torch.sigmoid(model(images)).cpu()
        for pred, target in zip(preds, targets):
            p = pred.tolist()
            t = target.tolist()
            pred_box = RoiBox.from_normalized(p[0], p[1], p[2], p[3], 1000, 1000)
            true_box = RoiBox.from_normalized(t[0], t[1], t[2], t[3], 1000, 1000)
            total_iou += box_iou(pred_box, true_box)
            count += 1
    return total_iou / max(1, count)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    dataset_root = Path(args.dataset_root)
    index_path = dataset_root / "manifests" / "clips_index.csv"
    models_dir = dataset_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    rows = load_clip_rows(index_path)
    clip_paths = sorted({r.clip_path for r in rows})
    if len(clip_paths) < 5:
        print("[game_roi] clips_index.csv 클립이 부족합니다.")
        return 1

    rng.shuffle(clip_paths)
    if args.max_clips > 0 and len(clip_paths) > args.max_clips:
        clip_paths = clip_paths[: args.max_clips]
    val_count = max(1, int(len(clip_paths) * args.val_ratio))
    val_paths = clip_paths[:val_count]
    train_paths = clip_paths[val_count:]
    print(f"[game_roi] clips train={len(train_paths)} val={len(val_paths)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[game_roi] device={device}", flush=True)

    train_ds = RoiFrameDataset(train_paths, args.samples_per_clip, args.seed)
    val_ds = RoiFrameDataset(val_paths, max(1, args.samples_per_clip), args.seed + 1)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = build_roi_model().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_iou = -1.0
    best_path = models_dir / "game_roi_best.pt"
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        t0 = time.time()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            preds = torch.sigmoid(model(images))
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * targets.size(0)
            total += targets.size(0)

        val_iou = evaluate_iou(model, val_loader, device)
        elapsed = time.time() - t0
        train_loss = running_loss / max(1, total)
        print(
            f"[game_roi] epoch {epoch}/{args.epochs} loss={train_loss:.4f} "
            f"val_iou~={val_iou:.3f} time={elapsed:.0f}s",
            flush=True,
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_iou_approx": val_iou,
            "elapsed_sec": elapsed,
        })

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_iou_approx": val_iou,
                    "input_size": ROI_INPUT_SIZE,
                },
                best_path,
            )
            print(f"[game_roi] saved best -> {best_path}", flush=True)

    meta_path = models_dir / "game_roi_meta.json"
    meta_path.write_text(
        json.dumps({
            "task": "game_screen_roi_regression",
            "best_val_iou_approx": best_iou,
            "epochs": args.epochs,
            "history": history,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[game_roi] meta -> {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
