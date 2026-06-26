# -*- coding: utf-8 -*-
"""1단계: 하이라이트 vs background 이진 분류기 학습."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from labeling_constants import BACKGROUND_LABEL, HIGHLIGHT_LABELS
from ml_train_common import (
    ClipFrameDataset,
    FocalLoss,
    WindowFrameDataset,
    build_model,
    build_train_rows_binary,
    build_train_rows_segments_binary,
    collate_raw_frames,
    evaluate,
    load_clip_rows,
    load_segment_rows,
    make_roi_processor,
    train_one_epoch,
)

BINARY_LABEL_TO_IDX = {"background": 0, "highlight": 1}
BINARY_IDX_TO_LABEL = {0: "background", 1: "highlight"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="하이라이트 이진 탐지기 학습")
    parser.add_argument("--dataset-root", default=r"E:\Highlights\ml_dataset")
    parser.add_argument(
        "--mode",
        choices=("clip", "window", "both"),
        default="window",
        help="clip=잘린 클립, window=OBS 구간, both=둘 다",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--max-background-train", type=int, default=150)
    parser.add_argument("--highlight-repeat", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _split_rows(rows, label_filter_hi, label_filter_bg):
    hi = [r for r in rows if r.label in label_filter_hi]
    bg = [r for r in rows if r.label == label_filter_bg]
    val = [r for r in rows if r.split == "val"]
    test = [r for r in rows if r.split == "test"]
    train_hi = [r for r in rows if r.split == "train" and r.label in label_filter_hi]
    train_bg = [r for r in rows if r.split == "train" and r.label == label_filter_bg]
    return hi, bg, val, test, train_hi, train_bg


def _make_binary_loaders(
    train_ds,
    val_ds,
    batch_size: int,
    num_workers: int,
    collate_fn,
):
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    dataset_root = Path(args.dataset_root)
    models_dir = dataset_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    use_clip = args.mode in ("clip", "both")
    use_window = args.mode in ("window", "both")

    clip_rows: list = []
    segment_rows: list = []
    if use_clip:
        clip_rows = load_clip_rows(dataset_root / "manifests" / "clips_index.csv")
        if not clip_rows and args.mode == "clip":
            print("[binary] clips_index.csv 없음. scan_clip_folders.py 먼저 실행.")
            return 1
    if use_window:
        segment_rows = load_segment_rows(
            dataset_root / "manifests" / "label_segments.csv", dataset_root
        )
        if not segment_rows and args.mode == "window":
            print("[binary] label_segments.csv 없음. build_label_manifest.py 먼저 실행.")
            return 1
        clip_ready = sum(1 for r in segment_rows if r.clip_path)
        if segment_rows and clip_ready == 0:
            print(
                "[binary] window 클립 미추출. extract_labeled_clips.py --overwrite 먼저 실행.",
                flush=True,
            )
            return 1
        print(f"[binary] window_clips={clip_ready}/{len(segment_rows)}", flush=True)

    clip_hi, clip_bg, clip_val, clip_test, clip_train_hi, clip_train_bg = ([], [], [], [], [], [])
    if clip_rows:
        clip_hi, clip_bg, clip_val, clip_test, clip_train_hi, clip_train_bg = _split_rows(
            clip_rows, HIGHLIGHT_LABELS, BACKGROUND_LABEL
        )

    seg_hi, seg_bg, seg_val, seg_test, seg_train_hi, seg_train_bg = ([], [], [], [], [], [])
    if segment_rows:
        seg_hi, seg_bg, seg_val, seg_test, seg_train_hi, seg_train_bg = _split_rows(
            segment_rows, HIGHLIGHT_LABELS, BACKGROUND_LABEL
        )

    print(
        f"[binary] mode={args.mode} "
        f"clip_hi={len(clip_hi)} clip_bg={len(clip_bg)} "
        f"window_hi={len(seg_hi)} window_bg={len(seg_bg)} "
        f"val={len(clip_val) + len(seg_val)} test={len(clip_test) + len(seg_test)}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[binary] device={device}", flush=True)

    roi_train = make_roi_processor(dataset_root, device, train=True)
    roi_val = make_roi_processor(dataset_root, device, train=False)
    collate_fn = collate_raw_frames if roi_train is not None else None
    if roi_train is not None:
        print("[binary] game_roi=gpu-batch", flush=True)

    model = build_model(2).to(device)
    best_highlight_recall = -1.0
    best_path = models_dir / "highlight_binary_best.pt"
    history: list[dict] = []

    ds_kwargs = dict(
        label_to_idx=BINARY_LABEL_TO_IDX,
        num_frames=args.num_frames,
        binary=True,
        dataset_root=dataset_root,
    )

    for epoch in range(1, args.epochs + 1):
        train_parts = []
        if use_clip and clip_train_hi:
            clip_train = build_train_rows_binary(
                clip_train_hi, clip_train_bg,
                args.max_background_train if not use_window else args.max_background_train // 2,
                args.highlight_repeat,
                rng,
            )
            train_parts.append(ClipFrameDataset(clip_train, train=True, **ds_kwargs))
        if use_window and seg_train_hi:
            window_bg_max = args.max_background_train if not use_clip else args.max_background_train // 2
            window_train = build_train_rows_segments_binary(
                seg_train_hi, seg_train_bg, window_bg_max, args.highlight_repeat, rng
            )
            train_parts.append(WindowFrameDataset(window_train, train=True, **ds_kwargs))

        if not train_parts:
            print("[binary] 학습 샘플 없음.")
            return 1

        train_ds = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)

        val_parts = []
        if use_clip and clip_val:
            val_parts.append(ClipFrameDataset(clip_val, train=False, **ds_kwargs))
        if use_window and seg_val:
            val_parts.append(WindowFrameDataset(seg_val, train=False, **ds_kwargs))
        val_ds = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)

        train_size = len(train_ds)
        print(f"[binary] epoch {epoch} train_size={train_size}", flush=True)

        train_loader, val_loader = _make_binary_loaders(
            train_ds, val_ds, args.batch_size, args.num_workers, collate_fn
        )

        weights = torch.tensor([1.0, 3.0], device=device)
        criterion = FocalLoss(gamma=2.0, weight=weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, roi_processor=roi_train
        )
        val_acc, val_per, highlight_recall = evaluate(
            model, val_loader, device, BINARY_IDX_TO_LABEL,
            positive_labels={"highlight"}, roi_processor=roi_val,
        )
        elapsed = time.time() - t0

        print(
            f"[binary] epoch {epoch}/{args.epochs} loss={train_loss:.4f} "
            f"val_acc={val_acc*100:.1f}% highlight_recall={highlight_recall*100:.1f}% "
            f"time={elapsed:.0f}s",
            flush=True,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "highlight_recall": highlight_recall,
            "elapsed_sec": elapsed,
        })

        if highlight_recall > best_highlight_recall:
            best_highlight_recall = highlight_recall
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label_to_idx": BINARY_LABEL_TO_IDX,
                    "idx_to_label": BINARY_IDX_TO_LABEL,
                    "epoch": epoch,
                    "highlight_recall": highlight_recall,
                    "num_frames": args.num_frames,
                    "train_mode": args.mode,
                },
                best_path,
            )
            print(f"[binary] saved best -> {best_path}", flush=True)

    test_recall = 0.0
    if best_path.exists():
        test_parts = []
        if use_clip and clip_test:
            test_parts.append(ClipFrameDataset(clip_test, train=False, **ds_kwargs))
        if use_window and seg_test:
            test_parts.append(WindowFrameDataset(seg_test, train=False, **ds_kwargs))
        if test_parts:
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            test_ds = test_parts[0] if len(test_parts) == 1 else ConcatDataset(test_parts)
            test_loader = DataLoader(
                test_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
            )
            test_acc, _, test_recall = evaluate(
                model, test_loader, device, BINARY_IDX_TO_LABEL,
                positive_labels={"highlight"}, roi_processor=roi_val,
            )
            print(
                f"[binary] test_acc={test_acc*100:.1f}% highlight_recall={test_recall*100:.1f}%",
                flush=True,
            )

    meta = {
        "task": "binary_highlight_vs_background",
        "train_mode": args.mode,
        "best_highlight_recall": best_highlight_recall,
        "test_highlight_recall": test_recall,
        "epochs": args.epochs,
        "max_background_train": args.max_background_train,
        "highlight_repeat": args.highlight_repeat,
        "window_highlights": len(seg_hi),
        "window_backgrounds": len(seg_bg),
        "history": history,
    }
    meta_path = models_dir / "highlight_binary_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[binary] meta -> {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
