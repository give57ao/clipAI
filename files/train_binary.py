# -*- coding: utf-8 -*-
"""1단계: 하이라이트 vs background 이진 분류기 학습."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from labeling_constants import BACKGROUND_LABEL, HIGHLIGHT_LABELS
from ml_train_common import (
    ClipFrameDataset,
    FocalLoss,
    build_model,
    build_train_rows_binary,
    evaluate,
    load_clip_rows,
    train_one_epoch,
)

BINARY_LABEL_TO_IDX = {"background": 0, "highlight": 1}
BINARY_IDX_TO_LABEL = {0: "background", 1: "highlight"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="하이라이트 이진 탐지기 학습")
    parser.add_argument("--dataset-root", default=r"E:\Highlights\ml_dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--max-background-train", type=int, default=150)
    parser.add_argument("--highlight-repeat", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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
        print("[binary] clips_index.csv 없음. scan_clip_folders.py 먼저 실행.")
        return 1

    highlights = [r for r in all_rows if r.label in HIGHLIGHT_LABELS]
    backgrounds = [r for r in all_rows if r.label == BACKGROUND_LABEL]
    val_rows = [r for r in all_rows if r.split == "val"]
    test_rows = [r for r in all_rows if r.split == "test"]

    print(
        f"[binary] highlights={len(highlights)} background_pool={len(backgrounds)} "
        f"val={len(val_rows)} test={len(test_rows)}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[binary] device={device}", flush=True)

    model = build_model(2).to(device)
    best_highlight_recall = -1.0
    best_path = models_dir / "highlight_binary_best.pt"
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_rows = build_train_rows_binary(
            [r for r in all_rows if r.split == "train" and r.label in HIGHLIGHT_LABELS],
            [r for r in all_rows if r.split == "train" and r.label == BACKGROUND_LABEL],
            args.max_background_train,
            args.highlight_repeat,
            rng,
        )
        print(f"[binary] epoch {epoch} train_size={len(train_rows)}", flush=True)

        train_ds = ClipFrameDataset(
            train_rows, BINARY_LABEL_TO_IDX, train=True,
            num_frames=args.num_frames, binary=True,
        )
        val_ds = ClipFrameDataset(
            val_rows, BINARY_LABEL_TO_IDX, train=False,
            num_frames=args.num_frames, binary=True,
        )

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        weights = torch.tensor([1.0, 3.0], device=device)
        criterion = FocalLoss(gamma=2.0, weight=weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_per, highlight_recall = evaluate(
            model, val_loader, device, BINARY_IDX_TO_LABEL, positive_labels={"highlight"}
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
                },
                best_path,
            )
            print(f"[binary] saved best -> {best_path}", flush=True)

    if best_path.exists() and test_rows:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test_loader = DataLoader(
            ClipFrameDataset(test_rows, BINARY_LABEL_TO_IDX, train=False,
                             num_frames=args.num_frames, binary=True),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        test_acc, _, test_recall = evaluate(
            model, test_loader, device, BINARY_IDX_TO_LABEL, positive_labels={"highlight"}
        )
        print(f"[binary] test_acc={test_acc*100:.1f}% highlight_recall={test_recall*100:.1f}%", flush=True)
    else:
        test_recall = 0.0

    meta = {
        "task": "binary_highlight_vs_background",
        "best_highlight_recall": best_highlight_recall,
        "test_highlight_recall": test_recall,
        "epochs": args.epochs,
        "max_background_train": args.max_background_train,
        "highlight_repeat": args.highlight_repeat,
        "history": history,
    }
    meta_path = models_dir / "highlight_binary_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[binary] meta -> {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
