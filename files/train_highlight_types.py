# -*- coding: utf-8 -*-
"""2단계: 하이라이트 클립만 4종 타입 분류 (double/multikill/save/allkill)."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from labeling_constants import HIGHLIGHT_LABELS
from ml_train_common import (
    ClipFrameDataset,
    FocalLoss,
    build_model,
    build_train_rows_types,
    evaluate,
    load_clip_rows,
    train_one_epoch,
)

LABEL_TO_IDX = {label: idx for idx, label in enumerate(HIGHLIGHT_LABELS)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="하이라이트 4종 타입 분류기")
    parser.add_argument("--dataset-root", default=r"E:\Highlights\ml_dataset")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--highlight-repeat", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_class_weights(rows: list, device) -> torch.Tensor:
    counts = [0] * len(HIGHLIGHT_LABELS)
    for row in rows:
        counts[LABEL_TO_IDX[row.label]] += 1
    total = sum(counts)
    weights = [total / (len(HIGHLIGHT_LABELS) * max(1, c)) for c in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    dataset_root = Path(args.dataset_root)
    index_path = dataset_root / "manifests" / "clips_index.csv"
    models_dir = dataset_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_clip_rows(index_path)
    highlight_rows = [r for r in all_rows if r.label in HIGHLIGHT_LABELS]
    if len(highlight_rows) < 10:
        print("[types] 하이라이트 클립이 너무 적습니다.", flush=True)
        return 1

    train_hi = [r for r in highlight_rows if r.split == "train"]
    val_hi = [r for r in highlight_rows if r.split == "val"]
    test_hi = [r for r in highlight_rows if r.split == "test"]

    print(f"[types] highlights train={len(train_hi)} val={len(val_hi)} test={len(test_hi)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(HIGHLIGHT_LABELS)).to(device)

    best_macro_recall = -1.0
    best_path = models_dir / "highlight_types_best.pt"
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_rows = build_train_rows_types(train_hi, args.highlight_repeat)
        train_ds = ClipFrameDataset(
            train_rows, LABEL_TO_IDX, train=True, num_frames=args.num_frames, binary=False
        )
        val_ds = ClipFrameDataset(
            val_hi, LABEL_TO_IDX, train=False, num_frames=args.num_frames, binary=False
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        weights = compute_class_weights(train_rows, device)
        criterion = FocalLoss(gamma=2.0, weight=weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_per, _ = evaluate(model, val_loader, device, IDX_TO_LABEL)
        macro_recall = sum(val_per.values()) / max(1, len(val_per))
        elapsed = time.time() - t0

        print(
            f"[types] epoch {epoch}/{args.epochs} loss={train_loss:.4f} "
            f"val_acc={val_acc*100:.1f}% macro_recall={macro_recall*100:.1f}% time={elapsed:.0f}s",
            flush=True,
        )
        for label in HIGHLIGHT_LABELS:
            if label in val_per:
                print(f"  val {label}: recall={val_per[label]*100:.1f}%", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "macro_recall": macro_recall,
            "per_label": val_per,
            "elapsed_sec": elapsed,
        })

        if macro_recall > best_macro_recall:
            best_macro_recall = macro_recall
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label_to_idx": LABEL_TO_IDX,
                    "idx_to_label": IDX_TO_LABEL,
                    "epoch": epoch,
                    "macro_recall": macro_recall,
                    "num_frames": args.num_frames,
                },
                best_path,
            )
            print(f"[types] saved best -> {best_path}", flush=True)

    if best_path.exists() and test_hi:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test_loader = DataLoader(
            ClipFrameDataset(test_hi, LABEL_TO_IDX, train=False,
                             num_frames=args.num_frames, binary=False),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        test_acc, test_per, _ = evaluate(model, test_loader, device, IDX_TO_LABEL)
        print(f"[types] test_acc={test_acc*100:.1f}%", flush=True)
        for label in HIGHLIGHT_LABELS:
            if label in test_per:
                print(f"  test {label}: recall={test_per[label]*100:.1f}%", flush=True)

    meta_path = models_dir / "highlight_types_meta.json"
    meta_path.write_text(
        json.dumps({
            "task": "highlight_4class_types",
            "best_macro_recall": best_macro_recall,
            "epochs": args.epochs,
            "history": history,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[types] meta -> {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
