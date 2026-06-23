# -*- coding: utf-8 -*-
"""label_segments.csv 기반으로 학습용 클립 추출 (4종 + background)."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

from labeling_constants import ALL_CLIP_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="라벨 세그먼트 클립 추출")
    parser.add_argument("--dataset-root", default=r"E:\Highlights\ml_dataset")
    parser.add_argument(
        "--only-pending",
        action="store_true",
        help="review_status=pending만 추출",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 추출된 클립도 덮어씀",
    )
    return parser.parse_args()


def run_ffmpeg_extract(video_path: Path, start_sec: float, end_sec: float, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        str(video_path),
        "-c",
        "copy",
        str(out_path),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return result.returncode == 0


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    manifests_path = dataset_root / "manifests" / "label_segments.csv"
    clips_root = dataset_root / "clips"

    if not manifests_path.exists():
        print(f"[extract] label_segments.csv 없음: {manifests_path}")
        return 1

    total = 0
    extracted = 0
    skipped = 0
    failed = 0
    by_label: dict[str, int] = {label: 0 for label in ALL_CLIP_LABELS}

    with manifests_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            review_status = (row.get("review_status") or "").strip().lower()
            if args.only_pending and review_status != "pending":
                skipped += 1
                continue

            segment_id = (row.get("segment_id") or "").strip()
            label = (row.get("label") or "").strip().lower()
            video_str = (row.get("video_path") or "").strip()
            start_sec = parse_float(row.get("start_sec", ""))
            end_sec = parse_float(row.get("end_sec", ""))

            if not segment_id or label not in ALL_CLIP_LABELS:
                failed += 1
                continue
            if not video_str:
                failed += 1
                continue
            if start_sec is None or end_sec is None or end_sec <= start_sec:
                failed += 1
                continue

            video_path = Path(video_str)
            if not video_path.exists():
                failed += 1
                continue

            out_path = clips_root / label / f"{segment_id}.mp4"
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            ok = run_ffmpeg_extract(video_path, start_sec, end_sec, out_path)
            if ok:
                extracted += 1
                by_label[label] = by_label.get(label, 0) + 1
            else:
                failed += 1

    print(f"[extract] total={total} extracted={extracted} skipped={skipped} failed={failed}")
    if extracted:
        print("[extract] by_label:", ", ".join(f"{k}={by_label.get(k, 0)}" for k in ALL_CLIP_LABELS))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
