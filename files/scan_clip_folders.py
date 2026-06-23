# -*- coding: utf-8 -*-
"""폴더에 정리된 클립 mp4를 스캔해 clips_index.csv 생성.

사용자가 이미 잘라둔 mp4를 아래처럼 넣으면 됩니다:
  clips/doublekill/*.mp4
  clips/multikill/*.mp4
  clips/save/*.mp4
  clips/allkill/*.mp4
  clips/background/*.mp4  (선택)

폴더명은 영문(label) 또는 한글(더블킬 등) 모두 가능.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from labeling_constants import ALL_CLIP_LABELS, BACKGROUND_LABEL, HIGHLIGHT_LABELS, normalize_label
from video_utils import probe_duration_sec

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm"}
BACKGROUND_CHUNKS_DIR = "_chunks"
BACKGROUND_FULL_VOD_MIN_SEC = 90.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="클립 폴더 스캔 → clips_index.csv")
    parser.add_argument("--dataset-root", default=r"E:\Highlights\ml_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="기존 clips_index.csv 덮어쓰기",
    )
    return parser.parse_args()


def resolve_folder_label(folder_name: str) -> str | None:
    return normalize_label(folder_name)


def stable_split(clip_path: str, seed: int) -> str:
    """클립 경로 기반 고정 split (재실행해도 동일)."""
    digest = hashlib.sha1(f"{seed}|{clip_path}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def should_index_clip(clip_path: Path, label: str, folder: Path) -> bool:
    """background 루트의 풀 녹화본은 제외, _chunks·짧은 클립만 인덱싱."""
    if label != BACKGROUND_LABEL:
        return True
    try:
        rel = clip_path.relative_to(folder)
    except ValueError:
        return True
    # background/_chunks/... → 항상 포함
    parts = rel.parts
    if parts and parts[0] == BACKGROUND_CHUNKS_DIR:
        return True
    # background 루트의 긴 파일 → slice 대상이므로 학습 인덱스에서 제외
    duration = probe_duration_sec(clip_path)
    if duration is not None and duration >= BACKGROUND_FULL_VOD_MIN_SEC:
        return False
    return True


def scan_clips(clips_root: Path, seed: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not clips_root.exists():
        return rows

    for folder in sorted(clips_root.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name.lower() == "review":
            continue

        label = resolve_folder_label(folder.name)
        if label not in ALL_CLIP_LABELS:
            print(f"[scan] skip unknown folder: {folder.name}")
            continue

        for clip_path in sorted(folder.rglob("*")):
            if not clip_path.is_file():
                continue
            if clip_path.suffix.lower() not in VIDEO_EXTS:
                continue
            if not should_index_clip(clip_path, label, folder):
                continue

            clip_id = f"{label}_{hashlib.sha1(str(clip_path).encode('utf-8')).hexdigest()[:12]}"
            rows.append(
                {
                    "clip_id": clip_id,
                    "clip_path": str(clip_path.resolve()),
                    "label": label,
                    "split": stable_split(str(clip_path.resolve()), seed),
                    "source": "user_folder",
                    "notes": "",
                }
            )

    rows.sort(key=lambda r: (r["label"], r["clip_path"]))
    return rows


def write_index(path: Path, rows: list[dict[str, str]], allow_overwrite: bool) -> None:
    if path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"이미 존재: {path}\n덮어쓰려면 --allow-overwrite 사용"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["clip_id", "clip_path", "label", "split", "source", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    clips_root = dataset_root / "clips"
    out_path = dataset_root / "manifests" / "clips_index.csv"

    rows = scan_clips(clips_root, args.seed)
    if not rows:
        print("[scan] 클립이 없습니다.")
        print(f"  아래 폴더에 mp4를 넣어주세요: {clips_root}")
        for label in HIGHLIGHT_LABELS:
            print(f"    - {clips_root / label}")
        return 1

    write_index(out_path, rows, args.allow_overwrite)

    counts: dict[str, int] = {label: 0 for label in ALL_CLIP_LABELS}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1

    print("[scan] saved:", out_path)
    print("[scan] total:", len(rows))
    for label in ALL_CLIP_LABELS:
        n = counts.get(label, 0)
        if n:
            print(f"  - {label}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
