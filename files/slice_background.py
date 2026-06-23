# -*- coding: utf-8 -*-
"""background 폴더의 긴 녹화본을 학습용 짧은 클립으로 분할.

입력:  clips/background/*.mp4  (풀 녹화본)
출력:  clips/background/_chunks/{원본이름}_part_0001.mp4

이미 짧은 mp4(기본 90초 미만)는 건너뜁니다.
_chunks 아래 파일은 재분할하지 않습니다.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from pathlib import Path

from labeling_constants import BACKGROUND_LABEL
from video_utils import probe_duration_sec

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm"}
CHUNKS_DIR = "_chunks"
DEFAULT_CHUNK_SEC = 12.0
DEFAULT_STRIDE_SEC = 12.0
DEFAULT_MIN_SOURCE_SEC = 90.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="background 풀영상 → 짧은 클립 분할")
    parser.add_argument("--dataset-root", default=r"E:\Highlights\ml_dataset")
    parser.add_argument("--chunk-sec", type=float, default=DEFAULT_CHUNK_SEC)
    parser.add_argument("--stride-sec", type=float, default=DEFAULT_STRIDE_SEC)
    parser.add_argument(
        "--min-source-sec",
        type=float,
        default=DEFAULT_MIN_SOURCE_SEC,
        help="이보다 짧은 파일은 이미 클립으로 간주하고 건너뜀",
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=0,
        help="처리할 원본 영상 수 제한 (0=전부)",
    )
    parser.add_argument(
        "--max-chunks-per-video",
        type=int,
        default=0,
        help="영상당 최대 생성 청크 수 (0=제한 없음)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="이미 존재하는 청크 파일은 건너뜀",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 ffmpeg 실행 없이 계획만 출력",
    )
    return parser.parse_args()


def list_source_videos(background_dir: Path) -> list[Path]:
    if not background_dir.exists():
        return []
    videos: list[Path] = []
    for path in sorted(background_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTS:
            continue
        videos.append(path)
    return videos


def build_chunk_plan(
    duration_sec: float,
    chunk_sec: float,
    stride_sec: float,
    max_chunks: int,
) -> list[tuple[float, float]]:
    if duration_sec < chunk_sec:
        return []
    if stride_sec <= 0:
        stride_sec = chunk_sec

    plan: list[tuple[float, float]] = []
    start = 0.0
    while start + chunk_sec <= duration_sec + 0.05:
        end = min(start + chunk_sec, duration_sec)
        if end - start < chunk_sec * 0.5:
            break
        plan.append((start, end))
        if max_chunks > 0 and len(plan) >= max_chunks:
            break
        start += stride_sec
    return plan


def safe_stem(path: Path) -> str:
    stem = path.stem.strip().replace(" ", "_")
    for ch in '<>:"/\\|?*':
        stem = stem.replace(ch, "_")
    return stem or "video"


def chunk_output_path(chunks_dir: Path, source: Path, index: int) -> Path:
    return chunks_dir / f"{safe_stem(source)}_part_{index:04d}.mp4"


def run_ffmpeg_slice(source: Path, start_sec: float, end_sec: float, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        str(source),
        "-c",
        "copy",
        str(out_path),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return result.returncode == 0


def write_log(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_path",
        "chunk_path",
        "start_sec",
        "end_sec",
        "duration_sec",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    background_dir = dataset_root / "clips" / BACKGROUND_LABEL
    chunks_dir = background_dir / CHUNKS_DIR
    log_path = dataset_root / "manifests" / "background_slice_log.csv"

    sources = list_source_videos(background_dir)
    if args.limit_videos > 0:
        sources = sources[: args.limit_videos]

    if not sources:
        print(f"[slice] 분할할 원본이 없습니다: {background_dir}")
        return 1

    log_rows: list[dict[str, str]] = []
    total_planned = 0
    total_created = 0
    total_skipped = 0
    total_failed = 0
    sources_processed = 0

    print(f"[slice] background dir: {background_dir}")
    print(f"[slice] output dir: {chunks_dir}")
    print(
        f"[slice] chunk={args.chunk_sec}s stride={args.stride_sec}s "
        f"min_source={args.min_source_sec}s dry_run={args.dry_run}"
    )

    for source in sources:
        duration = probe_duration_sec(source)
        if duration is None:
            print(f"[slice] skip (duration unknown): {source.name}")
            continue
        if duration < args.min_source_sec:
            print(f"[slice] skip short ({duration:.1f}s): {source.name}")
            continue

        plan = build_chunk_plan(
            duration, args.chunk_sec, args.stride_sec, args.max_chunks_per_video
        )
        if not plan:
            print(f"[slice] skip (no plan): {source.name}")
            continue

        sources_processed += 1
        est = len(plan)
        total_planned += est
        print(f"[slice] {source.name}  duration={duration/60:.1f}m  chunks={est}")

        for idx, (start_sec, end_sec) in enumerate(plan, start=1):
            out_path = chunk_output_path(chunks_dir, source, idx)
            if out_path.exists() and args.skip_existing:
                total_skipped += 1
                log_rows.append(
                    {
                        "source_path": str(source),
                        "chunk_path": str(out_path),
                        "start_sec": f"{start_sec:.3f}",
                        "end_sec": f"{end_sec:.3f}",
                        "duration_sec": f"{end_sec - start_sec:.3f}",
                        "status": "skipped_exists",
                    }
                )
                continue

            if args.dry_run:
                total_created += 1
                log_rows.append(
                    {
                        "source_path": str(source),
                        "chunk_path": str(out_path),
                        "start_sec": f"{start_sec:.3f}",
                        "end_sec": f"{end_sec:.3f}",
                        "duration_sec": f"{end_sec - start_sec:.3f}",
                        "status": "planned",
                    }
                )
                continue

            ok = run_ffmpeg_slice(source, start_sec, end_sec, out_path)
            if ok:
                total_created += 1
                status = "created"
            else:
                total_failed += 1
                status = "failed"
            log_rows.append(
                {
                    "source_path": str(source),
                    "chunk_path": str(out_path),
                    "start_sec": f"{start_sec:.3f}",
                    "end_sec": f"{end_sec:.3f}",
                    "duration_sec": f"{end_sec - start_sec:.3f}",
                    "status": status,
                }
            )

    if not args.dry_run:
        write_log(log_path, log_rows)

    print("\n[slice] summary")
    print(f"  sources processed: {sources_processed}")
    print(f"  chunks planned: {total_planned}")
    print(f"  chunks created/planned: {total_created}")
    print(f"  skipped existing: {total_skipped}")
    print(f"  failed: {total_failed}")
    if not args.dry_run:
        print(f"  log: {log_path}")
    print("\n다음 단계:")
    print(
        "  python scan_clip_folders.py "
        f"--dataset-root \"{dataset_root}\" --allow-overwrite"
    )
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
