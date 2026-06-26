# -*- coding: utf-8 -*-
"""known_highlights + (선택) background 영상으로 label_segments.csv 생성."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

from labeling_constants import (
    BACKGROUND_LABEL,
    HIGHLIGHT_LABELS,
    normalize_label,
)


@dataclass
class Segment:
    segment_id: str
    video_path: str
    start_sec: float
    end_sec: float
    label: str
    split: str
    source: str
    review_status: str
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="라벨링 세그먼트 매니페스트 생성")
    parser.add_argument("--dataset-root", default=r"E:\Highlights\ml_dataset")
    parser.add_argument(
        "--background-per-video",
        type=int,
        default=8,
        help="background_videos.txt 영상당 음성 샘플 수",
    )
    parser.add_argument("--background-clip-sec", type=float, default=12.0)
    parser.add_argument(
        "--obs-negative-per-video",
        type=int,
        default=30,
        help="known_highlights 영상당 hard negative 윈도우 수",
    )
    parser.add_argument(
        "--obs-negative-margin-sec",
        type=float,
        default=6.0,
        help="하이라이트 구간 주변 제외 마진(초)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="기존 label_segments.csv를 덮어씁니다.",
    )
    return parser.parse_args()


def read_known_highlights(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"known_highlights.csv 없음: {path}")

    # 구버전 호환: known_triples.csv
    legacy = path.parent / "known_triples.csv"
    if not path.exists() and legacy.exists():
        path = legacy

    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_background_videos(path: Path) -> list[Path]:
    if not path.exists():
        return []
    out: list[Path] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(Path(line))
    return out


def probe_duration_sec(video_path: Path) -> float | None:
    if not video_path.exists():
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def stable_segment_id(video_path: str, start_sec: float, end_sec: float, label: str) -> str:
    key = f"{video_path}|{start_sec:.3f}|{end_sec:.3f}|{label}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"{label}_{digest}"


def pick_split(rng: random.Random) -> str:
    p = rng.random()
    if p < 0.8:
        return "train"
    if p < 0.9:
        return "val"
    return "test"


def parse_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def resolve_highlight_label(row: dict[str, str]) -> str | None:
    raw = (row.get("label") or "").strip()
    if raw:
        return normalize_label(raw)
    # 구버전 known_triples.csv → multikill로 간주
    if row.get("video_path") and row.get("timestamp_sec"):
        return "multikill"
    return None


def build_highlight_segments(
    rows: list[dict[str, str]], rng: random.Random
) -> tuple[list[Segment], list[str]]:
    segments: list[Segment] = []
    skipped: list[str] = []

    for i, row in enumerate(rows, start=1):
        video_path = (row.get("video_path") or "").strip()
        if not video_path:
            continue

        label = resolve_highlight_label(row)
        if label not in HIGHLIGHT_LABELS:
            skipped.append(f"row {i}: invalid label ({row.get('label')})")
            continue

        timestamp_sec = parse_float(row.get("timestamp_sec", ""), -1.0)
        if timestamp_sec < 0:
            skipped.append(f"row {i}: invalid timestamp")
            continue

        before_sec = parse_float(row.get("window_before_sec", ""), 8.0)
        after_sec = parse_float(row.get("window_after_sec", ""), 8.0)
        start_sec = max(0.0, timestamp_sec - before_sec)
        end_sec = max(start_sec + 1.0, timestamp_sec + after_sec)

        segments.append(
            Segment(
                segment_id=stable_segment_id(video_path, start_sec, end_sec, label),
                video_path=video_path,
                start_sec=start_sec,
                end_sec=end_sec,
                label=label,
                split=pick_split(rng),
                source="known_highlights",
                review_status="confirmed",
                notes=(row.get("notes") or f"known_highlight_row_{i}").strip(),
            )
        )

    return segments, skipped


def _intervals_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def build_obs_hard_negative_segments(
    highlight_segments: list[Segment],
    per_video: int,
    clip_sec: float,
    margin_sec: float,
    rng: random.Random,
) -> list[Segment]:
    """known_highlights와 같은 OBS 영상에서 하이라이트 구간을 피한 negative."""
    by_video: dict[str, list[Segment]] = {}
    for seg in highlight_segments:
        by_video.setdefault(seg.video_path, []).append(seg)

    segments: list[Segment] = []
    max_attempts = max(50, per_video * 20)

    for video_str, hi_segs in by_video.items():
        video_path = Path(video_str)
        dur = probe_duration_sec(video_path)
        if dur is None or dur < clip_sec + 4:
            continue

        forbidden: list[tuple[float, float]] = []
        for hi in hi_segs:
            forbidden.append(
                (
                    max(0.0, hi.start_sec - margin_sec),
                    min(dur, hi.end_sec + margin_sec),
                )
            )

        added = 0
        attempts = 0
        while added < per_video and attempts < max_attempts:
            attempts += 1
            start_sec = rng.uniform(2.0, max(2.0, dur - clip_sec - 2.0))
            end_sec = start_sec + clip_sec
            blocked = any(
                _intervals_overlap(start_sec, end_sec, fb_start, fb_end)
                for fb_start, fb_end in forbidden
            )
            if blocked:
                continue

            segments.append(
                Segment(
                    segment_id=stable_segment_id(
                        video_str, start_sec, end_sec, BACKGROUND_LABEL
                    ),
                    video_path=video_str,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    label=BACKGROUND_LABEL,
                    split=pick_split(rng),
                    source="obs_hard_negative",
                    review_status="pending",
                    notes="hard negative from pilot OBS",
                )
            )
            added += 1

    return segments


def build_background_segments(
    videos: list[Path], per_video: int, clip_sec: float, rng: random.Random
) -> list[Segment]:
    segments: list[Segment] = []
    for video_path in videos:
        dur = probe_duration_sec(video_path)
        if dur is None or dur < clip_sec + 4:
            continue
        for _ in range(max(0, per_video)):
            start_sec = rng.uniform(2.0, max(2.0, dur - clip_sec - 2.0))
            end_sec = start_sec + clip_sec
            video_str = str(video_path)
            segments.append(
                Segment(
                    segment_id=stable_segment_id(
                        video_str, start_sec, end_sec, BACKGROUND_LABEL
                    ),
                    video_path=video_str,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    label=BACKGROUND_LABEL,
                    split=pick_split(rng),
                    source="random_from_background_video",
                    review_status="pending",
                    notes="",
                )
            )
    return segments


def write_manifest(path: Path, segments: list[Segment], allow_overwrite: bool) -> None:
    if path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"이미 파일이 존재합니다: {path}\n"
            "덮어쓰려면 --allow-overwrite 옵션을 사용하세요."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "segment_id",
                "video_path",
                "start_sec",
                "end_sec",
                "label",
                "split",
                "source",
                "review_status",
                "notes",
            ]
        )
        for s in segments:
            writer.writerow(
                [
                    s.segment_id,
                    s.video_path,
                    f"{s.start_sec:.3f}",
                    f"{s.end_sec:.3f}",
                    s.label,
                    s.split,
                    s.source,
                    s.review_status,
                    s.notes,
                ]
            )


def count_by_label(segments: list[Segment]) -> dict[str, int]:
    counts: dict[str, int] = {label: 0 for label in HIGHLIGHT_LABELS}
    counts[BACKGROUND_LABEL] = 0
    for s in segments:
        counts[s.label] = counts.get(s.label, 0) + 1
    return counts


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    dataset_root = Path(args.dataset_root)
    manifests = dataset_root / "manifests"
    known_path = manifests / "known_highlights.csv"
    background_path = manifests / "background_videos.txt"
    # 구버전 파일명 호환
    legacy_normal = manifests / "normal_videos.txt"
    out_manifest = manifests / "label_segments.csv"

    known_rows = read_known_highlights(known_path)
    background_videos = read_background_videos(background_path)
    if not background_videos and legacy_normal.exists():
        background_videos = read_background_videos(legacy_normal)

    highlights, skipped_rows = build_highlight_segments(known_rows, rng)
    hard_negatives = build_obs_hard_negative_segments(
        highlights,
        args.obs_negative_per_video,
        args.background_clip_sec,
        args.obs_negative_margin_sec,
        rng,
    )
    backgrounds = build_background_segments(
        background_videos,
        args.background_per_video,
        args.background_clip_sec,
        rng,
    )
    all_segments = highlights + hard_negatives + backgrounds
    all_segments.sort(key=lambda s: (s.label, s.video_path, s.start_sec))

    if not all_segments:
        print("[manifest] 생성할 세그먼트가 없습니다.")
        print("  - known_highlights.csv를 먼저 채워주세요.")
        print("  - (선택) background_videos.txt로 음성 샘플 추가 가능")
        return 1

    write_manifest(out_manifest, all_segments, args.allow_overwrite)
    counts = count_by_label(all_segments)

    print("[manifest] saved:", out_manifest)
    print(
        "[manifest] counts:",
        ", ".join(f"{k}={counts.get(k, 0)}" for k in list(HIGHLIGHT_LABELS) + [BACKGROUND_LABEL]),
    )
    print(f"[manifest] total={len(all_segments)}")
    if skipped_rows:
        print("[manifest] skipped rows:")
        for msg in skipped_rows[:10]:
            print("  -", msg)

    print("\n다음 단계:")
    print(
        "  python extract_labeled_clips.py "
        f"--dataset-root \"{dataset_root}\" --only-pending"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
