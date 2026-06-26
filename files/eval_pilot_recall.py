# -*- coding: utf-8 -*-
"""파일럿 OBS 추론 결과 vs known_highlights.csv recall 평가."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class KnownKill:
    video_path: str
    timestamp_sec: float
    margin_before: float
    margin_after: float


@dataclass
class PredictedSegment:
    start_sec: float
    end_sec: float
    highlight_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="파일럿 recall 평가")
    parser.add_argument("--dataset-root", default=r"E:\Highlights\ml_dataset")
    parser.add_argument("--result-root", default=r"E:\clipai_result")
    parser.add_argument(
        "--match-margin-sec",
        type=float,
        default=0.0,
        help="GT 구간 [T-before, T+after] 외 추가 여유(초)",
    )
    return parser.parse_args()


def load_known_kills(path: Path) -> list[KnownKill]:
    kills: list[KnownKill] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            video_path = (row.get("video_path") or "").strip()
            if not video_path:
                continue
            try:
                ts = float(row.get("timestamp_sec", ""))
                before = float(row.get("window_before_sec") or 6)
                after = float(row.get("window_after_sec") or 6)
            except (TypeError, ValueError):
                continue
            kills.append(
                KnownKill(
                    video_path=video_path,
                    timestamp_sec=ts,
                    margin_before=before,
                    margin_after=after,
                )
            )
    return kills


def load_predictions(csv_path: Path) -> list[PredictedSegment]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    segments: list[PredictedSegment] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                start_sec = float(row.get("start_sec", ""))
                end_sec = float(row.get("end_sec", ""))
                score = float(row.get("highlight_score") or 0)
            except (TypeError, ValueError):
                continue
            segments.append(
                PredictedSegment(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    highlight_score=score,
                )
            )
    return segments


def intervals_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def is_hit(kill: KnownKill, segments: list[PredictedSegment], extra_margin: float) -> tuple[bool, float | None]:
    gt_start = kill.timestamp_sec - kill.margin_before - extra_margin
    gt_end = kill.timestamp_sec + kill.margin_after + extra_margin
    best_score: float | None = None
    for seg in segments:
        if intervals_overlap(seg.start_sec, seg.end_sec, gt_start, gt_end):
            if best_score is None or seg.highlight_score > best_score:
                best_score = seg.highlight_score
            return True, best_score
    return False, None


def video_stem(video_path: str) -> str:
    return Path(video_path).stem


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    result_root = Path(args.result_root)
    known_path = dataset_root / "manifests" / "known_highlights.csv"

    kills = load_known_kills(known_path)
    if not kills:
        print("[recall] known_highlights.csv 비어 있음")
        return 1

    by_video: dict[str, list[KnownKill]] = {}
    for kill in kills:
        by_video.setdefault(video_stem(kill.video_path), []).append(kill)

    total = len(kills)
    hits = 0
    misses: list[str] = []

    print(f"[recall] known kills={total} videos={len(by_video)}")
    print("-" * 60)

    for stem, video_kills in sorted(by_video.items()):
        csv_path = result_root / f"{stem}_하이라이트" / "inferred_segments.csv"
        segments = load_predictions(csv_path)
        video_hits = 0

        print(f"\n{stem}.mp4  predictions={len(segments)}  known={len(video_kills)}")
        for kill in sorted(video_kills, key=lambda k: k.timestamp_sec):
            hit, score = is_hit(kill, segments, args.match_margin_sec)
            if hit:
                hits += 1
                video_hits += 1
                print(f"  HIT  {kill.timestamp_sec:7.1f}s  score={score:.3f}" if score else f"  HIT  {kill.timestamp_sec:7.1f}s")
            else:
                misses.append(f"{stem} @ {kill.timestamp_sec:.1f}s")
                print(f"  MISS {kill.timestamp_sec:7.1f}s")

        video_recall = video_hits / max(1, len(video_kills))
        print(f"  -> video recall {video_hits}/{len(video_kills)} ({video_recall*100:.1f}%)")

    recall = hits / max(1, total)
    print("\n" + "=" * 60)
    print(f"[recall] overall {hits}/{total} ({recall*100:.1f}%)")
    if misses:
        print(f"[recall] missed ({len(misses)}):")
        for msg in misses[:20]:
            print(f"  - {msg}")
        if len(misses) > 20:
            print(f"  ... +{len(misses) - 20} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
