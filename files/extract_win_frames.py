# -*- coding: utf-8 -*-
"""WIN/DEFEAT(라운드 종료) 화면 감지용 프레임 추출.

round_segments.csv(수동 타임라인)에서 scoreboard 구간 직전 프레임을 positive로,
movement/death/teammate/... 구간 프레임을 negative로 뽑습니다.

출력:
  dataset_root/win_frames/win/*.jpg
  dataset_root/win_frames/other/*.jpg
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

POS_LABEL = "scoreboard"  # scoreboard 직전 = win 후보
NEG_LABELS = ("movement", "death", "teammate", "single_kill", "loading", "lobby", "stream_end")


@dataclass
class Seg:
    video_path: str
    start_sec: float
    end_sec: float
    label: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WIN/DEFEAT 프레임 추출")
    p.add_argument("--dataset-root", default=r"E:\Highlights\ml_dataset")
    p.add_argument("--pos-fps", type=float, default=2.0)
    p.add_argument("--neg-fps", type=float, default=0.5)
    p.add_argument("--lookback-sec", type=float, default=2.5, help="scoreboard 직전 탐색 범위(초)")
    p.add_argument("--max-neg-per-seg", type=int, default=6)
    p.add_argument("--resize", type=int, default=224)
    p.add_argument("--clean", action="store_true")
    return p.parse_args()


def load_segments(path: Path) -> list[Seg]:
    out: list[Seg] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                start = float(row.get("start_sec", ""))
                end = float(row.get("end_sec", ""))
            except (TypeError, ValueError):
                continue
            video_path = (row.get("video_path") or "").strip()
            label = (row.get("label") or "").strip()
            if not video_path or end <= start:
                continue
            out.append(Seg(video_path, start, end, label))
    return out


def sample_times(start: float, end: float, fps: float, max_n: int | None) -> list[float]:
    if fps <= 0:
        times = [(start + end) / 2.0]
    else:
        step = 1.0 / fps
        times = []
        t = start
        while t < end:
            times.append(t)
            t += step
        if not times:
            times = [(start + end) / 2.0]
    if max_n is not None and len(times) > max_n:
        idxs = [round(i * (len(times) - 1) / (max_n - 1)) for i in range(max_n)] if max_n > 1 else [0]
        times = [times[i] for i in idxs]
    return times


def clean_dir(d: Path) -> None:
    if d.exists():
        for p in d.glob("*.jpg"):
            p.unlink()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    seg_path = dataset_root / "manifests" / "round_segments.csv"
    if not seg_path.exists():
        print(f"[winframe] round_segments.csv 없음: {seg_path}")
        return 1

    out_root = dataset_root / "win_frames"
    pos_dir = out_root / "win"
    neg_dir = out_root / "other"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        clean_dir(pos_dir)
        clean_dir(neg_dir)

    segs = load_segments(seg_path)
    by_video: dict[str, list[Seg]] = defaultdict(list)
    for s in segs:
        by_video[s.video_path].append(s)

    pos_count = 0
    neg_count = 0

    for video_path, v_segs in by_video.items():
        vp = Path(video_path)
        if not vp.exists():
            continue
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stem = vp.stem

        for s in v_segs:
            if s.label == POS_LABEL:
                # scoreboard 직전 lookback 구간에서 WIN/DEFEAT 프레임을 positive로 수집
                start = max(0.0, s.start_sec - args.lookback_sec)
                end = max(0.0, s.start_sec)
                times = sample_times(start, end, args.pos_fps, None)
                out_dir = pos_dir
                label = "win"
                is_pos = True
            elif s.label in NEG_LABELS:
                times = sample_times(s.start_sec, s.end_sec, args.neg_fps, args.max_neg_per_seg)
                out_dir = neg_dir
                label = s.label
                is_pos = False
            else:
                continue

            for t in times:
                frame_idx = int(t * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                if args.resize and args.resize > 0:
                    frame = cv2.resize(frame, (args.resize, args.resize), interpolation=cv2.INTER_AREA)
                fname = f"{stem}_{int(round(t*1000)):08d}ms_{label}.jpg"
                cv2.imwrite(str(out_dir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if is_pos:
                    pos_count += 1
                else:
                    neg_count += 1

        cap.release()

    print(f"[winframe] win(positive)={pos_count} -> {pos_dir}")
    print(f"[winframe] other(negative)={neg_count} -> {neg_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

