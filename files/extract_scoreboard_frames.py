# -*- coding: utf-8 -*-
"""round_segments.csv 기반 스코어 화면 감지용 프레임 추출.

scoreboard 구간 → positive, 나머지 → negative 로 jpg 저장.
1단계 라운드 분할용 "전체스코어 화면 분류기" 학습 데이터를 만듭니다.

출력:
  dataset_root/scoreboard_frames/scoreboard/*.jpg
  dataset_root/scoreboard_frames/other/*.jpg
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

# scoreboard = positive, 그 외 라벨 = negative 후보
POSITIVE_LABEL = "scoreboard"
# lobby/loading 은 화면이 명확히 다르므로 negative 에 포함 (다양성)
# single_kill = 일반 플레이 화면(스코어보드 아님) → negative
NEGATIVE_LABELS = (
    "movement", "death", "teammate", "single_kill", "loading", "lobby", "stream_end",
)


@dataclass
class Seg:
    video_path: str
    start_sec: float
    end_sec: float
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="스코어 화면 프레임 추출")
    parser.add_argument("--dataset-root", default=r"E:\Highlights\ml_dataset")
    parser.add_argument(
        "--pos-fps", type=float, default=2.0,
        help="scoreboard 구간 초당 추출 프레임 수",
    )
    parser.add_argument(
        "--neg-fps", type=float, default=0.5,
        help="negative 구간 초당 추출 프레임 수",
    )
    parser.add_argument(
        "--max-neg-per-seg", type=int, default=8,
        help="negative 세그먼트당 최대 프레임 수",
    )
    parser.add_argument(
        "--resize", type=int, default=224,
        help="저장 프레임 한 변 크기(정사각). 0이면 원본 유지",
    )
    parser.add_argument("--clean", action="store_true", help="기존 출력 폴더 비우기")
    return parser.parse_args()


def load_segments(csv_path: Path) -> list[Seg]:
    segs: list[Seg] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                start = float(row["start_sec"])
                end = float(row["end_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            video_path = (row.get("video_path") or "").strip()
            label = (row.get("label") or "").strip()
            if not video_path or end <= start:
                continue
            segs.append(Seg(video_path, start, end, label))
    return segs


def sample_times(start: float, end: float, fps: float, max_n: int | None) -> list[float]:
    if fps <= 0:
        return [(start + end) / 2.0]
    step = 1.0 / fps
    times: list[float] = []
    t = start
    while t < end:
        times.append(t)
        t += step
    if not times:
        times = [(start + end) / 2.0]
    if max_n is not None and len(times) > max_n:
        # 균등 다운샘플
        idxs = [round(i * (len(times) - 1) / (max_n - 1)) for i in range(max_n)] if max_n > 1 else [0]
        times = [times[i] for i in idxs]
    return times


def extract_frame(cap: cv2.VideoCapture, fps: float, time_sec: float, resize: int):
    frame_idx = max(0, int(time_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if resize and resize > 0:
        frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_AREA)
    return frame


def clean_dir(d: Path) -> None:
    if d.exists():
        for p in d.glob("*.jpg"):
            p.unlink()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    csv_path = dataset_root / "manifests" / "round_segments.csv"
    if not csv_path.exists():
        print(f"[sbframe] round_segments.csv 없음: {csv_path}")
        print("  parse_round_timeline.py 먼저 실행.")
        return 1

    out_root = dataset_root / "scoreboard_frames"
    pos_dir = out_root / "scoreboard"
    neg_dir = out_root / "other"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        clean_dir(pos_dir)
        clean_dir(neg_dir)

    segs = load_segments(csv_path)
    if not segs:
        print("[sbframe] 세그먼트 없음")
        return 1

    by_video: dict[str, list[Seg]] = defaultdict(list)
    for s in segs:
        by_video[s.video_path].append(s)

    pos_count = 0
    neg_count = 0
    missing_videos: list[str] = []

    for video_path, video_segs in by_video.items():
        vp = Path(video_path)
        if not vp.exists():
            missing_videos.append(video_path)
            continue
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            missing_videos.append(video_path)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stem = vp.stem

        for s in video_segs:
            if s.label == POSITIVE_LABEL:
                times = sample_times(s.start_sec, s.end_sec, args.pos_fps, None)
                out_dir = pos_dir
                is_pos = True
            elif s.label in NEGATIVE_LABELS:
                times = sample_times(s.start_sec, s.end_sec, args.neg_fps, args.max_neg_per_seg)
                out_dir = neg_dir
                is_pos = False
            else:
                continue

            for t in times:
                frame = extract_frame(cap, fps, t, args.resize)
                if frame is None:
                    continue
                fname = f"{stem}_{int(round(t*1000)):08d}ms_{s.label}.jpg"
                cv2.imwrite(str(out_dir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if is_pos:
                    pos_count += 1
                else:
                    neg_count += 1

        cap.release()

    print(f"[sbframe] scoreboard(positive)={pos_count} -> {pos_dir}")
    print(f"[sbframe] other(negative)={neg_count} -> {neg_dir}")
    if missing_videos:
        print(f"[sbframe] 영상 열기 실패 ({len(missing_videos)}):")
        for m in missing_videos:
            print("  -", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
