# -*- coding: utf-8 -*-
"""주어진 구간에서 프레임별 (red_wins, blue_wins) 판독 타임라인 덤프.
가짜 hud_elim 경계(탭 스코어보드) vs 진짜 라운드 종료 구분용 —
진짜 종료라면 이 순간 승수가 실제로 +1 돼야 한다.

사용:
    python -u _probe_wins_window.py "E:\\OBS\\<영상>.mp4" <시작 M:SS> <끝 M:SS> [--scan-fps 4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
from hud_score_wins import find_score_anchor, read_wins  # noqa: E402


def _s(mss: str) -> float:
    parts = [int(p) for p in mss.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def mss(sec: float) -> str:
    m, s = divmod(sec, 60.0)
    return f"{int(m)}:{s:05.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("start")
    ap.add_argument("end")
    ap.add_argument("--scan-fps", type=float, default=4.0)
    args = ap.parse_args()

    t0, t1 = _s(args.start), _s(args.end)
    anchor = find_score_anchor(args.video)
    print(f"# anchor: {anchor}")
    if anchor is None:
        print("앵커 탐지 실패 — 이 영상은 판독 불가")
        return 1

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"열기 실패: {args.video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / args.scan_fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))

    print(f"# {Path(args.video).name}  {args.start}-{args.end}  scan_fps={args.scan_fps}")
    print(f"# {'t':>9}   R   B")
    frame_i = int(t0 * fps)
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t = frame_i / fps
        if t >= t1:
            break
        if (frame_i - int(t0 * fps)) % step == 0:
            r, b = read_wins(frame, anchor)
            fmt = lambda v: "-" if v is None else str(v)  # noqa: E731
            print(f"{mss(t):>10}   {fmt(r):>2}  {fmt(b):>2}")
        frame_i += 1
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
