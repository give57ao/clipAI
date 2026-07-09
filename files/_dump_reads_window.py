# -*- coding: utf-8 -*-
"""임의 영상의 지정 구간만 디코드해 원시 K/D/A 판독 덤프 — 캐시 불필요.

배치 영상(sig_cache 없음)의 오탐/미탐 구간 진단용. seek 후 창만 읽으므로 빠름(수 초).

사용:
    python -u _dump_reads_window.py "E:\\OBS\\2026-03-29 01-01-04.mp4" 18:20 19:00
    python -u _dump_reads_window.py <mp4> <start M:SS> <end M:SS> [--scan-fps 4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
from game_frame import extract_game_crop_bgr  # noqa: E402
from hud_digit_match import get_hud_digit_matcher  # noqa: E402
from hud_kda import read_kda_triple_from_game  # noqa: E402


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
    get_hud_digit_matcher()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"열기 실패: {args.video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / args.scan_fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))

    print(f"# {Path(args.video).name}  {args.start}-{args.end}  scan_fps={args.scan_fps}")
    print(f"# {'t':>9}  {'K':>4} {'D':>4} {'A':>4}  conf   method")
    frame_i = int(t0 * fps)
    n_hit = n_total = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t = frame_i / fps
        if t >= t1:
            break
        if (frame_i - int(t0 * fps)) % step == 0:
            game, _ = extract_game_crop_bgr(frame)
            k, d, a, conf, method = read_kda_triple_from_game(game)
            n_total += 1
            if k is not None:
                n_hit += 1
            fmt = lambda v: "-" if v is None else str(v)  # noqa: E731
            print(f"{mss(t):>10}  {fmt(k):>4} {fmt(d):>4} {fmt(a):>4}  {conf:.2f}   {method}")
        frame_i += 1
    cap.release()
    print(f"# hit {n_hit}/{n_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
