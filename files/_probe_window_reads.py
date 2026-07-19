# -*- coding: utf-8 -*-
"""영상의 특정 구간만 표적 디코딩해 원시 K/D 판독을 덤프 — 캐시 없는 영상의 국소 진단용.

전체 스캔(collect_reads) 없이 [t1,t2] 구간만 seek해서 읽으므로 수 초면 끝남.
사용: python -u _probe_window_reads.py "stem" 4:30 5:10 [--obs-dir E:\\OBS] [--fps 4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import cv2

from game_frame import extract_game_crop_bgr
from hud_digit_match import get_hud_digit_matcher
from hud_kda import read_kda_triple_from_game


def _parse_mss(s: str) -> float:
    parts = [int(p) for p in s.split(":")]
    return parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("t1")
    ap.add_argument("t2")
    ap.add_argument("--obs-dir", default=r"E:\OBS")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--all", action="store_true", help="row_miss 포함 전체 출력")
    args = ap.parse_args()

    video = Path(args.obs_dir) / f"{args.stem}.mp4"
    if not video.exists():
        print(f"영상 없음: {video}")
        return 1
    t1, t2 = _parse_mss(args.t1), _parse_mss(args.t2)

    get_hud_digit_matcher()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    step = max(1, int(round(fps / args.fps)))
    start_frame = int(t1 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fi = start_frame
    n_rowmiss = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        if t > t2:
            break
        if (fi - start_frame) % step == 0:
            game, _ = extract_game_crop_bgr(frame)
            k, d, a, conf, method = read_kda_triple_from_game(game)
            if k is not None and d is None and a is None:
                k, method = None, "triple_incomplete"
            if method == "row_miss":
                n_rowmiss += 1
                if args.all:
                    print(f"  {int(t//60)}:{t % 60:05.2f}  row_miss")
            else:
                m = int(t // 60)
                print(f"  {m}:{t % 60:05.2f}  K={k} D={d} A={a}  conf={conf:.2f}  {method}")
        fi += 1
    cap.release()
    print(f"[probe] row_miss {n_rowmiss}프레임 (구간 {args.t1}-{args.t2})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
