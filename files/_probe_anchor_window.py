# -*- coding: utf-8 -*-
"""주어진 구간에서 프레임별 "005" 점수 앵커 매치 신뢰도(conf) 타임라인을 덤프.
탭으로 스코어보드를 잠깐 연 순간(가짜 hud_elim 경계) vs 진짜 라운드 종료를
구분하는 실측용 — 앵커가 사라지면(conf 급락) 가짜 경계 가설 지지.

사용:
    python -u _probe_anchor_window.py "E:\\OBS\\<영상>.mp4" <시작 M:SS> <끝 M:SS> [--scan-fps 6]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
from hud_score_wins import _load_anchor_template, _match_anchor_in_frame, find_score_anchor  # noqa: E402


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
    ap.add_argument("--scan-fps", type=float, default=6.0)
    args = ap.parse_args()

    t0, t1 = _s(args.start), _s(args.end)
    tmpl = _load_anchor_template()
    if tmpl is None:
        print("앵커 템플릿 없음")
        return 1

    anchor = find_score_anchor(args.video)
    print(f"# anchor(전체영상 기준): {anchor}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"열기 실패: {args.video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / args.scan_fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))

    print(f"# {Path(args.video).name}  {args.start}-{args.end}  scan_fps={args.scan_fps}")
    print(f"# {'t':>9}  conf   x    y   scale")
    frame_i = int(t0 * fps)
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t = frame_i / fps
        if t >= t1:
            break
        if (frame_i - int(t0 * fps)) % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            r = _match_anchor_in_frame(gray, tmpl)
            if r is None:
                print(f"{mss(t):>10}  ----   (매치 없음)")
            else:
                conf, x, y, s = r
                flag = "" if conf >= 0.75 else "  <<< 낮음"
                print(f"{mss(t):>10}  {conf:.3f}  {x:4d} {y:4d} {s:.2f}{flag}")
        frame_i += 1
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
