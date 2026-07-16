# -*- coding: utf-8 -*-
"""seek(cap.set) vs 순차 디코드로 같은 시각의 프레임을 뽑아 비교 — CNN 판정까지.

가설(2026-07-16): hud_boundary_verify.verify_runs_live는 run당 3프레임을 seek로
뽑는데, 긴 영상에서 seek가 부정확하면 스코어보드가 아닌 엉뚱한 프레임이 CNN에
들어가 '가짜 경계'로 오기각 → 라운드 대량 병합 → 올킬 소실.

사용:
    python -u _probe_seek_accuracy.py "E:\\OBS\\<영상>.mp4" 58:34.25 58:35.5 58:36.75
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
from hud_boundary_verify import classify_frame, get_boundary_verifier  # noqa: E402


def _s(v: str) -> float:
    parts = v.split(":")
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])


def grab_seek(video: str, t: float):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    actual_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
    ok, frame = cap.read()
    cap.release()
    return (frame if ok else None), fps, actual_idx


def grab_sequential(video: str, t: float):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target = int(t * fps)
    i = 0
    frame = None
    while i <= target:
        ok, f = cap.read()
        if not ok:
            break
        if i == target:
            frame = f
            break
        i += 1
    cap.release()
    return frame, fps


def main() -> int:
    video = sys.argv[1]
    times = [_s(a) for a in sys.argv[2:]]
    model, transform, device = get_boundary_verifier()

    print(f"# {Path(video).name}")
    print(f"# {'요청시각':>10}  {'seek 착지프레임':>14} {'요청프레임':>10}  {'오차(초)':>8}  {'seek CNN':>9} {'순차 CNN':>9}  일치?")
    for t in times:
        f_seek, fps, actual_idx = grab_seek(video, t)
        f_seq, _ = grab_sequential(video, t)
        want_idx = int(t * fps)
        err_sec = (actual_idx - want_idx) / fps
        p_seek = classify_frame(f_seek, model, transform, device) if f_seek is not None else -1
        p_seq = classify_frame(f_seq, model, transform, device) if f_seq is not None else -1
        same = "-"
        if f_seek is not None and f_seq is not None:
            same = "같음" if np.array_equal(f_seek, f_seq) else "★다름★"
        print(
            f"{t:>10.2f}  {actual_idx:>14.0f} {want_idx:>10d}  {err_sec:>8.2f}  "
            f"{p_seek:>9.4f} {p_seq:>9.4f}  {same}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
