# -*- coding: utf-8 -*-
"""구간 내 row_miss(HUD 사라짐) 프레임을 찾아 전광판 CNN이 뭐라고 판정하는지 덤프.

R5 CNN 경계검증기(hud_boundary_verify.verify_runs_live)가 진짜 스코어보드를
'가짜 경계'로 기각하는지 직접 확인하는 진단 도구 (2026-07-16).

사용:
    python -u _probe_cnn_boundary.py "E:\\OBS\\<영상>.mp4" <시작 M:SS> <끝 M:SS> [--save-dir DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
from game_frame import extract_game_crop_bgr  # noqa: E402
from hud_boundary_verify import classify_frame, get_boundary_verifier  # noqa: E402
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
    ap.add_argument("--save-dir", default=None, help="row_miss 프레임 저장 (육안 확인용)")
    args = ap.parse_args()

    t0, t1 = _s(args.start), _s(args.end)
    get_hud_digit_matcher()
    model, transform, device = get_boundary_verifier()
    print(f"# CNN device={device}  SCORE_THRESHOLD=0.6 (이 미만이면 경계 폐기)")

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"열기 실패: {args.video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / args.scan_fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))

    print(f"# {Path(args.video).name}  {args.start}-{args.end}")
    print(f"# {'t':>9}  {'HUD K':>5}  {'CNN scoreboard prob':>20}  판정")
    frame_i = int(t0 * fps)
    n_saved = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t = frame_i / fps
        if t >= t1:
            break
        if (frame_i - int(t0 * fps)) % step == 0:
            game, _ = extract_game_crop_bgr(frame)
            k, _d, _a, _conf, method = read_kda_triple_from_game(game)
            if method == "row_miss":  # HUD 사라짐 = 경계 후보 프레임
                prob = classify_frame(frame, model, transform, device)
                verdict = "진짜경계 유지" if prob >= 0.6 else "★가짜로 폐기★"
                print(f"{mss(t):>10}  {'-':>5}  {prob:>20.4f}  {verdict}")
                if save_dir and n_saved < 12:
                    cv2.imwrite(str(save_dir / f"{mss(t).replace(':','m')}_p{prob:.3f}.png"), frame)
                    n_saved += 1
        frame_i += 1
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
