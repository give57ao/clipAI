# -*- coding: utf-8 -*-
"""2026-01-08 02-33-22.mp4 직접 대조 — K 3분할 + ΔK 검증."""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import cv2

from detect_ace_hud import scan_hud_aces, format_report
from game_frame import extract_game_crop_bgr
from hud_digit_match import calibrate_from_video, reset_matcher, get_hud_digit_matcher, DEFAULT_TEMPLATE_DIR
from hud_kda import (
    _MID_ROUND_X,
    _RED_WINS_X,
    _BLUE_WINS_X,
    _TOP_Y,
    _crop_ratio,
    _kda_numbers_crop,
    read_kda_triple_from_game,
    read_top_hud_scores,
)

VIDEO = Path(r"D:\2026-01-08 02-33-22.mp4")
DS = Path(r"E:\Highlights\ml_dataset")
OUT = Path("_tmp_hud/verify")
OUT.mkdir(parents=True, exist_ok=True)

# 프레임 육안 대조 정답 (K/D/A 첫·둘·셋 슬롯)
KDA_GROUND_TRUTH = [
    (60, 0, None, None),
    (120, 1, 3, 0),
    (124, 1, 3, 0),
    (150, 0, 0, 0),
    (180, 1, None, None),
    (240, 0, 1, 0),
    (306, 3, 1, 0),
    (348, 6, 1, 0),
    (420, 6, 2, 0),
    (488, 3, None, 0),
]

TOP_GROUND_TRUTH = {
    120: (0, 5, 3),
}


def _k_crop(game):
    from hud_kda import _split_kda_crops
    nums = _kda_numbers_crop(game)
    kc, _, _ = _split_kda_crops(nums)
    return kc


def calibrate_and_save() -> None:
    def white_crop(game, slot):
        xs = {"R": _RED_WINS_X, "M": _MID_ROUND_X, "B": _BLUE_WINS_X}
        return _crop_ratio(game, _TOP_Y, xs[slot])

    k_samples = [(t, k) for t, k, _d, _a in KDA_GROUND_TRUTH if k is not None]
    white_samples = [(120, "R", 0), (120, "M", 5), (120, "B", 3)]
    m = calibrate_from_video(
        VIDEO,
        samples=k_samples,
        dataset_root=DS,
        k_crop_fn=_k_crop,
        white_crop_fn=white_crop,
        white_samples=white_samples,
    )
    m.save_templates(DEFAULT_TEMPLATE_DIR)
    print(f"[calib] K={sorted(m.k_templates.keys())} white={sorted(m.white_templates.keys())}")


def verify_kda_ocr() -> tuple[int, int]:
    reset_matcher()
    get_hud_digit_matcher()
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    ok_n = 0
    total = 0
    print("\n=== K/D/A 3분할 직접 대조 (K슬롯=올킬) ===")
    for t, ek, ed, ea in KDA_GROUND_TRUTH:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        game, _ = extract_game_crop_bgr(frame, dataset_root=DS)
        k, d, a, conf, method = read_kda_triple_from_game(game)
        nums = _kda_numbers_crop(game)
        if nums.size:
            cv2.imwrite(str(OUT / f"kda_{t}s.jpg"), nums)
        if ek is not None:
            total += 1
            mark = "OK" if k == ek else "NG"
            if k == ek:
                ok_n += 1
            print(f"  t={t:3d}s K exp={ek} got={k} (D={d} A={a}) {method} [{mark}]")
        else:
            print(f"  t={t:3d}s K got={k} (D={d} A={a}) {method}")
    cap.release()
    print(f"K 정확도: {ok_n}/{total}")
    return ok_n, total


def verify_top_ocr() -> None:
    reset_matcher()
    get_hud_digit_matcher()
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    print("\n=== 상단 스코어 (템플릿, 보조) ===")
    for t, (er, em, eb) in TOP_GROUND_TRUTH.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        game, _ = extract_game_crop_bgr(frame, dataset_root=DS)
        r, m, b = read_top_hud_scores(game)
        print(f"  t={t}s exp=({er},{em},{eb}) got=({r},{m},{b})")
    cap.release()


def run_full_scan() -> None:
    reset_matcher()
    t0 = time.perf_counter()
    tl = scan_hud_aces(VIDEO, scan_fps=4.0, dataset_root=DS)
    elapsed = time.perf_counter() - t0
    print(f"\n=== 전체 스캔 ({elapsed:.1f}s) ===")
    print(format_report(tl))
    if tl.ace_rounds:
        print("\n올킬 후보:")
        for r in tl.rounds:
            if r.ace:
                print(
                    f"  R{r.round_index:02d} {sec_to_mss(r.start_sec)}-{sec_to_mss(r.end_sec)} "
                    f"baseK={r.k_base} maxΔK={r.max_delta_k}"
                )


def sec_to_mss(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}:{s:02d}"


def main() -> None:
    if not VIDEO.exists():
        print(f"영상 없음: {VIDEO}")
        return
    print(f"대조: {VIDEO}")
    calibrate_and_save()
    verify_kda_ocr()
    verify_top_ocr()
    run_full_scan()


if __name__ == "__main__":
    main()
