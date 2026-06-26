# -*- coding: utf-8 -*-
"""실패 라운드의 스코어보드 ROI 덤프 - 원인 파악용."""
import sys
sys.path.insert(0, ".")

import cv2
import numpy as np
from pathlib import Path

from game_frame import extract_game_crop_bgr
from scoreboard_layout import (
    _ROW_Y_CENTERS, _ROW_HALF_H,
    _RED_NICK, _RED_K, _BLUE_NICK, _BLUE_K,
    read_scoreboard_rows,
)

VIDEO = Path(r"E:\OBS\2026-03-19 23-00-50.mp4")
DATASET_ROOT = Path(r"E:\Highlights\ml_dataset")
DEBUG_DIR = Path(r"E:\clipai_result\_roi_debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# R04≈469s, R00≈132s, R01≈290s 샘플
SAMPLE_TIMES = {
    "R00": 136.0,
    "R01": 293.0,
    "R02": 337.0,   # 성공 라운드 (기준)
    "R04": 469.5,
    "R05": 507.0,
}

cap = cv2.VideoCapture(str(VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0


def save_crop(img, label, out_dir):
    path = out_dir / f"{label}.jpg"
    cv2.imwrite(str(path), img)


from scoreboard_layout import _is_scoreboard_noise
from scouter_nick import get_reader


def _ocr_raw(crop_bgr):
    if crop_bgr.size == 0:
        return "", 0.0
    reader = get_reader()
    up = cv2.resize(crop_bgr, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    results = reader.readtext(up, detail=1, paragraph=False)
    if not results:
        return "", 0.0
    best_text, best_conf = "", 0.0
    for _, text, conf in results:
        t = str(text).strip()
        if float(conf) > best_conf and t:
            best_text, best_conf = t, float(conf)
    return best_text, best_conf


for name, t in SAMPLE_TIMES.items():
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    if not ok:
        print(f"[!] {name} 프레임 읽기 실패")
        continue

    game_bgr, box = extract_game_crop_bgr(frame, dataset_root=DATASET_ROOT)
    h, w = game_bgr.shape[:2]
    out_dir = DEBUG_DIR / name
    out_dir.mkdir(exist_ok=True)

    # 전체 game crop 저장
    cv2.imwrite(str(out_dir / "00_game_crop.jpg"), game_bgr)

    print(f"\n=== {name} (t={t}s) game={w}x{h} ===")
    row_index = 0
    for team, nick_x, k_x in [
        ("red",  _RED_NICK,  _RED_K),
        ("blue", _BLUE_NICK, _BLUE_K),
    ]:
        for i, yc in enumerate(_ROW_Y_CENTERS):
            y1 = max(0, int((yc - _ROW_HALF_H) * h))
            y2 = min(h, int((yc + _ROW_HALF_H) * h))

            red_nick_crop = game_bgr[y1:y2, int(nick_x[0]*w):int(nick_x[1]*w)]
            red_k_crop    = game_bgr[y1:y2, int(k_x[0]*w):int(k_x[1]*w)]

            nick_raw, nick_conf = _ocr_raw(red_nick_crop)
            k_raw, k_conf       = _ocr_raw(red_k_crop)

            noise = _is_scoreboard_noise(nick_raw)
            flag = "SKIP" if noise else "OK  "

            print(f"  [{row_index}] {team:4} row{i} [{flag}] nick_raw={nick_raw!r:22} conf={nick_conf:.2f}  k={k_raw!r} kconf={k_conf:.2f}")

            # 크롭 이미지 저장
            lbl = f"{team}_row{i}"
            if red_nick_crop.size > 0:
                cv2.imwrite(str(out_dir / f"{lbl}_nick.jpg"), red_nick_crop)
            if red_k_crop.size > 0:
                cv2.imwrite(str(out_dir / f"{lbl}_k.jpg"), red_k_crop)

            row_index += 1

cap.release()
print(f"\n[debug] 이미지 저장 완료: {DEBUG_DIR}")
