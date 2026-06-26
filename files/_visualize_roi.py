# -*- coding: utf-8 -*-
"""현재 ROI 경계를 game_crop 위에 그려서 저장."""
import sys
sys.path.insert(0, ".")

import cv2
import numpy as np
from pathlib import Path

from game_frame import extract_game_crop_bgr
from scoreboard_layout import (
    _ROW_Y_CENTERS, _ROW_HALF_H,
    _RED_NICK, _RED_K, _BLUE_NICK, _BLUE_K,
)

VIDEO = Path(r"E:\OBS\2026-03-19 23-00-50.mp4")
DATASET_ROOT = Path(r"E:\Highlights\ml_dataset")
OUT_DIR = Path(r"E:\clipai_result\_roi_debug\overlay")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# R01, R04 두 프레임 시각화
SAMPLES = {"R01": 293.5, "R02": 339.5, "R04": 471.5}

cap = cv2.VideoCapture(str(VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

COLORS = {
    "red_nick":  (0, 80, 255),    # 빨강계
    "red_k":     (0, 200, 255),   # 주황
    "blue_nick": (255, 80, 0),    # 파랑계
    "blue_k":    (255, 200, 0),   # 하늘
}

for name, t in SAMPLES.items():
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    if not ok:
        continue

    game_bgr, _ = extract_game_crop_bgr(frame, dataset_root=DATASET_ROOT)
    h, w = game_bgr.shape[:2]
    vis = game_bgr.copy()

    for i, yc in enumerate(_ROW_Y_CENTERS):
        y1 = int((yc - _ROW_HALF_H) * h)
        y2 = int((yc + _ROW_HALF_H) * h)

        for (x1r, x2r), key in [
            (_RED_NICK,  "red_nick"),
            (_RED_K,     "red_k"),
            (_BLUE_NICK, "blue_nick"),
            (_BLUE_K,    "blue_k"),
        ]:
            x1 = int(x1r * w)
            x2 = int(x2r * w)
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS[key], 2)
            cv2.putText(vis, f"{key[:6]}r{i}", (x1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS[key], 1)

    cv2.imwrite(str(OUT_DIR / f"{name}_overlay.jpg"), vis)
    print(f"[overlay] {name} saved  game={w}x{h}")

cap.release()
print("[overlay] 완료:", OUT_DIR)
