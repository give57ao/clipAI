# -*- coding: utf-8 -*-
"""전처리 후 크롭 이미지 저장 — overclock 행 OCR 실패 원인 파악."""
import sys
sys.path.insert(0, ".")

import cv2
import numpy as np
from pathlib import Path

from game_frame import extract_game_crop_bgr
from scoreboard_layout import _ROW_Y_CENTERS, _ROW_HALF_H, _RED_NICK, _preprocess_ocr
from scouter_nick import get_reader

VIDEO = Path(r"E:\OBS\2026-03-19 23-00-50.mp4")
DATASET_ROOT = Path(r"E:\Highlights\ml_dataset")
OUT_DIR = Path(r"E:\clipai_result\_roi_debug\preprocess")
OUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

for rnd, t in [("R04_t469", 469.5), ("R04_t471", 471.5), ("R04_t473", 473.0)]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    if not ok:
        continue

    game_bgr, _ = extract_game_crop_bgr(frame, dataset_root=DATASET_ROOT)
    h, w = game_bgr.shape[:2]

    # row2 (overclock 예상)
    yc = _ROW_Y_CENTERS[2]
    y1 = int((yc - _ROW_HALF_H) * h)
    y2 = int((yc + _ROW_HALF_H) * h)
    x1 = int(_RED_NICK[0] * w)
    x2 = int(_RED_NICK[1] * w)

    crop = game_bgr[y1:y2, x1:x2]
    up_raw = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    up_clahe = _preprocess_ocr(crop)
    up_inv = cv2.bitwise_not(up_raw)

    cv2.imwrite(str(OUT_DIR / f"{rnd}_raw.jpg"), up_raw)
    cv2.imwrite(str(OUT_DIR / f"{rnd}_clahe.jpg"), up_clahe)
    cv2.imwrite(str(OUT_DIR / f"{rnd}_inv.jpg"), up_inv)

    reader = get_reader()
    for suffix, img in [("raw", up_raw), ("clahe", up_clahe), ("inv", up_inv)]:
        results = reader.readtext(img, detail=1, paragraph=False)
        texts = [(t, round(c, 2)) for _, t, c in results]
        print(f"[{rnd}][{suffix}] {texts}")

cap.release()
