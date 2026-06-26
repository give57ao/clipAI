# -*- coding: utf-8 -*-
"""K열 크롭 + OCR 결과 확인 (R05, R06, R07)."""
import sys
sys.path.insert(0, ".")
import cv2
from pathlib import Path
from game_frame import extract_game_crop_bgr
from scoreboard_layout import _ROW_Y_CENTERS, _ROW_HALF_H, _RED_K, _preprocess_ocr
from scouter_nick import get_reader

VIDEO = Path(r"E:\OBS\2026-03-19 23-00-50.mp4")
DATASET_ROOT = Path(r"E:\Highlights\ml_dataset")
OUT_DIR = Path(r"E:\clipai_result\_roi_debug\k_col"); OUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
reader = get_reader()

# 각 라운드 중간 시각 샘플
SAMPLES = {"R05": 507.75, "R06": 662.75, "R07": 702.25}

for name, t in SAMPLES.items():
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    if not ok:
        continue
    game, _ = extract_game_crop_bgr(frame, dataset_root=DATASET_ROOT)
    h, w = game.shape[:2]
    print(f"\n=== {name} (t={t}) game={w}x{h} ===")
    for i, yc in enumerate(_ROW_Y_CENTERS):
        y1 = int((yc - _ROW_HALF_H) * h)
        y2 = int((yc + _ROW_HALF_H) * h)
        x1 = int(_RED_K[0] * w)
        x2 = int(_RED_K[1] * w)
        crop = game[y1:y2, x1:x2]
        up = _preprocess_ocr(crop)
        results = reader.readtext(up, detail=1, paragraph=False)
        txt = [(t2, round(c, 2)) for _, t2, c in results]
        # 이미지 저장
        cv2.imwrite(str(OUT_DIR / f"{name}_row{i}_k.jpg"), crop)
        print(f"  red_row{i}: {txt}")

cap.release()
print("\n[done]", OUT_DIR)
