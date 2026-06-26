# -*- coding: utf-8 -*-
"""OBS 프레임 → game_roi ML 게임 영역 crop (스카우터·스코어보드 공용)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from game_roi import GameRoiPredictor, RoiBox

DEFAULT_DATASET_ROOT = Path(r"E:\Highlights\ml_dataset")
SPONSOR_DARK_THRESHOLD = 35.0

_predictor: GameRoiPredictor | None = None
_predictor_root: Path | None = None


def _trim_sponsor_panel_right(crop_bgr: np.ndarray) -> np.ndarray:
    """game_roi crop 우측 검은 후원패널 제거 (구 detect_game_width 로직)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
    h, w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    band = gray[int(h * 0.15) : int(h * 0.85), :]
    if band.size == 0:
        return crop_bgr
    col_mean = band.mean(axis=0)
    is_dark = col_mean < SPONSOR_DARK_THRESHOLD
    for x in range(w - 1, w // 2, -1):
        if not is_dark[x]:
            boundary = x + 1
            if w - boundary >= w * 0.10:
                return crop_bgr[:, :boundary]
            return crop_bgr
    return crop_bgr


def get_game_roi_predictor(dataset_root: Path | None = None) -> GameRoiPredictor:
    """game_roi 모델 싱글톤 (없으면 teacher fallback)."""
    global _predictor, _predictor_root
    root = Path(dataset_root) if dataset_root else DEFAULT_DATASET_ROOT
    if _predictor is None or _predictor_root != root:
        _predictor = GameRoiPredictor.from_dataset_root(root)
        _predictor_root = root
    return _predictor


def reset_game_roi_predictor() -> None:
    """테스트/재로드용."""
    global _predictor, _predictor_root
    _predictor = None
    _predictor_root = None


def extract_game_crop_bgr(
    frame_bgr: np.ndarray,
    *,
    dataset_root: Path | None = None,
    predictor: GameRoiPredictor | None = None,
) -> tuple[np.ndarray, RoiBox]:
    """전체 프레임에서 게임 화면 crop(BGR) + 프레임 좌표 RoiBox 반환."""
    if frame_bgr is None or frame_bgr.size == 0:
        h, w = (0, 0) if frame_bgr is None else frame_bgr.shape[:2]
        return frame_bgr, RoiBox(0, 0, w, h)

    roi = predictor or get_game_roi_predictor(dataset_root)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    box = roi.predict_box(rgb)
    h, w = frame_bgr.shape[:2]
    box = box.clamp(w, h)
    crop = frame_bgr[box.y1 : box.y2, box.x1 : box.x2]
    if crop.size == 0:
        return frame_bgr, RoiBox(0, 0, w, h)
    crop = _trim_sponsor_panel_right(crop)
    return crop, box


def game_crop_size(box: RoiBox) -> tuple[int, int]:
    """게임 crop (width, height)."""
    return max(1, box.x2 - box.x1), max(1, box.y2 - box.y1)
