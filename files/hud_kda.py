# -*- coding: utf-8 -*-
"""라이브 HUD OCR — K/D/A 3분할 + 템플릿 매칭."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from game_frame import extract_game_crop_bgr
from hud_digit_match import get_hud_digit_matcher, red_mask
from scoreboard_layout import _ocr_mask_digit, get_reader

_TOP_Y = (0.038, 0.092)
_RED_WINS_X = (0.392, 0.448)
_MID_ROUND_X = (0.448, 0.504)
_BLUE_WINS_X = (0.504, 0.560)

_KDA_LINE_Y = (0.235, 0.305)
_KDA_LINE_X = (0.055, 0.220)
_KDA_LABEL_SKIP = 0.30

_KDA_NUM_Y = _KDA_LINE_Y
_KDA_NUM_X = _KDA_LINE_X
_K_ONLY_Y = _KDA_LINE_Y
_K_ONLY_X = _KDA_LINE_X

_MAX_ROUND_K = 15
_MIN_K_CONF = 0.25


@dataclass
class HudSnapshot:
    time_sec: float
    red_wins: int | None
    blue_wins: int | None
    round_index: int | None
    round_k: int | None
    round_d: int | None
    round_a: int | None
    k_conf: float = 0.0
    k_method: str = ""
    source: str = ""


def _crop_ratio(img: np.ndarray, y: tuple[float, float], x: tuple[float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    y1, y2 = int(y[0] * h), int(y[1] * h)
    x1, x2 = int(x[0] * w), int(x[1] * w)
    if y2 <= y1 or x2 <= x1:
        return np.array([])
    return img[y1:y2, x1:x2]


def _kda_numbers_crop(game_bgr: np.ndarray) -> np.ndarray:
    line = _crop_ratio(game_bgr, _KDA_LINE_Y, _KDA_LINE_X)
    if line.size == 0:
        return line
    w = line.shape[1]
    return line[:, int(w * _KDA_LABEL_SKIP) :]


def _split_kda_crops(nums_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if nums_bgr is None or nums_bgr.size == 0:
        empty = np.array([])
        return empty, empty, empty
    w = nums_bgr.shape[1]
    return nums_bgr[:, : w // 3], nums_bgr[:, w // 3 : 2 * w // 3], nums_bgr[:, 2 * w // 3 :]


def _ocr_k_easyocr(crop_bgr: np.ndarray) -> tuple[int | None, float]:
    if crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0
    mask = red_mask(crop_bgr)
    up = cv2.resize(mask, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
    up = cv2.copyMakeBorder(up, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=0)
    results = get_reader().readtext(up, detail=1, paragraph=False, allowlist="0123456789")
    best_val: int | None = None
    best_conf = 0.0
    for _, text, conf in results:
        t = str(text).strip()
        if t.isdigit() and len(t) == 1:
            v = int(t)
            if 0 <= v <= _MAX_ROUND_K and float(conf) > best_conf:
                best_val, best_conf = v, float(conf)
    if best_val is not None and best_conf >= _MIN_K_CONF:
        return best_val, best_conf
    val, conf = _ocr_mask_digit(up)
    if val is not None and 0 <= val <= _MAX_ROUND_K and conf >= _MIN_K_CONF:
        return val, conf
    return None, 0.0


def read_k_digit(crop_bgr: np.ndarray, *, template_only: bool = False) -> tuple[int | None, float, str]:
    if crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0, ""
    matcher = get_hud_digit_matcher()
    k, sc, method = matcher.read_k(crop_bgr)
    if k is not None and 0 <= k <= _MAX_ROUND_K:
        return k, sc, method
    if template_only:
        return None, max(sc, 0.0), "template_miss"
    k2, sc2 = _ocr_k_easyocr(crop_bgr)
    if k2 is not None:
        return k2, sc2, "easyocr"
    return None, max(sc, 0.0), "miss"


def read_kda_triple_from_game(
    game_bgr: np.ndarray,
    *,
    template_only: bool = False,
) -> tuple[int | None, int | None, int | None, float, str]:
    nums = _kda_numbers_crop(game_bgr)
    kc, dc, ac = _split_kda_crops(nums)
    k, kconf, km = read_k_digit(kc, template_only=template_only)
    if template_only:
        return k, None, None, kconf, km
    d, _, _ = read_k_digit(dc)
    a, _, _ = read_k_digit(ac)
    return k, d, a, kconf, km


def read_top_hud_scores(game_bgr: np.ndarray) -> tuple[int | None, int | None, int | None]:
    matcher = get_hud_digit_matcher()
    crops = [
        _crop_ratio(game_bgr, _TOP_Y, _RED_WINS_X),
        _crop_ratio(game_bgr, _TOP_Y, _MID_ROUND_X),
        _crop_ratio(game_bgr, _TOP_Y, _BLUE_WINS_X),
    ]
    vals = []
    for c in crops:
        v, _ = matcher.read_white_score3(c)
        vals.append(v)
    return vals[0], vals[1], vals[2]


def read_round_k(game_bgr: np.ndarray, *, template_only: bool = False) -> tuple[int | None, float, str]:
    k, _, _, conf, method = read_kda_triple_from_game(game_bgr, template_only=template_only)
    return k, conf, method


def read_hud_snapshot(
    frame_bgr: np.ndarray,
    time_sec: float,
    *,
    dataset_root=None,
    game_bgr: np.ndarray | None = None,
) -> HudSnapshot:
    if game_bgr is None:
        game_bgr, _ = extract_game_crop_bgr(frame_bgr, dataset_root=dataset_root)
    red, mid, blue = read_top_hud_scores(game_bgr)
    k, d, a, k_conf, k_method = read_kda_triple_from_game(game_bgr)
    return HudSnapshot(
        time_sec=time_sec,
        red_wins=red,
        blue_wins=blue,
        round_index=mid,
        round_k=k,
        round_d=d,
        round_a=a,
        k_conf=k_conf,
        k_method=k_method,
        source="game_crop",
    )
