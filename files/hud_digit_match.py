# -*- coding: utf-8 -*-
"""HUD 숫자 템플릿 매칭 — EasyOCR 대비 빠르고 K 슬롯 정확도 향상."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DEFAULT_TEMPLATE_DIR = Path(__file__).parent / "hud_templates"

_GLYPH_SIZE = (24, 32)
_MATCH_MIN_SCORE = 0.52
_K_MATCH_MIN = 0.50
# IoU 매칭 (K 숫자 경로) — TM_CCOEFF는 박스형 폰트(0/5/6)에서 변별력 부족(실측)
_K_IOU_MIN = 0.55
_K_IOU_MARGIN = 0.06   # 1등-2등 IoU 차이가 이보다 작으면 모호 → None (안전 미스)


def red_mask(crop_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 90, 90), (10, 255, 255))
    m2 = cv2.inRange(hsv, (168, 90, 90), (180, 255, 255))
    return cv2.bitwise_or(m1, m2)


def white_mask(crop_bgr: np.ndarray, thresh: int = 155) -> np.ndarray:
    up = cv2.resize(crop_bgr, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    _, binimg = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    if (binimg > 127).mean() > 0.5:
        binimg = cv2.bitwise_not(binimg)
    return binimg


def normalize_glyph(mask: np.ndarray) -> np.ndarray | None:
    """글리프 마스크 → 24x32 letterbox (가로세로비 보존, 중앙 정렬).

    스트레치 방식은 '1'도 꽉 찬 블록이 되어 숫자 간 변별력이 사라짐(2026-07-06).
    """
    if mask is None or mask.size == 0:
        return None
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w * h < 20:
        return None
    g = mask[y : y + h, x : x + w]
    tw, th = _GLYPH_SIZE
    nw = max(1, min(tw, int(round(w * th / h))))
    g = cv2.resize(g, (nw, th), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw), dtype=np.uint8)
    x0 = (tw - nw) // 2
    canvas[:, x0 : x0 + nw] = g
    return canvas


def leftmost_red_glyph(crop_bgr: np.ndarray) -> np.ndarray | None:
    """K 슬롯 — 빨간 마스크에서 가장 왼쪽 글리프."""
    mask = red_mask(crop_bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    h, w = mask.shape[:2]
    candidates = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh < 15 or bh < h * 0.25 or bw > w * 0.85:
            continue
        candidates.append((x, mask[y : y + bh, x : x + bw]))
    if not candidates:
        return normalize_glyph(mask)
    candidates.sort(key=lambda t: t[0])
    return cv2.resize(candidates[0][1], _GLYPH_SIZE, interpolation=cv2.INTER_AREA)


def match_glyph(
    glyph: np.ndarray | None,
    templates: dict,
    min_score: float,
) -> tuple[int | None, float]:
    """키는 int 또는 '3_b' 같은 변형 문자열 — 앞 숫자만 취함 (레이아웃별 보조 템플릿)."""
    if glyph is None or not templates:
        return None, 0.0
    best_key = None
    best_score = -1.0
    for key, tmpl in templates.items():
        if tmpl.shape != glyph.shape:
            tmpl = cv2.resize(tmpl, _GLYPH_SIZE, interpolation=cv2.INTER_AREA)
        score = float(cv2.matchTemplate(glyph, tmpl, cv2.TM_CCOEFF_NORMED).max())
        if score > best_score:
            best_score, best_key = score, key
    if best_key is not None and best_score >= min_score:
        return int(str(best_key).split("_")[0]), best_score
    return None, max(0.0, best_score)


def _iou_shifted(a: np.ndarray, b: np.ndarray) -> float:
    """이진 글리프 IoU — letterbox 정렬 지터(±1px) 흡수를 위해 x/y 시프트 탐색."""
    ab = a > 127
    best = 0.0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            bb = np.roll(b, (dy, dx), axis=(0, 1)) > 127
            union = np.logical_or(ab, bb).sum()
            if union == 0:
                continue
            iou = np.logical_and(ab, bb).sum() / union
            if iou > best:
                best = float(iou)
    return best


def match_glyph_iou(
    glyph: np.ndarray | None,
    templates: dict,
    min_score: float = _K_IOU_MIN,
    min_margin: float = _K_IOU_MARGIN,
) -> tuple[int | None, float]:
    """IoU + 마진 매칭: 1등이 min_score 이상이고 2등(다른 숫자)과 min_margin 이상
    차이날 때만 채택. 모호하면 None — 상태머신이 미스로 안전 처리."""
    if glyph is None or not templates:
        return None, 0.0
    by_digit: dict[int, float] = {}
    for key, tmpl in templates.items():
        if tmpl.shape != glyph.shape:
            tmpl = cv2.resize(tmpl, _GLYPH_SIZE, interpolation=cv2.INTER_AREA)
        digit = int(str(key).split("_")[0])
        s = _iou_shifted(glyph, tmpl)
        if s > by_digit.get(digit, 0.0):
            by_digit[digit] = s
    ranked = sorted(by_digit.items(), key=lambda kv: -kv[1])
    best_digit, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if best >= min_score and (best - second) >= min_margin:
        return best_digit, best
    return None, best


class HudDigitMatcher:
    """빨간 K·흰 상단스코어 숫자 템플릿 매칭."""

    def __init__(
        self,
        k_templates: dict[int, np.ndarray] | None = None,
        white_templates: dict[int, np.ndarray] | None = None,
    ):
        self.k_templates = k_templates or {}
        self.white_templates = white_templates or {}

    @classmethod
    def from_template_dir(cls, template_dir: Path | None = None) -> HudDigitMatcher:
        d = Path(template_dir) if template_dir else DEFAULT_TEMPLATE_DIR
        k_t: dict[int, np.ndarray] = {}
        w_t: dict[int, np.ndarray] = {}
        if d.is_dir():
            for p in d.glob("k_*.png"):
                try:
                    key = p.stem.split("_", 1)[1]  # "3" 또는 "3_b" (변형)
                    int(key.split("_")[0])  # 숫자 검증
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        k_t[key] = cv2.resize(img, _GLYPH_SIZE, interpolation=cv2.INTER_AREA)
                except ValueError:
                    pass
            for p in d.glob("white_*.png"):
                try:
                    digit = int(p.stem.split("_", 1)[1])
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        w_t[digit] = cv2.resize(img, _GLYPH_SIZE, interpolation=cv2.INTER_AREA)
                except ValueError:
                    pass
        return cls(k_templates=k_t, white_templates=w_t)

    def save_templates(self, template_dir: Path | None = None) -> None:
        d = Path(template_dir) if template_dir else DEFAULT_TEMPLATE_DIR
        d.mkdir(parents=True, exist_ok=True)
        for digit, img in self.k_templates.items():
            cv2.imwrite(str(d / f"k_{digit}.png"), img)
        for digit, img in self.white_templates.items():
            cv2.imwrite(str(d / f"white_{digit}.png"), img)

    def read_k(self, crop_bgr: np.ndarray) -> tuple[int | None, float, str]:
        glyph = leftmost_red_glyph(crop_bgr)
        digit, score = match_glyph(glyph, self.k_templates, _K_MATCH_MIN)
        if digit is not None:
            return digit, score, "template"
        return None, max(0.0, score), "template_miss"

    def read_white_digit(self, crop_bgr: np.ndarray) -> tuple[int | None, float, str]:
        mask = white_mask(crop_bgr)
        glyph = normalize_glyph(mask)
        digit, score = match_glyph(glyph, self.white_templates, _MATCH_MIN_SCORE)
        if digit is not None:
            return digit, score, "template"
        return None, max(0.0, score), "template_miss"

    def read_white_score3(self, crop_bgr: np.ndarray) -> tuple[int | None, float]:
        if crop_bgr is None or crop_bgr.size == 0 or not self.white_templates:
            return None, 0.0
        h, w = crop_bgr.shape[:2]
        digits: list[int] = []
        scores: list[float] = []
        for i in range(3):
            sl = crop_bgr[:, int(w * i / 3) : int(w * (i + 1) / 3)]
            d, sc, _ = self.read_white_digit(sl)
            if d is None:
                return None, 0.0
            digits.append(d)
            scores.append(sc)
        val = digits[0] * 100 + digits[1] * 10 + digits[2]
        return val, min(scores)


def calibrate_from_video(
    video_path: Path,
    *,
    samples: list[tuple[float, int]],
    dataset_root: Path | None = None,
    k_crop_fn=None,
    white_crop_fn=None,
    white_samples: list[tuple[float, str, int]] | None = None,
) -> HudDigitMatcher:
    from game_frame import extract_game_crop_bgr

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    k_templates: dict[int, np.ndarray] = {}
    white_templates: dict[int, np.ndarray] = {}

    for t_sec, label in samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
        ok, frame = cap.read()
        if not ok or k_crop_fn is None:
            continue
        game, _ = extract_game_crop_bgr(frame, dataset_root=dataset_root)
        glyph = leftmost_red_glyph(k_crop_fn(game))
        if glyph is not None and label not in k_templates:
            k_templates[label] = glyph.copy()

    if white_samples and white_crop_fn:
        for t_sec, slot, exp in white_samples:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
            ok, frame = cap.read()
            if not ok:
                continue
            game, _ = extract_game_crop_bgr(frame, dataset_root=dataset_root)
            box = white_crop_fn(game, slot)
            bh, bw = box.shape[:2]
            for i in range(3):
                sl = box[:, int(bw * i / 3) : int(bw * (i + 1) / 3)]
                g = normalize_glyph(white_mask(sl))
                if g is None:
                    continue
                digit = (exp // (100 ** (2 - i))) % 10
                if digit not in white_templates:
                    white_templates[digit] = g.copy()

    cap.release()
    return HudDigitMatcher(k_templates=k_templates, white_templates=white_templates)


_matcher: HudDigitMatcher | None = None


def get_hud_digit_matcher() -> HudDigitMatcher:
    global _matcher
    if _matcher is None:
        _matcher = HudDigitMatcher.from_template_dir()
        if not _matcher.k_templates:
            # 자동 캘리브(_build_default_matcher)는 구 고정 ROI 기반이라
            # 조각 글리프를 템플릿으로 만드는 함정 → 사용 금지 (2026-07-06).
            # harvest_hud_digits.py 로 수확·설치할 것.
            print(
                "[hud_digit_match] ⚠ hud_templates/ 비어있음 — "
                "harvest_hud_digits.py --harvest/--cluster/--install 필요",
                flush=True,
            )
    return _matcher


def _build_default_matcher() -> HudDigitMatcher:
    from hud_kda import (
        _KDA_LINE_X,
        _KDA_LINE_Y,
        _MID_ROUND_X,
        _RED_WINS_X,
        _BLUE_WINS_X,
        _TOP_Y,
        _crop_ratio,
        _kda_numbers_crop,
        _split_kda_crops,
    )

    vp = Path(r"D:\2026-01-08 02-33-22.mp4")
    if not vp.exists():
        return HudDigitMatcher()

    ds = Path(r"E:\Highlights\ml_dataset")

    def k_crop(game):
        nums = _kda_numbers_crop(game)
        kc, _, _ = _split_kda_crops(nums)
        return kc

    def white_crop(game, slot):
        xs = {"R": _RED_WINS_X, "M": _MID_ROUND_X, "B": _BLUE_WINS_X}
        return _crop_ratio(game, _TOP_Y, xs[slot])

    k_samples = [
        (60, 0), (120, 1), (124, 1), (150, 0), (240, 0), (306, 3), (488, 3),
    ]
    white_samples = [
        (120, "R", 0), (120, "M", 5), (120, "B", 3),
        (300, "R", 0), (300, "M", 5), (300, "B", 5),
    ]
    m = calibrate_from_video(
        vp,
        samples=k_samples,
        dataset_root=ds,
        k_crop_fn=k_crop,
        white_crop_fn=white_crop,
        white_samples=white_samples,
    )
    m.save_templates(DEFAULT_TEMPLATE_DIR)
    return m


def reset_matcher() -> None:
    global _matcher
    _matcher = None
