# -*- coding: utf-8 -*-
"""라이브 HUD OCR — K/D/A 자동 위치 탐지(빨간 blob 행) + 템플릿 매칭.

2026-07-06 개편: 고정 ROI(_KDA_LINE_*)는 16:9 캘리브 영상에서만 우연히
걸리는 수준이었고 OBS 4:3 레이아웃에선 숫자를 완전히 벗어남(판독률 0.6%).
→ 좌측 밴드에서 빨간 글리프 행을 찾아 슬래시(/) 2개로 K/D/A를 분리하는
`locate_kda_glyphs()`가 기본 경로. 다자리 K(누적 10+)도 지원.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from game_frame import extract_game_crop_bgr
from hud_digit_match import get_hud_digit_matcher, normalize_glyph, red_mask
from scoreboard_layout import _ocr_mask_digit, get_reader

# 2026-07-13 재보정: 기존 값은 구 16:9 캘리브 영상 기준으로 어긋나 있어 실제로는
# 숫자 박스가 아니라 그 아래 팀 인원 아이콘 줄을 잡고 있었음(OBS 4:3 game crop
# 실측으로 재측정, 판정 대상 = 상단 라이브 HUD 팀별 라운드 승수 박스).
_TOP_Y = (0.0, 0.036)
_RED_WINS_X = (0.382, 0.458)
_MID_ROUND_X = (0.462, 0.538)
_BLUE_WINS_X = (0.542, 0.618)

_KDA_LINE_Y = (0.235, 0.305)
_KDA_LINE_X = (0.055, 0.220)
_KDA_LABEL_SKIP = 0.30

_KDA_NUM_Y = _KDA_LINE_Y
_KDA_NUM_X = _KDA_LINE_X
_K_ONLY_Y = _KDA_LINE_Y
_K_ONLY_X = _KDA_LINE_X

_MAX_ROUND_K = 15
_MIN_K_CONF = 0.25

# --- KDA 행 자동 탐지 (레이아웃 불문) ---
# 좌측 밴드: 두 레이아웃 모두 숫자 행이 game 비율 gy≈0.272~0.295, gx 0.03~0.11
_KDA_BAND_Y = (0.20, 0.34)
_KDA_BAND_X = (0.0, 0.22)
_GLYPH_H_MIN = 0.011   # game 높이 대비 글리프 높이 (실측 15~20px @ ~1050p)
_GLYPH_H_MAX = 0.024
_GLYPH_W_MAX = 0.020   # game 높이 대비 폭 상한 (배너 등 큰 blob 배제)
_GLYPH_MIN_AREA = 12
_ROW_Y_TOL = 4         # 같은 행 판정 y 오차(px)
_SLASH_SLOPE = -0.30   # 행별 x-centroid 기울기: '/'는 ≈-0.45, 숫자는 |s|<0.1
_MAX_GROUP_DIGITS = 2  # K/D/A 각 슬롯 최대 자릿수 (누적 K 2자리까지)


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


@dataclass
class KdaGlyphs:
    """탐지된 K/D/A 글리프(이진 마스크 패치, 좌→우) — 값 판독 전 단계.

    k_raw: K 글리프의 **원본 BGR 크롭**(패딩 2px) — R8 8-위상 프로브용
    (`hud_digit_match.probe_eight_topology`). 이진 마스크는 8의 가운데 획을
    뭉개므로 위상 판정은 원본 색상에서 해야 한다.
    """
    k: list[np.ndarray] = field(default_factory=list)
    d: list[np.ndarray] = field(default_factory=list)
    a: list[np.ndarray] = field(default_factory=list)
    row_gy: float = 0.0  # game 비율 y (디버그)
    k_raw: list[np.ndarray] = field(default_factory=list)


def _glyph_slope(patch: np.ndarray) -> float:
    """행별 x-centroid 기울기. '/'는 강한 음수(≈-0.45), 숫자는 ≈0."""
    ys, xs = np.nonzero(patch)
    if len(xs) < 8:
        return 0.0
    rows: dict[int, list[int]] = {}
    for x, y in zip(xs, ys):
        rows.setdefault(int(y), []).append(int(x))
    if len(rows) < 4:
        return 0.0
    ys_s = sorted(rows)
    cx = [float(np.mean(rows[y])) for y in ys_s]
    return float(np.polyfit(ys_s, cx, 1)[0])


def locate_kda_glyphs(game_bgr: np.ndarray) -> KdaGlyphs | None:
    """좌측 밴드에서 빨간 K/D/A 행을 찾아 슬래시 2개 기준으로 3그룹 분리.

    실패(행 미발견·슬래시≠2·그룹 크기 이상) 시 None — 안전한 miss.
    """
    if game_bgr is None or game_bgr.size == 0:
        return None
    gh, gw = game_bgr.shape[:2]
    y1, y2 = int(_KDA_BAND_Y[0] * gh), int(_KDA_BAND_Y[1] * gh)
    x1, x2 = int(_KDA_BAND_X[0] * gw), int(_KDA_BAND_X[1] * gw)
    band = game_bgr[y1:y2, x1:x2]
    if band.size == 0:
        return None
    mask = red_mask(band)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    hmin, hmax = gh * _GLYPH_H_MIN, gh * _GLYPH_H_MAX
    blobs = []
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if area < _GLYPH_MIN_AREA or not (hmin <= bh <= hmax) or bw > gh * _GLYPH_W_MAX:
            continue
        patch = (labels[by : by + bh, bx : bx + bw] == i).astype(np.uint8) * 255
        blobs.append((int(bx), int(by), int(bw), int(bh), patch))
    if len(blobs) < 5:
        return None

    # 같은 행 클러스터 (y 오차 _ROW_Y_TOL) 중 최대
    best_row: list = []
    for anchor in blobs:
        row = [b for b in blobs if abs(b[1] - anchor[1]) <= _ROW_Y_TOL]
        if len(row) > len(best_row):
            best_row = row
    if len(best_row) < 5:
        return None
    best_row.sort(key=lambda b: b[0])

    slash_idx = [
        i
        for i, b in enumerate(best_row)
        if _glyph_slope(b[4]) < _SLASH_SLOPE and (b[2] / max(1, b[3])) < 0.75
    ]
    if len(slash_idx) != 2:
        return None
    s1, s2 = slash_idx
    groups = [best_row[:s1], best_row[s1 + 1 : s2], best_row[s2 + 1 :]]
    if any(not (1 <= len(g) <= _MAX_GROUP_DIGITS) for g in groups):
        return None
    k_raw = []
    for (bx, by, bw, bh, _p) in groups[0]:
        pad = 2
        rx1, ry1 = max(0, bx - pad), max(0, by - pad)
        rx2 = min(band.shape[1], bx + bw + pad)
        ry2 = min(band.shape[0], by + bh + pad)
        k_raw.append(band[ry1:ry2, rx1:rx2].copy())
    return KdaGlyphs(
        k=[b[4] for b in groups[0]],
        d=[b[4] for b in groups[1]],
        a=[b[4] for b in groups[2]],
        row_gy=(y1 + best_row[0][1]) / gh,
        k_raw=k_raw,
    )


def _read_digit_group(
    patches: list[np.ndarray],
    matcher,
    raw_patches: list[np.ndarray] | None = None,
) -> tuple[int | None, float]:
    """글리프 마스크 리스트(좌→우) → 다자리 정수. 하나라도 미매칭이면 None.

    raw_patches (R8, 2026-07-14): 원본 BGR 크롭 — 전달되면(K 슬롯) IoU가 0 또는
    None을 반환한 자리에서 8-위상 프로브(`probe_eight_topology`)를 추가로 시도.
    실측 근거·채택 규칙은 `hud_digit_match.py` R8 주석 참고. 확신 있는 다른 숫자
    판독은 절대 뒤집지 않음(0/None만 프로브) — R4 CNN 실패의 재발 방지 원칙.

    ⚠ R4(2026-07-07) CNN 폴백 실험 — 순손실로 되돌림(현재 비활성):
    "IoU가 None이면 CNN에 물어 8만 채택"으로 통합했다가 recall 88.5%→80.8%
    순손실 실측(TP 23→21). 원인: IoU가 None을 반환하는 건 흔한 정상 노이즈
    프레임(모션블러 등)이지 "진짜 8" 신호가 아닌데, 그때 CNN이 **고신뢰(0.90~0.98)로
    "0"을 "8"로 오분류**하는 경우가 많아 원래 안정적인 0 구간을 스퓨리어스 8로
    오염시킴(00-40-56 24:04~24:12 실측 — 원래 전부 "0"이어야 할 자리에 8이 8회
    끼어듦). 신뢰도 임계값을 올려도 해결 안 됨 — 오분류 자체가 고신뢰이기 때문
    (모델이 "모른다"를 표현 못 함, 8 표본 36개가 여전히 협소).
    54:20·79:51은 실제로 복원됐지만(사용자 확인 케이스), 다른 4곳(00-40-56 24:10,
    02-21-23 64:54, 02-34-09 2:20·51:40)이 새로 깨져 순손실 — `_tp_diff` 확인 완료.
    재시도하려면: (a) 단일 프레임이 아니라 연속 2프레임 이상 CNN이 8로 동의할 때만
    채택(시간적 corroboration), (b) IoU가 실패하는 지점들을 모아 그 실제 정답
    라벨(대개 0/6/9)을 하드네거티브로 재학습에 추가 — 8만 편식된 데이터셋 탈피.
    `train_hud_digit_cnn.py`/`hud_digit_match.match_glyph_cnn`은 인프라로 보존.
    """
    from hud_digit_match import _EIGHT_PROBE_CONF, match_glyph_iou, probe_eight_topology

    digits: list[int] = []
    min_score = 1.0
    for gi, p in enumerate(patches):
        glyph = normalize_glyph(p)
        d, sc = match_glyph_iou(glyph, matcher.k_templates)
        if raw_patches is not None and gi < len(raw_patches) and (d is None or d == 0):
            # R8: '0' 또는 미판독 자리만 8-위상 프로브 (다른 확신 판독은 불변)
            if probe_eight_topology(raw_patches[gi]):
                d, sc = 8, _EIGHT_PROBE_CONF
        if d is None:
            return None, max(0.0, sc)
        digits.append(d)
        min_score = min(min_score, sc)
    val = 0
    for d in digits:
        val = val * 10 + d
    return val, min_score


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


_MAX_CUM_K = 60  # 매치 누적 K 상한 (오독 가드)


def read_kda_triple_from_game(
    game_bgr: np.ndarray,
    *,
    template_only: bool = False,
) -> tuple[int | None, int | None, int | None, float, str]:
    """KDA 행 자동 탐지 → 템플릿 판독. K는 매치 누적값(다자리 가능)."""
    glyphs = locate_kda_glyphs(game_bgr)
    if glyphs is None:
        return None, None, None, 0.0, "row_miss"
    matcher = get_hud_digit_matcher()
    k, kconf = _read_digit_group(glyphs.k, matcher, raw_patches=glyphs.k_raw)
    if k is not None and not (0 <= k <= _MAX_CUM_K):
        k, kconf = None, 0.0
    km = "template" if k is not None else "template_miss"
    if template_only:
        return k, None, None, kconf, km
    d, _ = _read_digit_group(glyphs.d, matcher)
    a, _ = _read_digit_group(glyphs.a, matcher)
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
