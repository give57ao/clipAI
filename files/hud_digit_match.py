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

# --- R4 (2026-07-07): 숫자 CNN 판독 — IoU는 8을 0/6/9와 구분 못 해 EXCLUDE_DIGITS로
# 뺐던 근본 한계를 학습 분류기로 해소(train_hud_digit_cnn.py, held-out val_acc 99.7%,
# 실프레임 스팟체크로 8 고신뢰 판독 확인). IoU는 CNN 모델이 없을 때만 폴백.
_CNN_MODEL_PATH = Path(r"E:\Highlights\ml_dataset\models\hud_digit_clf_best.pt")
_CNN_MIN_P = 0.85
_cnn_model = None
_cnn_classes: list = []
_cnn_device = None


def _get_cnn():
    global _cnn_model, _cnn_classes, _cnn_device
    if _cnn_model is not None or not _CNN_MODEL_PATH.exists():
        return _cnn_model
    import json as _json

    import torch as _torch

    from train_hud_digit_cnn import TinyDigitCNN

    meta_path = _CNN_MODEL_PATH.parent / "hud_digit_clf_meta.json"
    meta = _json.loads(meta_path.read_text(encoding="utf-8"))
    _cnn_classes = [int(c) if c != "junk" else "junk" for c in meta["classes"]]
    _cnn_device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    model = TinyDigitCNN(num_classes=len(_cnn_classes)).to(_cnn_device)
    model.load_state_dict(_torch.load(_CNN_MODEL_PATH, map_location=_cnn_device))
    model.eval()
    _cnn_model = model
    return _cnn_model


def match_glyph_cnn(glyph: np.ndarray | None) -> tuple[int | None, float]:
    """CNN 분류 — junk 또는 최고확률 < `_CNN_MIN_P`면 None(판독실패, 안전 미스)."""
    import torch as _torch

    model = _get_cnn()
    if model is None or glyph is None:
        return None, 0.0
    x = glyph.astype(np.float32) / 255.0
    t = _torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(_cnn_device)
    with _torch.no_grad():
        p = _torch.softmax(model(t), dim=1)[0]
    top_i = int(_torch.argmax(p).item())
    top_p = float(p[top_i].item())
    label = _cnn_classes[top_i]
    if label == "junk" or top_p < _CNN_MIN_P:
        return None, top_p
    return label, top_p


def red_mask(crop_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 90, 90), (10, 255, 255))
    m2 = cv2.inRange(hsv, (168, 90, 90), (180, 255, 255))
    return cv2.bitwise_or(m1, m2)


# --- R8 (2026-07-14): 8 위상(topology) 프로브 — "구멍 2개"로 8을 직접 포착 ---
# 8은 EXCLUDE_DIGITS(IoU 템플릿 제외)라 지금껏 판독 불가였고, 화면의 8은 상습적으로
# '가짜 0'(conf 0.7~0.95)으로 오독돼 왔다(수많은 폭0 FP·미탐의 근원). 과거 두 실패
# (IoU 템플릿 부활→마진 역전으로 0/6/9 동반 파괴, CNN 폴백→흐릿한 0을 고신뢰 8로
# 오분류) 는 둘 다 "모양 전체의 닮음"에 의존한 접근이었다. 이번엔 위상 불변량:
# **8=구멍 2개, 0/6/9=1개, 7=0개** — 모양이 뭉개져도 0이 위조할 수 없는 구조.
# 단일 HSV 이진화(red_mask)가 8의 가운데 획을 뭉개는 게 오독의 기전이므로,
# 원본 색상에서 redness(R-max(G,B)) 맵을 만들어 **임계값 7단계를 훑으며** 구멍
# 수를 센다 — 8의 두 구멍이 살아나는 임계값 구간이 실측상 거의 항상 존재.
#
# 실측 검증 (2026-07-14, 사용자 육안확인 구간 337글리프, scratchpad probe8):
#   판독기가 '0'이라 한 글리프 중 — 진짜 8: 98%(64/65)가 구멍2 / 진짜 0: 0%(0/18)
#   대조군 — 7: 구멍0 100%, 9·10: 구멍2 오발 3%(2/69, 단발이라 지지필터가 흡수)
# 채택 규칙: max구멍==2 일 때만 8 (3개 이상은 노이즈로 보고 불채택).
_EIGHT_PROBE_THRS = (25, 40, 55, 70, 85, 100, 120)
_EIGHT_PROBE_CONF = 0.80  # 지지필터 단발예외(0.88) 미만 — 반드시 2회 이상 관측돼야 채택됨


def _count_holes(binimg: np.ndarray) -> int:
    """최대 전경 CC 내부의 '갇힌 배경 구멍' 수 (테두리 접촉 배경 제외). 전경 없으면 -1."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binimg, connectivity=8)
    if n < 2:
        return -1
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = stats[big, cv2.CC_STAT_LEFT]
    y = stats[big, cv2.CC_STAT_TOP]
    w = stats[big, cv2.CC_STAT_WIDTH]
    h = stats[big, cv2.CC_STAT_HEIGHT]
    sub = (labels[y : y + h, x : x + w] == big).astype(np.uint8)
    inv = (1 - sub).astype(np.uint8)
    ni, _li, si, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    holes = 0
    for i in range(1, ni):
        bx, by, bw, bh, area = si[i]
        if area < 2:
            continue
        if bx == 0 or by == 0 or bx + bw == w or by + bh == h:
            continue
        holes += 1
    return holes


def probe_eight_topology(raw_bgr_patch: np.ndarray | None) -> bool:
    """원본 BGR 글리프 패치가 '8'인지 위상으로 판정 (상단 R8 주석 참고).

    IoU가 0 또는 None을 반환한 K 글리프에만 호출할 것 — 확신 있는 다른 숫자
    판독을 뒤집는 용도가 아니다.
    """
    if raw_bgr_patch is None or raw_bgr_patch.size == 0 or raw_bgr_patch.ndim != 3:
        return False
    r = raw_bgr_patch[:, :, 2].astype(np.int16)
    g = raw_bgr_patch[:, :, 1].astype(np.int16)
    b = raw_bgr_patch[:, :, 0].astype(np.int16)
    red = np.clip(r - np.maximum(g, b), 0, 255).astype(np.uint8)
    max_holes = -1
    for thr in _EIGHT_PROBE_THRS:
        binimg = (red >= thr).astype(np.uint8) * 255
        h = _count_holes(binimg)
        if h > max_holes:
            max_holes = h
    return max_holes == 2


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
    차이날 때만 채택. 모호하면 None — 상태머신이 미스로 안전 처리.

    ⚠ 8-거부권(0 매칭 시 _removed_8 템플릿과 재대조해 근접하면 None) 시도·철회
    (2026-07-09 Fable): 프레임 검증으로 "화면의 8이 0으로 오독됨"은 확정했으나
    (05-26 26:10, 실제 K/D/A 8/9/0 → K가 conf 0.7대 '0'), 글리프 IoU 마진으로는
    분리 불가 실측 — 진짜 8의 (iou0−iou8) 마진 0.07~0.10 vs **깨끗한 진짜 0**의
    마진 0.001~0.05로 역전돼 있어 어떤 임계값도 진짜 0을 먼저 죽임. 대응은
    hud_round_settle._quarantine_zeros(K 단조성 도메인 규칙)로 이동 — 재시도 금지."""
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
