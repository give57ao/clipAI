# -*- coding: utf-8 -*-
"""Clan Match 전체스코어 6행 ROI + 닉/K OCR.

좌표는 game_roi ML 게임 crop 기준 상대 비율 (0~1).
캘리브레이션 기준: 2026-03-19 23-00-50.mp4 t≈136s.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from game_frame import extract_game_crop_bgr
from nick_fuzzy import canonicalize_nick
from scouter_nick import _is_valid_player_nick, get_reader

# 플레이어 행 y 중심 (게임 crop 높이 비율) — 킬순 3행 × 양팀 동일 밴드
_ROW_Y_CENTERS = (0.414, 0.453, 0.493)
_ROW_HALF_H = 0.020  # 행 높이 여유 (소폭 확대)

# game crop 너비로 레이아웃 분기 (player_identity·scouter와 동일 기준)
SPONSOR_PANEL_MAX_GAME_WIDTH = 1800

# 팀별 x 구간 (게임 crop 너비 비율) — layout별 테이블
# 열 순서: 마크/게급 | 닉네임 | 클랜명 | K | D | A | 킬바
_LAYOUT_ROI: dict[str, dict[str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]] = {
    # 풀스크린형 (gw≥1800) — 03-12-36·00-42-33 캘리브레이션
    "fullscreen": {
        "red": ((0.192, 0.312), (0.407, 0.426), (0.430, 0.450)),
        "blue": ((0.566, 0.668), (0.724, 0.743), (0.747, 0.767)),
    },
    # 후원패널형 (gw<1800) — 02-21-23 R110 닉 열 재캘리브 (클랜명 침범 방지)
    "sponsor": {
        "red": ((0.192, 0.312), (0.407, 0.426), (0.430, 0.450)),
        "blue": ((0.545, 0.625), (0.724, 0.743), (0.747, 0.767)),
    },
}

# 하위 호환·캘리브 스크립트용 (풀스크린 기본값)
_RED_NICK  = _LAYOUT_ROI["fullscreen"]["red"][0]
_RED_K     = _LAYOUT_ROI["fullscreen"]["red"][1]
_RED_D     = _LAYOUT_ROI["fullscreen"]["red"][2]
_BLUE_NICK = _LAYOUT_ROI["fullscreen"]["blue"][0]
_BLUE_K    = _LAYOUT_ROI["fullscreen"]["blue"][1]
_BLUE_D    = _LAYOUT_ROI["fullscreen"]["blue"][2]

_SCOREBOARD_NOISE = re.compile(
    r"(?i)(clan|match|mission|redtea|bluete|kill|death|head|nick|"
    r"닉|킬|데스|헤드|승리|패배|win|lose)"
)


@dataclass
class ScoreboardRow:
    row_index: int  # 0~5 (red 0~2, blue 3~5)
    team: str  # red | blue
    nickname: str
    nick_conf: float
    kills: int | None = None
    kills_conf: float = 0.0
    deaths: int | None = None
    deaths_conf: float = 0.0


@dataclass
class ScoreboardWindow:
    start_sec: float
    end_sec: float

    @property
    def mid_sec(self) -> float:
        return (self.start_sec + self.end_sec) / 2.0


def layout_from_game_width(game_width: int) -> str:
    """game_roi crop 너비 → 스코어보드 ROI 테이블 키."""
    if game_width > 0 and game_width < SPONSOR_PANEL_MAX_GAME_WIDTH:
        return "sponsor"
    return "fullscreen"


def team_rois_for_layout(layout: str) -> dict[str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
    return _LAYOUT_ROI.get(layout, _LAYOUT_ROI["fullscreen"])


def _roi_to_slice(
    game_h: int,
    game_w: int,
    y_center: float,
    x1_ratio: float,
    x2_ratio: float,
) -> tuple[int, int, int, int]:
    y1 = int(game_h * (y_center - _ROW_HALF_H))
    y2 = int(game_h * (y_center + _ROW_HALF_H))
    x1 = int(game_w * x1_ratio)
    x2 = int(game_w * x2_ratio)
    y1 = max(0, y1)
    y2 = min(game_h, max(y1 + 1, y2))
    x1 = max(0, x1)
    x2 = min(game_w, max(x1 + 1, x2))
    return y1, y2, x1, x2


def _preprocess_ocr(crop_bgr: np.ndarray) -> np.ndarray:
    """CLAHE 대비 보정 + 3x upscale."""
    up = cv2.resize(crop_bgr, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _ocr_best(img: np.ndarray) -> tuple[str, float]:
    """단일 이미지 OCR → 최고 신뢰도 결과."""
    results = get_reader().readtext(img, detail=1, paragraph=False)
    best_text, best_conf = "", 0.0
    for _, text, conf in results:
        t = str(text).strip()
        if float(conf) > best_conf and t:
            best_text, best_conf = t, float(conf)
    return best_text, best_conf


def _ocr_crop(crop_bgr: np.ndarray) -> tuple[str, float]:
    """K/숫자 전용 OCR (CLAHE)."""
    if crop_bgr.size == 0:
        return "", 0.0
    return _ocr_best(_preprocess_ocr(crop_bgr))


def _ocr_mask_digit(mask: np.ndarray) -> tuple[int | None, float]:
    """이진 마스크(흰글씨/검은배경)에서 최고 신뢰도 숫자."""
    mask = cv2.copyMakeBorder(mask, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=0)
    results = get_reader().readtext(
        mask, detail=1, paragraph=False, allowlist="0123456789"
    )
    best_val: int | None = None
    best_conf = 0.0
    for _, text, conf in results:
        digits = "".join(ch for ch in str(text) if ch.isdigit())
        if digits and float(conf) > best_conf:
            best_val, best_conf = int(digits), float(conf)
    return best_val, best_conf


def _red_watermark_ratio(crop_bgr: np.ndarray) -> float:
    """스코프 워터마크/킬바 등 고채도 빨강 픽셀 비율."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(crop_bgr)
    red = ((h < 15) | (h > 165)) & (s > 60) & (v > 40)
    red |= (
        (r.astype(np.int16) > g.astype(np.int16) + 35)
        & (r.astype(np.int16) > b.astype(np.int16) + 35)
        & (s > 40)
    )
    return float(red.mean())


def _ocr_kill_digit(crop_bgr: np.ndarray) -> tuple[int | None, float]:
    """K열 OCR — SNIPER 워터마크가 우측을 덮을 때 좌측 절반 폴백."""
    val, conf = _ocr_digit(crop_bgr)
    w = crop_bgr.shape[1] if crop_bgr is not None and crop_bgr.size else 0
    val2: int | None = None
    conf2 = 0.0
    if w >= 4:
        left = crop_bgr[:, : max(1, int(w * 0.52))]
        val2, conf2 = _ocr_digit(left)

    if val2 is not None and 0 <= val2 <= 40 and conf2 >= 0.45:
        # 워터마크가 8·9·91 등으로 오독 → 좌측 한 자리가 더 신뢰
        if val is None or val >= 15:
            return val2, conf2
        if (
            val >= 8
            and val < 10
            and val2 < val
            and (val - val2) >= 4
            and conf2 >= 0.5
        ):
            return val2, conf2

    if val is not None and 0 <= val <= 40 and conf >= 0.45:
        return val, conf
    if val2 is not None and 0 <= val2 <= 40 and conf2 >= 0.45:
        return val2, conf2
    return val, conf


def _ocr_death_digit(crop_bgr: np.ndarray) -> tuple[int | None, float]:
    """D열 OCR — 누적 10·20·30 오독 시 어시스트 열 혼입 의심."""
    val, conf = _ocr_digit(crop_bgr)
    if val is not None and val >= 10 and val % 10 == 0 and conf < 0.92:
        # 좌측으로 0.008 이동한 좁은 crop 재시도
        h, w = crop_bgr.shape[:2]
        x1 = max(0, int(w * 0.05))
        x2 = max(x1 + 1, int(w * 0.82))
        val2, conf2 = _ocr_digit(crop_bgr[:, x1:x2])
        if val2 is not None and 0 <= val2 <= 15 and conf2 >= conf:
            return val2, conf2
    return val, conf


def _ocr_digit(crop_bgr: np.ndarray) -> tuple[int | None, float]:
    """킬 단독 숫자 OCR — Otsu + 흰글씨분리 결합, 더 높은 신뢰도 채택.

    두 전처리를 병행하는 이유:
    - Otsu 이진화: 일반(어두운 배경) 행의 단일/2자리 숫자에 강함
    - 흰글씨 분리(HSV: 밝고 채도낮음): 본인/1위 행에 깔리는 **빨간 킬바**가
      숫자에 겹쳐 Otsu가 망가질 때(예: '3'→'93', '7'→없음) 흰 숫자만 남겨 복구
    K는 누적 킬이라 2자리(10·11·23…)가 나오므로 ROI는 2자리를 담는 폭으로 잡는다.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0
    up = cv2.resize(crop_bgr, None, fx=5.0, fy=5.0, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if (binimg > 127).mean() > 0.5:  # 흰 비율 과반이면 전경/배경 반전
        binimg = cv2.bitwise_not(binimg)
    val_otsu, conf_otsu = _ocr_mask_digit(binimg)

    hsv = cv2.cvtColor(up, cv2.COLOR_BGR2HSV)
    _, sat, val = cv2.split(hsv)
    white = ((val > 120) & (sat < 90)).astype("uint8") * 255  # 밝고 채도낮은 흰글씨만
    val_white, conf_white = _ocr_mask_digit(white)

    if conf_otsu >= conf_white:
        return val_otsu, conf_otsu
    return val_white, conf_white


def _ocr_nick(crop_bgr: np.ndarray) -> tuple[str, float]:
    """닉네임 전용 OCR: 반전 우선 시도 후 CLAHE 폴백.

    로컬 플레이어 행은 밝은 텍스트 on 어두운 배경 → 반전이 더 잘 읽힘.
    반전으로 conf≥0.35 이상이면 채택, 아니면 CLAHE 결과와 비교.
    """
    if crop_bgr.size == 0:
        return "", 0.0
    up = _preprocess_ocr(crop_bgr)
    up_inv = cv2.bitwise_not(up)

    text_inv, conf_inv = _ocr_best(up_inv)
    # 임계값을 낮춰 로컬 플레이어 행(밝은 글씨) 반전 결과도 활용
    if conf_inv >= 0.20:
        text_norm, conf_norm = _ocr_best(up)
        return (text_inv, conf_inv) if conf_inv >= conf_norm else (text_norm, conf_norm)
    # 반전 결과 없음 → CLAHE만 사용
    return _ocr_best(up)


def _parse_kills(text: str) -> int | None:
    nums = re.findall(r"\d+", text)
    if not nums:
        return None
    try:
        return int(nums[0])
    except ValueError:
        return None


def _is_scoreboard_noise(text: str) -> bool:
    t = text.strip()
    if not t or not _is_valid_player_nick(t):
        return True
    compact = t.replace(" ", "")
    if _SCOREBOARD_NOISE.search(compact):
        return True
    if re.fullmatch(r"[\d\s/\.]+", compact):
        return True
    if len(compact) <= 1:
        return True
    return False


def read_scoreboard_rows(
    frame_bgr: np.ndarray,
    *,
    dataset_root: Path | None = None,
    game_bgr: np.ndarray | None = None,
    layout: str | None = None,
) -> list[ScoreboardRow]:
    """전체스코어 프레임에서 6행 닉(+K) OCR (game_roi crop 기준)."""
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    if game_bgr is None:
        game_bgr, _ = extract_game_crop_bgr(frame_bgr, dataset_root=dataset_root)

    gh, gw = game_bgr.shape[:2]
    roi_layout = layout or layout_from_game_width(gw)
    team_rois = team_rois_for_layout(roi_layout)

    rows: list[ScoreboardRow] = []
    row_index = 0
    for team in ("red", "blue"):
        nick_x, k_x, d_x = team_rois[team]
        for y_center in _ROW_Y_CENTERS:
            ny1, ny2, nx1, nx2 = _roi_to_slice(gh, gw, y_center, nick_x[0], nick_x[1])
            ky1, ky2, kx1, kx2 = _roi_to_slice(gh, gw, y_center, k_x[0], k_x[1])
            dy1, dy2, dx1, dx2 = _roi_to_slice(gh, gw, y_center, d_x[0], d_x[1])
            nick_text, nick_conf = _ocr_nick(game_bgr[ny1:ny2, nx1:nx2])
            k_crop = game_bgr[ky1:ky2, kx1:kx2]
            d_crop = game_bgr[dy1:dy2, dx1:dx2]
            k_val, k_conf = _ocr_kill_digit(k_crop)
            d_val, d_conf = _ocr_death_digit(d_crop)

            nick_is_noise = _is_scoreboard_noise(nick_text)
            if nick_is_noise:
                nick = ""
                nick_conf = 0.0
            else:
                nick = canonicalize_nick(nick_text)

            rows.append(
                ScoreboardRow(
                    row_index=row_index,
                    team=team,
                    nickname=nick,
                    nick_conf=nick_conf,
                    kills=k_val,
                    kills_conf=k_conf,
                    deaths=d_val,
                    deaths_conf=d_conf,
                )
            )
            row_index += 1
    return rows


# 승리라운드 숫자 (팀 헤더 하단) — 캘리브레이션 2026-06-29
_RED_WINS_Y  = (0.365, 0.405)
_RED_WINS_X  = (0.26, 0.34)
_BLUE_WINS_Y = (0.355, 0.395)
_BLUE_WINS_X = (0.60, 0.68)


def _read_wins_digit(game_bgr: np.ndarray, team: str) -> int | None:
    """팀 승리라운드 숫자 OCR (0~20)."""
    h, w = game_bgr.shape[:2]
    if team == "red":
        y1, y2 = int(_RED_WINS_Y[0] * h), int(_RED_WINS_Y[1] * h)
        x1, x2 = int(_RED_WINS_X[0] * w), int(_RED_WINS_X[1] * w)
    else:
        y1, y2 = int(_BLUE_WINS_Y[0] * h), int(_BLUE_WINS_Y[1] * h)
        x1, x2 = int(_BLUE_WINS_X[0] * w), int(_BLUE_WINS_X[1] * w)
    crop = game_bgr[y1:y2, x1:x2]
    val, conf = _ocr_digit(crop)
    if val is not None and 0 <= val <= 20 and conf >= 0.35:
        return val
    return None


def read_team_wins(
    frame_bgr: np.ndarray,
    *,
    dataset_root: Path | None = None,
    game_bgr: np.ndarray | None = None,
) -> tuple[int | None, int | None]:
    """(red_wins, blue_wins) — 스코어보드 프레임에서 승리라운드."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None, None
    if game_bgr is None:
        game_bgr, _ = extract_game_crop_bgr(frame_bgr, dataset_root=dataset_root)
    return _read_wins_digit(game_bgr, "red"), _read_wins_digit(game_bgr, "blue")


def team_kills_sum(rows: list[ScoreboardRow], team: str) -> int | None:
    """팀 3행 K 합산 — 개별 OCR 오류(>40) 제외 후 합산."""
    vals = [
        r.kills for r in rows
        if r.team == team and r.kills is not None and 0 <= r.kills <= 40
    ]
    if len(vals) < 2:
        return None
    return sum(vals)


def load_scoreboard_windows(csv_path: Path) -> list[ScoreboardWindow]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    windows: list[ScoreboardWindow] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = float(row["start_sec"])
                end = float(row["end_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            if end > start:
                windows.append(ScoreboardWindow(start, end))
    return windows


def find_scoreboard_csv(video_path: Path, rounds_dir: Path | None, dataset_root: Path) -> Path | None:
    if rounds_dir:
        candidate = Path(rounds_dir) / "detected_scoreboards.csv"
        if candidate.exists():
            return candidate
    auto = dataset_root / "rounds" / video_path.stem / "detected_scoreboards.csv"
    return auto if auto.exists() else None


def collect_scoreboard_nick_votes(
    cap: cv2.VideoCapture,
    fps: float,
    windows: list[ScoreboardWindow],
    *,
    ocr_min_conf: float = 0.35,
    max_windows: int = 20,
    dataset_root: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """소스 B: 스코어보드 중앙 프레임 6행 닉 투표 + 프레임별 닉 목록(교차검증용)."""
    votes: list[dict] = []
    frame_nicks: list[dict] = []

    if not windows:
        return votes, frame_nicks

    picked = windows
    if len(windows) > max_windows:
        step = len(windows) / max_windows
        idxs = sorted({int(i * step) for i in range(max_windows)})
        picked = [windows[i] for i in idxs if i < len(windows)]

    for window in picked:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(window.mid_sec * fps))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        sb_rows = read_scoreboard_rows(frame, dataset_root=dataset_root)
        nicks = [r.nickname for r in sb_rows if r.nick_conf >= ocr_min_conf]
        frame_nicks.append({"sec": round(window.mid_sec, 1), "nicks": nicks})

        for row in sb_rows:
            if row.nick_conf < ocr_min_conf:
                continue
            votes.append(
                {
                    "text": row.nickname,
                    "weight": 0.5,
                    "conf": row.nick_conf,
                    "mode": "scoreboard",
                    "kind": "scoreboard",
                    "team": row.team,
                    "row_index": row.row_index,
                }
            )
    return votes, frame_nicks
