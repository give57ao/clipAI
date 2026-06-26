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

from game_frame import extract_game_crop_bgr
from nick_fuzzy import canonicalize_nick
from scouter_nick import _is_valid_player_nick, get_reader

# 플레이어 행 y 중심 (게임 crop 높이 비율) — 킬순 3행 × 양팀 동일 밴드
_ROW_Y_CENTERS = (0.414, 0.453, 0.493)
_ROW_HALF_H = 0.020  # 행 높이 여유 (소폭 확대)

# 팀별 x 구간 (게임 crop 너비 비율)
# 열 순서: 마크/게급 | 닉네임 | 클랜명 | 킬데스도움
# 이전 ROI가 '클랜명' 열을 읽었던 문제를 수정 (2026-06-26 재보정)
_RED_NICK  = (0.192, 0.312)   # 닉네임 열 (순위배지/아이콘 이후 시작)
_RED_K     = (0.400, 0.452)   # KDA 첫 번째(K) 값
_BLUE_NICK = (0.566, 0.668)   # 블루팀 닉네임 열 (아이콘 이후)
_BLUE_K    = (0.760, 0.815)   # 블루팀 KDA 첫 번째(K) 값

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


@dataclass
class ScoreboardWindow:
    start_sec: float
    end_sec: float

    @property
    def mid_sec(self) -> float:
        return (self.start_sec + self.end_sec) / 2.0


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
) -> list[ScoreboardRow]:
    """전체스코어 프레임에서 6행 닉(+K) OCR (game_roi crop 기준)."""
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    if game_bgr is None:
        game_bgr, _ = extract_game_crop_bgr(frame_bgr, dataset_root=dataset_root)

    gh, gw = game_bgr.shape[:2]

    rows: list[ScoreboardRow] = []
    row_index = 0
    for team, nick_x, k_x in (
        ("red", _RED_NICK, _RED_K),
        ("blue", _BLUE_NICK, _BLUE_K),
    ):
        for y_center in _ROW_Y_CENTERS:
            ny1, ny2, nx1, nx2 = _roi_to_slice(gh, gw, y_center, nick_x[0], nick_x[1])
            ky1, ky2, kx1, kx2 = _roi_to_slice(gh, gw, y_center, k_x[0], k_x[1])
            nick_text, nick_conf = _ocr_nick(game_bgr[ny1:ny2, nx1:nx2])
            k_text, k_conf = _ocr_crop(game_bgr[ky1:ky2, kx1:kx2])

            if _is_scoreboard_noise(nick_text):
                row_index += 1
                continue

            nick = canonicalize_nick(nick_text)
            rows.append(
                ScoreboardRow(
                    row_index=row_index,
                    team=team,
                    nickname=nick,
                    nick_conf=nick_conf,
                    kills=_parse_kills(k_text),
                    kills_conf=k_conf,
                )
            )
            row_index += 1
    return rows


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
