# -*- coding: utf-8 -*-
"""B안: 스카우터 패널에서 플레이어 닉네임 추출.

핵심 처리:
1) game_roi ML로 게임 화면 crop (레이아웃 무관)
2) 게임 crop 기준 우하단 스카우터 패널 crop
3) 헤더 판별: "스카우터2" → 맨 위 행 = 본인 / "스카우터" → 점(●) 행 = 본인
   ※ 닉 문자열이 `null`이면 진짜 닉이름 null (프로그래밍 널값 아님)
4) 특수문자 닉 포함 OCR
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from game_frame import extract_game_crop_bgr

# 게임 crop 기준 스카우터 패널 상대 좌표 (0~1)
PANEL_X1 = 0.78
PANEL_X2 = 0.995
PANEL_Y1 = 0.61
PANEL_Y2 = 0.80

# null 닉 OCR 변형 (진짜 닉이름 "null")
_NULL_VARIANTS = {
    "null", "nul", "nuli", "nu11", "nu1l", "nill", "mull", "nu", "jyu", "nuii",
}

# 헤더 괄호 잔여 OCR: (L), {L), [L] 등
_HEADER_FRAGMENT_RE = re.compile(r"^[\{\[\(]?[lL1Iİ\|]+[\}\]\)]?$")

_reader = None


def get_reader():
    global _reader
    if _reader is None:
        import easyocr

        _reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
    return _reader


@dataclass
class ScouterReadout:
    mode: str  # "scouter2" | "scouter" | "unknown"
    player_nick: str
    player_conf: float
    is_spectator: bool  # 스카우터(점 기반) vs 스카우터2(맨 위) — 둘 다 본인 닉
    rows: list[tuple[str, float]] = field(default_factory=list)
    game_width: int = 0


def detect_game_width(frame_bgr: np.ndarray) -> int:
    """게임 화면 폭(px). game_roi crop 기준 (하위 호환·레이아웃 표시용)."""
    game_bgr, _ = extract_game_crop_bgr(frame_bgr)
    return int(game_bgr.shape[1])


def crop_scouter_panel(game_bgr: np.ndarray) -> np.ndarray:
    """게임 crop에서 스카우터 패널 영역 추출."""
    gh, gw = game_bgr.shape[:2]
    x1 = int(gw * PANEL_X1)
    x2 = int(gw * PANEL_X2)
    y1 = int(gh * PANEL_Y1)
    y2 = int(gh * PANEL_Y2)
    x1 = max(0, min(x1, gw - 1))
    x2 = max(x1 + 1, min(x2, gw))
    y1 = max(0, min(y1, gh - 1))
    y2 = max(y1 + 1, min(y2, gh))
    return game_bgr[y1:y2, x1:x2]


def _ocr_lines(panel_bgr: np.ndarray) -> list[dict]:
    """패널 crop을 OCR해 (text, conf, cx, cy, box) 라인 목록을 y 순으로 반환."""
    if panel_bgr.size == 0:
        return []
    reader = get_reader()
    up = cv2.resize(panel_bgr, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    results = reader.readtext(up, detail=1, paragraph=False)
    lines = []
    for box, text, conf in results:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        lines.append(
            {
                "text": str(text).strip(),
                "conf": float(conf),
                "cx": float(np.mean(xs)),
                "cy": float(np.mean(ys)),
                "x_left": float(min(xs)),
                "y_top": float(min(ys)),
                "y_bot": float(max(ys)),
            }
        )
    lines.sort(key=lambda d: d["cy"])
    return lines


def levenshtein(a: str, b: str) -> int:
    """두 문자열의 편집 거리(삽입/삭제/치환). 헤더·닉 fuzzy 매칭용."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _header_base(text: str) -> str:
    """헤더 문자열에서 괄호·숫자·기호를 제거한 한글/영문 본체만 추출."""
    head = text.replace(" ", "").split("(")[0]
    return re.sub(r"[^가-힣a-zA-Z]", "", head)


def _is_header(text: str) -> bool:
    """'스카우터'/'스카우터2' 헤더 여부. OCR 변형(스카무터·스무무터 등)을 fuzzy 허용."""
    base = _header_base(text)
    if not base:
        return False
    if "scouter" in base.lower():
        return True
    # '스카우터'(4글자) 기준 편집거리 2 이내면 헤더로 인정
    return levenshtein(base, "스카우터") <= 2


def _header_is_scouter2(text: str) -> bool:
    # "스카우터2 (L)" → 2 포함, "스카우터 (L)" → 미포함
    # 숫자 2가 z/Z/²로 오인식되는 경우까지 포함
    head = text.replace(" ", "").split("(")[0]
    return any(ch in head for ch in ("2", "z", "Z", "²"))


def _normalize_nick(text: str) -> str:
    return text.strip().lower().replace(" ", "")


def _is_null_variant(text: str) -> bool:
    norm = re.sub(r"[^0-9a-z]", "", _normalize_nick(text))
    if not norm:
        return False
    if norm in _NULL_VARIANTS:
        return True
    return len(norm) <= 5 and levenshtein(norm, "null") <= 1


def _is_header_fragment(text: str) -> bool:
    """헤더 '스카우터 (L)' OCR 분할 잔여 또는 무효 텍스트."""
    t = text.strip().replace(" ", "")
    if not t:
        return True
    if _is_header(t):
        return True
    if _HEADER_FRAGMENT_RE.match(t):
        return True
    if re.fullmatch(r"[\(\)\{\}\[\]\|]+", t):
        return True
    return False


def _is_valid_player_nick(text: str) -> bool:
    return bool(text.strip()) and not _is_header_fragment(text)


def _filter_data_lines(lines: list[dict]) -> list[dict]:
    """데이터 행에서 헤더 잔여·무효 OCR 라인 제거."""
    return [line for line in lines if _is_valid_player_nick(line["text"])]


def _lines_on_same_row(a: dict, b: dict, y_tol: float = 18.0) -> bool:
    return abs(a["cy"] - b["cy"]) <= y_tol


def _merge_row_text(lines: list[dict], anchor_idx: int) -> dict | None:
    """같은 행(y 근접) OCR 조각을 합쳐 닉 후보 1개로 만든다."""
    anchor = lines[anchor_idx]
    parts = [anchor]
    for idx, line in enumerate(lines):
        if idx == anchor_idx:
            continue
        if _lines_on_same_row(anchor, line):
            parts.append(line)
    parts.sort(key=lambda d: d["x_left"])
    merged_text = "".join(p["text"] for p in parts).strip()
    if not _is_valid_player_nick(merged_text):
        return None
    return {
        "text": merged_text,
        "conf": max(p["conf"] for p in parts),
        "cx": anchor["cx"],
        "cy": anchor["cy"],
        "x_left": min(p["x_left"] for p in parts),
        "y_top": min(p["y_top"] for p in parts),
        "y_bot": max(p["y_bot"] for p in parts),
    }


def _pick_scouter_player(panel_bgr: np.ndarray, data_lines: list[dict]) -> dict | None:
    """스카우터 모드: 점(●) 행에서 본인 닉 선택. 헤더 잔여·null 폴백 포함."""
    if not data_lines:
        return None

    dot_idx = _detect_white_dot_row(panel_bgr, data_lines)
    if dot_idx is not None:
        target = data_lines[dot_idx]
        if _is_valid_player_nick(target["text"]):
            return target
        merged = _merge_row_text(data_lines, dot_idx)
        if merged is not None:
            return merged

    valid_lines = _filter_data_lines(data_lines)
    if not valid_lines:
        return None

    null_lines = [line for line in valid_lines if _is_null_variant(line["text"])]
    if null_lines:
        return max(null_lines, key=lambda d: d["conf"])

    if dot_idx is not None:
        anchor = data_lines[dot_idx]
        same_row = [
            line for line in valid_lines
            if _lines_on_same_row(anchor, line)
        ]
        if same_row:
            return max(same_row, key=lambda d: d["conf"])

    return max(valid_lines, key=lambda d: d["conf"])


def _pick_scouter2_player(data_lines: list[dict]) -> dict | None:
    """스카우터2 모드: 맨 위 유효 행 = 본인."""
    for idx, line in enumerate(data_lines):
        if _is_valid_player_nick(line["text"]):
            return line
        merged = _merge_row_text(data_lines, idx)
        if merged is not None:
            return merged
    return None


def _detect_white_dot_row(panel_bgr: np.ndarray, data_lines: list[dict]) -> int | None:
    """스카우터 모드: 각 행 왼쪽의 흰 점(●)을 찾아 해당 행 인덱스를 반환."""
    if not data_lines or panel_bgr.size == 0:
        return None
    up_scale = 2.5
    ph, pw = panel_bgr.shape[:2]
    hsv = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2HSV)
    # 흰색(고명도/저채도) 마스크
    white = cv2.inRange(hsv, (0, 0, 200), (180, 60, 255))

    best_idx = None
    best_score = 0
    for idx, line in enumerate(data_lines):
        # 헤더 잔여 (L) 등은 점 행 후보에서 제외
        if _is_header_fragment(line["text"]):
            continue
        # OCR 좌표는 2.5배 업스케일 기준 → 원본으로 환산
        y_top = int(line["y_top"] / up_scale)
        y_bot = int(line["y_bot"] / up_scale)
        x_left = int(line["x_left"] / up_scale)
        y1 = max(0, y_top - 4)
        y2 = min(ph, y_bot + 4)
        # 텍스트 왼쪽의 좁은 띠에서 흰 점 탐색
        x1 = max(0, x_left - int(pw * 0.18))
        x2 = max(x1 + 1, x_left - int(pw * 0.02))
        roi = white[y1:y2, x1:x2]
        score = int(roi.sum() / 255) if roi.size else 0
        if score > best_score:
            best_score = score
            best_idx = idx
    # 흰 점이 충분히 클 때만 인정
    if best_score >= 6:
        return best_idx
    return None


def read_scouter(frame_bgr: np.ndarray, *, dataset_root: Path | None = None) -> ScouterReadout:
    game_bgr, _ = extract_game_crop_bgr(frame_bgr, dataset_root=dataset_root)
    game_width = int(game_bgr.shape[1])
    panel = crop_scouter_panel(game_bgr)
    lines = _ocr_lines(panel)

    if not lines:
        return ScouterReadout("unknown", "", 0.0, False, [], game_width)

    header_idx = next((i for i, l in enumerate(lines) if _is_header(l["text"])), None)
    if header_idx is None:
        return ScouterReadout("unknown", "", 0.0, False, [(l["text"], l["conf"]) for l in lines], game_width)

    is_scouter2 = _header_is_scouter2(lines[header_idx]["text"])
    data_lines = lines[header_idx + 1 :]
    valid_rows = _filter_data_lines(data_lines)
    rows = [(l["text"], l["conf"]) for l in valid_rows]

    if not data_lines:
        mode = "scouter2" if is_scouter2 else "scouter"
        return ScouterReadout(mode, "", 0.0, not is_scouter2, rows, game_width)

    if is_scouter2:
        top = _pick_scouter2_player(data_lines)
        if top is not None:
            return ScouterReadout("scouter2", top["text"], top["conf"], False, rows, game_width)
        return ScouterReadout("scouter2", "", 0.0, False, rows, game_width)

    player = _pick_scouter_player(panel, data_lines)
    if player is not None:
        nick = player["text"]
        if _is_null_variant(nick):
            nick = "null"
        return ScouterReadout("scouter", nick, player["conf"], True, rows, game_width)
    return ScouterReadout("scouter", "", 0.0, True, rows, game_width)
