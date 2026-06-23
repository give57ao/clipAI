# -*- coding: utf-8 -*-
"""라벨링 클래스 정의 (4종 하이라이트 + 선택적 배경)."""

from __future__ import annotations

# 학습 대상 하이라이트 4종
HIGHLIGHT_LABELS: tuple[str, ...] = (
    "doublekill",  # 더블킬
    "multikill",   # 멀티킬 (트리플 이상)
    "save",        # 세이브
    "allkill",     # 올킬
)

HIGHLIGHT_LABEL_KO: dict[str, str] = {
    "doublekill": "더블킬",
    "multikill": "멀티킬",
    "save": "세이브",
    "allkill": "올킬",
}

# 학습용 음성(일반 플레이) — 선택
BACKGROUND_LABEL = "background"

ALL_CLIP_LABELS: tuple[str, ...] = HIGHLIGHT_LABELS + (BACKGROUND_LABEL,)

# 한글 입력 별칭 → 내부 label
LABEL_ALIASES: dict[str, str] = {
    "doublekill": "doublekill",
    "double": "doublekill",
    "double_kill": "doublekill",
    "더블킬": "doublekill",
    "더블": "doublekill",
    "multikill": "multikill",
    "multi": "multikill",
    "multi_kill": "multikill",
    "triple": "multikill",
    "triplekill": "multikill",
    "멀티킬": "multikill",
    "멀티": "multikill",
    "트리플": "multikill",
    "트리플킬": "multikill",
    "save": "save",
    "clutch": "save",
    "세이브": "save",
    "allkill": "allkill",
    "ace": "allkill",
    "올킬": "allkill",
    "에이스": "allkill",
    "background": BACKGROUND_LABEL,
    "normal": BACKGROUND_LABEL,
    "bg": BACKGROUND_LABEL,
    "일반": BACKGROUND_LABEL,
    "배경": BACKGROUND_LABEL,
}


def normalize_label(raw: str) -> str | None:
    key = (raw or "").strip().lower()
    if not key:
        return None
    return LABEL_ALIASES.get(key) or (key if key in ALL_CLIP_LABELS else None)
