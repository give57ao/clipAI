# -*- coding: utf-8 -*-
"""상단 라이브 HUD 인원 아이콘 기반 라운드 종료 탐지 (규칙/색상 CV, ML 아님).

서든어택 상단 HUD: 레드점수 | 라운드# | 블루점수, 각 팀 점수 아래 생존 인원 아이콘.
한쪽 팀 아이콘이 0이면 해당 라운드 종료(엘리미네이션).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class HudState(str, Enum):
    ACTIVE = "active"
    RED_ELIMINATED = "red_eliminated"
    BLUE_ELIMINATED = "blue_eliminated"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class HudIconCounts:
    red_icons: int
    blue_icons: int

    @property
    def state(self) -> HudState:
        red = self.red_icons
        blue = self.blue_icons
        if red >= 1 and blue >= 1:
            return HudState.ACTIVE
        if red == 0 and blue >= 1:
            return HudState.RED_ELIMINATED
        if blue == 0 and red >= 1:
            return HudState.BLUE_ELIMINATED
        return HudState.UNKNOWN


@dataclass
class HudRoundEnd:
    time_sec: float
    eliminated_team: str
    red_icons: int
    blue_icons: int


# 1920x1080 기준 상단 HUD (우측 방송 오버레이 때문에 중앙이 아니라 좌측에 위치)
_HUD_Y = (0.01, 0.12)
_HUD_X = (0.30, 0.70)
_ICON_ROW_Y = 0.50
_RED_COL_X = (0.0, 0.34)
_BLUE_COL_X = (0.58, 1.0)
_MAX_ICONS_PER_TEAM = 3
_MIN_BLOB_AREA = 10
_MAX_BLOB_AREA = 500


def _count_team_icons(roi_bgr: np.ndarray, team: str) -> int:
    if roi_bgr.size == 0:
        return 0
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    if team == "red":
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 90, 90), (12, 255, 255)),
            cv2.inRange(hsv, (168, 90, 90), (180, 255, 255)),
        )
    else:
        mask = cv2.inRange(hsv, (95, 60, 60), (130, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    icon_count = sum(
        1
        for i in range(1, n_labels)
        if _MIN_BLOB_AREA <= stats[i, cv2.CC_STAT_AREA] <= _MAX_BLOB_AREA
    )
    return min(icon_count, _MAX_ICONS_PER_TEAM)


def analyze_hud_icons(frame_bgr: np.ndarray) -> HudIconCounts:
    """프레임에서 상단 HUD 생존 인원 아이콘 수를 추정."""
    h, w = frame_bgr.shape[:2]
    y1, y2 = int(h * _HUD_Y[0]), int(h * _HUD_Y[1])
    x1, x2 = int(w * _HUD_X[0]), int(w * _HUD_X[1])
    hud = frame_bgr[y1:y2, x1:x2]
    if hud.size == 0:
        return HudIconCounts(0, 0)

    hh, hw = hud.shape[:2]
    icon_y1 = int(hh * _ICON_ROW_Y)
    icon_row = hud[icon_y1:, :]
    red_roi = icon_row[:, int(hw * _RED_COL_X[0]) : int(hw * _RED_COL_X[1])]
    blue_roi = icon_row[:, int(hw * _BLUE_COL_X[0]) : int(hw * _BLUE_COL_X[1])]
    return HudIconCounts(
        red_icons=_count_team_icons(red_roi, "red"),
        blue_icons=_count_team_icons(blue_roi, "blue"),
    )


def scan_hud_round_ends(
    video_path,
    scan_fps: float = 4.0,
    min_active_streak: int = 3,
    min_ended_streak: int = 3,
) -> list[HudRoundEnd]:
    """영상을 스캔해 active → 한쪽 전멸 전환 시점을 수집."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(round(fps / scan_fps))) if scan_fps > 0 else 1

    events: list[HudRoundEnd] = []
    active_streak = 0
    ended_streak = 0
    ended_state: HudState | None = None
    ended_counts: HudIconCounts | None = None
    ended_start_sec = 0.0
    awaiting_next_round = False
    frame_idx = 0

    while True:
        if frame_idx % frame_step == 0:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            t = frame_idx / fps
            counts = analyze_hud_icons(frame)
            state = counts.state

            if state == HudState.ACTIVE:
                active_streak += 1
                ended_streak = 0
                ended_state = None
                awaiting_next_round = False
            elif state in (HudState.RED_ELIMINATED, HudState.BLUE_ELIMINATED):
                if ended_state != state:
                    ended_streak = 1
                    ended_state = state
                    ended_counts = counts
                    ended_start_sec = t
                else:
                    ended_streak += 1

                if (
                    not awaiting_next_round
                    and active_streak >= min_active_streak
                    and ended_streak >= min_ended_streak
                    and ended_counts is not None
                ):
                    eliminated = "red" if state == HudState.RED_ELIMINATED else "blue"
                    events.append(
                        HudRoundEnd(
                            time_sec=ended_start_sec,
                            eliminated_team=eliminated,
                            red_icons=ended_counts.red_icons,
                            blue_icons=ended_counts.blue_icons,
                        )
                    )
                    awaiting_next_round = True
                    active_streak = 0
            else:
                active_streak = 0
                ended_streak = 0
                ended_state = None
        else:
            if not cap.grab():
                break
        frame_idx += 1

    cap.release()
    return _merge_nearby_events(events, merge_gap_sec=3.0)


def _merge_nearby_events(events: list[HudRoundEnd], merge_gap_sec: float) -> list[HudRoundEnd]:
    if not events:
        return []
    merged: list[HudRoundEnd] = [events[0]]
    for event in events[1:]:
        prev = merged[-1]
        if event.time_sec - prev.time_sec <= merge_gap_sec:
            continue
        merged.append(event)
    return merged


def find_hud_end_before_scoreboard(
    hud_ends: list[HudRoundEnd],
    scoreboard_start_sec: float,
    lookback_sec: float = 8.0,
    min_gap_sec: float = 0.3,
) -> HudRoundEnd | None:
    """스코어보드 직전 구간에서 가장 가까운 HUD 라운드 종료 이벤트."""
    t_min = scoreboard_start_sec - lookback_sec
    t_max = scoreboard_start_sec - min_gap_sec
    candidates = [e for e in hud_ends if t_min <= e.time_sec <= t_max]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e.time_sec)
