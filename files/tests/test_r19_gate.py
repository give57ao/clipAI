# -*- coding: utf-8 -*-
"""R19 적팀-승리 blind-gap 게이트 회귀 고정 (ANALYSIS_R102_R19.md).

2026-06-24 01-55-38 R102(boundary_merge FP, 폭 16.9s로 R14 폭게이트 미달)의
지문을 합성 픽스처로 고정 — E: 드라이브 캐시 없이도 이 게이트의 핵심 판별을
회귀 검증한다. 전체 GT 코퍼스 재확인(recall 59.8% 불변, FP 122→121)은
hud_from_cache.py + _compare_hud_gt.py 수동 실행으로 별도 확인 완료.
"""

from __future__ import annotations

from detect_ace_hud import RoundTrack, _r19_enemy_win_in_gap


def _round(start_sec, kill_times, ace_sec):
    return RoundTrack(
        round_index=0,
        start_sec=start_sec,
        kill_times=list(kill_times),
        kills=len(kill_times),
        ace=True,
        first_kill_sec=kill_times[0],
        ace_sec=ace_sec,
    )


def test_r102_fingerprint_flagged():
    """R102 실측: gap-stamp 킬 2개(4118.125) + 적팀(B) win이 blind gap 안(4112.5)에
    있음 → True(FP 신호)."""
    r = _round(4118.125, [4118.125, 4118.125, 4135.0], 4135.0)
    ok_reads = [(4105.5, 5), (4128.5, 7)]  # carry_t=4105.5(gap전 마지막), first_obs=4128.5
    events = [
        {"kind": "win", "side": "B", "t_hi": 4112.5},   # 적팀 승리 — blind gap 내부
        {"kind": "win", "side": "R", "t_hi": 4135.0},   # 우리팀 승리 — ace_sec 근방
    ]
    assert _r19_enemy_win_in_gap(r, ok_reads, events) is True


def test_no_enemy_win_in_gap_not_flagged():
    """같은 gap-stamp 구조지만 blind gap 안에 적팀 win이 없으면(진짜 올킬 다수 패턴)
    False — TP 보호."""
    r = _round(4118.125, [4118.125, 4118.125, 4135.0], 4135.0)
    ok_reads = [(4105.5, 5), (4128.5, 7)]
    events = [
        {"kind": "win", "side": "R", "t_hi": 4135.0},   # 우리팀 승리만 존재
    ]
    assert _r19_enemy_win_in_gap(r, ok_reads, events) is False


def test_no_gap_stamp_kill_not_flagged():
    """gap-stamp 킬이 아예 없으면(모든 킬이 라운드 내부에서 관측) 적팀 win이 있어도
    신호 미적용 — within-chain 올킬은 이 게이트의 대상이 아님."""
    r = _round(100.0, [110.0, 115.0, 120.0], 120.0)
    ok_reads = [(105.0, 3), (120.0, 6)]
    events = [
        {"kind": "win", "side": "B", "t_hi": 112.0},
        {"kind": "win", "side": "R", "t_hi": 120.0},
    ]
    assert _r19_enemy_win_in_gap(r, ok_reads, events) is False


def test_no_own_win_near_ace_sec_skips_conservatively():
    """ace_sec 근방에 우리팀 win이 아예 없으면(승수 채널 공백 등) 신호 적용 불가로
    스킵 — 보수적 설계, FP를 놓치더라도 TP를 깨지 않는다."""
    r = _round(4118.125, [4118.125, 4118.125, 4135.0], 4135.0)
    ok_reads = [(4105.5, 5), (4128.5, 7)]
    events = [
        {"kind": "win", "side": "B", "t_hi": 4112.5},  # 적팀 win은 있으나
        # 우리팀 win이 ace_sec(4135) 근방에 없음
    ]
    assert _r19_enemy_win_in_gap(r, ok_reads, events) is False
