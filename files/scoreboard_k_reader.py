# -*- coding: utf-8 -*-
"""1b-2 단계: 확정 닉으로 전체스코어 K 읽기 → 라운드 간 ΔK → 올킬 후보.

설계 문서: PLAYER_IDENTITY_AND_K_READER.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import cv2

from nick_fuzzy import normalize_nick, nick_match_text, nick_core, cores_match
from player_identity import PlayerIdentity, resolve_player_identity
from scoreboard_layout import (
    ScoreboardRow,
    ScoreboardWindow,
    find_scoreboard_csv,
    load_scoreboard_windows,
    read_scoreboard_rows,
    read_team_wins,
    team_kills_sum,
)
from scouter_nick import levenshtein

DEFAULT_DATASET_ROOT = Path(r"E:\Highlights\ml_dataset")
ACE_KILL_THRESHOLD = 3
# 직전 스코어보드 대비 플레이 구간이 중앙값의 N배 초과 → 중간 라운드 누락 의심
PLAY_GAP_ACE_MAX_RATIO = 1.55
# 직전 라운드 올킬 직후 또 ΔK=3 + 짧은 플레이 구간 → 중간 SB 누락(2+1 합산) 의심
PLAY_GAP_CONSECUTIVE_ACE_MAX_RATIO = 0.55


@dataclass
class RoundKillReadout:
    round_index: int
    scoreboard_start_sec: float
    scoreboard_end_sec: float
    kills: int | None
    nick_matched: str
    match_score: float
    deaths: int | None = None
    team_k_sum: int | None = None
    red_wins: int | None = None
    blue_wins: int | None = None
    play_gap_sec: float | None = None
    row_index: int | None = None
    samples_used: int = 0
    rejoin_reset: bool = False
    # 안전장치: 팀 고정 + K 연속성으로 추론한 경우 True (재검증 필요)
    inferred_by_team: bool = False
    # ΔK >= 4 → OCR 오류 의심 (올킬 아님)
    k_read_error: bool = False
    ace_reject_reason: str = ""
    # 올킬 '후보' (직접매칭/인접성 등 게이트 미통과) — 리콜 확보용
    ace_candidate_reason: str = ""
    # 안전장치에서 후보로 올라온 팀원 닉/K 목록 (디버그용)
    team_candidates: list[dict] = field(default_factory=list)


@dataclass
class VideoKillTimeline:
    video_path: str
    player_nick: str
    # 레이아웃 진단용 (스카우터 기반)
    game_width_median: int = 0
    layout: str = ""
    rounds: list[RoundKillReadout] = field(default_factory=list)
    delta_kills: list[int | None] = field(default_factory=list)
    ace_rounds: list[int] = field(default_factory=list)
    ace_candidates: list[int] = field(default_factory=list)
    k_error_rounds: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # 팀 고정으로 라운드 수 / 안전장치 사용 수
    team_lock_rounds: int = 0
    inferred_rounds: int = 0


def nick_match_score(player_nick: str, row_nick: str) -> float:
    """0~1 fuzzy 매칭 점수 — 정규화 편집거리 기반.

    기존 공식(1 - dist/(threshold+1))은 오류 2개짜리 9자 닉에서 0.33만 반환해
    threshold=0.65를 통과하지 못하는 문제가 있었음.
    새 공식은 max_len 대비 편집거리 비율을 사용해 문자열 길이에 따라 스케일.
    """
    if not player_nick or not row_nick:
        return 0.0
    if not nick_match_text(player_nick, row_nick):
        # 핵심부 포함 매칭 (특수문자·잘림 닉)
        if cores_match(player_nick, row_nick):
            ca, cb = nick_core(player_nick), nick_core(row_nick)
            if ca and cb:
                shorter = min(len(ca), len(cb))
                longer = max(len(ca), len(cb))
                return 0.82 + 0.18 * (shorter / longer) if longer else 0.82
        return 0.0
    a, b = normalize_nick(player_nick), normalize_nick(row_nick)
    if a == b:
        return 1.0
    max_len = max(len(a), len(b))
    dist = levenshtein(a, b)
    return max(0.0, 1.0 - dist / max_len) if max_len > 0 else 0.0


def find_player_row(
    rows: list[ScoreboardRow],
    player_nick: str,
    *,
    nick_match_threshold: float = 0.65,
) -> tuple[ScoreboardRow | None, float]:
    best_row: ScoreboardRow | None = None
    best_score = 0.0
    for row in rows:
        score = nick_match_score(player_nick, row.nickname)
        if score > best_score:
            best_row, best_score = row, score
    if best_score >= nick_match_threshold:
        return best_row, best_score
    return None, best_score


def _rows_matching_player(
    rows: list[ScoreboardRow],
    player_nick: str,
    *,
    nick_match_threshold: float,
) -> list[tuple[float, ScoreboardRow]]:
    matched: list[tuple[float, ScoreboardRow]] = []
    for row in rows:
        score = nick_match_score(player_nick, row.nickname)
        if score >= nick_match_threshold:
            matched.append((score, row))
    return matched


# ──────────────────────────────────────────────
# 팀 고정 + K 연속성 안전장치
# ──────────────────────────────────────────────

# 한 라운드 최대 허용 K 증가량 (올킬=3 + 여유치)
_MAX_K_DELTA_PER_ROUND = 5


@dataclass
class TeamLock:
    """여러 라운드 성공에서 축적된 팀 소속 확정 정보."""
    team: str = ""                            # "red" | "blue" | ""(미확정)
    confirmed_rounds: int = 0                 # 직접 닉 매칭 성공 횟수
    teammate_nicks: set[str] = field(default_factory=set)   # 같은 팀 닉 집합
    opponent_nicks: set[str] = field(default_factory=set)   # 상대 팀 닉 집합

    @property
    def is_locked(self) -> bool:
        return bool(self.team) and self.confirmed_rounds >= 1


def _update_team_lock(
    lock: TeamLock,
    matched_row: ScoreboardRow,
    all_rows: list[ScoreboardRow],
) -> None:
    """직접 매칭 성공 시 팀 정보 갱신."""
    team = matched_row.team
    if not lock.team:
        lock.team = team
    elif lock.team != team:
        # 팀이 바뀌면 (잘못된 매칭 가능성) 확정 해제
        lock.confirmed_rounds = max(0, lock.confirmed_rounds - 1)
        return
    lock.confirmed_rounds += 1
    for row in all_rows:
        if not row.nickname:
            continue
        if row.team == team:
            lock.teammate_nicks.add(row.nickname)
        else:
            lock.opponent_nicks.add(row.nickname)


def _team_fallback_kill(
    rows: list[ScoreboardRow],
    lock: TeamLock,
    prev_kills: int | None,
) -> tuple[ScoreboardRow | None, list[dict]]:
    """
    안전장치: 닉 직접 매칭 실패 시 팀 고정 + K 연속성으로 후보 행 추론.

    반환:
        (선택된 행 또는 None, 후보 목록[디버그용])
    """
    if not lock.is_locked:
        return None, []

    team_rows = [r for r in rows if r.team == lock.team]
    if not team_rows:
        return None, []

    candidates: list[dict] = []
    for row in team_rows:
        reasons: list[str] = []
        k = row.kills

        # K 연속성 검사
        k_ok = True
        if k is None:
            k_ok = False
            reasons.append("K=None")
        elif prev_kills is not None:
            delta = k - prev_kills
            if delta < 0:
                # 리조인 가능성 — K가 0이면 허용, 아니면 의심
                if k == 0:
                    reasons.append("K=0(rejoin?)")
                else:
                    k_ok = False
                    reasons.append(f"K감소({prev_kills}→{k})")
            elif delta > _MAX_K_DELTA_PER_ROUND:
                k_ok = False
                reasons.append(f"K과다증가(Δ{delta})")

        # 상대팀 닉과 유사하면 신뢰도 하락
        is_opponent_nick = any(
            nick_match_text(row.nickname, opp)
            for opp in lock.opponent_nicks
            if opp
        )
        if is_opponent_nick:
            k_ok = False
            reasons.append("상대팀닉유사")

        candidates.append({
            "nickname": row.nickname,
            "nick_conf": row.nick_conf,
            "kills": k,
            "row_index": row.row_index,
            "k_ok": k_ok,
            "reasons": reasons,
            "_row": row,
        })

    # K 조건 통과 후보만 남김
    valid = [c for c in candidates if c["k_ok"]]
    if not valid:
        # 모두 탈락 → None, 후보 목록은 디버그용으로 반환
        return None, candidates

    # nick_conf 기준 최선 선택
    best = max(valid, key=lambda c: c["nick_conf"])
    return best["_row"], candidates


def _resolve_team_by_teammates(
    rows: list["ScoreboardRow"],
    teammate_nicks: set[str],
) -> str | None:
    """이번 라운드의 행 목록에서 아군 닉을 찾아 플레이어의 팀 반환.

    팀 교체(빨강↔파랑 스왑) 상황에서도 아군 닉을 통해 올바른 팀을 결정.
    """
    for team in ("red", "blue"):
        for row in (r for r in rows if r.team == team):
            if not row.nickname:
                continue
            if any(
                nick_match_text(row.nickname, tmate)
                for tmate in teammate_nicks
                if tmate
            ):
                return team
    return None


def sample_times_in_window(window: ScoreboardWindow, count: int = 3) -> list[float]:
    """스코어보드 구간 내 OCR 샘플 시각 (다수결용)."""
    duration = window.end_sec - window.start_sec
    if duration <= 0:
        return [window.mid_sec]
    margin = min(0.6, duration * 0.12)
    t_start = window.start_sec + margin
    t_end = window.end_sec - margin
    if t_end <= t_start:
        return [window.mid_sec]
    if count <= 1:
        return [window.mid_sec]
    return [t_start + (t_end - t_start) * i / (count - 1) for i in range(count)]


def dense_sample_times_in_window(
    window: ScoreboardWindow,
    *,
    step_sec: float = 0.4,
) -> list[float]:
    """닉 매칭 실패 시 스코어보드 전체(~4초)를 촘촘히 재스캔."""
    duration = window.end_sec - window.start_sec
    if duration <= 0:
        return [window.mid_sec]
    # 페이드 인/아웃 여유만 두고 구간 전체를 훑음
    margin = min(0.25, duration * 0.05)
    t_start = window.start_sec + margin
    t_end = window.end_sec - margin
    if t_end <= t_start:
        return [window.mid_sec]
    times: list[float] = []
    t = t_start
    while t <= t_end + 1e-6:
        times.append(t)
        t += step_sec
    return times


def _collect_frame_hits(
    cap: cv2.VideoCapture,
    fps: float,
    sample_secs: list[float],
    player_nick: str,
    *,
    nick_match_threshold: float,
    dataset_root: Path | None,
) -> tuple[
    list[tuple[float, ScoreboardRow]],
    list[ScoreboardRow],
    list[ScoreboardRow],
    object | None,
    int,
]:
    """샘플 시각마다 닉 매칭 행 수집."""
    frame_hits: list[tuple[float, ScoreboardRow]] = []
    all_rows_seen: list[ScoreboardRow] = []
    last_rows: list[ScoreboardRow] = []
    last_frame = None
    samples_used = 0

    for sample_sec in sample_secs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sample_sec * fps))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        samples_used += 1
        last_frame = frame
        rows = read_scoreboard_rows(frame, dataset_root=dataset_root)
        last_rows = rows
        all_rows_seen.extend(rows)
        for score, row in _rows_matching_player(
            rows, player_nick, nick_match_threshold=nick_match_threshold
        ):
            frame_hits.append((score, row))

    return frame_hits, all_rows_seen, last_rows, last_frame, samples_used


def _majority_k(values: list[int]) -> int | None:
    if not values:
        return None
    top, count = Counter(values).most_common(1)[0]
    if count >= 2 or len(values) == 1:
        return top
    return top


def _filter_death_ocr_glitch(
    candidates: list[int],
    prev_deaths: int | None,
) -> list[int]:
    """D열 10·20·30 누적 오독(어시스트 열 혼입) 후보 제거."""
    if not candidates:
        return candidates
    filtered: list[int] = []
    for d in candidates:
        if d >= 10 and d % 10 == 0:
            if prev_deaths is not None and d - prev_deaths == 10:
                continue
            if prev_deaths is None and d >= 10:
                continue
        filtered.append(d)
    return filtered if filtered else candidates


def read_round_kill(
    cap: cv2.VideoCapture,
    fps: float,
    window: ScoreboardWindow,
    round_index: int,
    player_nick: str,
    *,
    nick_match_threshold: float = 0.65,
    frame_samples: int = 3,
    dataset_root: Path | None = None,
    _rows_cache: dict | None = None,
    prev_deaths: int | None = None,
) -> RoundKillReadout:
    """한 스코어보드 구간에서 본인 K 읽기 (프레임 다수결)."""
    kill_candidates: list[int] = []
    death_candidates: list[int] = []
    last_rows: list[ScoreboardRow] = []
    last_frame = None

    sample_secs = sample_times_in_window(window, frame_samples)
    frame_hits, all_rows_seen, last_rows, last_frame, samples_used = _collect_frame_hits(
        cap, fps, sample_secs, player_nick,
        nick_match_threshold=nick_match_threshold,
        dataset_root=dataset_root,
    )

    # 1차 실패 → 스코어보드 유지 구간(~4초) 전체를 0.4초 간격 재스캔
    if not frame_hits:
        dense_secs = dense_sample_times_in_window(window)
        dense_hits, dense_rows, dense_last_rows, dense_frame, dense_n = _collect_frame_hits(
            cap, fps, dense_secs, player_nick,
            nick_match_threshold=nick_match_threshold,
            dataset_root=dataset_root,
        )
        if dense_hits:
            frame_hits = dense_hits
            all_rows_seen.extend(dense_rows)
            if dense_last_rows:
                last_rows = dense_last_rows
            if dense_frame is not None:
                last_frame = dense_frame
            samples_used += dense_n

    # 안전장치용 캐시: 마지막 샘플 프레임의 6행 저장
    if _rows_cache is not None and all_rows_seen:
        # 마지막 프레임 row만 (row_index 기준 dedup)
        seen_idx: dict[int, ScoreboardRow] = {}
        for r in all_rows_seen:
            seen_idx[r.row_index] = r
        _rows_cache[round_index] = list(seen_idx.values())

    if not frame_hits:
        return RoundKillReadout(
            round_index=round_index,
            scoreboard_start_sec=window.start_sec,
            scoreboard_end_sec=window.end_sec,
            kills=None,
            nick_matched="",
            match_score=0.0,
            samples_used=samples_used,
        )

    best_score, matched_row = max(frame_hits, key=lambda item: item[0])

    # 동일 row_index + 닉 일치 프레임만 K/D 수집 (다른 행 OCR 혼입 방지)
    kill_candidates = [
        r.kills for r in all_rows_seen
        if r.row_index == matched_row.row_index
        and nick_match_score(player_nick, r.nickname) >= nick_match_threshold
        and r.kills is not None
    ]
    death_candidates = [
        r.deaths for r in all_rows_seen
        if r.row_index == matched_row.row_index
        and nick_match_score(player_nick, r.nickname) >= nick_match_threshold
        and r.deaths is not None
    ]

    death_candidates = _filter_death_ocr_glitch(death_candidates, prev_deaths)

    kills = _majority_k(kill_candidates)
    deaths = _majority_k(death_candidates)
    team_k = team_kills_sum(last_rows, matched_row.team) if last_rows else None
    red_w, blue_w = (None, None)
    if last_frame is not None:
        red_w, blue_w = read_team_wins(last_frame, dataset_root=dataset_root)

    return RoundKillReadout(
        round_index=round_index,
        scoreboard_start_sec=window.start_sec,
        scoreboard_end_sec=window.end_sec,
        kills=kills,
        deaths=deaths,
        team_k_sum=team_k,
        red_wins=red_w,
        blue_wins=blue_w,
        nick_matched=matched_row.nickname,
        match_score=round(best_score, 3),
        row_index=matched_row.row_index,
        samples_used=samples_used,
    )


def _median_play_gap(scoreboards: list[ScoreboardWindow]) -> float:
    gaps = []
    for i in range(1, len(scoreboards)):
        g = scoreboards[i].start_sec - scoreboards[i - 1].end_sec
        if g > 0:
            gaps.append(g)
    if not gaps:
        return 45.0
    gaps.sort()
    return gaps[len(gaps) // 2]


def _player_team_wins(readout: RoundKillReadout) -> int | None:
    if readout.row_index is None:
        return None
    if readout.row_index < 3:
        return readout.red_wins
    return readout.blue_wins


def _match_rounds_total(readout: RoundKillReadout) -> int | None:
    if readout.red_wins is None or readout.blue_wins is None:
        return None
    return readout.red_wins + readout.blue_wins


def compute_delta_kills(
    readouts: list[RoundKillReadout],
    *,
    ace_threshold: int = ACE_KILL_THRESHOLD,
    median_play_gap: float = 45.0,
) -> tuple[list[int | None], list[int], list[int], list[int]]:
    """라운드 간 ΔK + 올킬(ΔK==3) / K 읽기 오류(ΔK>=4) 라운드 인덱스.

    올킬 추가 검증 (2026-06-29):
      - 직접매칭 + 인접 라운드
      - 플레이 구간이 중앙값 대비 과대 → 중간 라운드 누락
      - 경기 총 라운드(레드승+블루승) 증가가 1이 아니면 단일 라운드 올킬 아님
      - 플레이어 팀 승리라운드 증가가 1이 아니면 해당 라운드 미승리
    """
    deltas: list[int | None] = []
    ace_rounds: list[int] = []
    ace_candidates: list[int] = []
    k_error_rounds: list[int] = []
    prev_kills: int | None = None
    prev_deaths: int | None = None
    prev_inferred: bool = False
    prev_round_index: int | None = None
    prev_pt_wins: int | None = None
    prev_match_total: int | None = None
    prev_team_k_sum: int | None = None

    for readout in readouts:
        kills = readout.kills
        if kills is None:
            deltas.append(None)
            continue

        if prev_kills is None:
            deltas.append(None)
            prev_kills = kills
            prev_deaths = readout.deaths
            prev_inferred = readout.inferred_by_team
            prev_round_index = readout.round_index
            prev_pt_wins = _player_team_wins(readout)
            prev_match_total = _match_rounds_total(readout)
            prev_team_k_sum = readout.team_k_sum
            continue

        if kills < prev_kills:
            readout.rejoin_reset = True
            deltas.append(None)
            prev_kills = kills
            prev_deaths = readout.deaths
            prev_inferred = readout.inferred_by_team
            prev_round_index = readout.round_index
            prev_pt_wins = _player_team_wins(readout)
            prev_match_total = _match_rounds_total(readout)
            prev_team_k_sum = readout.team_k_sum
            continue

        delta = kills - prev_kills
        deltas.append(delta)

        both_direct = (not readout.inferred_by_team) and (not prev_inferred)
        adjacent = prev_round_index is not None and readout.round_index == prev_round_index + 1

        if delta > ace_threshold:
            readout.k_read_error = True
            k_error_rounds.append(readout.round_index)
        elif delta == ace_threshold and adjacent:
            # 기본 게이트: 직접매칭+인접 라운드만 "확정 올킬"
            # 단, 클립이 너무 적을 때를 위해 직접매칭이 아니면 "후보"로 따로 수집한다.
            reject = ""

            gap = readout.play_gap_sec
            if gap is not None and median_play_gap > 0:
                if gap > median_play_gap * PLAY_GAP_ACE_MAX_RATIO:
                    reject = f"play_gap_wide({gap:.0f}s>{median_play_gap*PLAY_GAP_ACE_MAX_RATIO:.0f}s)"
                elif (
                    prev_round_index is not None
                    and prev_round_index in ace_rounds
                    and gap < median_play_gap * PLAY_GAP_CONSECUTIVE_ACE_MAX_RATIO
                ):
                    reject = (
                        f"consecutive_ace_short_gap({gap:.0f}s"
                        f"<{median_play_gap*PLAY_GAP_CONSECUTIVE_ACE_MAX_RATIO:.0f}s)"
                    )

            cur_total = _match_rounds_total(readout)
            if not reject and prev_match_total is not None and cur_total is not None:
                round_passed = cur_total - prev_match_total
                if round_passed != 1:
                    reject = f"match_rounds_delta={round_passed}"

            cur_pt = _player_team_wins(readout)
            if not reject and prev_pt_wins is not None and cur_pt is not None:
                if cur_pt - prev_pt_wins != 1:
                    reject = f"team_wins_delta={cur_pt - prev_pt_wins}"

            # 팀 K 합산(가능할 때만): 플레이어 ΔK=3인데 팀 합산 증가가 3 미만이면 물리적으로 불가능
            if (
                not reject
                and prev_team_k_sum is not None
                and readout.team_k_sum is not None
                and (readout.team_k_sum - prev_team_k_sum) < ace_threshold
            ):
                reject = f"team_k_sum_delta={readout.team_k_sum - prev_team_k_sum}"

            # D 증가 = 해당 라운드 사망 (라벨: 죽음/사망/1킬1데스) → 올킬 클립 아님
            if (
                not reject
                and prev_deaths is not None
                and readout.deaths is not None
                and readout.deaths > prev_deaths
            ):
                reject = f"deaths_increased({prev_deaths}->{readout.deaths})"

            # K/D 교차(약한 신호): 데스 동일이면 "더블킬일 수도" 있으나,
            # 실제 올킬(ΔK=3)도 데스가 그대로일 수 있어 확정 거절로 쓰면 리콜이 크게 떨어진다.
            # 따라서 reject 대신 candidate 사유로만 기록한다.
            if (
                not reject
                and prev_deaths is not None
                and readout.deaths is not None
                and readout.deaths == prev_deaths
                and delta == ace_threshold
            ):
                readout.ace_candidate_reason = readout.ace_candidate_reason or "kd_same_deaths_check"

            # D=0 갑자기 등장 + 이전 D>0 → K열 OCR 오염 (R39 스코프 워터마크)
            if (
                not reject
                and prev_deaths is not None
                and readout.deaths == 0
                and prev_deaths >= 1
                and delta == ace_threshold
            ):
                reject = "deaths_read_zero"

            if reject:
                readout.ace_reject_reason = reject
            else:
                if both_direct:
                    ace_rounds.append(readout.round_index)
                else:
                    # 후보도 품질 하한선을 둔다 (라벨링된 오답 다수가 inferred 저신뢰)
                    if readout.match_score < 0.55:
                        readout.ace_reject_reason = "cand_match_low"
                    else:
                        readout.ace_candidate_reason = readout.ace_candidate_reason or "inferred_or_non_direct"
                        ace_candidates.append(readout.round_index)

        prev_kills = kills
        prev_deaths = readout.deaths
        prev_inferred = readout.inferred_by_team
        prev_round_index = readout.round_index
        prev_pt_wins = _player_team_wins(readout)
        prev_match_total = _match_rounds_total(readout)
        prev_team_k_sum = readout.team_k_sum

    return deltas, ace_rounds, ace_candidates, k_error_rounds


def read_kills_per_round(
    video_path: Path,
    identity: PlayerIdentity,
    scoreboards: list[ScoreboardWindow],
    *,
    nick_match_threshold: float = 0.65,
    frame_samples: int = 3,
    ace_threshold: int = ACE_KILL_THRESHOLD,
    dataset_root: Path | None = None,
) -> VideoKillTimeline:
    video_path = Path(video_path)
    timeline = VideoKillTimeline(
        video_path=str(video_path),
        player_nick=identity.nickname,
        game_width_median=int(getattr(identity, "game_width_median", 0) or 0),
        layout=("후원패널형" if (getattr(identity, "game_width_median", 0) or 0) and (getattr(identity, "game_width_median", 0) or 0) < 1800 else "풀스크린형"),
    )

    if not identity.nickname:
        timeline.warnings.append("player_nick_empty")
        return timeline
    if not scoreboards:
        timeline.warnings.append("no_scoreboard_windows")
        return timeline

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        timeline.warnings.append("video_open_failed")
        return timeline

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # ── 1차 패스: 전체 OCR + 성공 매칭으로 전역 팀 고정 정보 선수집 ──────────────────────────────
    readouts: list[RoundKillReadout] = []
    frame_rows_cache: dict[int, list[ScoreboardRow]] = {}

    prev_deaths_track: int | None = None
    for idx, window in enumerate(scoreboards):
        readout = read_round_kill(
            cap,
            fps,
            window,
            idx,
            identity.nickname,
            nick_match_threshold=nick_match_threshold,
            frame_samples=frame_samples,
            dataset_root=dataset_root,
            _rows_cache=frame_rows_cache,
            prev_deaths=prev_deaths_track,
        )
        readouts.append(readout)
        if readout.deaths is not None:
            prev_deaths_track = readout.deaths

    # 전역 팀 고정 정보 — 영상 전체 성공 매칭으로 누적
    global_lock = TeamLock()
    for idx, readout in enumerate(readouts):
        if readout.match_score >= nick_match_threshold and readout.row_index is not None:
            cached = frame_rows_cache.get(idx, [])
            matched = next((r for r in cached if r.row_index == readout.row_index), None)
            if matched:
                _update_team_lock(global_lock, matched, cached)

    timeline.team_lock_rounds = global_lock.confirmed_rounds

    # ── 2차 패스: 안전장치 (실패 라운드만, 역방향 적용 포함) ─────────────────────
    # K 연속성 추적
    confirmed_k_by_round: dict[int, int] = {}
    for r in readouts:
        if r.match_score >= nick_match_threshold and r.kills is not None:
            confirmed_k_by_round[r.round_index] = r.kills

    if global_lock.is_locked:
        all_teammate_nicks = global_lock.teammate_nicks  # 전체 아군 닉 집합

        for idx, (readout, window) in enumerate(zip(readouts, scoreboards)):
            if readout.match_score >= nick_match_threshold:
                continue  # 이미 직접 매칭 성공

            # 가장 가까운 앞/뒤 확정 K 탐색
            prev_k_sorted = sorted(
                ((r_idx, k) for r_idx, k in confirmed_k_by_round.items() if r_idx < readout.round_index),
                key=lambda x: x[0],
            )
            prev_kills = prev_k_sorted[-1][1] if prev_k_sorted else None

            # 캐시된 6행 가져오기 (없으면 재 OCR)
            rows = frame_rows_cache.get(idx, [])
            if not rows:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(window.mid_sec * fps))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                rows = read_scoreboard_rows(frame, dataset_root=dataset_root)
                frame_rows_cache[idx] = rows

            # 이번 라운드 팀 동적 결정: 아군 닉 위치로 팀 교체(swap) 자동 감지
            team_in_round = _resolve_team_by_teammates(rows, all_teammate_nicks)
            if team_in_round:
                round_lock = TeamLock(
                    team=team_in_round,
                    confirmed_rounds=1,
                    teammate_nicks=all_teammate_nicks,
                    opponent_nicks=global_lock.opponent_nicks,
                )
            else:
                round_lock = global_lock  # 아군 닉 감지 실패 → 전역 팀 사용

            inferred_row, candidates = _team_fallback_kill(rows, round_lock, prev_kills)

            if inferred_row is not None:
                readout.kills = inferred_row.kills
                readout.nick_matched = inferred_row.nickname
                readout.match_score = round(inferred_row.nick_conf * 0.6, 3)
                readout.row_index = inferred_row.row_index
                readout.inferred_by_team = True
                readout.team_candidates = [
                    {k: v for k, v in c.items() if k != "_row"} for c in candidates
                ]
                timeline.inferred_rounds += 1
                if inferred_row.kills is not None:
                    confirmed_k_by_round[readout.round_index] = inferred_row.kills

    cap.release()

    # 플레이 구간(직전 SB 종료 → 이번 SB 시작) 기록
    prev_end = 0.0
    for readout, window in zip(readouts, scoreboards):
        readout.play_gap_sec = max(0.0, window.start_sec - prev_end)
        prev_end = window.end_sec
    median_gap = _median_play_gap(scoreboards)

    # ── 경고 집계 ──────────────────────────────────────────
    fail_streak = 0
    for readout in readouts:
        if readout.kills is None:
            fail_streak += 1
        else:
            fail_streak = 0
        if fail_streak >= 3:
            timeline.warnings.append("consecutive_round_read_failures")
            break

    if not global_lock.is_locked:
        timeline.warnings.append("team_lock_not_acquired")

    timeline.rounds = readouts
    (
        timeline.delta_kills,
        timeline.ace_rounds,
        timeline.ace_candidates,
        timeline.k_error_rounds,
    ) = compute_delta_kills(
        readouts, ace_threshold=ace_threshold, median_play_gap=median_gap
    )
    return timeline


def load_identity_json(path: Path) -> PlayerIdentity:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return PlayerIdentity(
        nickname=data.get("nickname", ""),
        confidence=float(data.get("confidence", 0.0)),
        mode=data.get("mode", "unknown"),
        sources=data.get("sources", {}),
        game_width_median=int(data.get("game_width_median", 0)),
        samples_total=int(data.get("samples_total", 0)),
        samples_hit=int(data.get("samples_hit", 0)),
    )


def write_kill_timeline_csv(timeline: VideoKillTimeline, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "round_index",
                "start_sec",
                "end_sec",
                "start_mss",
                "kills",
                "delta_k",
                "ace",
                "k_read_error",
                "nick_matched",
                "match_score",
                "row_index",
                "rejoin_reset",
                "inferred_by_team",
            ]
        )
        ace_set = set(timeline.ace_rounds)
        for readout, delta in zip(timeline.rounds, timeline.delta_kills):
            writer.writerow(
                [
                    readout.round_index,
                    f"{readout.scoreboard_start_sec:.2f}",
                    f"{readout.scoreboard_end_sec:.2f}",
                    sec_to_mss(readout.scoreboard_start_sec),
                    "" if readout.kills is None else readout.kills,
                    "" if delta is None else delta,
                    1 if readout.round_index in ace_set else 0,
                    1 if readout.k_read_error else 0,
                    readout.nick_matched,
                    f"{readout.match_score:.3f}",
                    "" if readout.row_index is None else readout.row_index,
                    1 if readout.rejoin_reset else 0,
                    1 if readout.inferred_by_team else 0,
                ]
            )


def sec_to_mss(sec: float) -> str:
    """초 → M:SS 표기 (예: 217.3 → '3:37')."""
    total = int(sec)
    return f"{total // 60}:{total % 60:02d}"


def format_report(timeline: VideoKillTimeline) -> str:
    inferred_cnt = sum(1 for r in timeline.rounds if r.inferred_by_team)
    ace_mss = [sec_to_mss(r.scoreboard_start_sec)
               for r in timeline.rounds if r.round_index in timeline.ace_rounds]
    cand_mss = [sec_to_mss(r.scoreboard_start_sec)
                for r in timeline.rounds if r.round_index in timeline.ace_candidates]
    lines = [
        f"## {Path(timeline.video_path).name}",
        f"player: {timeline.player_nick!r}  ace_rounds={timeline.ace_rounds} ({ace_mss})",
        f"candidates: {timeline.ace_candidates} ({cand_mss})",
        (
            f"rounds={len(timeline.rounds)}  warnings={timeline.warnings}  "
            f"team_lock={timeline.team_lock_rounds}  inferred={inferred_cnt}  "
            f"k_errors={len(timeline.k_error_rounds)}"
        ),
        "timeline:",
    ]
    ace_set = set(timeline.ace_rounds)
    cand_set = set(timeline.ace_candidates)
    err_set = set(timeline.k_error_rounds)
    for readout, delta in zip(timeline.rounds, timeline.delta_kills):
        delta_txt = "" if delta is None else str(delta)
        ace = " ACE" if readout.round_index in ace_set else ""
        cand = " CAND" if readout.round_index in cand_set else ""
        kerr = " K_ERR" if readout.k_read_error else ""
        kills_txt = "?" if readout.kills is None else str(readout.kills)
        deaths_txt = "" if readout.deaths is None else f"/{readout.deaths}"
        inferred_tag = " [T]" if readout.inferred_by_team else ""
        reject = f" REJ:{readout.ace_reject_reason}" if readout.ace_reject_reason else ""
        cand_reason = (
            f" CAND:{readout.ace_candidate_reason}"
            if readout.ace_candidate_reason
            else ""
        )
        time_txt = sec_to_mss(readout.scoreboard_start_sec)
        lines.append(
            f"  R{readout.round_index:02d} {time_txt:>6} "
            f"K={kills_txt}{deaths_txt} ΔK={delta_txt}{ace}{cand}{kerr}{inferred_tag}{reject}{cand_reason} "
            f"nick={readout.nick_matched!r} match={readout.match_score:.2f}"
        )
        if readout.inferred_by_team and readout.team_candidates:
            for c in readout.team_candidates:
                tag = "O" if c.get("k_ok") else "X"
                lines.append(
                    f"       {tag} [{c.get('row_index','-')}] {c.get('nickname','?')!r} "
                    f"K={c.get('kills','?')} conf={c.get('nick_conf',0):.2f} "
                    f"{c.get('reasons','')}"
                )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="스코어보드 K 읽기 + ΔK 올킬 판별 (v1)")
    parser.add_argument("video_path", help="입력 mp4/mkv")
    parser.add_argument("--identity", default=None, help="player_identity.json (없으면 자동 확정)")
    parser.add_argument(
        "--player-nick",
        default=None,
        help="본인 닉 수동 지정 (특수문자·잘림 닉 — 자동 확정 실패 시)",
    )
    parser.add_argument("--rounds-dir", default=None)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--nick-match-threshold", type=float, default=0.65)
    parser.add_argument("--frame-samples", type=int, default=3)
    parser.add_argument("--ace-threshold", type=int, default=ACE_KILL_THRESHOLD)
    parser.add_argument("--out", default=None, help="kill_timeline.csv 경로")
    parser.add_argument("--json-out", default=None, help="전체 결과 JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    video_path = Path(args.video_path)
    dataset_root = Path(args.dataset_root)
    rounds_dir = Path(args.rounds_dir) if args.rounds_dir else None

    sb_csv = find_scoreboard_csv(video_path, rounds_dir, dataset_root)
    sb_count = len(load_scoreboard_windows(sb_csv)) if sb_csv else 0

    if args.player_nick:
        identity = PlayerIdentity(
            nickname=args.player_nick.strip(),
            confidence=1.0,
            mode="manual",
            sources={"accepted": True, "manual": True},
        )
        print(f"[k-reader] 닉 수동 지정 → {identity.nickname!r}", flush=True)
    elif args.identity:
        identity = load_identity_json(Path(args.identity))
    else:
        print("[k-reader] identity JSON 없음 → player_identity 자동 실행", flush=True)
        identity = resolve_player_identity(
            video_path,
            rounds_dir=rounds_dir,
            dataset_root=dataset_root,
            max_samples=80,
            max_scoreboards=min(40, max(sb_count, 12)),
        )

    if not identity.nickname:
        print(f"[k-reader] 닉 확정 실패 → SKIP ({identity.sources.get('reason', 'unknown')})")
        print(
            "[k-reader] 힌트: 특수문자·잘림 닉은 --player-nick '핵심4글자' 로 수동 지정",
            flush=True,
        )
        return 2

    if not sb_csv:
        print("[k-reader] detected_scoreboards.csv 없음 → detect_rounds 먼저 실행")
        return 1

    windows = load_scoreboard_windows(sb_csv)
    print(
        f"[k-reader] {video_path.name} nick={identity.nickname!r} "
        f"scoreboards={len(windows)}",
        flush=True,
    )

    timeline = read_kills_per_round(
        video_path,
        identity,
        windows,
        nick_match_threshold=args.nick_match_threshold,
        frame_samples=args.frame_samples,
        ace_threshold=args.ace_threshold,
        dataset_root=dataset_root,
    )

    print(format_report(timeline), flush=True)

    if args.out:
        write_kill_timeline_csv(timeline, Path(args.out))
        print(f"[k-reader] saved -> {args.out}", flush=True)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(timeline)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[k-reader] json -> {args.json_out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
