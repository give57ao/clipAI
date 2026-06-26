# -*- coding: utf-8 -*-
"""1b-2 단계: 확정 닉으로 전체스코어 K 읽기 → 라운드 간 ΔK → 올킬 후보.

설계 문서: PLAYER_IDENTITY_AND_K_READER.md
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2

from nick_fuzzy import normalize_nick, nick_match_text
from player_identity import PlayerIdentity, resolve_player_identity
from scoreboard_layout import (
    ScoreboardRow,
    ScoreboardWindow,
    find_scoreboard_csv,
    load_scoreboard_windows,
    read_scoreboard_rows,
)
from scouter_nick import levenshtein

DEFAULT_DATASET_ROOT = Path(r"E:\Highlights\ml_dataset")
ACE_KILL_THRESHOLD = 3


@dataclass
class RoundKillReadout:
    round_index: int
    scoreboard_start_sec: float
    scoreboard_end_sec: float
    kills: int | None
    nick_matched: str
    match_score: float
    row_index: int | None = None
    samples_used: int = 0
    rejoin_reset: bool = False
    # 안전장치: 팀 고정 + K 연속성으로 추론한 경우 True (재검증 필요)
    inferred_by_team: bool = False
    # 안전장치에서 후보로 올라온 팀원 닉/K 목록 (디버그용)
    team_candidates: list[dict] = field(default_factory=list)


@dataclass
class VideoKillTimeline:
    video_path: str
    player_nick: str
    rounds: list[RoundKillReadout] = field(default_factory=list)
    delta_kills: list[int | None] = field(default_factory=list)
    ace_rounds: list[int] = field(default_factory=list)
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


def _majority_k(values: list[int]) -> int | None:
    if not values:
        return None
    top, count = Counter(values).most_common(1)[0]
    if count >= 2 or len(values) == 1:
        return top
    return top


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
) -> RoundKillReadout:
    """한 스코어보드 구간에서 본인 K 읽기 (프레임 다수결)."""
    frame_hits: list[tuple[float, ScoreboardRow]] = []
    kill_candidates: list[int] = []
    samples_used = 0
    all_rows_seen: list[ScoreboardRow] = []

    for sample_sec in sample_times_in_window(window, frame_samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sample_sec * fps))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        samples_used += 1
        rows = read_scoreboard_rows(frame, dataset_root=dataset_root)
        all_rows_seen.extend(rows)
        for score, row in _rows_matching_player(
            rows, player_nick, nick_match_threshold=nick_match_threshold
        ):
            frame_hits.append((score, row))
            if row.kills is not None:
                kill_candidates.append(row.kills)

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

    # 닉 매칭 프레임에서 K를 못 읽은 경우, 동일 row_index의 다른 프레임 K도 포함
    if not kill_candidates:
        kill_candidates = [
            r.kills for r in all_rows_seen
            if r.row_index == matched_row.row_index and r.kills is not None
        ]

    kills = _majority_k(kill_candidates)
    return RoundKillReadout(
        round_index=round_index,
        scoreboard_start_sec=window.start_sec,
        scoreboard_end_sec=window.end_sec,
        kills=kills,
        nick_matched=matched_row.nickname,
        match_score=round(best_score, 3),
        row_index=matched_row.row_index,
        samples_used=samples_used,
    )


def compute_delta_kills(
    readouts: list[RoundKillReadout],
    *,
    ace_threshold: int = ACE_KILL_THRESHOLD,
) -> tuple[list[int | None], list[int]]:
    """라운드 간 ΔK + 올킬(ΔK>=3) 라운드 인덱스. 리조인 시 기준점 리셋."""
    deltas: list[int | None] = []
    ace_rounds: list[int] = []
    prev_kills: int | None = None

    for readout in readouts:
        kills = readout.kills
        if kills is None:
            deltas.append(None)
            continue

        if prev_kills is None:
            deltas.append(None)
            prev_kills = kills
            continue

        if kills < prev_kills:
            readout.rejoin_reset = True
            deltas.append(None)
            prev_kills = kills
            continue

        delta = kills - prev_kills
        deltas.append(delta)
        if delta >= ace_threshold:
            ace_rounds.append(readout.round_index)
        prev_kills = kills

    return deltas, ace_rounds


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
        )
        readouts.append(readout)

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
    timeline.delta_kills, timeline.ace_rounds = compute_delta_kills(readouts, ace_threshold=ace_threshold)
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
                "kills",
                "delta_k",
                "ace",
                "nick_matched",
                "match_score",
                "row_index",
                "rejoin_reset",
                "inferred_by_team",  # ★ 안전장치 사용 여부 (1=재검증 필요)
            ]
        )
        for readout, delta in zip(timeline.rounds, timeline.delta_kills):
            writer.writerow(
                [
                    readout.round_index,
                    f"{readout.scoreboard_start_sec:.2f}",
                    f"{readout.scoreboard_end_sec:.2f}",
                    "" if readout.kills is None else readout.kills,
                    "" if delta is None else delta,
                    1 if readout.round_index in timeline.ace_rounds else 0,
                    readout.nick_matched,
                    f"{readout.match_score:.3f}",
                    "" if readout.row_index is None else readout.row_index,
                    1 if readout.rejoin_reset else 0,
                    1 if readout.inferred_by_team else 0,
                ]
            )


def format_report(timeline: VideoKillTimeline) -> str:
    inferred_cnt = sum(1 for r in timeline.rounds if r.inferred_by_team)
    lines = [
        f"## {Path(timeline.video_path).name}",
        f"player: {timeline.player_nick!r}  ace_rounds={timeline.ace_rounds}",
        (
            f"rounds={len(timeline.rounds)}  warnings={timeline.warnings}  "
            f"team_lock={timeline.team_lock_rounds}  inferred={inferred_cnt}"
        ),
        "timeline:",
    ]
    for readout, delta in zip(timeline.rounds, timeline.delta_kills):
        delta_txt = "" if delta is None else str(delta)
        ace = " ACE" if readout.round_index in timeline.ace_rounds else ""
        kills_txt = "?" if readout.kills is None else str(readout.kills)
        # ★ 안전장치 사용 라운드는 [T] 표시 → 재검증 필요
        inferred_tag = " [T]" if readout.inferred_by_team else ""
        lines.append(
            f"  R{readout.round_index:02d} {readout.scoreboard_start_sec:7.1f}s "
            f"K={kills_txt} ΔK={delta_txt}{ace}{inferred_tag} "
            f"nick={readout.nick_matched!r} match={readout.match_score:.2f}"
        )
        # 안전장치 후보 목록 (디버그)
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

    if args.identity:
        identity = load_identity_json(Path(args.identity))
    else:
        print("[k-reader] identity JSON 없음 → player_identity 자동 실행", flush=True)
        identity = resolve_player_identity(
            video_path,
            rounds_dir=rounds_dir,
            dataset_root=dataset_root,
            max_samples=40,
            max_scoreboards=12,
        )

    if not identity.nickname:
        print(f"[k-reader] 닉 확정 실패 → SKIP ({identity.sources.get('reason', 'unknown')})")
        return 2

    sb_csv = find_scoreboard_csv(video_path, rounds_dir, dataset_root)
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
