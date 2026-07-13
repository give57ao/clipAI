# -*- coding: utf-8
"""HUD K/D/A 기반 올킬 탐지 — 단일 패스 + ΔK + 템플릿."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import cv2

from game_frame import extract_game_crop_bgr
from hud_digit_match import get_hud_digit_matcher
from hud_kda import read_kda_triple_from_game
from hud_round_settle import SettledRound, settle_rounds

DEFAULT_DATASET_ROOT = Path(r"E:\Highlights\ml_dataset")
DEFAULT_OUTPUT_DIR = Path(r"E:\clipai_result\ace_clips_hud")
DEFAULT_JSON_DIR = Path(r"E:\clipai_result\hud_timeline")
ACE_KILLS = 3
# --- _KTracker v2: 증거창(evidence window) 확정 (2026-07-06 Sonnet, SONNET_TASK.md §3) ---
# v1("연속 N회 동일판독")은 라운드 막판 HUD 깜빡임에 취약 — 중간값(예: 8)이 연속을
# 못 채우면 확정이 뒤처지다 한 번에 큰 폭(예: 6→9, +3)으로 뭉쳐 가짜 올킬 생성.
# v2는 "최근 _EV_WINDOW초 안에 증거가 몇 번 쌓였는지"로 확정하고, confirmed+1부터
# 오름차순으로 하나씩 확정(체인)해 각 킬을 그 값이 처음 관측된 시각에 정확히 귀속.
_EV_WINDOW = 4.0        # 증거 수집 창(초). 6초 이상 금지 — 이전 라운드 잔존증거가 새는 창이 됨
_EV_CONFIRM_HI = 2      # conf>=_CONF_STRONG 판독 포함 시 확정에 필요한 창내 증거 수
_EV_CONFIRM_LO = 3      # 전부 저신뢰면 요구 증거 수 (가드 G2: 흐릿한 프레임 연쇄 오독 방지)
_CONF_STRONG = 0.75
_EV_REBASE = 5          # 하향/+4↑ 리베이스 확정 증거 수 (가드 G1: 큰 점프 오독 방지)
# _EV_REBASE_DOWN = 8   # Task 4a: 순손실 없이 79:51만 해결은 phantom-8로 충분, FP 증가로 비활성
_ROLLBACK_WINDOW = 4.0  # 킬 확정 직후 되돌림 감시 창 (가드 G4)
_EV_ROLLBACK = 3        # 되돌림에 필요한 이전값(from_k) 재관측 증거 수
# ★ 라운드 경계 = 연속 row_miss (KDA HUD 사라짐 = 풀 스코어보드/전환 화면).
#   도메인 정의 "라운드 = 풀 스코어보드 사이 구간"과 일치. 전멸 아이콘(analyze_hud_icons)보다
#   훨씬 깨끗한 신호 — 캐시 스윕에서 recall 33%→79%, FP 대폭 감소 (2026-07-06 Opus).
_BOUNDARY_ROWMISS = 18     # 경계 인정 최소 연속 row_miss 프레임(4fps≈4.5s). 실측: 18~20 안정,
                           #   <16이면 플레이 중 순간 가림에 과분할, ≥24면 스코어보드 놓쳐 병합
_MAX_ACE_SPAN_SEC = 90.0   # 올킬 3킬은 한 라운드(≤90s) 내 (그 이상은 경계 놓친 병합 의심)
# G5: 라운드 신뢰도 가드 — 라운드 전체가 거의 row_miss인데 그 틈에 잠깐 보인 값이
# 이전 확정값보다 커서(예: 5→8 한 번의 점프) ace로 오판되는 사례 실측(23-00-50 R15:
# 22초 라운드에 K 판독 성공 단 1건, 그마저 3만큼 뛴 값). 라운드 내 K 판독 성공 총
# 횟수가 너무 적으면(경계 오검출로 잘려나온 가짜 라운드일 위험) ace 제외.
_MIN_ROUND_K_SAMPLES = 10  # 임계값 스윕 실측: 6~8보다 FP를 더 줄이면서 recall 손실 없음
_KILL_GRACE_SEC = 10.0     # R2 Task 2: 경계 직후 킬이 이전 라운드 막판일 수 있는 최대 간격

# R6(2026-07-13): 상단 스코어(팀 승수) 게이트 — 폭0(kill_times 전부 동일 시각)
# ace 후보 전용. Task 3 실측(HUD_ACE_HANDOFF.md R6 절): 라운드 span 안에 승수
# win 이벤트가 "있냐 없냐"는 TP/FP가 안 갈림(둘 다 대부분 1개 — 라운드가 이겨서
# 끝나면 당연히 뜨는 정상 신호라 변별력이 없음). **2개 이상**(=그 라운드 안에서
# 승부가 두 번 결정됐다는 뜻 = 라운드 병합 직접 증거)만 유의미했음 — 폭0 FP 18건
# 중 1건(01-01-04)만 이 조건으로 걸러짐(약한 신호, 완전 기각은 아님). TP 손실 0
# 보장(TP 3건 전부 1로 안 걸림) — 위험 없는 국소 게이트라 채택.
_SCORE_SPLIT_MIN_EVENTS = 2

# 클립 구간 — end-35 방식 대신 라운드 시작·첫 킬 기준
_CLIP_PRE_ROUND_SEC = 6.0   # 라운드 시작 직전
_CLIP_PRE_KILL_SEC = 14.0   # 첫 킬 이전 (킬 장면이 클립 앞쪽에 오도록)
_CLIP_POST_END_SEC = 1.5    # 라운드 종료 직후 (스코어 잠깐만)
_CLIP_MAX_SEC = 55.0


@dataclass
class RoundTrack:
    round_index: int
    start_sec: float
    end_sec: float = 0.0
    kills: int = 0
    kill_times: list[float] = field(default_factory=list)
    resets: int = 0
    reset_times: list[float] = field(default_factory=list)
    k_samples: int = 0  # 라운드 구간 내 K 판독 성공(원시 프레임) 총 횟수 — 라운드 신뢰도(G5) 지표
    ace: bool = False
    end_reason: str = "hud_elim"
    first_kill_sec: float | None = None
    ace_sec: float | None = None


@dataclass
class KillEvent:
    t: float          # 새 K값이 처음 관측된 시각
    from_k: int
    to_k: int


@dataclass
class KRead:
    """프레임 1개의 원시 K 판독 — 신호 캐시의 단위 (hud_sig_cache.py).

    method: 'template'(성공) | 'template_miss' | 'row_miss' | 'triple_incomplete'
    d/a: D(데스)/A(어시) 슬롯 — 2026-07-09 오탐 검수로 필요성 확정 (사망·관전 오독
         판별에 D 채널이 결정적). 구 캐시에는 없음 → None (가드 자동 비활성).
    """
    t: float
    k: int | None
    conf: float
    method: str
    d: int | None = None
    a: int | None = None


@dataclass
class HudAceTimeline:
    video_path: str
    scan_fps: float
    ace_kills: int
    rounds: list[RoundTrack] = field(default_factory=list)
    ace_rounds: list[int] = field(default_factory=list)
    hud_end_count: int = 0
    k_template_hits: int = 0
    k_template_miss: int = 0
    k_row_miss: int = 0
    kill_events: list[KillEvent] = field(default_factory=list)
    reset_events: list[list[float]] = field(default_factory=list)  # [t, old, new]
    warnings: list[str] = field(default_factory=list)


class _KTracker:
    """누적 K 확정값 상태머신 v2 — 증거창(evidence window) 확정.

    값마다 "최근 _EV_WINDOW초 안에 몇 번, 어떤 신뢰도로 보였는가"를 누적하고,
    confirmed+1부터 오름차순으로 하나씩 확정(체인)한다. 이렇게 하면 중간값(예: 8)이
    띄엄띄엄(창 안이면 연속 아니어도 OK) 보이기만 해도 확정되어, v1처럼 확정이
    뒤처지다 한 번에 큰 폭으로 뭉쳐 가짜 올킬을 만드는 문제가 구조적으로 사라진다.
    킬은 그 값이 처음 관측된 시각(first_t)에 기록 — 실제 킬 타이밍과 라운드 귀속이 정확해짐.

    가드 5종 (오독 FP 방어, G5는 _assign_events에 위치):
      G1 리베이스(_EV_REBASE=5) — confirmed+3 초과·하향은 증거 5회 필요 (큰 점프 오독 방지)
      G2 신뢰도 연동(_EV_CONFIRM_HI/LO) — 저신뢰뿐이면 3회 요구 (흐릿한 프레임 연쇄 오독 방지)
      G3 트리플가드 — collect_reads에서 K/D/A 셋 다 파싱된 프레임만 채택 (배너/페이드 오염 방지)
      G4 킬 롤백 — 킬 확정 직후 이전 값이 다시 쌓이고 확정값 증거가 안 늘면 되돌림
          (순간 오독 2프레임이 킬로 확정된 경우의 안전핀)
      G5 라운드 신뢰도(_assign_events, _MIN_ROUND_K_SAMPLES) — 판독 성공이 극히 드문
          라운드(row_miss로 잘려나온 가짜 라운드일 위험)는 킬 합이 맞아도 ace 제외
    """

    def __init__(self) -> None:
        self.confirmed: int | None = None
        self.ev: dict[int, list[tuple[float, float]]] = {}  # k값 → [(t, conf), ...] (창 내)
        self.kills: list[KillEvent] = []
        self.resets: list[tuple[float, int, int]] = []
        self.last_kill: tuple[float, int, int] | None = None  # (kill_t, from_k, to_k)

    def _prune(self, t: float) -> None:
        # 리셋(하프타임/리조인)은 도메인상 항상 K=0으로 감 — 그 값의 증거만 시간
        # 만료 없이 누적. HUD 블랙아웃이 길어 "0" 판독이 수십 초에 걸쳐 드물게만
        # 보일 수 있음(실측: 02-34-09 12:48~13:37, 48초간 "0" 판독 6회뿐).
        # 값별로 독립 누적하므로 v1과 달리 중간에 다른 값(오독)이 끼어도 끊기지 않음.
        #
        # ⚠ 상향 킬 후보(confirmed<v<=confirmed+3)에도 같은 무제한 누적을 시도했으나
        # (00-40-56 65:15/65:24, 9.25초 간격 "8" 사례 개선 목적) 여러 영상에서 오탐이
        # FP 7→20으로 폭증해 순손실 확인·되돌림(2026-07-06). 스파스 중간값 문제는
        # v==0(리셋)에서만 안전하게 해결됐고, 일반 킬체인 중간값 문제는 미해결로 남음
        # — HUD_ACE_HANDOFF.md에 다음 과제로 기록.
        cutoff = t - _EV_WINDOW
        for key in list(self.ev.keys()):
            if key == 0 and self.confirmed is not None and self.confirmed != 0:
                continue
            kept = [(tt, c) for tt, c in self.ev[key] if tt >= cutoff]
            if kept:
                self.ev[key] = kept
            else:
                del self.ev[key]

    @staticmethod
    def _strong(entries: list[tuple[float, float]]) -> bool:
        if len(entries) >= _EV_CONFIRM_LO:
            return True
        return len(entries) >= _EV_CONFIRM_HI and max(c for _, c in entries) >= _CONF_STRONG

    @staticmethod
    def _first_t(entries: list[tuple[float, float]]) -> float:
        return min(tt for tt, _ in entries)

    def _rebase_threshold(self, v: int) -> int:
        return _EV_REBASE

    def _confirm_to(self, k: int) -> None:
        self.confirmed = k
        self.ev = {v: e for v, e in self.ev.items() if v > k}

    def update(self, t: float, k: int | None, conf: float = 1.0) -> None:
        if k is None:
            return
        self._prune(t)
        self.ev.setdefault(k, []).append((t, conf))

        if self.confirmed is None:
            if self._strong(self.ev.get(k, [])):
                self._confirm_to(k)
            return

        if k == self.confirmed:
            return

        # G4: 킬 롤백 — 확정 직후(_ROLLBACK_WINDOW 내) 이전 값(from_k)이 다시 쌓이고,
        # 확정값(to_k)의 증거가 그 뒤로 더 늘지 않았으면 순간 오독으로 보고 되돌림.
        if (
            self.last_kill is not None
            and k == self.last_kill[1]
            and (t - self.last_kill[0]) <= _ROLLBACK_WINDOW
            and len(self.ev.get(k, [])) >= _EV_ROLLBACK
            and len(self.ev.get(self.last_kill[2], [])) == 0
        ):
            lk = self.last_kill
            if self.kills and self.kills[-1].from_k == lk[1] and self.kills[-1].to_k == lk[2]:
                self.kills.pop()
            self.confirmed = lk[1]
            self.last_kill = None
            self.ev = {}
            return

        # 상향 체인: confirmed+1부터 오름차순으로 확정 시도 (핵심).
        # 9의 증거가 먼저 쌓여도 7·8 증거가 있으면 7→8→9 순서로 확정되어
        # 각 킬이 제 시각·제 라운드로 흩어짐 (6→9 +3 뭉침 방지).
        while True:
            candidates = [
                v for v, entries in self.ev.items()
                if self.confirmed < v <= self.confirmed + ACE_KILLS and self._strong(entries)
            ]
            if not candidates:
                break
            nxt = min(candidates)
            kill_t = self._first_t(self.ev[nxt])
            self.kills.append(KillEvent(kill_t, self.confirmed, nxt))
            self.last_kill = (kill_t, self.confirmed, nxt)
            self._confirm_to(nxt)

        # G1: 리베이스 (하향 또는 confirmed+3 초과) — 진짜 리셋/큰 점프는 증거 5회로 확정.
        #
        # ⚠ 0이 아닌 하향(v<confirmed, v!=0)을 리베이스 대상에서 제외해봤으나
        # (03-02-03 10:22 "8→7" 오독 뭉침이 라운드를 쪼개는 사례 개선 목적) 트래커가
        # 전역 순차 상태를 유지하는 구조라 이 시점 이후 confirmed 궤적 전체가 달라져
        # 다른 3곳(00-44-50 9:00, 02-03-10 46:32, 00-42-33 36:40)의 TP를 깨뜨림
        # (recall 51.9%→44.4%, 순손실) → 되돌림(2026-07-06). 국소 수정이 전역
        # 리플렉트를 일으키는 구조적 한계 — 라운드별 독립 트래커로 재설계해야 안전할 수 있음.
        reb_candidates = [
            v for v, entries in self.ev.items()
            if len(entries) >= self._rebase_threshold(v)
            and (v < self.confirmed or v > self.confirmed + ACE_KILLS)
        ]
        if reb_candidates:
            v_reb = max(reb_candidates, key=lambda v: len(self.ev[v]))
            reset_t = self._first_t(self.ev[v_reb])
            self.resets.append((reset_t, self.confirmed, v_reb))
            self.confirmed = v_reb
            self.ev = {}
            self.last_kill = None


def rowmiss_runs(reads: list[KRead]) -> list[tuple[float, float, int]]:
    """row_miss 연속 구간(경계 후보, ≥`_BOUNDARY_ROWMISS`) 목록: (start, end, n_frames).

    `timeline_from_reads`의 경계 검출과 `hud_boundary_verify.py`(R2 Task 1, 전광판
    CNN 스팟검증)가 이 함수를 공유 — 두 곳의 "run" 정의가 어긋나면 검증 결과를
    올바른 run에 매칭할 수 없으므로 반드시 이 함수 하나로만 계산할 것.
    """
    runs: list[tuple[float, float, int]] = []
    run_start = 0.0
    run_last = 0.0
    n = 0
    for r in reads:
        if r.method == "row_miss":
            if n == 0:
                run_start = r.t
            n += 1
            run_last = r.t
        else:
            if n >= _BOUNDARY_ROWMISS:
                runs.append((run_start, run_last, n))
            n = 0
    if n >= _BOUNDARY_ROWMISS:
        runs.append((run_start, run_last, n))
    return runs


def _lookup_boundary_verdict(
    verdicts: list[list], start: float, last: float, tol: float = 0.05
) -> bool | None:
    """(start, last)에 맞는 검증 결과를 찾음. 못 찾으면 None(안전하게 경계 유지 취급)."""
    for vs, vl, verdict in verdicts:
        if abs(vs - start) < tol and abs(vl - last) < tol:
            return bool(verdict)
    return None


def _has_successful_k_read(
    reads: list[KRead], t_lo: float, t_hi: float, k_val: int
) -> bool:
    """(t_lo, t_hi) 구간에 from_k 성공 판독(template)이 있는지."""
    for r in reads:
        if t_lo < r.t < t_hi and r.k == k_val and r.method == "template":
            return True
    return False


def _miss_gap_starts(reads: list[KRead], t_lo: float, t_hi: float) -> list[float]:
    """template_miss 연속 구간의 시작 시각 (R2 Task 3: 팬텀 8)."""
    gaps: list[float] = []
    in_gap = False
    for r in sorted(reads, key=lambda x: x.t):
        if not (t_lo < r.t < t_hi):
            continue
        if r.method == "template_miss":
            if not in_gap:
                gaps.append(r.t)
                in_gap = True
        else:
            in_gap = False
    return gaps


def _split_phantom_eight_kills(
    kills: list[KillEvent],
    reads: list[KRead],
    boundaries: list[float],
) -> list[KillEvent]:
    """7→9(+2) 킬을 7→8 + 8→9로 분리 — 8은 EXCLUDE_DIGITS로 template_miss만 남음."""
    out: list[KillEvent] = []
    for ev in kills:
        if ev.from_k != 7 or ev.to_k != 9:
            out.append(ev)
            continue
        last_7_t = max(
            (r.t for r in reads if r.k == 7 and r.method == "template" and r.t < ev.t),
            default=None,
        )
        if last_7_t is None:
            out.append(ev)
            continue
        bounds = [b for b in boundaries if last_7_t < b < ev.t]
        if not bounds:
            out.append(ev)
            continue
        bnd = bounds[0]
        gap_starts = _miss_gap_starts(reads, last_7_t, bnd)
        if not gap_starts:
            out.append(ev)
            continue
        t8 = gap_starts[-1]
        out.append(KillEvent(t8, 7, 8))
        out.append(KillEvent(ev.t, 8, 9))
    return out


def _inject_phantom_eight_kills(
    kills: list[KillEvent],
    reads: list[KRead],
    boundaries: list[float],
) -> list[KillEvent]:
    """7 확정 후 9로 스킵된 체인에 7→8, 8→9 삽입 (7→9 이벤트 없을 때)."""
    out: list[KillEvent] = []
    for i, ev in enumerate(kills):
        out.append(ev)
        if ev.to_k != 7:
            continue
        nxt = kills[i + 1] if i + 1 < len(kills) else None
        if nxt is None or nxt.from_k < 9:
            continue
        first9 = min(
            (
                r.t
                for r in reads
                if r.k == 9 and r.method == "template" and ev.t < r.t <= nxt.t + 1.0
            ),
            default=nxt.t,
        )
        gaps = _miss_gap_starts(reads, ev.t, first9)
        if not gaps:
            continue
        bounds = [b for b in boundaries if ev.t < b < first9]
        if bounds:
            before = [g for g in gaps if g < bounds[0]]
            t8 = before[-1] if before else gaps[0]
        else:
            t8 = gaps[0]
        out.append(KillEvent(t8, 7, 8))
        out.append(KillEvent(first9, 8, 9))
    return out


def _postprocess_kills(
    kills: list[KillEvent],
    reads: list[KRead],
    boundaries: list[float],
) -> list[KillEvent]:
    k = _split_phantom_eight_kills(kills, reads, boundaries)
    return _inject_phantom_eight_kills(k, reads, boundaries)


def _kill_round_index(
    ev: KillEvent,
    rounds: list[RoundTrack],
    boundaries: list[float],
    reads: list[KRead],
) -> int | None:
    """킬 이벤트가 속할 라운드 인덱스 (R2 Task 2 그레이스 포함)."""

    def round_for(t: float) -> int | None:
        for i, r in enumerate(rounds):
            if r.start_sec <= t < r.end_sec:
                return i
        return len(rounds) - 1 if rounds and t >= rounds[-1].end_sec else None

    idx = round_for(ev.t)
    if idx is None:
        return None
    # R2 Task 2: 경계 직후 막판 킬 귀속 (조건 엄격 — 이전 라운드에 from_k 있고 경계 후엔 없을 때만)
    prev_bounds = [b for b in boundaries if b < ev.t]
    if not prev_bounds:
        return idx
    bnd = max(prev_bounds)
    if ev.t - bnd > _KILL_GRACE_SEC or ev.t <= bnd:
        return idx
    prev_idx = round_for(bnd - 0.001)
    if prev_idx is None or prev_idx == idx:
        return idx
    r_prev = rounds[prev_idx]
    if not _has_successful_k_read(reads, r_prev.start_sec, bnd, ev.from_k):
        return idx
    if _has_successful_k_read(reads, bnd, ev.t, ev.from_k):
        return idx
    # +1/+2만 귀속 (큰 점프는 다음 라운드 진짜 킬일 수 있음)
    if ev.to_k - ev.from_k > 2:
        return idx
    return prev_idx


def _build_rounds(
    boundaries: list[float],
    tracker: _KTracker,
    duration: float,
) -> list[RoundTrack]:
    """경계(연속 row_miss 중앙) + 리셋 시각으로 라운드 세그먼트 생성.

    리셋(하프타임·리조인)도 라운드 경계 — 서로 다른 매치/반의 킬이 섞이지 않도록.
    """
    reset_ts = [t for (t, _o, _n) in tracker.resets]
    seps = sorted(set(boundaries) | set(reset_ts))
    rounds: list[RoundTrack] = []
    prev = 0.0
    idx = 0
    for sp in seps + [duration]:
        if sp - prev > 0.5:
            rounds.append(RoundTrack(round_index=idx, start_sec=prev, end_sec=sp))
            idx += 1
        prev = sp
    return rounds


def _assign_events(
    rounds: list[RoundTrack],
    kills: list[KillEvent],
    k_read_times: list[float],
    boundaries: list[float],
    reads: list[KRead],
    resets: list[tuple[float, int, int]],
) -> None:
    """킬/리셋 이벤트를 라운드(세그먼트)에 귀속하고 ace 판정.

    ace = 세그먼트 내 킬 합 정확히 3 + 리셋 없음 + 3킬 스팬 ≤ _MAX_ACE_SPAN_SEC
    + 라운드 내 K 판독 성공 횟수 ≥ _MIN_ROUND_K_SAMPLES(G5).
    경계가 row_miss(스코어보드)라 세그먼트가 곧 도메인상 라운드와 일치.
    """
    if not rounds:
        return

    def round_for(t: float) -> int | None:
        for i, r in enumerate(rounds):
            if r.start_sec <= t < r.end_sec:
                return i
        return len(rounds) - 1 if t >= rounds[-1].end_sec else None

    for ev in kills:
        idx = _kill_round_index(ev, rounds, boundaries, reads)
        if idx is None:
            continue
        r = rounds[idx]
        r.kills += ev.to_k - ev.from_k
        r.kill_times.append(ev.t)
        if r.first_kill_sec is None or ev.t < r.first_kill_sec:
            r.first_kill_sec = ev.t
        if r.ace_sec is None and r.kills >= ACE_KILLS:
            r.ace_sec = ev.t

    # G5: 라운드 신뢰도 — 판독 성공(원시 프레임) 총 횟수. kill_times(확정 전이 수)와
    # 별개로, 라운드가 거의 전부 row_miss인데 그 틈의 값이 우연히 +3 떨어져 있어
    # ace로 오판되는 사례 방지(23-00-50 R15: 22s 라운드에 판독 성공 단 1회).
    for t in k_read_times:
        idx = round_for(t)
        if idx is not None:
            rounds[idx].k_samples += 1

    # 리셋은 세그먼트 경계(_build_rounds에서 분리) — 경계에 걸친 리셋을 내부로
    # 오귀속하면 그 세그먼트의 정상 올킬을 잘못 기각(02-21-23 64:44 실측).
    # 엄격 내부(start < t < end)만 카운트: 정상적으론 발생 안 하는 방어용.
    for (t, _old, _new) in resets:
        for r in rounds:
            if r.start_sec < t < r.end_sec:
                r.resets += 1
                r.reset_times.append(t)
                break

    for r in rounds:
        span = (
            (r.ace_sec - r.first_kill_sec)
            if (r.ace_sec is not None and r.first_kill_sec is not None)
            else 0.0
        )
        r.ace = (
            r.kills == ACE_KILLS
            and r.resets == 0
            and span <= _MAX_ACE_SPAN_SEC
            and r.k_samples >= _MIN_ROUND_K_SAMPLES
        )


def scan_hud_aces(
    video_path: Path,
    *,
    scan_fps: float = 4.0,
    ace_kills: int = ACE_KILLS,
    dataset_root: Path | None = None,
    verify_boundaries: bool = True,
    verify_score_wins: bool = True,
) -> HudAceTimeline:
    """R5(2026-07-09): verify_boundaries=True(기본)면 row_miss 경계 후보를 전광판
    CNN으로 스팟 검증(R2 Task 1) — 지금까지 hud_from_cache.py(라벨 10영상 전용
    실험 경로)에만 연결돼 있었고, 실제 배치 스캔(이 함수)에는 연결이 안 돼 있던
    구조적 공백이었음. 실측(2026-07-09): MULTI/DOUBLE KILL 연출이 HUD를 4~5초
    가리는 게 짧은 row_miss run으로 잡혀 진짜 라운드가 쪼개지는 사례를 04-13
    22-46-39(12:46-12:52, K 3→3 불변·D 1 불변인데 라운드 분리됨), 04-24 00-43-29
    등에서 확인 — 검증기가 이미 겨냥하던 실패 모드와 정확히 일치.
    모델 로드 실패(CUDA 문제 등) 시 경고만 남기고 검증 없이 진행(회귀 없음)."""
    reads, duration, err = collect_reads(
        Path(video_path), scan_fps=scan_fps, dataset_root=dataset_root
    )
    boundary_verdicts = None
    boundary_warning = None
    if verify_boundaries:
        try:
            from hud_boundary_verify import get_boundary_verifier, verify_runs_live
            model, transform, device = get_boundary_verifier()
            boundary_verdicts = verify_runs_live(Path(video_path), reads, model, transform, device)
        except Exception as exc:  # noqa: BLE001 — 검증은 부가 기능, 실패해도 스캔은 진행
            boundary_warning = f"boundary_verify_failed:{exc}"
    score_win_events = None
    score_warning = None
    if verify_score_wins:
        try:
            from dataclasses import asdict as _asdict
            from hud_score_wins import load_score_timeline, scan_score_timeline, score_events
            score_tl = load_score_timeline(Path(video_path).stem)
            if score_tl is None:
                score_tl = scan_score_timeline(Path(video_path))
            if score_tl is not None:
                score_win_events = [_asdict(e) for e in score_events(score_tl)]
        except Exception as exc:  # noqa: BLE001 — R6 부가 기능, 실패해도 스캔은 진행
            score_warning = f"score_wins_verify_failed:{exc}"
    timeline = timeline_from_reads(
        reads,
        duration=duration,
        video_path=Path(video_path),
        scan_fps=scan_fps,
        ace_kills=ace_kills,
        boundary_verdicts=boundary_verdicts,
        score_win_events=score_win_events,
    )
    if err:
        timeline.warnings.append(err)
    if boundary_warning:
        timeline.warnings.append(boundary_warning)
    if score_warning:
        timeline.warnings.append(score_warning)
    return timeline


def collect_reads(
    video_path: Path,
    *,
    scan_fps: float = 4.0,
    dataset_root: Path | None = None,
) -> tuple[list[KRead], float, str | None]:
    """영상 1패스 디코드 → 프레임별 원시 K 판독 리스트.

    이 결과(신호 캐시)만 있으면 판정 로직은 영상 재판독 없이 무한 재실험 가능
    — hud_sig_cache.py 로 저장, hud_from_cache.py / timeline_from_reads 로 소비.
    """
    get_hud_digit_matcher()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0, "video_open_failed"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_total / fps if frame_total > 0 else 0.0
    step = max(1, int(round(fps / scan_fps))) if scan_fps > 0 else int(fps)

    reads: list[KRead] = []
    frame_idx = 0
    while True:
        if frame_idx % step == 0:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            t = frame_idx / fps
            game, _ = extract_game_crop_bgr(frame, dataset_root=dataset_root)
            # 오독 방어: 배너·페이드 오염 시 보통 세 슬롯이 다 깨지므로 동반 슬롯을
            # 요구한다. 단 R5(2026-07-09): 트리플(D와 A 모두) 요구는 D=8 정전을 유발
            # — 8은 EXCLUDE_DIGITS라 D가 8인 구간은 모든 프레임이 통째로 버려져
            # 완전 미탐 발생 (실측: 05-23 02-48-06 '7/8→10/8', 05-26 23-45-51
            # '6/8→9/8' 구간). D 또는 A 중 하나만 파싱돼도 채택(더블 가드)으로 완화.
            k, d, a, conf, method = read_kda_triple_from_game(game)
            if k is not None and d is None and a is None:
                k = None
                method = "triple_incomplete"
            reads.append(KRead(t=t, k=k, conf=float(conf), method=method, d=d, a=a))
        else:
            if not cap.grab():
                break
            frame_idx += 1
            if frame_idx / fps >= duration:
                break
            continue

        frame_idx += 1
        if frame_idx / fps >= duration:
            break

    cap.release()
    return reads, duration, None


def _rounds_from_settled(
    settled: list[SettledRound], ace_kills: int
) -> tuple[list[RoundTrack], list[KillEvent], list[list[float]]]:
    """SettledRound(라운드별 독립 정산 결과) → RoundTrack/JSON 스키마 매핑 + ace 판정.

    R3(2026-07-07): `_KTracker`(전역 순차 상태머신) + `_build_rounds`/`_assign_events`
    (킬 귀속부)를 대체. ace 판정식(kills==정확히3·resets==0·span·G5)은 기존과 동일 —
    `hud_round_settle.settle_rounds`가 라운드별 kills/kill_times/reset/k_samples까지
    정산해 넘겨주므로 여기서는 그 값으로 판정만 한다.
    """
    rounds: list[RoundTrack] = []
    kill_events: list[KillEvent] = []
    reset_events: list[list[float]] = []

    for sr in settled:
        kt = sorted(sr.kill_times)
        r = RoundTrack(
            round_index=sr.index,
            start_sec=sr.start_sec,
            end_sec=sr.end_sec,
            kills=sr.kills,
            kill_times=kt,
            resets=1 if sr.reset else 0,
            reset_times=[sr.start_sec] if sr.reset else [],
            k_samples=sr.k_samples,
        )
        if kt:
            r.first_kill_sec = kt[0]
            if len(kt) >= ace_kills:
                r.ace_sec = kt[ace_kills - 1]
        span = (
            (r.ace_sec - r.first_kill_sec)
            if (r.ace_sec is not None and r.first_kill_sec is not None)
            else 0.0
        )
        r.ace = (
            r.kills == ace_kills
            and r.resets == 0
            and span <= _MAX_ACE_SPAN_SEC
            and r.k_samples >= _MIN_ROUND_K_SAMPLES
        )
        rounds.append(r)

        # 진단용 재구성(kill_events/reset_events) — GT 비교 도구는 rounds[]만 읽으므로
        # 여기 값이 소수점 단위로 정확할 필요는 없음(사람이 눈으로 볼 진단 필드).
        base = sr.k_base if sr.k_base is not None else 0
        for i, t in enumerate(kt):
            kill_events.append(KillEvent(t=t, from_k=base + i, to_k=base + i + 1))
        if sr.reset:
            reset_events.append(
                [sr.start_sec, float(sr.k_base or 0), float(sr.k_start or 0)]
            )

    return rounds, kill_events, reset_events


def _is_width0(kill_times: list[float]) -> bool:
    """폭0(3킬이 전부 동일 시각) 여부 — R6 게이트 적용 대상 식별."""
    return len(kill_times) >= 2 and max(kill_times) == min(kill_times)


def _count_score_win_events(
    score_win_events: list[dict], start_sec: float, end_sec: float
) -> int:
    """라운드 구간 [start_sec, end_sec] 안에 t_hi가 떨어지는 'win' 이벤트 수."""
    return sum(
        1
        for e in score_win_events
        if e.get("kind") == "win" and start_sec <= e.get("t_hi", -1.0) <= end_sec
    )


def timeline_from_reads(
    reads: list[KRead],
    *,
    duration: float,
    video_path: Path,
    scan_fps: float = 4.0,
    ace_kills: int = ACE_KILLS,
    boundary_verdicts: list[list] | None = None,
    score_win_events: list[dict] | None = None,
) -> HudAceTimeline:
    """원시 판독 리스트 → 라운드 분할·킬 이벤트·ace 판정 (영상 접근 없음).

    판정 로직의 단일 진입점: 실스캔(scan_hud_aces)과 캐시 재계산(hud_from_cache)이
    모두 이 함수를 쓰므로 로직 수정 시 두 경로가 자동 일치.

    R3(2026-07-07): 라운드 분할 후 킬 귀속·ace 판정은 `hud_round_settle.settle_rounds`
    (라운드별 독립 정산 — 전역 상태 없음)로 위임. 구 `_KTracker` 경로는 참고용으로
    파일 하단에 남겨둠(대체 완료, 미사용).

    boundary_verdicts: R2 Task 1(전광판 CNN 스팟검증) 결과 — `[[start, last, bool], ...]`.
    None이면 검증 없이 모든 row_miss run을 경계로 인정(기존 동작과 완전 동일, 회귀 없음).
    특정 run이 False(가짜 경계)로 판정되면 그 run은 경계 목록에서 제외 — 두 라운드가
    합쳐진다. 검증 결과가 없는 run(캐시 갱신 등)은 None 반환 → 안전하게 경계 유지.

    score_win_events: R6(2026-07-13) 상단 스코어(팀 승수) 이벤트 —
    `hud_score_wins.score_events()`를 `dataclasses.asdict()`로 변환한 dict 리스트.
    None이면 게이트 완전 비활성(하위호환, 회귀 없음). 폭0 ace 후보의 라운드 구간
    안에 win 이벤트가 `_SCORE_SPLIT_MIN_EVENTS`개 이상이면 그 라운드가 실제로는
    여러 라운드가 병합된 것으로 보고 ace를 기각한다 (Task 3 실측 근거는 위 상수
    주석 참고 — 정상(폭>0) ace는 절대 건드리지 않음).
    """
    timeline = HudAceTimeline(
        video_path=str(video_path),
        scan_fps=scan_fps,
        ace_kills=ace_kills,
    )

    for r in reads:
        if r.k is not None:
            timeline.k_template_hits += 1
        elif r.method in ("template_miss", "triple_incomplete"):
            timeline.k_template_miss += 1
        elif r.method == "row_miss":
            timeline.k_row_miss += 1

    boundaries: list[float] = []   # 라운드 경계 시각 (연속 row_miss run 중앙)
    for (start, last, _n) in rowmiss_runs(reads):
        if boundary_verdicts is not None:
            verdict = _lookup_boundary_verdict(boundary_verdicts, start, last)
            if verdict is False:
                continue  # 가짜 경계(전광판 미확인) — 폐기, 두 라운드 병합
        boundaries.append((start + last) / 2)

    settled = settle_rounds(
        [(r.t, r.k, r.conf) for r in reads], boundaries, duration,
        # D(데스) 채널 — 사후킬/관전 가드용 (2026-07-09). 구 캐시엔 d=None → 가드 no-op.
        d_reads=[(r.t, r.d, r.conf) for r in reads if r.d is not None],
    )
    rounds, kill_events, reset_events = _rounds_from_settled(settled, ace_kills)

    if score_win_events:
        for r in rounds:
            if r.ace and _is_width0(r.kill_times):
                n = _count_score_win_events(score_win_events, r.start_sec, r.end_sec)
                if n >= _SCORE_SPLIT_MIN_EVENTS:
                    r.ace = False
                    r.end_reason = "score_wins_split"

    timeline.rounds = rounds
    timeline.hud_end_count = len(boundaries)
    timeline.ace_rounds = [r.round_index for r in rounds if r.ace]
    timeline.kill_events = kill_events
    timeline.reset_events = reset_events

    if timeline.k_template_hits == 0:
        timeline.warnings.append("k_never_read")

    return timeline


def ace_clip_window(r: RoundTrack) -> tuple[float, float]:
    """올킬 클립 [start, end] — 라운드 시작·첫 킬 기준, 종료 후 tail 최소."""
    start = r.start_sec - _CLIP_PRE_ROUND_SEC
    if r.first_kill_sec is not None:
        start = min(start, r.first_kill_sec - _CLIP_PRE_KILL_SEC)
    start = max(0.0, start)

    end = r.end_sec + _CLIP_POST_END_SEC
    if r.ace_sec is not None:
        end = min(end, max(r.ace_sec + 12.0, r.end_sec + 0.5))

    if end - start > _CLIP_MAX_SEC:
        start = end - _CLIP_MAX_SEC
    if end - start < 12.0:
        start = max(0.0, end - 12.0)
    return start, end


def extract_ace_clips(
    video_path: Path,
    timeline: HudAceTimeline,
    output_dir: Path,
) -> list[Path]:
    from extract_labeled_clips import run_ffmpeg_extract

    out_dir = output_dir / Path(timeline.video_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for r in timeline.rounds:
        if not r.ace:
            continue
        clip_start, clip_end = ace_clip_window(r)
        label = sec_to_mss(r.end_sec).replace(":", "m")
        out_path = out_dir / f"{Path(timeline.video_path).stem}_R{r.round_index:02d}_{label}s_hud_ace.mp4"
        ok = run_ffmpeg_extract(video_path, clip_start, clip_end, out_path)
        tag = "OK" if ok else "FAIL"
        print(
            f"  R{r.round_index} {sec_to_mss(clip_start)}-{sec_to_mss(clip_end)} "
            f"(round {sec_to_mss(r.start_sec)}-{sec_to_mss(r.end_sec)}) -> {out_path.name} {tag}",
            flush=True,
        )
        if ok:
            written.append(out_path)
    return written


def sec_to_mss(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}:{s:02d}"


def format_report(timeline: HudAceTimeline) -> str:
    lines = [
        f"## {Path(timeline.video_path).name}",
        f"scan_fps={timeline.scan_fps}  rounds={len(timeline.rounds)}  "
        f"ace={timeline.ace_rounds}  hud_ends={timeline.hud_end_count}",
        f"k_read: hit={timeline.k_template_hits} template_miss={timeline.k_template_miss} "
        f"row_miss={timeline.k_row_miss}  kills={len(timeline.kill_events)} "
        f"resets={len(timeline.reset_events)}",
    ]
    if timeline.warnings:
        lines.append(f"warnings: {timeline.warnings}")
    for r in timeline.rounds:
        tag = " **ACE**" if r.ace else ""
        kt = ",".join(sec_to_mss(t) for t in r.kill_times)
        lines.append(
            f"  R{r.round_index:02d} {sec_to_mss(r.start_sec)}-{sec_to_mss(r.end_sec)} "
            f"kills={r.kills}[{kt}] resets={r.resets} n={r.k_samples}{tag}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HUD K/D/A 올킬 스캔 (닉 불필요)")
    p.add_argument("video_path")
    p.add_argument("--scan-fps", type=float, default=4.0)
    p.add_argument("--ace-kills", type=int, default=ACE_KILLS)
    p.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    p.add_argument("--json-out", default=None)
    p.add_argument("--extract", action="store_true")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"[hud-ace] 영상 없음: {video_path}")
        return 1

    print(f"[hud-ace] 스캔: {video_path.name} (fps={args.scan_fps})", flush=True)
    timeline = scan_hud_aces(
        video_path,
        scan_fps=args.scan_fps,
        ace_kills=args.ace_kills,
        dataset_root=Path(args.dataset_root),
    )
    print(format_report(timeline), flush=True)

    json_out = Path(args.json_out) if args.json_out else (
        DEFAULT_JSON_DIR / f"{video_path.stem}.json"
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(asdict(timeline), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[hud-ace] saved -> {json_out}", flush=True)

    if args.extract and timeline.ace_rounds:
        extract_ace_clips(video_path, timeline, Path(args.output_dir))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
