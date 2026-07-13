# -*- coding: utf-8 -*-
"""라운드별 독립 정산(settlement) 디코더 — 전역 상태머신(_KTracker) 대체 핵심 로직.

배경 (2026-07-07 사용자 육안 피드백 + 캐시 덤프 대조, HUD_ACE_HANDOFF.md R2 미탐 3건):
남은 미탐의 공통점은 "라운드 끝 K/D/A가 마지막 킬 후 수 초간 안정 노출되어 캐시에
이미 정답이 있는데" 전역 순차 상태머신이 버리는 것 —
  (a) 산발적 고신뢰 "0" 오독(conf 0.9+ 실측 다수)이 v==0 무기한 누적 예외를 타고
      가짜 리셋을 확정 → 라운드가 쪼개지며 킬 증발 (00-40-56 42:33 실측)
  (b) 먼 과거 오독 리베이스로 오염된 confirmed가 이월돼 정상 킬 체인(9→10→11)이
      영영 확정 못 됨 (02-21-23 52:32 오염 → 54:20 미탐)
라운드(경계 사이 세그먼트) 안에서만 최적 K 궤적을 정산하면 "국소 수정 → 원거리
회귀"의 구조적 문제(핸드오프 '구조적 시사점' 절)가 사라진다.

3단 구조:
  1) 지지 필터 — 세그먼트 내 관측 횟수 < _SUPPORT_MIN_READS 이고
     conf < _SUPPORT_SINGLE_CONF 인 값은 노이즈로 제외 (42:00 "7" conf 0.73 단발 등)
  2) DP 최적 체인 — 비감소(상승폭 ≤ _MAX_STEP) 경로 중 conf 합 최대.
     0-리셋 간선은 _RESET_PENALTY 를 물려 "산발 0 오독"(높은 값이 뒤따라 나옴 →
     리셋 경로 점수 열세)과 "진짜 하프타임"(이후 저값만 지속 → 리셋 경로 승리)을
     문맥 점수로 자연 구분 — HMM 전역 디코딩의 라운드-국소 축소판.
  3) k_base 이월 — 직전 세그먼트 k_end 를 기준으로 라운드 ΔK = k_end - k_base 산출.
     경계 직후 그레이스(_GRACE 규칙, R2 Task 2와 동일 의미)로 경계 걸친 킬을
     직전 라운드에 귀속.

ace 판정 자체(kills==3, G5, span)는 detect_ace_hud.py 소관 — 이 모듈은 라운드별
kills/kill_times/reset/k_samples 까지만 책임진다 (순수 함수, 영상·캐시 접근 없음).
"""

from __future__ import annotations

from dataclasses import dataclass, field

_MAX_STEP = 3              # 라운드 내 인접 관측 간 최대 상승폭 (ace=정확히 3이므로 충분)
_RESET_PENALTY = 2.0       # 리셋 간선 페널티 (conf 합 단위) — 산발 0 오독 5회(≈4.6)로는
                           #   뒤따르는 정상 체인(6→9, 보통 conf 합 5+)을 못 이기게
_SUPPORT_MIN_READS = 2     # 값 지지 최소 관측 수
_SUPPORT_SINGLE_CONF = 0.88  # 단발 관측 허용 conf (라운드 끝값이 1프레임만 읽힌 실측:
                             #   02-21-23 "11" conf 0.90 1회)
_CARRY_MAX_GAP_SEC = 180.0   # k_base 이월 유효 시간 (이보다 오래 판독 공백이면 이월 폐기)
_GRACE_SEC = 10.0            # 경계 직후 첫 관측이 곧 증가값이면 직전 라운드 킬로 귀속
_GRACE_MAX_KILLS = 2         # 그레이스로 넘길 수 있는 최대 킬 수 (R2 Task 2와 동일)

# --- 0-격리 (R5, 2026-07-09 Fable — "스퓨리어스 0"의 정체 규명) ---
# 프레임 육안 검증(05-26 23-45-51 26:10 — 실제 K/D/A "8/9/0"인데 K가 conf 0.7대
# '0'으로 판독)으로, 만성적 "고신뢰 스퓨리어스 0"(00-40-56 42:24 conf 0.95,
# 02-21-23 54:28, 05-23 45:58 등)의 정체가 **화면의 8이 가장 닮은 0으로
# 오독되는 것**임을 확정. 0은 리셋 신호라 킬 체인을 통째로 파괴해 왔음.
# 글리프 점수로는 0/8 분리 불가(실측, hud_digit_match.py 주석) → 도메인 규칙:
# K는 리셋 외 단조증가 — **0 전후의 K가 이어지면(다음 K ≥ 이전 K) 그 0은 가짜**.
# 진짜 리셋이면 이후 K는 반드시 이전보다 작게 다시 시작한다(0→1→2…).
_ZERO_QUAR_LOOKAHEAD = 120.0  # 0 이후 다음 양수 K를 찾는 최대 시간 (초)

# --- 8-엔드포인트 보정 (R5, 2026-07-09 미탐 43건 검수 — 9건이 K 시작/끝값 8) ---
# 8은 EXCLUDE_DIGITS(구조적 미판독)라 K가 8인 상태는 절대 관측되지 않는다.
# Rule A: 이월 무효 상태에서 라운드가 9로 시작 + 마지막 알려진 K ≤7
#         → 보이지 않는 8을 거쳐온 것. 이 라운드 몫은 8→9 한 킬뿐 (k_base=8).
#         (실측: 02-21-23 54:20 '8/8→11/8' — base 8이라 ΔK 산정 불가였던 케이스)
# Rule B: 직전 라운드가 2킬로 7에서 끝났고 이 라운드가 9로 시작(7→9, 사이 8 미판독)
#         → 3번째 킬(7→8)은 직전 라운드 말미 — 브리지해 직전 라운드를 3킬로 완성.
#         (실측: 02-21-23 79:51 '5/6→8/6', 00-36-50 40:37 '5/1→8/1' 등 end-8 4건+)
# ⚠ 8 템플릿이 부활하면(EXCLUDE_DIGITS 해제) 두 규칙 모두 전제가 깨짐 — 함께 제거할 것.
_BRIDGE_8 = True

# --- 약한 판독의 gap 상한 (R5, 2026-07-09 Fable — (a)형 폭0 오탐 대응) ---
# carry~첫관측 사이에 지지 필터(n≥2 or conf≥0.88)에 걸러진 약한 판독이 중간값을
# 증언하면, gap 킬 수를 그만큼 줄인다. 약한 증거는 "킬 수를 늘리는 방향"으로는
# 못 쓰지만 "줄이는 방향"으로는 안전. 실측: 05-31 21-57-14 101:47의 '6'(conf
# 0.67 단발)이 걸러져 carry(4)→7이 gap=3 거짓 올킬이 됨 — 6을 쓰면 gap=1로 정정.
_GAP_WEAK_CAP = True

# --- R7 (2026-07-14): 격리된 '가짜 0'(=화면의 8)을 carry 전진 증거로 재활용 ---
# 사용자 육안 검수(폭0 오탐 9건 전부 = 원본 존재분)로 지배적 원인 확정:
# 8은 EXCLUDE_DIGITS라 판독 불가 → 누적 K가 8을 지나면 시스템이 마지막 깨끗값 7에
# 얼어붙고, 다음 깨끗값 10을 읽는 순간 carry(7)→k_start(10) gap=3 = 가짜 올킬.
# 실측 3케이스 덤프 전부 동일 지문: 7(직전라운드, conf 0.9) → 가짜0=8(격리) → 10.
# 그 8은 _quarantine_zeros가 이미 "가짜 0"으로 격리하는 바로 그 판독. blind gap
# (carry_t~first_t)에 지지된 가짜-0이 있으면 k_base를 8로 전진 → 7→10(+3)이
# 8→10(+2)로 정정돼 가짜 올킬 소멸. ⚠ gap-path 전용(k_start>k_base)이라 킬 수를
# 줄이는 방향으로만 작동 — 정탐 폭0(7이 라운드 안에서 관측되는 within-chain 상승,
# 03-12-36 R010 실측)은 gap=0이라 안 건드림 → 새 오탐 생성 원리적으로 불가.
_EIGHT_EVID = True
_EIGHT_EVID_MIN = 2          # blind gap 내 가짜-0(=8) 최소 지지 관측 수

# --- D(데스) 채널 가드 (2026-07-09, 사용자 오탐 12건 검수 기반) ---
# 도메인: 본인 D는 라운드 내 사망(+1) 외엔 변할 수 없다. 라운드 중 지지된 D 변화
# = 사망(상승) 또는 관전 화면 오독(임의 변화, 죽으면 팀원 K/D가 패널에 노출됨 —
# 실측: 2026-05-31 21-57-14 R139, 14/10 관전 오독 올킬 FP). 사망/관전 시각 이후의
# K 상승은 본인 킬이 아니므로 무효화. 트레이드킬(사망과 동시 킬)은 마진으로 허용.
_D_GUARD_MARGIN_SEC = 2.0    # 사망과 사실상 동시(트레이드킬)의 킬은 유효
_D_SUPPORT_MIN = 2           # 가드용 D 지지: 같은 값 2회 이상 (단발 conf 예외 없음 — 보수적)


@dataclass
class SettledRound:
    index: int
    start_sec: float
    end_sec: float
    k_base: int | None = None   # 라운드 시작 기준값 (이월 or 라운드 첫 체인값)
    k_start: int | None = None  # 라운드 내 체인 첫 값
    k_end: int | None = None    # 라운드 내 체인 끝 값 (다음 라운드로 이월됨)
    kills: int = 0
    kill_times: list[float] = field(default_factory=list)
    reset: bool = False         # 체인이 리셋 간선을 썼음 (하프타임/리조인) → ace 제외 대상
    k_samples: int = 0          # 라운드 내 성공 판독 총 수 (G5 지표, 필터 전 원시 기준)
    chain_reads: int = 0        # 체인에 채택된 판독 수 (진단용)
    d_guard_dropped: int = 0    # D-가드가 무효화한 킬 수 (진단용, 2026-07-09)


def _supported_obs(
    obs: list[tuple[float, int, float]],
) -> list[tuple[float, int, float]]:
    """지지 필터: 세그먼트 내 n>=2 또는 conf>=_SUPPORT_SINGLE_CONF 인 값만 통과."""
    count: dict[int, int] = {}
    best: dict[int, float] = {}
    for _t, k, c in obs:
        count[k] = count.get(k, 0) + 1
        best[k] = max(best.get(k, 0.0), c)
    ok = {
        k
        for k in count
        if count[k] >= _SUPPORT_MIN_READS or best[k] >= _SUPPORT_SINGLE_CONF
    }
    return [(t, k, c) for (t, k, c) in obs if k in ok]


def _quarantine_zeros(
    reads: list[tuple[float, int, float]],
) -> list[tuple[float, int, float]]:
    """'가짜 0(=화면의 8 오독)' 격리 (상단 0-격리 주석 참고). 전체 판독 1패스.

    0 관측이 가짜인 조건: (±_ZERO_QUAR_LOOKAHEAD 내) 직전 양수 K가 있고,
    직후 첫 양수 K가 그 이상(리셋 없이 이어짐). 진짜 리셋(하프타임/리조인)은
    이후 K가 반드시 더 작게 재시작(0→1→2…)하므로 보존된다.
    """
    n = len(reads)
    out: list[tuple[float, int, float]] = []
    for i, (t, k, c) in enumerate(reads):
        if k != 0:
            out.append((t, k, c))
            continue
        prev_nz = None
        for j in range(i - 1, -1, -1):
            t2, k2, _c2 = reads[j]
            if t - t2 > _ZERO_QUAR_LOOKAHEAD:
                break
            if k2 > 0:
                prev_nz = k2
                break
        if prev_nz is None:
            out.append((t, k, c))  # 앞선 양수 K 없음 — 매치 초반/리셋 직후의 진짜 0
            continue
        next_nz = None
        for j in range(i + 1, n):
            t2, k2, _c2 = reads[j]
            if t2 - t > _ZERO_QUAR_LOOKAHEAD:
                break
            if k2 > 0:
                next_nz = k2
                break
        if next_nz is not None and next_nz >= prev_nz:
            continue  # K가 리셋 없이 이어짐 → 이 0은 가짜(8 오독) — 격리
        out.append((t, k, c))
    return out


def _d_anomaly_t(d_obs: list[tuple[float, int, float]]) -> float | None:
    """라운드 내 D 채널 이상 시각 — 지지(같은 값 ≥_D_SUPPORT_MIN회) D가 기준값에서
    변하는 첫 시각. 상승=본인 사망, 하락=관전/오독 — 어느 쪽이든 이후 K 상승은
    본인 킬이 아님. 이상 없으면 None."""
    count: dict[int, int] = {}
    for _t, d, _c in d_obs:
        count[d] = count.get(d, 0) + 1
    ok = {d for d in count if count[d] >= _D_SUPPORT_MIN}
    base: int | None = None
    for t, d, _c in sorted(d_obs, key=lambda x: x[0]):
        if d not in ok:
            continue
        if base is None:
            base = d
        elif d != base:
            return t
    return None


def _best_chain(
    obs: list[tuple[float, int, float]],
) -> tuple[list[tuple[float, int, float]], bool]:
    """비감소(상승폭 ≤_MAX_STEP) 최적 체인 — conf 합 최대 경로.

    리셋 간선(v>0 → 0)은 1회 허용·_RESET_PENALTY 차감. 반환: (채택 판독, 리셋 사용 여부).
    O(n²) DP — 세그먼트 판독 수는 수백 이하(4fps×라운드 길이)라 충분히 빠름.
    """
    n = len(obs)
    if n == 0:
        return [], False
    NEG = float("-inf")
    # score[i][r]: i번째 판독을 채택하며 끝나는 체인의 최대 점수 (r=리셋 사용 여부)
    score = [[NEG, NEG] for _ in range(n)]
    back: list[list[tuple[int, int] | None]] = [[None, None] for _ in range(n)]
    for i, (_ti, ki, ci) in enumerate(obs):
        score[i][0] = ci  # 체인 시작
        for j in range(i):
            _tj, kj, _cj = obs[j]
            if kj <= ki <= kj + _MAX_STEP:  # 유지 또는 상승 (킬)
                for r in (0, 1):
                    if score[j][r] + ci > score[i][r]:
                        score[i][r] = score[j][r] + ci
                        back[i][r] = (j, r)
            if ki == 0 and kj > 0:  # 리셋 간선 (하프타임/리조인) — 1회, 페널티
                cand = score[j][0] + ci - _RESET_PENALTY
                if cand > score[i][1]:
                    score[i][1] = cand
                    back[i][1] = (j, 0)
    end_i, end_r, best = 0, 0, NEG
    for i in range(n):
        for r in (0, 1):
            if score[i][r] > best:
                best, end_i, end_r = score[i][r], i, r
    chain: list[tuple[float, int, float]] = []
    cur: tuple[int, int] | None = (end_i, end_r)
    used_reset = False
    while cur is not None:
        i, r = cur
        chain.append(obs[i])
        prev = back[i][r]
        if prev is not None and prev[1] != r:
            used_reset = True
        cur = prev
    chain.reverse()
    return chain, used_reset


def settle_rounds(
    reads: list[tuple[float, int | None, float]],
    boundaries: list[float],
    duration: float,
    d_reads: list[tuple[float, int, float]] | None = None,
) -> list[SettledRound]:
    """세그먼트별 독립 정산 + k_base 이월 + 경계 그레이스 귀속.

    reads: (t, k|None, conf) — k=None(판독 실패)은 k_samples 집계에서만 제외.
    boundaries: 라운드 경계 시각 (row_miss run 중앙, CNN 검증 반영 후) — 정렬 가정 안 함.
    d_reads: (t, d, conf) — D(데스) 슬롯 판독 (d=None 제외 후 전달). None/빈 리스트면
             D-가드 완전 비활성 (구 캐시 하위호환 — 기존 동작과 100% 동일).
    """
    seps = sorted(set(b for b in boundaries if 0.0 < b < duration))
    segs: list[tuple[float, float]] = []
    prev = 0.0
    for sp in seps + [duration]:
        if sp - prev > 0.5:
            segs.append((prev, sp))
        prev = sp

    ok_reads = [(t, k, c) for (t, k, c) in reads if k is not None]
    clean_reads = _quarantine_zeros(ok_reads)  # 가짜 0(8 오독) 격리 — 상단 주석

    rounds: list[SettledRound] = []
    carry_k: int | None = None
    carry_t: float = float("-inf")

    for idx, (s, e) in enumerate(segs):
        r = SettledRound(index=idx, start_sec=s, end_sec=e)
        obs = [(t, k, c) for (t, k, c) in clean_reads if s <= t < e]
        # k_samples는 정의상 "필터 전 원시 기준"(G5 지표) — 격리 전 개수로 센다.
        # 격리 후 개수로 세면 G5(>=10) 임계가 기존 캘리브와 어긋남 (실측: 00-40-56
        # R83 65:01 TP가 k_samples 10→9로 떨어져 G5에 기각되는 회귀).
        r.k_samples = sum(1 for (t, _k, _c) in ok_reads if s <= t < e)
        # ⚠ 시도했으나 순손실로 되돌림 (2026-07-07 Sonnet R3):
        # "carry_k 확립 시 0<obs_k<carry_k 관측은 항상 오독"이라는 강한 불변식을
        # 걸어봤으나(02-21-23 80:37 등 폭0 FP 다수 제거에는 성공) 02-21-23 64:54-65:20·
        # 00-42-33 12:13-12:47(둘 다 기존 TP)이 깨져 recall 88.5%→80.8% 순손실
        # (TP 23→21). 핸드오프에 이미 기록된 "G1 v==0-only 제한 실험"과 동일한 함정 —
        # 실제 영상엔 정당한 하향 보정(오독으로 부풀려진 값의 재교정)이 존재해
        # 하드 불변식으로는 막을 수 없음. 재시도하려면 "몇 프레임 이상 지속" 등
        # 추가 조건으로 노이즈와 정당한 보정을 구분해야 함 — 다음 세션 과제로 이월.
        chain, used_reset = _best_chain(_supported_obs(obs))
        r.chain_reads = len(chain)
        r.reset = used_reset

        if chain:
            r.k_start = chain[0][1]
            r.k_end = chain[-1][1]
            # 체인 내부 전이 → 킬 이벤트 (값이 바뀌는 첫 판독 시각)
            for (t_a, k_a, _), (t_b, k_b, _) in zip(chain, chain[1:]):
                if k_b > k_a:
                    r.kill_times.extend([t_b] * (k_b - k_a))
            # k_base: 직전 세그먼트 k_end 이월 (0 <= 상승폭 <= _MAX_STEP 일 때만)
            first_t = chain[0][0]
            if (
                carry_k is not None
                and first_t - carry_t <= _CARRY_MAX_GAP_SEC
                and 0 <= r.k_start - carry_k <= _MAX_STEP
            ):
                r.k_base = carry_k
            elif _BRIDGE_8 and r.k_start == 9 and carry_k is not None and carry_k <= 7:
                r.k_base = 8  # Rule A — 보이지 않는 8 경유, 8→9만 이 라운드 몫
            else:
                r.k_base = r.k_start  # 이월 불가(하프타임 하락·큰 점프·공백) → 새 기준
            # Rule B — end-8 크로스라운드 브리징 (상단 _BRIDGE_8 주석 참고)
            if (
                _BRIDGE_8
                and r.k_base == 7
                and r.k_start == 9
                and rounds
                and rounds[-1].kills == 2
                and not rounds[-1].reset
            ):
                prev_r = rounds[-1]
                prev_r.kills += 1
                prev_r.kill_times.append(prev_r.end_sec)
                if prev_r.k_end is not None:
                    prev_r.k_end = 8
                r.k_base = 8
            # 약한 판독 gap 상한 (상단 _GAP_WEAK_CAP 주석 참고) — 지지 미달로
            # 걸러진 중간값 판독이 carry~첫관측 사이에 있으면 k_base를 끌어올림.
            # 킬 수를 줄이는 방향으로만 작동(폭0 거짓 올킬 방지), 늘리진 못함.
            if (
                _GAP_WEAK_CAP
                and r.k_base is not None
                and r.k_start - r.k_base > 1
                and carry_k is not None
            ):
                weak = [
                    v for (t, v, _c) in ok_reads
                    if carry_t < t < first_t and r.k_base < v < r.k_start
                ]
                if weak:
                    r.k_base = max(weak)
            # R7: 격리된 가짜-0(=8) 증거로 k_base 전진 (상단 _EIGHT_EVID 주석 참고).
            # blind gap의 raw 0은 quarantine 규칙상(직전 양수7 · 직후 양수10≥7) 전부
            # 가짜-0=8이므로, 지지되면 k_base를 8로 올려 gap +3 가짜 올킬을 +2로 정정.
            if (
                _EIGHT_EVID
                and r.k_base is not None
                and r.k_base < 8 < r.k_start
                and r.k_start - r.k_base > 1
                and carry_k is not None
            ):
                gap_zeros = [
                    t for (t, v, _c) in ok_reads
                    if carry_t < t < first_t and v == 0
                ]
                if len(gap_zeros) >= _EIGHT_EVID_MIN:
                    r.k_base = 8
            gap = r.k_start - r.k_base
            if gap > 0:
                # 경계 그레이스: 경계 직후 곧바로 +1/+2 상태로 관측되면 그 킬은
                # 전광판 직전(직전 라운드 막판) 것 — 스폰 직후 킬 불가(도메인 확정).
                if (
                    rounds
                    and gap <= _GRACE_MAX_KILLS
                    and first_t - s <= _GRACE_SEC
                ):
                    prev_r = rounds[-1]
                    prev_r.kills += gap
                    prev_r.kill_times.extend([s] * gap)
                    if prev_r.k_end is not None:
                        prev_r.k_end = max(prev_r.k_end, r.k_start)
                    r.k_base = r.k_start
                else:
                    # gap 킬 시각 = 라운드 시작(s) — 킬은 첫 관측(first_t) 시점이
                    # 아니라 그 이전(직전 경계 부근)에 일어난 것. first_t로 찍으면
                    # 판독 공백이 길 때 GT 창과 어긋남 (실측: 04-24 00-43-29 5:50
                    # GT vs 6:17 첫관측 — 15s 허용오차 밖으로 밀려 미탐).
                    r.kill_times = [s] * gap + r.kill_times
            # D-채널 가드 (2026-07-09): 라운드 내 사망/관전 이상 시각 이후 킬 무효화.
            # 사용자 오탐 검수 실측 — 죽으면 팀원 관전 화면의 K/D가 본인 것으로 오독됨.
            if d_reads and r.kill_times:
                d_obs = [(t, d, c) for (t, d, c) in d_reads if s <= t < e]
                anom_t = _d_anomaly_t(d_obs)
                if anom_t is not None:
                    kept = [t for t in r.kill_times if t <= anom_t + _D_GUARD_MARGIN_SEC]
                    r.d_guard_dropped = len(r.kill_times) - len(kept)
                    r.kill_times = kept
            r.kills = len(r.kill_times)
            carry_k = r.k_end
            carry_t = chain[-1][0]
        rounds.append(r)
    return rounds


# ---------------------------------------------------------------------------
# 자가 검증 — 실측 캐시 덤프(2026-07-07) 기반 픽스처. python -u hud_round_settle.py
# ---------------------------------------------------------------------------

def _fx(*groups: tuple[float, int | None, float, int, float]) -> list:
    """(시작t, k, conf, 개수, 간격) 그룹들을 판독 리스트로 전개."""
    out = []
    for t0, k, c, n, dt in groups:
        out.extend((t0 + i * dt, k, c) for i in range(n))
    return sorted(out, key=lambda x: x[0])


def _selftest() -> None:
    m = lambda mm, ss: mm * 60 + ss  # noqa: E731

    # ① 00-40-56 41:54-42:39 — 산발 고신뢰 0 오독 5회에도 6→9 체인이 이겨야 (ACE)
    reads1 = _fx(
        (m(41, 55.5), 6, 0.90, 14, 1.0),   # 6 안정
        (m(42, 0.0), 7, 0.73, 1, 1.0),     # 단발 저신뢰 7 → 지지 필터에서 탈락
        (m(42, 4.25), 6, 0.88, 6, 3.0),
        (m(42, 24.75), 0, 0.95, 1, 1.0),   # 스퓨리어스 0 (conf 0.95!)
        (m(42, 29.75), 6, 0.82, 1, 1.0),
        (m(42, 32.75), 0, 0.93, 4, 0.33),  # 스퓨리어스 0 연속 4회
        (m(42, 35.25), 9, 0.90, 6, 0.55),  # 라운드 끝 9 안정 (사용자 육안: 9/4/0)
    )
    rs = settle_rounds(reads1, [m(41, 50)], m(42, 50))
    r = rs[-1]
    assert r.kills == 3 and not r.reset, f"case1: {r}"
    assert r.k_base == 6 and r.k_end == 9

    # ②' 00-40-56 변형 — 가짜 경계(42:14)가 라운드를 쪼개도 6 재관측 이월로 ACE 유지
    rs = settle_rounds(reads1, [m(41, 50), m(42, 14.75)], m(42, 50))
    r = rs[-1]
    assert r.kills == 3 and r.k_base == 6, f"case1-split: {r}"

    # ③ 03-02-03 14:15-14:57 — k_base=0 이월 + 라운드 내 1→2→3 (ACE)
    reads3 = _fx(
        (m(13, 59.75), 0, 0.88, 9, 0.25),  # 이전 세그먼트: 0 안정
        (m(14, 4.0), 0, 0.88, 4, 0.25),
        (m(14, 7.25), 0, 0.85, 3, 0.25),
        (m(14, 41.25), 1, 0.94, 12, 0.8),  # 긴 row_miss 후 첫 관측 1 (경계+14.5s → 그레이스 아님)
        (m(14, 51.25), 2, 0.80, 3, 1.0),
        (m(14, 55.25), 3, 0.80, 6, 0.3),   # 사용자 육안: 3/0/0 안정
    )
    rs = settle_rounds(reads3, [m(13, 49), m(14, 26.75)], m(15, 0))
    r = rs[-1]
    assert r.kills == 3 and r.k_base == 0 and r.k_end == 3, f"case3: {r}"

    # ④ 02-21-23 54:20-54:41 — 오염 이월 없이 체인 9→10→11은 복원되나 base 8이
    #    EXCLUDE_DIGITS(8 미판독)이라 ΔK 확정 불가 → 미탐 유지가 '정상' (HMM 카드로 이월)
    reads2 = _fx(
        (m(53, 43.75), 6, 0.85, 5, 1.0),   # 직전 라운드 6 (킬 7·8은 미판독 구간)
        (m(54, 14.75), 6, 0.86, 1, 1.0),   # 단발 stale 6
        (m(54, 23.5), 9, 0.72, 3, 1.4),
        (m(54, 28.75), 0, 0.92, 1, 1.0),   # 스퓨리어스 0
        (m(54, 32.0), 10, 0.73, 3, 0.9),
        (m(54, 38.25), 11, 0.90, 1, 1.0),  # 단발 conf 0.90 → SUPPORT_SINGLE_CONF 통과
    )
    rs = settle_rounds(reads2, [m(53, 41), m(54, 10.6)], m(54, 50))
    r = rs[-1]
    assert not r.reset and r.k_end == 11, f"case2: {r}"
    assert r.kills != 3, f"case2는 8-미판독 한계로 미탐이 정상: {r}"

    # ⑤ 진짜 하프타임 — 7 이후 0이 지속·저값만 뒤따름 → 리셋 경로 승리, ace 아님
    reads5 = _fx(
        (10.0, 7, 0.9, 6, 1.0),
        (30.0, 0, 0.9, 8, 1.0),
        (45.0, 1, 0.9, 4, 1.0),
        (55.0, 2, 0.9, 4, 1.0),
    )
    rs = settle_rounds(reads5, [], 60.0)
    r = rs[0]
    assert r.reset, f"case5: {r}"

    # ⑥ 경계 그레이스 — 23-51-52 16:23 유형: 경계 직후 첫 관측이 곧 +1 → 직전 라운드 귀속
    reads6 = _fx(
        (m(16, 30), 9, 0.9, 8, 1.0),       # 직전 라운드 9
        (m(16, 49), 10, 0.9, 6, 1.0),      # 경계(16:46) 직후 3초 만에 10
    )
    rs = settle_rounds(reads6, [m(16, 46)], m(17, 10))
    assert rs[0].kills == 1 and rs[0].kill_times == [m(16, 46)], f"case6 prev: {rs[0]}"
    assert rs[1].kills == 0, f"case6 cur: {rs[1]}"

    # ⑦ D-가드 — 라운드 중 사망(D 4→5 지지 상승) 후의 K 상승은 관전 오독 → 무효
    #    (실측 유형: 2026-05-31 21-57-14 R139 — 죽고 팀원 관전, 팀원 K/D 14/10 오독)
    reads7 = _fx(
        (10.0, 6, 0.9, 6, 1.0),            # 본인 K=6 안정
        (30.0, 7, 0.9, 4, 1.0),            # 진짜 킬 +1 (사망 전) → 유효
        (40.0, 9, 0.9, 6, 1.0),            # 사망(35s) 후 관전 K → 무효여야 함
    )
    d7 = _fx(
        (10.0, 4, 0.9, 24, 1.0),           # D=4 안정
        (35.0, 5, 0.9, 10, 1.0),           # 사망: D 4→5 (지지 2회 이상)
    )
    rs = settle_rounds(reads7, [], 60.0, d_reads=d7)
    r = rs[0]
    assert r.kills == 1 and r.d_guard_dropped == 2, f"case7: {r}"

    # ⑦' 같은 판독에서 d_reads 없으면(구 캐시) 가드 비활성 — 기존 동작 그대로
    rs = settle_rounds(reads7, [], 60.0)
    assert rs[0].kills == 3 and rs[0].d_guard_dropped == 0, f"case7-nod: {rs[0]}"

    # ⑧ 트레이드킬 — 3번째 킬이 사망과 사실상 동시(마진 내)면 ACE 유지
    reads8 = _fx(
        (10.0, 0, 0.9, 6, 1.0),
        (20.0, 1, 0.9, 4, 1.0),
        (30.0, 2, 0.9, 4, 1.0),
        (40.0, 3, 0.9, 6, 1.0),            # 3번째 킬 (40s) — 사망(41s)과 1초 차
    )
    d8 = _fx(
        (10.0, 2, 0.9, 30, 1.0),
        (41.0, 3, 0.9, 8, 1.0),            # 트레이드 사망
    )
    rs = settle_rounds(reads8, [], 60.0, d_reads=d8)
    assert rs[0].kills == 3 and rs[0].d_guard_dropped == 0, f"case8: {rs[0]}"

    # ⑨ Rule A — 이월 만료 + 라운드가 9로 시작 + 마지막 K는 7 이하 → k_base=8
    #    (02-21-23 54:20 '8/8→11/8' 유형: 8이 미판독이라 base를 못 잡던 케이스)
    reads9 = _fx(
        (10.0, 7, 0.9, 6, 1.0),             # 마지막 알려진 K=7 (이후 200s 공백 → 이월 만료)
        (235.0, 9, 0.9, 4, 1.0),            # 라운드 시작값 9 (보이지 않는 8 경유, 경계+15s)
        (245.0, 10, 0.9, 3, 1.0),
        (255.0, 11, 0.9, 6, 1.0),
    )
    rs = settle_rounds(reads9, [220.0], 270.0)
    r = rs[-1]
    assert r.k_base == 8 and r.kills == 3, f"case9: {r}"

    # ⑩ Rule B — 직전 라운드 2킬로 7 종료 + 다음 라운드 9 시작 → 3번째 킬(7→8) 브리지
    #    (02-21-23 79:51 '5/6→8/6' 유형: end-8 미탐)
    reads10 = _fx(
        (10.0, 5, 0.9, 6, 1.0),
        (25.0, 6, 0.9, 4, 1.0),
        (35.0, 7, 0.9, 6, 1.0),             # 2킬로 7에서 라운드 종료 (진짜 3번째 킬 8은 미판독)
        (75.0, 9, 0.9, 6, 1.0),             # 다음 라운드, 경계+15s (그레이스 밖)에 9 등장
    )
    rs = settle_rounds(reads10, [60.0], 100.0)
    assert rs[0].kills == 3 and rs[0].k_end == 8, f"case10 prev: {rs[0]}"
    assert rs[1].kills == 1 and rs[1].k_base == 8, f"case10 cur: {rs[1]}"

    # ⑪ 0-격리 — 8 오독 유래 고신뢰 0 지속(05-23 45:58 실측 유형): 7 안정 →
    #    '0' 여러 번(진짜 K=8) → 세그먼트 끝. 다음 세그먼트 10 → 리셋 오염 없이
    #    carry 7 유지 + gap 3 킬 (진짜 리셋이었다면 다음 K가 7보다 작아야 함)
    #    ★ R7(2026-07-14) 갱신: 이 판독의 '0'은 주석대로 "진짜 K=8" — 따라서 seg1은
    #    8→10=2킬이 정답이지, 7→10=3킬(가짜 올킬)이 아니다. 기존 테스트가 FP 패턴을
    #    '정답 3킬'로 잘못 인코딩하고 있었음(사용자 폭0 오탐 검수로 확정). R7이 정정.
    reads11 = _fx(
        (10.0, 7, 0.9, 8, 1.0),
        (20.0, 0, 0.75, 7, 0.25),          # 가짜 0 (=8) — 격리 + R7 carry 전진 증거
        (40.0, 10, 0.8, 6, 1.0),           # 다음 세그먼트, K 이어짐 (경계+10s 밖)
    )
    rs = settle_rounds(reads11, [28.0], 60.0)
    assert not rs[0].reset and rs[0].k_end == 7, f"case11 seg0: {rs[0]}"
    assert rs[1].kills == 2 and rs[1].k_base == 8, f"case11 seg1: {rs[1]}"
    assert rs[1].kill_times == [28.0] * 2, f"case11 kill_times: {rs[1].kill_times}"

    # ⑤' 진짜 하프타임은 격리되지 않아야 (case5와 동일 판독 재확인 — 0 이후 저값 재시작)
    rs = settle_rounds(reads5, [], 60.0)
    assert rs[0].reset, f"case5-requar: {rs[0]}"

    # ⑫ 약한 판독 gap 상한 — carry 4에서 다음 라운드 7 시작(gap 3 = 거짓 올킬
    #    후보)이지만, 사이에 지지 미달 '6'(단발 conf 0.67)이 있으면 gap=1로 정정
    #    (05-31 21-57-14 101:47 실측 유형)
    reads12 = _fx(
        (10.0, 4, 0.9, 8, 1.0),
        (25.0, 6, 0.67, 1, 1.0),           # 약한 6 — 지지 필터엔 걸러지나 gap 상한엔 사용
        (45.0, 7, 0.9, 8, 1.0),            # 다음 세그먼트 (경계+15s, 그레이스 밖)
    )
    rs = settle_rounds(reads12, [30.0], 60.0)
    assert rs[1].k_base == 6 and rs[1].kills == 1, f"case12: {rs[1]}"

    # ⑬ R7 8-증거 carry 전진 — FP: 7(직전라운드) → 가짜0=8(blind gap) → 10(이 라운드).
    #    실측 3케이스(00-07-24 R055, 22-50-54 R096, 01-14-15 R040) 동일 지문.
    #    7→10(+3 가짜 올킬)이 8→10(+2)로 정정돼야 함.
    reads13 = _fx(
        (10.0, 7, 0.9, 8, 1.0),            # 직전 라운드 깨끗한 7
        (23.0, 0, 0.75, 4, 1.0),           # blind gap: 가짜0=8 지지 4회 (경계 전후 걸침)
        (45.0, 10, 0.8, 8, 1.0),           # 이 라운드: 10만 관측 (경계+15s, 그레이스 밖)
    )
    rs = settle_rounds(reads13, [30.0], 60.0)
    assert rs[1].k_base == 8 and rs[1].kills == 2, f"case13 FP정정: {rs[1]}"

    # ⑬' 정탐 미영향 — within-chain 7→10(진짜 올킬, 7이 라운드 안에서 관측).
    #    03-12-36 R010 유형: 7·(가짜0=8)·10 모두 같은 라운드 → gap=0이라 R7 미발동.
    reads13b = _fx(
        (10.0, 7, 0.9, 8, 1.0),            # 같은 라운드 안: 7
        (25.0, 0, 0.75, 5, 1.0),           # 같은 라운드 안: 가짜0=8 (격리됨)
        (40.0, 10, 0.9, 8, 1.0),           # 같은 라운드 안: 10 (경계 없음)
    )
    rs = settle_rounds(reads13b, [], 60.0)
    assert rs[0].kills == 3 and rs[0].k_start == 7, f"case13b 정탐유지: {rs[0]}"

    print("hud_round_settle selftest: 15/15 OK")


if __name__ == "__main__":
    _selftest()
