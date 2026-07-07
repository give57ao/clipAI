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
) -> list[SettledRound]:
    """세그먼트별 독립 정산 + k_base 이월 + 경계 그레이스 귀속.

    reads: (t, k|None, conf) — k=None(판독 실패)은 k_samples 집계에서만 제외.
    boundaries: 라운드 경계 시각 (row_miss run 중앙, CNN 검증 반영 후) — 정렬 가정 안 함.
    """
    seps = sorted(set(b for b in boundaries if 0.0 < b < duration))
    segs: list[tuple[float, float]] = []
    prev = 0.0
    for sp in seps + [duration]:
        if sp - prev > 0.5:
            segs.append((prev, sp))
        prev = sp

    ok_reads = [(t, k, c) for (t, k, c) in reads if k is not None]

    rounds: list[SettledRound] = []
    carry_k: int | None = None
    carry_t: float = float("-inf")

    for idx, (s, e) in enumerate(segs):
        r = SettledRound(index=idx, start_sec=s, end_sec=e)
        obs = [(t, k, c) for (t, k, c) in ok_reads if s <= t < e]
        r.k_samples = len(obs)
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
            else:
                r.k_base = r.k_start  # 이월 불가(하프타임 하락·큰 점프·공백) → 새 기준
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
                    r.kill_times = [first_t] * gap + r.kill_times
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

    print("hud_round_settle selftest: 6/6 OK")


if __name__ == "__main__":
    _selftest()
