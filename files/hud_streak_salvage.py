# -*- coding: utf-8 -*-
"""킬 스트릭 구제 정산 (R13, 2026-07-22 Fable) — KILLS_LOW 미탐 65건 원인 분석 대응.

배경 (미탐 71건 전수 분석, SONNET_TASK_UNDERREAD.md):
GT 미탐의 최다 유형은 "원시 캐시에 ΔK≥3이 멀쩡히 존재하는데 정산이 3킬을 한
라운드에 못 귀속"(캐시 보유 33건 중 25건 = RAW_DK_OK). 두 메커니즘이 겹친다:
  (1) 킬 스트릭 중 가짜 경계 — 킬 배너/연출이 HUD를 가려 row_miss 연속 구간이
      생기고 그게 라운드 경계로 오인돼 킬 체인이 2+1로 쪼개짐
      (실측: 03-12-36 R4/R5 — 진짜 3킬 3:10·3:22·3:49 가운데 3:33에 경계 삽입,
       02-15-28 R42~R45 — 100초에 라운드 4개로 과분할)
  (2) 지지 필터가 빠른 전이를 버림 — 스트릭 중 K가 3→4→5로 수 초 만에 지나가
      각 값이 1프레임(conf 0.6~0.86)만 읽히는데, 단발+conf<0.88이라 탈락
      (실측: 03-12-36 3:22 '4' conf 0.86이 _SUPPORT_SINGLE_CONF=0.88에 걸림)

원리 — 라운드 정산과 독립적으로, 원시 판독에서 "+3 사다리"를 직접 찾는다:
  안정 기준값 p 이후 p+1, p+2, p+3 증거가 시간순으로 나타나고(단발 허용 — 이게
  지지 필터 우회의 핵심), 그 사이에
    · 강한 경계(CNN 확인 or 승수 인접 — 진짜 라운드 종료) 없음
    · 승수(win) 이벤트 없음 (라운드가 중간에 끝났다면 스트릭이 아님)
  이고, 3번째 킬 직후 승수 이벤트가 있으면(올킬 = 상대 전멸 = 라운드 즉시 승리 —
  도메인 확정 신호) 올킬로 구제한다.

정밀도 방어선 (오탐 방지가 최우선 — kill_shortfall 사후 검수 체계와 별개로 설계 단계 방어):
  D1. 승수 채널 필수 — score_win_events 없으면 전체 비활성 (증거 부족 시 침묵)
  D2. "직후 승리" 필수 — 3번째 킬 증거 후 _WIN_AFTER_SEC 내 win 이벤트.
      2+1 스트래들(직전 라운드 2킬 + 다음 라운드 1킬)은 중간에 win이 찍혀 D3에
      걸리거나, 마지막 킬 후 라운드가 계속돼 D2에 걸린다.
  D3. 창 내부 win 금지 — 첫 킬 증거~3번째 킬 증거 사이 win 이벤트가 있으면 기각
  D4. 강한 경계 금지 — CNN이 전광판을 확인했거나 승수가 인접한 경계는 진짜
      라운드 종료 — 창을 가로지르면 기각. (스트릭 중 가짜 경계는 이 검증이 없는
      '약한' 경계라 통과 — R10 안전핀이 무검증 통과시킨 경계가 바로 이 부류)
  D5. D(데스) 이상 금지 — 창 내 사망/관전 전환(D 채널 변동) 시 기각 (관전 오독 방어)
  D6. 역행 금지 — 창 내에 p 미만의 '지지된' 판독이 있으면 기각 (오염 신호)
  D7. 기왕 올킬 라운드와 겹치면 스킵 (중복 방지)
  D8. 단발 증거 conf 하한(_EVID_MIN_CONF) — 완전 쓰레기 판독은 사다리 증거 불가

산출: SalvageRec 리스트. apply_salvage()가 3번째 킬 시각이 속한 라운드에
ace=True + first_kill_sec/ace_sec + salvage 태그를 스탬프 (settled kills 수치는
정직하게 그대로 둠 — ace 플래그와 클립 창만 세팅).

측정 루프: python -u hud_from_cache.py && python -u _compare_hud_gt.py
자가 검증:  python -u hud_streak_salvage.py  (실측 유형 픽스처 8케이스)
"""

from __future__ import annotations

from dataclasses import dataclass

_MAX_SPAN_SEC = 90.0        # 첫 킬 증거~3번째 킬 증거 최대 폭 (detect_ace_hud._MAX_ACE_SPAN_SEC와 동일 의미)
_STEP_MAX_SEC = 50.0        # 사다리 인접 단(p+i → p+i+1) 최대 간격
_WIN_AFTER_SEC = 25.0       # 3번째 킬 증거 후 win 이벤트 허용 창 (D2)
_WIN_BEFORE_SLACK = 10.0    # win이 3번째 킬 '판독'보다 살짝 먼저 찍히는 것 허용
                            #   (배너가 HUD를 가려 p+3 판독이 win보다 늦게 잡히는 실측 패턴)
_EVID_MIN_CONF = 0.55       # 단발 사다리 증거 최소 conf (D8)
_MIN_WINDOW_SAMPLES = 6     # 창 내 K 판독 성공 최소 수 (G5 축소판 — 암전 환각 방지)
_STABLE_GAP_SEC = 6.0       # 안정 상태 병합: 같은 값 연속 판독 허용 간격
_STABLE_MIN_READS = 2       # 안정 상태 최소 관측 수 …또는
_STABLE_SINGLE_CONF = 0.88  # 단발 안정 인정 conf (hud_round_settle._SUPPORT_SINGLE_CONF와 동일)
_SUPPORT_MIN = 2            # D6 '지지된 역행' 판단 최소 관측 수


@dataclass
class SalvageRec:
    round_index: int        # ace 스탬프 대상 라운드 (3번째 킬 증거 시각 소속)
    k_from: int             # 기준값 p
    kill_evid: list[float]  # p+1, p+2, p+3 첫 증거 시각 (오름차순 3개)
    win_t: float            # D2를 충족시킨 win 이벤트 시각
    reason: str = ""        # 진단용 (채택 근거 요약)


def _stable_states(k_reads: list[tuple[float, int, float]]) -> list[tuple[float, float, int]]:
    """연속 동일값 run → 안정 상태 [(t0, t1, v)]. n>=2 또는 conf>=0.88 run만."""
    states: list[tuple[float, float, int]] = []
    cur_v: int | None = None
    t0 = t1 = 0.0
    n = 0
    cmax = 0.0

    def flush() -> None:
        if cur_v is not None and (n >= _STABLE_MIN_READS or cmax >= _STABLE_SINGLE_CONF):
            states.append((t0, t1, cur_v))

    for t, k, c in k_reads:
        if k == cur_v and t - t1 <= _STABLE_GAP_SEC:
            t1 = t
            n += 1
            cmax = max(cmax, c)
        else:
            flush()
            cur_v, t0, t1, n, cmax = k, t, t, 1, c
    flush()
    return states


def _wins_in(win_events: list[dict], lo: float, hi: float) -> list[float]:
    return sorted(
        e.get("t_hi", -1e9)
        for e in win_events
        if e.get("kind") == "win" and lo <= e.get("t_hi", -1e9) <= hi
    )


def _d_anomaly_in(d_reads: list[tuple[float, int, float]], lo: float, hi: float) -> bool:
    """창 내 D 채널 이상 (hud_round_settle._d_anomaly_t와 동일 원리의 국소판)."""
    obs = [(t, d, c) for (t, d, c) in d_reads if lo <= t <= hi]
    if not obs:
        return False
    count: dict[int, int] = {}
    for _t, d, _c in obs:
        count[d] = count.get(d, 0) + 1
    ok = {d for d in count if count[d] >= _SUPPORT_MIN}
    base: int | None = None
    for t, d, _c in sorted(obs):
        if d not in ok:
            continue
        if base is None:
            base = d
        elif d != base:
            return True
    return False


def salvage_streak_aces(
    k_reads: list[tuple[float, int, float]],
    rounds: list,                      # RoundTrack 리스트 (정산 완료, ace 스탬프 후)
    boundary_strength: list[tuple[float, bool]],  # (경계 시각, 강한 경계 여부)
    win_events: list[dict] | None,
    d_reads: list[tuple[float, int, float]] | None = None,
    *,
    ace_kills: int = 3,
) -> list[SalvageRec]:
    """원시 K 판독에서 +ace_kills 사다리를 직접 탐색해 구제 후보를 반환.

    k_reads는 0-격리(_quarantine_zeros) 이후 판독이어야 한다 — 가짜 0(=8 오독)이
    남아 있으면 D6 역행 검사에 오염된다.
    """
    if not win_events:
        return []  # D1: 승수 채널 없으면 침묵
    d_reads = d_reads or []
    strong_bounds = sorted(t for t, strong in boundary_strength if strong)

    # 기존 올킬 라운드 구간 (D7)
    ace_spans = [
        (r.start_sec, r.end_sec) for r in rounds if getattr(r, "ace", False)
    ]

    recs: list[SalvageRec] = []
    claimed_until = float("-inf")  # 같은 스트릭에 중복 발화 방지

    for s0, s1, p in _stable_states(k_reads):
        if s1 <= claimed_until:
            continue
        # p 이후 사다리 증거 수집: p+1..p+ace_kills 각각의 '첫' 등장 시각 (시간순 강제)
        evid: list[float] = []
        t_cursor = s1
        ok = True
        for step in range(1, ace_kills + 1):
            target = p + step
            found = None
            for t, k, c in k_reads:
                if t <= t_cursor:
                    continue
                if t - t_cursor > _STEP_MAX_SEC:
                    break
                if k == target and c >= _EVID_MIN_CONF:
                    found = t
                    break
            if found is None:
                ok = False
                break
            evid.append(found)
            t_cursor = found
        if not ok:
            continue
        e1, e3 = evid[0], evid[-1]
        if e3 - e1 > _MAX_SPAN_SEC:
            continue

        w_lo, w_hi = s1, e3  # 킬 발생 창 (기준값 마지막 관측 ~ 3번째 킬 증거)

        # D7: 기존 올킬 라운드와 겹치면 스킵 (이미 잡힌 스트릭)
        if any(a <= w_hi and w_lo <= b for a, b in ace_spans):
            continue
        # D4: 강한 경계가 창을 가로지르면 진짜 라운드 종료 — 기각
        if any(w_lo < bt < w_hi for bt in strong_bounds):
            continue
        # D3: 창 내부 win — 라운드가 중간에 끝남 — 기각
        if _wins_in(win_events, w_lo + 1.0, e3 - _WIN_BEFORE_SLACK):
            continue
        # D2: 3번째 킬 직후 승리 필수
        wins_after = _wins_in(win_events, e3 - _WIN_BEFORE_SLACK, e3 + _WIN_AFTER_SEC)
        if not wins_after:
            continue
        # D5: 창 내 사망/관전 전환 — 관전 오독 방어
        if _d_anomaly_in(d_reads, w_lo, w_hi + 2.0):
            continue
        # D6: 창 내 '지지된' p 미만 값 — 체인 오염 신호
        low = [k for (t, k, _c) in k_reads if w_lo < t < w_hi and k < p]
        if any(low.count(v) >= _SUPPORT_MIN for v in set(low)):
            continue
        # 표본 하한 (암전 환각 방지)
        n_win = sum(1 for (t, _k, _c) in k_reads if w_lo <= t <= w_hi + 2.0)
        if n_win < _MIN_WINDOW_SAMPLES:
            continue

        # 대상 라운드: 3번째 킬 증거 시각이 속한 라운드
        target_r = None
        for r in rounds:
            if r.start_sec <= e3 <= r.end_sec:
                target_r = r
                break
        if target_r is None:
            continue

        recs.append(SalvageRec(
            round_index=target_r.round_index,
            k_from=p,
            kill_evid=list(evid),
            win_t=wins_after[0],
            reason=f"p={p} evid={[round(t,1) for t in evid]} win@{wins_after[0]:.1f}",
        ))
        claimed_until = e3  # 이 스트릭 소비 — 겹침 재발화 방지
    return recs


def apply_salvage(rounds: list, recs: list[SalvageRec]) -> int:
    """구제 레코드를 라운드에 스탬프. settled kills 수치는 그대로 두고
    ace/first_kill_sec/ace_sec/salvage 만 세팅 (클립 창과 GT 비교에 필요한 필드).
    반환: 실제 스탬프 수."""
    by_idx = {r.round_index: r for r in rounds}
    n = 0
    for rec in recs:
        r = by_idx.get(rec.round_index)
        if r is None or r.ace:
            continue
        r.ace = True
        r.salvage = rec.reason or "streak"
        e1, e3 = rec.kill_evid[0], rec.kill_evid[-1]
        r.first_kill_sec = min(r.first_kill_sec, e1) if r.first_kill_sec is not None else e1
        r.ace_sec = e3
        n += 1
    return n


# ---------------------------------------------------------------------------
# 자가 검증 — 실측 미탐/오탐 유형 픽스처. python -u hud_streak_salvage.py
# ---------------------------------------------------------------------------

def _selftest() -> None:
    from dataclasses import dataclass as _dc, field as _f

    @_dc
    class _R:  # RoundTrack 대역 (필요 필드만)
        round_index: int
        start_sec: float
        end_sec: float
        ace: bool = False
        salvage: str = ""
        first_kill_sec: float | None = None
        ace_sec: float | None = None

    def reads_of(*groups):
        out = []
        for t0, k, c, n, dt in groups:
            out.extend((t0 + i * dt, k, c) for i in range(n))
        return sorted(out)

    WIN = lambda t: {"kind": "win", "t_hi": t}  # noqa: E731

    # ① 03-12-36 유형 — 가짜 경계(213s)가 스트릭을 쪼갬 + '4' 단발 conf 0.86 필터 탈락.
    #    안정 2 (~190s) → 3(190s) 4(202s) 5(229s) 단발들 → 235s win. 구제되어야 함.
    reads1 = reads_of(
        (136.0, 2, 0.80, 30, 1.8),   # 기준값 2 안정
        (190.0, 3, 0.81, 3, 4.0),
        (202.0, 4, 0.86, 1, 1.0),    # 단발 0.86 — 정산 지지 필터에선 탈락하는 값
        (229.0, 5, 0.65, 2, 2.0),
    )
    rounds1 = [_R(4, 176.0, 213.0), _R(5, 213.0, 235.0)]
    bs1 = [(213.0, False)]           # 약한 경계 (무검증 — R10 안전핀 통과분)
    recs = salvage_streak_aces(reads1, rounds1, bs1, [WIN(235.0)])
    assert len(recs) == 1 and recs[0].round_index == 5, f"case1: {recs}"
    assert apply_salvage(rounds1, recs) == 1 and rounds1[1].ace
    assert rounds1[1].first_kill_sec == 190.0 and rounds1[1].ace_sec == 229.0

    # ② 스트래들 반례 — 같은 판독인데 2번째 킬 후 win(210s)이 창 내부에 찍힘
    #    (직전 라운드 2킬 + 다음 라운드 1킬) → D3 기각.
    recs = salvage_streak_aces(reads1, rounds1[:], bs1, [WIN(210.0), WIN(235.0)])
    assert recs == [], f"case2: {recs}"

    # ③ '직후 승리' 없음 — 3번째 킬 후 라운드가 계속됨(win이 한참 뒤) → D2 기각.
    recs = salvage_streak_aces(reads1, [_R(4, 176.0, 213.0), _R(5, 213.0, 280.0)],
                               bs1, [WIN(275.0)])
    assert recs == [], f"case3: {recs}"

    # ④ 강한 경계가 창을 가로지름 (CNN 전광판 확인 = 진짜 라운드 종료) → D4 기각.
    recs = salvage_streak_aces(reads1, rounds1[:], [(213.0, True)], [WIN(235.0)])
    assert recs == [], f"case4: {recs}"

    # ⑤ D-이상 — 창 내 사망(D 상승) → 이후 상승분은 관전 오독 가능 → D5 기각.
    d5 = reads_of((136.0, 1, 0.9, 30, 2.0), (205.0, 2, 0.9, 10, 2.0))
    recs = salvage_streak_aces(reads1, rounds1[:], bs1, [WIN(235.0)], d5)
    assert recs == [], f"case5: {recs}"

    # ⑥ 역행 오염 — 창 안에 지지된 p 미만 값(1이 2회) → D6 기각.
    reads6 = sorted(reads1 + [(210.0, 1, 0.7), (212.0, 1, 0.7)])
    recs = salvage_streak_aces(reads6, rounds1[:], bs1, [WIN(235.0)])
    assert recs == [], f"case6: {recs}"

    # ⑦ 이미 잡힌 올킬과 겹침 → D7 스킵 (중복 방지).
    rounds7 = [_R(4, 176.0, 213.0, ace=True), _R(5, 213.0, 235.0)]
    recs = salvage_streak_aces(reads1, rounds7, bs1, [WIN(235.0)])
    assert recs == [], f"case7: {recs}"

    # ⑧ 승수 채널 없음 → D1 전체 비활성.
    recs = salvage_streak_aces(reads1, rounds1[:], bs1, None)
    assert recs == [], f"case8: {recs}"

    print("hud_streak_salvage selftest: 8/8 OK")


if __name__ == "__main__":
    _selftest()
