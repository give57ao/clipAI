# -*- coding: utf-8 -*-
"""R13(hud_streak_salvage) 미구제 케이스 진단 (T2, SONNET_TASK_UNDERREAD.md).

score_cache가 있는데도 구제 안 된 RAW_DK_OK 케이스마다 salvage_streak_aces의
방어선(D1~D8) 로직을 그대로 재현하며 어디서 멈췄는지 출력한다.
salvage_streak_aces() 자체는 건드리지 않고 복사해 진단 print만 추가.

사용:
    python -u _salvage_why.py                  # miss_diag_full.txt 기준 전수
    python -u _salvage_why.py --stem "<stem>" --t0 <sec>   # 단건
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
from hud_round_settle import _quarantine_zeros
from hud_score_wins import load_score_timeline, score_events
from hud_streak_salvage import (
    _EVID_MIN_CONF,
    _MAX_SPAN_SEC,
    _MIN_WINDOW_SAMPLES,
    _STEP_MAX_SEC,
    _WIN_AFTER_SEC,
    _WIN_BEFORE_SLACK,
    _d_anomaly_in,
    _stable_states,
    _wins_in,
)
from detect_ace_hud import _BOUNDARY_WIN_MARGIN_BEFORE, _BOUNDARY_WIN_MARGIN_AFTER


def _strong_boundary_crossed(rounds, win_events, w_lo: float, w_hi: float) -> float | None:
    """D4 근사 재현: w_lo~w_hi 사이 라운드 경계(=인접 라운드 start_sec) 중
    승수 인접(강한 경계)인 것이 있으면 그 시각을 반환. CNN 검증(boundary_verdicts)은
    미반영 — 있으면 더 많은 경계가 강한 걸로 잡혀 이 함수는 D4를 과소평가(누락)할 수 있음."""
    for r in rounds:
        bt = r.get("start_sec", 0)
        if not (w_lo < bt < w_hi):
            continue
        near = any(
            e.get("kind") == "win"
            and (bt - _BOUNDARY_WIN_MARGIN_BEFORE) <= e.get("t_hi", -1e9) <= (bt + _BOUNDARY_WIN_MARGIN_AFTER)
            for e in win_events
        )
        if near:
            return bt
    return None
from dataclasses import asdict

SIG_DIR = Path(r"E:\clipai_result\sig_cache_v2")
TL_DIR = Path(r"E:\clipai_result\hud_timeline")


def mss(s: float) -> str:
    s = int(s)
    return f"{s // 60}:{s % 60:02d}"


def load_reads(stem: str):
    d = json.loads((SIG_DIR / f"{stem}.json").read_text(encoding="utf-8"))
    k_reads = [(r[0], r[1], r[2]) for r in d["reads"] if r[1] is not None]
    d_reads = [(r[0], r[4], r[2]) for r in d["reads"] if len(r) > 4 and r[4] is not None]
    return _quarantine_zeros(k_reads), d_reads


def load_wins(stem: str):
    tl = load_score_timeline(stem)
    if tl is None:
        return None
    return [asdict(e) for e in score_events(tl)]


def load_bounds(stem: str):
    tl = json.loads((TL_DIR / f"{stem}.json").read_text(encoding="utf-8"))
    return tl.get("rounds", [])


def diagnose(stem: str, g0: float, ace_kills: int = 3) -> str:
    k_reads, d_reads = load_reads(stem)
    win_events = load_wins(stem)
    rounds = load_bounds(stem)
    ace_spans = [(r["start_sec"], r["end_sec"]) for r in rounds if r.get("ace")]

    if not win_events:
        return "D1: score_cache는 있으나 win 이벤트 0건 (앵커/판독 실패) — win채널 문제"

    # GT 시각(g0) 근방 안정 상태에서 시작하는 사다리만 진단 (전수 대신 GT 주변에 한정)
    cands = [(s0, s1, p) for (s0, s1, p) in _stable_states(k_reads) if g0 - 90 <= s1 <= g0 + 10]
    if not cands:
        return "사다리 탐색: GT 근방에 '안정 상태(p)' 자체가 없음 — 근본 한계(원시 부족)"

    msgs = []
    for s0, s1, p in cands:
        t_cursor = s1
        evid = []
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
                msgs.append(f"p={p}@{mss(s1)}: {step}단계(K={target}) 증거 없음"
                            f" (STEP_MAX={_STEP_MAX_SEC}s 내) — 근본한계 or STEP_MAX 튜닝대상")
                ok = False
                break
            evid.append(found)
            t_cursor = found
        if not ok:
            continue
        e1, e3 = evid[0], evid[-1]
        if e3 - e1 > _MAX_SPAN_SEC:
            msgs.append(f"p={p}: 사다리 폭 {e3-e1:.0f}s > MAX_SPAN {_MAX_SPAN_SEC}s — 근본한계")
            continue
        w_lo, w_hi = s1, e3
        if any(a <= w_hi and w_lo <= b for a, b in ace_spans):
            msgs.append(f"p={p} evid={[round(t,1) for t in evid]}: D7 스킵"
                        f" (기존 ace 라운드와 겹침 — 이미 다른 라운드가 이 킬 이벤트를 정탐한 상태,"
                        f" GT 중복/근접 스팬 의심)")
            continue
        strong_bt = _strong_boundary_crossed(rounds, win_events, w_lo, w_hi)
        if strong_bt is not None:
            msgs.append(f"p={p} evid={[round(t,1) for t in evid]}: D4 위반"
                        f" (강한 경계@{mss(strong_bt)} 관통) — 완화 금지, 진짜 라운드 종료로 판단됨")
            continue
        win_in_window = _wins_in(win_events, w_lo + 1.0, e3 - _WIN_BEFORE_SLACK)
        if win_in_window:
            msgs.append(f"p={p} evid={[round(t,1) for t in evid]}: D3 위반"
                        f" (창 내부 win@{win_in_window}) — 스트래들, 완화 금지")
            continue
        wins_after = _wins_in(win_events, e3 - _WIN_BEFORE_SLACK, e3 + _WIN_AFTER_SEC)
        if not wins_after:
            nearest = min(
                (e.get("t_hi") for e in win_events if e.get("kind") == "win"),
                key=lambda t: abs(t - e3), default=None,
            )
            gap = abs(nearest - e3) if nearest is not None else None
            msgs.append(f"p={p} evid={[round(t,1) for t in evid]}: D2 위반"
                        f" (직후 {_WIN_AFTER_SEC}s 내 win 없음, 최근접 win gap={gap}) — WIN_AFTER 튜닝대상 or win채널 누락")
            continue
        if _d_anomaly_in(d_reads, w_lo, w_hi + 2.0):
            msgs.append(f"p={p}: D5 위반 (창 내 D-이상) — 완화 금지")
            continue
        low = [k for (t, k, _c) in k_reads if w_lo < t < w_hi and k < p]
        from collections import Counter
        cnt = Counter(low)
        bad = [v for v, n in cnt.items() if n >= 2]
        if bad:
            msgs.append(f"p={p}: D6 위반 (역행 지지값 {bad}) — 근본한계(오염)")
            continue
        n_win = sum(1 for (t, _k, _c) in k_reads if w_lo <= t <= w_hi + 2.0)
        if n_win < _MIN_WINDOW_SAMPLES:
            msgs.append(f"p={p}: 표본부족 (n={n_win}<{_MIN_WINDOW_SAMPLES})")
            continue
        msgs.append(f"p={p} evid={[round(t,1) for t in evid]} win@{wins_after[0]:.1f}: "
                     f"방어선 전부 통과 — 구제됐어야 함 (D4/대상라운드 확인 필요)")
    return " | ".join(msgs) if msgs else "사다리 후보 전부 STEP1에서 탈락 (근본한계)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem")
    ap.add_argument("--t0", type=float)
    ap.add_argument("--src", default="miss_diag_full.txt")
    args = ap.parse_args()

    if args.stem is not None:
        cases = [(args.stem, args.t0)]
    else:
        src = Path(args.src)
        if not src.exists():
            print(f"{src} 없음 — 먼저 python _miss_diag.py > {src} 실행할 것")
            return 1
        cases = []
        for l in src.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\[KILLS_LOW \] (.+?)  (\d+):(\d+)   R(\d+) kills=(\d+)", l)
            if m:
                cases.append((m.group(1), int(m.group(2)) * 60 + int(m.group(3))))

    buckets = {"tunable": [], "win채널": [], "근본한계": [], "GT중복의심": [], "확인필요": [], "캐시없음": []}
    for stem, g0 in cases:
        if not (SIG_DIR / f"{stem}.json").exists():
            buckets["캐시없음"].append((stem, g0, "sig_cache 없음"))
            continue
        verdict = diagnose(stem, g0)
        print(f"### {stem} {mss(g0)}\n    {verdict}\n")
        if "win채널" in verdict or "win 이벤트 0건" in verdict:
            buckets["win채널"].append((stem, g0, verdict))
        elif "D7 스킵" in verdict and "튜닝대상" not in verdict:
            buckets["GT중복의심"].append((stem, g0, verdict))
        elif "튜닝대상" in verdict:
            buckets["tunable"].append((stem, g0, verdict))
        elif "근본한계" in verdict or "완화 금지" in verdict:
            buckets["근본한계"].append((stem, g0, verdict))
        else:
            buckets["확인필요"].append((stem, g0, verdict))

    print("===== 버킷 집계 =====")
    for k, v in buckets.items():
        print(f"  {k:<10} {len(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
