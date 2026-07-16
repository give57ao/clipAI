# -*- coding: utf-8 -*-
"""미탐 34건 원인 분류: 각 GT 미탐 구간에 대응하는 HUD 타임라인 라운드를 찾아
어디서 떨어졌는지 분류.

분류:
  NO_SCAN   : hud_timeline JSON 자체가 없음
  NO_ROUND  : JSON은 있으나 GT 시각을 덮는 라운드가 없음(라운드 분할 실패/병합)
  KILLS_LOW : 라운드는 있고 kills<3 (K 과소판독) — kills 값 함께 표시
  ACE_FALSE : kills>=3인데 ace=False (게이트 기각) — 있으면 사유
  DETECTED  : 사실 ace=True인데 tol 밖 (경계 문제)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
from _compare_hud_gt import GT  # noqa

TL = Path(r"E:\clipai_result\hud_timeline")


def mss(sec: float) -> str:
    s = int(sec)
    return f"{s // 60}:{s % 60:02d}"


def find_round(rounds, g0, g1):
    """GT 구간 [g0,g1]과 겹치는(또는 가장 가까운) 라운드."""
    best = None
    best_ov = 0.0
    nearest = None
    nearest_gap = 1e9
    for r in rounds:
        rs, re_ = r.get("start_sec", 0), r.get("end_sec", 0)
        ov = max(0, min(g1, re_) - max(g0, rs))
        if ov > best_ov:
            best_ov = ov
            best = r
        mid = (g0 + g1) / 2
        gap = min(abs(mid - rs), abs(mid - re_))
        if gap < nearest_gap:
            nearest_gap = gap
            nearest = r
    return best, nearest, nearest_gap


rows = []
for stem, spans in GT.items():
    if not spans:
        continue
    jp = TL / f"{stem}.json"
    if not jp.exists():
        for (g0, g1) in spans:
            rows.append((stem, mss(g0), "NO_SCAN", ""))
        continue
    d = json.loads(jp.read_text(encoding="utf-8"))
    rounds = d.get("rounds") or []
    aces = set(d.get("ace_rounds") or [])
    for (g0, g1) in spans:
        best, nearest, gap = find_round(rounds, g0, g1)
        if best and best.get("ace"):
            continue  # 탐지됨
        if best is None:
            r = nearest
            ri = r.get("round_index") if r else "?"
            det = f"근접R{ri} {mss(r['start_sec'])}-{mss(r['end_sec'])} gap={gap:.0f}s" if r else ""
            rows.append((stem, mss(g0), "NO_ROUND", det))
        else:
            k = best.get("kills")
            ks = best.get("k_samples")
            kt = best.get("kill_times") or []
            ri = best.get("round_index")
            er = best.get("end_reason")
            if best.get("ace"):
                rows.append((stem, mss(g0), "DETECTED?", f"R{ri} tol밖"))
            elif isinstance(k, int) and k >= 3:
                rows.append((stem, mss(g0), "ACE_FALSE", f"R{ri} kills={k} end={er}"))
            else:
                rows.append((stem, mss(g0),
                             "KILLS_LOW", f"R{ri} kills={k} k_samp={ks} ktimes={len(kt)} end={er}"))

# 집계
from collections import Counter
cnt = Counter(r[2] for r in rows)
print("===== 미탐 원인 집계 =====")
for cat, n in cnt.most_common():
    print(f"  {cat:12s} {n}")
print(f"  {'TOTAL':12s} {len(rows)}")
print()
print("===== 상세 =====")
cur = None
for stem, ts, cat, det in sorted(rows, key=lambda x: (x[2], x[0])):
    print(f"[{cat:10s}] {stem}  {ts}   {det}")
