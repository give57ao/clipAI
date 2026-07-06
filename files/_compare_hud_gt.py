# -*- coding: utf-8 -*-
"""HUD 타임라인 vs 수동 정답(HUD_ACE_HANDOFF.md §3) 자동 recall/precision.

사용:
    python -u _compare_hud_gt.py                 # E:\clipai_result\hud_timeline 기준
    python -u _compare_hud_gt.py --tol 15        # 매칭 허용 오차(초)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

TIMELINE_DIR = Path(r"E:\clipai_result\hud_timeline")


def _s(mss: str) -> float:
    parts = [int(p) for p in mss.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


# 정답 올킬 구간 (플레이 구간, HUD_ACE_HANDOFF.md §3 — 27건)
GT: dict[str, list[tuple[float, float]]] = {
    "2026-03-19 23-00-50": [(_s("11:47"), _s("12:16"))],
    "2026-03-19 23-13-48": [],
    "2026-03-21 00-40-56": [
        (_s("24:10"), _s("24:32")), (_s("37:25"), _s("38:05")),
        (_s("41:54"), _s("42:39")), (_s("1:05:01"), _s("1:05:31")),
    ],
    "2026-03-21 02-21-23": [
        (_s("54:20"), _s("54:41")), (_s("1:04:54"), _s("1:05:20")),
        (_s("1:19:51"), _s("1:20:10")),
    ],
    "2026-03-22 00-44-50": [(_s("9:00"), _s("9:10")), (_s("40:40"), _s("41:45"))],
    "2026-03-22 02-03-10": [
        (_s("5:38"), _s("5:55")), (_s("14:30"), _s("15:26")),
        (_s("22:37"), _s("23:18")), (_s("29:45"), _s("30:13")),
        (_s("46:32"), _s("46:50")),
    ],
    "2026-03-22 03-02-03": [(_s("10:18"), _s("10:48")), (_s("14:15"), _s("14:57"))],
    "2026-03-22 23-51-52": [(_s("9:20"), _s("9:27")), (_s("16:23"), _s("16:43"))],
    "2026-03-24 00-42-33": [
        (_s("4:05"), _s("4:45")), (_s("12:13"), _s("12:47")),
        (_s("36:40"), _s("36:50")),
    ],
    "2026-03-24 02-34-09": [
        (_s("2:20"), _s("2:35")), (_s("2:49"), _s("3:10")),
        (_s("13:38"), _s("14:04")), (_s("38:14"), _s("38:35")),
        (_s("51:40"), _s("52:10")),
    ],
}


def mss(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}:{s:02d}"


def _overlaps(a1: float, a2: float, b1: float, b2: float, tol: float) -> bool:
    return a1 - tol <= b2 and b1 - tol <= a2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(TIMELINE_DIR))
    ap.add_argument("--tol", type=float, default=15.0)
    args = ap.parse_args()
    tdir = Path(args.dir)

    tot_gt = tot_hit = tot_det = tot_tp = 0
    for stem, gts in GT.items():
        jp = tdir / f"{stem}.json"
        if not jp.exists():
            print(f"## {stem}: JSON 없음 — 스캔 필요")
            tot_gt += len(gts)
            continue
        data = json.loads(jp.read_text(encoding="utf-8"))
        aces = [r for r in data.get("rounds", []) if r.get("ace")]
        # 탐지 구간: ace_sec 있으면 [first_kill, ace_sec], 없으면 라운드 구간
        det = []
        for r in aces:
            d1 = r.get("first_kill_sec") or r["start_sec"]
            d2 = r.get("ace_sec") or r["end_sec"]
            det.append((r["round_index"], d1, max(d1, d2)))

        hits = []
        used = set()
        for (g1, g2) in gts:
            match = None
            for (ri, d1, d2) in det:
                if ri in used:
                    continue
                if _overlaps(g1, g2, d1, d2, args.tol):
                    match = ri
                    used.add(ri)
                    break
            hits.append(((g1, g2), match))
        n_hit = sum(1 for _, m in hits if m is not None)
        n_fp = len(det) - len(used)
        tot_gt += len(gts)
        tot_hit += n_hit
        tot_det += len(det)
        tot_tp += len(used)

        print(f"## {stem}: GT {len(gts)}건 중 {n_hit} 탐지, 오탐 {n_fp}")
        for (g1, g2), m in hits:
            tag = f"→ R{m:02d} ✓" if m is not None else "→ 미탐 ✗"
            print(f"   GT {mss(g1)}-{mss(g2)} {tag}")
        for (ri, d1, d2) in det:
            if ri not in used:
                print(f"   FP R{ri:02d} {mss(d1)}-{mss(d2)}")

    print("\n===== 합계 =====")
    recall = tot_hit / tot_gt if tot_gt else 0.0
    prec = tot_tp / tot_det if tot_det else 0.0
    print(f"GT {tot_gt}건 | 탐지 {tot_hit} (recall {recall:.1%}) | "
          f"검출 {tot_det}건 중 TP {tot_tp} (precision {prec:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
