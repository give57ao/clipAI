# -*- coding: utf-8 -*-
"""gate-on(r10_gt107) vs gate-off(재구성) FP 목록 diff — precision 하락 원인 조사용.

캐시 기반 재계산 결과 두 디렉터리(hud_from_cache.py --out-dir)를 비교해,
gate-on 상태의 각 FP가 gate-off 쪽에도 시간대가 겹치는 검출이 있는지 확인.
없으면 "R10 게이트가 새로 만든 FP"로 표시.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from _compare_hud_gt import GT, mss

ON_DIR = Path(r"E:\clipai_result\_precision_investigate\gate_on")
OFF_DIR = Path(r"E:\clipai_result\_precision_investigate\gate_off")
TOL = 15.0


def _dets(jp: Path) -> list[tuple[int, float, float]]:
    if not jp.exists():
        return []
    data = json.loads(jp.read_text(encoding="utf-8"))
    out = []
    for r in data.get("rounds", []):
        if not r.get("ace"):
            continue
        d1 = r.get("first_kill_sec") or r["start_sec"]
        d2 = r.get("ace_sec") or r["end_sec"]
        out.append((r["round_index"], d1, max(d1, d2)))
    return out


def _overlaps(a1, a2, b1, b2, tol) -> bool:
    return a1 - tol <= b2 and b1 - tol <= a2


def main() -> int:
    new_fp = []
    preexisting_fp = []
    for stem, gts in GT.items():
        on_det = _dets(ON_DIR / f"{stem}.json")
        off_det = _dets(OFF_DIR / f"{stem}.json")

        def mark_tp(dets):
            used = set()
            tp = set()
            for (g1, g2) in gts:
                for (ri, d1, d2) in dets:
                    if ri in used:
                        continue
                    if _overlaps(g1, g2, d1, d2, TOL):
                        used.add(ri)
                        tp.add(ri)
                        break
            return tp

        on_tp = mark_tp(on_det)
        off_tp = mark_tp(off_det)
        on_fp = [(ri, d1, d2) for (ri, d1, d2) in on_det if ri not in on_tp]

        for (ri, d1, d2) in on_fp:
            matched_off = any(
                _overlaps(d1, d2, od1, od2, TOL) for (_ori, od1, od2) in off_det
            )
            entry = f"{stem} R{ri:02d} {mss(d1)}-{mss(d2)}"
            if matched_off:
                preexisting_fp.append(entry)
            else:
                new_fp.append(entry)

    print(f"[r10_fp_diff] gate-on FP 총 {len(new_fp) + len(preexisting_fp)}건")
    print(f"\n=== 게이트 무관 기존 FP ({len(preexisting_fp)}) ===")
    for e in preexisting_fp:
        print(" ", e)
    print(f"\n=== R10 게이트가 새로 만든 FP ({len(new_fp)}) ===")
    for e in new_fp:
        print(" ", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
