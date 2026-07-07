# -*- coding: utf-8 -*-
"""GT별 TP 획득/상실 diff — 실험 전후 두더지잡기 방지.

사용:
    python -u _tp_diff.py --save-baseline c2_gt26
    python -u _tp_diff.py --compare-to c2_gt26
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from _compare_hud_gt import GT, TIMELINE_DIR, _overlaps, mss

BASELINE_DIR = Path(__file__).parent / "_tp_baselines"


def _gt_key(stem: str, g1: float, g2: float) -> str:
    return f"{stem}|{mss(g1)}-{mss(g2)}"


def _collect_hits(tdir: Path, tol: float) -> dict[str, str | None]:
    """GT 키 → 탐지 라운드 문자열(Rxx) 또는 None."""
    hits: dict[str, str | None] = {}
    for stem, gts in GT.items():
        jp = tdir / f"{stem}.json"
        if not jp.exists():
            for g1, g2 in gts:
                hits[_gt_key(stem, g1, g2)] = None
            continue
        data = json.loads(jp.read_text(encoding="utf-8"))
        aces = [r for r in data.get("rounds", []) if r.get("ace")]
        det = []
        for r in aces:
            d1 = r.get("first_kill_sec") or r["start_sec"]
            d2 = r.get("ace_sec") or r["end_sec"]
            det.append((r["round_index"], d1, max(d1, d2)))
        used: set[int] = set()
        for g1, g2 in gts:
            key = _gt_key(stem, g1, g2)
            match = None
            for ri, d1, d2 in det:
                if ri in used:
                    continue
                if _overlaps(g1, g2, d1, d2, tol):
                    match = f"R{ri:02d}"
                    used.add(ri)
                    break
            hits[key] = match
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(TIMELINE_DIR))
    ap.add_argument("--tol", type=float, default=15.0)
    ap.add_argument("--save-baseline", metavar="NAME")
    ap.add_argument("--compare-to", metavar="NAME")
    args = ap.parse_args()

    tdir = Path(args.dir)
    current = _collect_hits(tdir, args.tol)

    if args.save_baseline:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        out = BASELINE_DIR / f"{args.save_baseline}.json"
        out.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        n_hit = sum(1 for v in current.values() if v is not None)
        print(f"[tp_diff] 베이스라인 저장: {out.name} ({n_hit}/{len(current)} TP)")
        return 0

    if args.compare_to:
        base_path = BASELINE_DIR / f"{args.compare_to}.json"
        if not base_path.exists():
            print(f"[tp_diff] 베이스라인 없음: {base_path}")
            return 1
        baseline = json.loads(base_path.read_text(encoding="utf-8"))
        gained: list[str] = []
        lost: list[str] = []
        kept: list[str] = []
        for key in sorted(set(baseline) | set(current)):
            b, c = baseline.get(key), current.get(key)
            if b and c:
                kept.append(f"  유지 {key} ({c})")
            elif not b and c:
                gained.append(f"  +획득 {key} → {c}")
            elif b and not c:
                lost.append(f"  -상실 {key} (was {b})")
        print(f"[tp_diff] vs {args.compare_to}")
        print(f"  유지 {len(kept)} | +획득 {len(gained)} | -상실 {len(lost)}")
        for line in gained:
            print(line)
        for line in lost:
            print(line)
        n_cur = sum(1 for v in current.values() if v is not None)
        n_base = sum(1 for v in baseline.values() if v is not None)
        print(f"  TP {n_base} → {n_cur} (Δ{n_cur - n_base:+d})")
        return 0

    n_hit = sum(1 for v in current.values() if v is not None)
    print(f"[tp_diff] 현재 TP {n_hit}/{len(current)}")
    for key, val in sorted(current.items()):
        tag = val if val else "미탐"
        print(f"  {key} → {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
