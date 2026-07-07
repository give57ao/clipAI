# -*- coding: utf-8 -*-
"""TP 잠금 리스트 — 실험 전후 recall/precision 변화의 내부 손익(획득/상실)을 diff로 보여줌.

두더지잡기 방지: "총점만 보면 +2-2=0"인 변경도 어느 GT를 얻고 어느 GT를 잃었는지 드러남.

사용:
    python -u _tp_diff.py --dir "E:\\clipai_result\\hud_timeline"          # 현재 상태 vs 고정 베이스라인
    python -u _tp_diff.py --save-baseline BASELINE_NAME                    # 현재 상태를 새 베이스라인으로 저장
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import _compare_hud_gt as gt

BASELINE_DIR = Path(__file__).parent / "_tp_baselines"

# 고정 베이스라인 3종 (2026-07-06/07 기록, 근거는 HUD_ACE_HANDOFF.md 참고)
FIXED_BASELINES = {
    "a_opus_v1cache": None,   # 구 sig_cache(v1) 방식 — 파이프라인이 달라 GT별 매칭 재현 불가, 수치만 기록
    "b_sonnet_v2_g5_with8": None,  # sig_cache_v2 + v2트래커 + G5, 8템플릿 포함 (recall 51.9%/66.7%)
}


def _overlaps(a1, a2, b1, b2, tol=15):
    return a1 - tol <= b2 and b1 - tol <= a2


def current_hits(tdir: Path) -> dict[str, list[str]]:
    """{stem: [hit GT라벨, ...]} — 현재 hud_timeline JSON 기준."""
    out: dict[str, list[str]] = {}
    for stem, gts in gt.GT.items():
        jp = tdir / f"{stem}.json"
        if not jp.exists():
            continue
        data = json.loads(jp.read_text(encoding="utf-8"))
        aces = [r for r in data.get("rounds", []) if r.get("ace")]
        det = []
        for r in aces:
            d1 = r.get("first_kill_sec") or r["start_sec"]
            d2 = r.get("ace_sec") or r["end_sec"]
            det.append((r["round_index"], d1, max(d1, d2)))
        hits = []
        used = set()
        for (g1, g2) in gts:
            for (ri, d1, d2) in det:
                if ri in used:
                    continue
                if _overlaps(g1, g2, d1, d2):
                    used.add(ri)
                    hits.append(f"{gt.mss(g1)}-{gt.mss(g2)}")
                    break
        out[stem] = hits
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(gt.TIMELINE_DIR))
    ap.add_argument("--save-baseline", default=None, help="현재 상태를 이 이름으로 저장")
    ap.add_argument("--compare-to", default="b_sonnet_v2_g5_with8", help="비교할 베이스라인 이름")
    args = ap.parse_args()

    BASELINE_DIR.mkdir(exist_ok=True)
    cur = current_hits(Path(args.dir))
    cur_flat = {f"{stem}::{h}" for stem, hits in cur.items() for h in hits}

    if args.save_baseline:
        out = BASELINE_DIR / f"{args.save_baseline}.json"
        out.write_text(json.dumps(sorted(cur_flat), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[baseline] {args.save_baseline} 저장 -> {out} ({len(cur_flat)}건)")
        return 0

    base_path = BASELINE_DIR / f"{args.compare_to}.json"
    if not base_path.exists():
        print(f"[diff] 베이스라인 없음: {base_path} — 먼저 --save-baseline 으로 저장할 것")
        print(f"[diff] 현재 TP {len(cur_flat)}건: {sorted(cur_flat)}")
        return 0

    base_flat = set(json.loads(base_path.read_text(encoding="utf-8")))
    gained = sorted(cur_flat - base_flat)
    lost = sorted(base_flat - cur_flat)
    kept = sorted(cur_flat & base_flat)

    print(f"=== TP diff: 현재 vs '{args.compare_to}' ({len(base_flat)}건) ===")
    print(f"유지: {len(kept)}건")
    for x in kept:
        print(f"   = {x}")
    print(f"\n획득: {len(gained)}건")
    for x in gained:
        print(f"   + {x}")
    print(f"\n상실: {len(lost)}건")
    for x in lost:
        print(f"   - {x}")
    print(f"\n순변화: {len(cur_flat) - len(base_flat):+d}  (현재 {len(cur_flat)} vs 베이스라인 {len(base_flat)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
