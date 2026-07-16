# -*- coding: utf-8 -*-
"""HUD 타임라인 vs 수동 정답(files/gt_aces.json) 자동 recall/precision.

사용:
    python -u _compare_hud_gt.py                 # E:\clipai_result\hud_timeline 기준
    python -u _compare_hud_gt.py --tol 15        # 매칭 허용 오차(초)
    python -u _compare_hud_gt.py --source-available-only  # 원본 소실 GT 제외한 서브셋 지표

정답 원본은 files/gt_aces.json (2026-07-17 GT 이관, IMPROVEMENT_REPORT.md §B-1) —
과거 이 dict와 HUD_ACE_HANDOFF.md §3 표가 이중 관리되며 GT 충돌 사고가 났었다.
개별 GT 항목의 확정/수정 이력(육안 재확인, 정정 사유 등)은 git 히스토리(이 파일의
이관 전 버전, `git log -p -- files/_compare_hud_gt.py`)에서 확인.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

TIMELINE_DIR = Path(r"E:\clipai_result\hud_timeline")
GT_JSON_PATH = Path(__file__).parent / "gt_aces.json"

_GT_DATA: dict = json.loads(GT_JSON_PATH.read_text(encoding="utf-8"))

# 정답 올킬 구간 (플레이 구간) — 값의 원본은 gt_aces.json, 이 이름은 하위호환 유지
# (_gt_source_audit.py 등 기존 코드가 `from _compare_hud_gt import GT`로 참조)
GT: dict[str, list[tuple[float, float]]] = {
    stem: [tuple(span) for span in entry["spans"]] for stem, entry in _GT_DATA.items()
}

# 원본 영상이 소실돼 재스캔이 불가능한 GT는 별도 표시(§D-1) — 이 dict는 gt_aces.json의
# source_available 필드를 그대로 노출
GT_SOURCE_AVAILABLE: dict[str, bool] = {
    stem: entry["source_available"] for stem, entry in _GT_DATA.items()
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
    ap.add_argument(
        "--source-available-only", action="store_true",
        help="원본 소실 GT 영상의 상세 출력을 생략(집계는 항상 전체+서브셋 둘 다 계산됨)",
    )
    args = ap.parse_args()
    tdir = Path(args.dir)

    tot_gt = tot_hit = tot_det = tot_tp = 0
    sub_gt = sub_hit = sub_det = sub_tp = 0  # source_available 서브셋(§D-1) 전용 집계
    for stem, gts in GT.items():
        available = GT_SOURCE_AVAILABLE.get(stem, True)
        if args.source_available_only and not available:
            continue
        jp = tdir / f"{stem}.json"
        if not jp.exists():
            print(f"## {stem}: JSON 없음 — 스캔 필요")
            tot_gt += len(gts)
            if available:
                sub_gt += len(gts)
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
        if available:
            sub_gt += len(gts)
            sub_hit += n_hit
            sub_det += len(det)
            sub_tp += len(used)

        print(f"## {stem}: GT {len(gts)}건 중 {n_hit} 탐지, 오탐 {n_fp}"
              + ("" if available else "  [원본 소실 — 재스캔 불가]"))
        for (g1, g2), m in hits:
            tag = f"→ R{m:02d} ✓" if m is not None else "→ 미탐 ✗"
            print(f"   GT {mss(g1)}-{mss(g2)} {tag}")
        for (ri, d1, d2) in det:
            if ri not in used:
                print(f"   FP R{ri:02d} {mss(d1)}-{mss(d2)}")

    print("\n===== 합계 (전체 GT) =====")
    recall = tot_hit / tot_gt if tot_gt else 0.0
    prec = tot_tp / tot_det if tot_det else 0.0
    print(f"GT {tot_gt}건 | 탐지 {tot_hit} (recall {recall:.1%}) | "
          f"검출 {tot_det}건 중 TP {tot_tp} (precision {prec:.1%})")

    n_available = sum(1 for a in GT_SOURCE_AVAILABLE.values() if a)
    sub_recall = sub_hit / sub_gt if sub_gt else 0.0
    sub_prec = sub_tp / sub_det if sub_det else 0.0
    print(f"\n===== 서브셋 (원본 보유 {n_available}영상 — 재스캔 가능) =====")
    print(f"GT {sub_gt}건 | 탐지 {sub_hit} (recall {sub_recall:.1%}) | "
          f"검출 {sub_det}건 중 TP {sub_tp} (precision {sub_prec:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
