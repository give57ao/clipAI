# -*- coding: utf-8 -*-
"""GT 영상 자산 전수 감사 — 원본·캐시·타임라인 보유 현황 (IMPROVEMENT_REPORT §D-1).

드라이브 용량 확보를 위해 원본을 지워야 할 때 "지우면 안 되는 것"을 먼저
확정하는 읽기 전용 도구. 어떤 파일도 수정·삭제하지 않는다.

사용:
    python -u _gt_source_audit.py                # 표 출력 + gt_source_audit.json 저장
    python -u _gt_source_audit.py --no-write     # 표만
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from _compare_hud_gt import GT  # 정답 원본 (61영상) — §B-1 이관 전까지 유일 소스

SOURCE_DIRS = [Path(r"E:\OBS"), Path(r"D:\\")]
SIG_CACHE_DIR = Path(r"E:\clipai_result\sig_cache")
TIMELINE_DIR = Path(r"E:\clipai_result\hud_timeline")
OUT_JSON = Path(__file__).parent / "gt_source_audit.json"


def find_source(stem: str) -> Path | None:
    for d in SOURCE_DIRS:
        p = d / f"{stem}.mp4"
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="GT 원본/캐시/타임라인 보유 감사")
    ap.add_argument("--no-write", action="store_true", help="JSON 저장 생략")
    args = ap.parse_args()

    rows = []
    for stem, spans in sorted(GT.items()):
        src = find_source(stem)
        rows.append({
            "stem": stem,
            "gt_spans": len(spans),
            "source": str(src) if src else None,
            "sig_cache": (SIG_CACHE_DIR / f"{stem}.json").exists(),
            "hud_timeline": (TIMELINE_DIR / f"{stem}.json").exists(),
        })

    lost = [r for r in rows if r["source"] is None]
    lost_spans = sum(r["gt_spans"] for r in lost)
    total_spans = sum(r["gt_spans"] for r in rows)
    # 원본도 캐시도 없는 GT: 판독 재실험조차 불가 — 최우선 보호 대상은 아님(이미 소실)
    # 원본 有 + 캐시 無: 지우기 전에 캐시부터 만들어야 하는 영상
    need_cache = [r for r in rows if r["source"] and not r["sig_cache"]]

    print(f"GT 영상 {len(rows)}개 / 구간 {total_spans}건")
    print(f"  원본 보유: {len(rows) - len(lost)}개  |  소실: {len(lost)}개 (구간 {lost_spans}건 재검증 불가)")
    print(f"  원본 있으나 sig_cache 없음(삭제 전 캐시 필수): {len(need_cache)}개")
    print()
    print(f"{'상태':<6} {'구간':>3}  {'캐시':<4} {'stem'}")
    for r in rows:
        st = "소실" if r["source"] is None else ("D:" if r["source"].startswith("D:") else "OBS")
        print(f"{st:<6} {r['gt_spans']:>3}  {'유' if r['sig_cache'] else '무':<4} {r['stem']}")

    if lost:
        print("\n[재검증 불가 — 원본 소실 GT]")
        for r in lost:
            print(f"  {r['stem']} (구간 {r['gt_spans']}건, timeline {'유' if r['hud_timeline'] else '무'})")
    if need_cache:
        print("\n[원본 삭제 전 sig_cache 구축 필요 (단일 프로세스로!)]")
        for r in need_cache:
            print(f"  {r['stem']}")

    if not args.no_write:
        OUT_JSON.write_text(
            json.dumps({"generated": "gt_source_audit", "rows": rows}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"\n저장: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
