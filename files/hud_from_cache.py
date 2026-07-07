# -*- coding: utf-8 -*-
"""신호 캐시 → 판정 재계산 → hud_timeline JSON (영상 재판독 0회).

detect_ace_hud.timeline_from_reads(실스캔과 동일 로직)를 그대로 호출하므로
_KTracker·경계·게이트를 수정하면 이 도구 결과에 즉시 반영됨.

측정 루프 (판정 로직 수정 → 3초 재평가):
    python -u hud_from_cache.py && python -u _compare_hud_gt.py

특정 구간 원시 판독 눈으로 보기 (진단):
    python -u hud_from_cache.py --dump "2026-03-22 02-03-10" 41:50 42:10
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from detect_ace_hud import DEFAULT_JSON_DIR, KRead, format_report, timeline_from_reads
from hud_sig_cache import DEFAULT_CACHE_DIR, METHOD_DECODE


def _parse_mss(s: str) -> float:
    parts = [int(p) for p in s.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def load_reads(cache_path: Path) -> tuple[list[KRead], float, float, str]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    reads = [
        KRead(t=t, k=k, conf=c, method=METHOD_DECODE.get(m, m))
        for t, k, c, m in data["reads"]
    ]
    return reads, data["duration"], data.get("scan_fps", 4.0), data["stem"]


def load_boundary_verdicts(stem: str, cache_dir: Path) -> list[list] | None:
    """R2 Task 1(hud_boundary_verify.py) 산출물. 없으면 None(검증 없이 전체 유지)."""
    p = cache_dir / f"{stem}.boundary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("runs", [])


def main() -> int:
    ap = argparse.ArgumentParser(description="신호 캐시 → 판정 재계산")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_JSON_DIR))
    ap.add_argument("--report", action="store_true", help="라운드별 리포트 출력")
    ap.add_argument("--dump", nargs=3, metavar=("STEM", "FROM", "TO"),
                    help='원시 판독 덤프: --dump "stem" 41:50 42:10')
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)

    if args.dump:
        stem, t1s, t2s = args.dump
        t1, t2 = _parse_mss(t1s), _parse_mss(t2s)
        reads, _, _, _ = load_reads(cache_dir / f"{stem}.json")
        for r in reads:
            if t1 <= r.t <= t2:
                m = int(r.t // 60)
                print(f"  {m}:{r.t % 60:05.2f}  K={r.k}  conf={r.conf:.2f}  {r.method}")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cp in sorted(cache_dir.glob("*.json")):
        if cp.name.endswith(".boundary.json"):
            continue
        reads, duration, scan_fps, stem = load_reads(cp)
        boundary_verdicts = load_boundary_verdicts(stem, cache_dir)
        tl = timeline_from_reads(
            reads,
            duration=duration,
            video_path=Path(rf"E:\OBS\{stem}.mp4"),
            scan_fps=scan_fps,
            boundary_verdicts=boundary_verdicts,
        )
        (out_dir / f"{stem}.json").write_text(
            json.dumps(asdict(tl), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[from-cache] {stem}: rounds={len(tl.rounds)} ace={tl.ace_rounds}")
        if args.report:
            print(format_report(tl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
