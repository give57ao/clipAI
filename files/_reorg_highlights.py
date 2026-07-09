# -*- coding: utf-8 -*-
"""ace_clips_hud/<date>/<clips> -> 평탄화 + 오탐 분리.

- 현재 JSON의 ace round와 매칭되는 클립만 이동 대상 (구버전 잔재는 건드리지 않음)
- FP_SET에 있는 (stem, round_idx)는 _오탐/ 로, 나머지는 루트로 이동
- 이동 후 비어버린 날짜 폴더는 삭제, 잔재 파일이 남아있으면 폴더 유지

사용:
    python -u _reorg_highlights.py --dry-run
    python -u _reorg_highlights.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"E:\clipai_result\ace_clips_hud")
TIMELINE_DIR = Path(r"E:\clipai_result\hud_timeline")
FP_DIR = ROOT / "_오탐"

# 2026-07-09 GT 대조로 확인된 현재(최신 JSON 기준) 오탐 라운드
FP_SET: set[tuple[str, int]] = {
    ("2026-03-21 00-40-56", 94), ("2026-03-21 00-40-56", 101), ("2026-03-21 00-40-56", 104),
    ("2026-03-21 02-21-23", 5), ("2026-03-21 02-21-23", 64),
    ("2026-03-21 02-21-23", 107), ("2026-03-21 02-21-23", 124),
    ("2026-03-22 02-03-10", 20), ("2026-03-22 02-03-10", 58), ("2026-03-22 02-03-10", 62),
    ("2026-03-24 02-34-09", 30), ("2026-03-24 02-34-09", 41),
    ("2026-03-26 01-26-52", 49),
    ("2026-03-29 01-01-04", 46), ("2026-03-29 01-01-04", 70),
    ("2026-04-08 00-36-50", 16),
    ("2026-04-08 02-15-28", 69),
    ("2026-04-09 02-05-29", 39), ("2026-04-09 02-05-29", 161),
    # 3차 대조 추가 (2026-07-09)
    ("2026-04-23 23-41-07", 6),
    ("2026-04-28 00-07-24", 58),
    ("2026-04-29 01-14-15", 100),
    ("2026-05-12 00-23-40", 30),
    ("2026-05-18 01-54-57", 19),
    ("2026-05-19 01-36-22", 5),
    # 4차 대조 추가 (2026-07-09)
    ("2026-05-31 21-57-14", 139),
}

NAME_RE = re.compile(r"^(.*)_R(\d+)_")


def mss(sec: float) -> str:
    sec = int(round(sec))
    h, sec = divmod(sec, 3600)
    m, s = divmod(sec, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    FP_DIR.mkdir(parents=True, exist_ok=True)

    n_hl = n_fp = n_stale = n_nojson = 0
    stale_report: list[str] = []
    nojson_report: list[str] = []

    for stem_dir in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name != "_오탐"):
        stem = stem_dir.name
        jp = TIMELINE_DIR / f"{stem}.json"
        current_idx: dict[int, dict] = {}
        if jp.exists():
            data = json.loads(jp.read_text(encoding="utf-8"))
            for r in data.get("rounds", []):
                if r.get("ace"):
                    current_idx[r["round_index"]] = r
        else:
            nojson_report.append(stem)

        ordered_current = sorted(current_idx.keys())
        seq = {ri: i + 1 for i, ri in enumerate(ordered_current)}

        files = sorted(stem_dir.glob("*.mp4"))
        moved_any = False
        for f in files:
            m = NAME_RE.match(f.name)
            if not m:
                stale_report.append(f"{stem}/{f.name} (이름 패턴 불일치)")
                continue
            ridx = int(m.group(2))
            if ridx not in current_idx:
                stale_report.append(f"{stem}/{f.name} (구버전 잔재, 현재 JSON ace 아님)")
                continue

            if (stem, ridx) in FP_SET:
                dest = FP_DIR / f"{stem}_오탐_R{ridx:02d}.mp4"
                kind = "FP"
                n_fp += 1
            else:
                n = seq[ridx]
                dest = ROOT / f"{stem}_하이라이트({n}).mp4"
                kind = "HL"
                n_hl += 1

            if dest.exists():
                k = 2
                while dest.with_name(f"{dest.stem}_{k}{dest.suffix}").exists():
                    k += 1
                dest = dest.with_name(f"{dest.stem}_{k}{dest.suffix}")

            print(f"[{kind}] {stem}/{f.name} -> {dest.name}")
            if not args.dry_run:
                f.rename(dest)
            moved_any = True

        remaining = list(stem_dir.glob("*"))
        if not remaining:
            print(f"  (empty) rmdir {stem}")
            if not args.dry_run:
                stem_dir.rmdir()
        elif moved_any:
            print(f"  남은 잔재 {len(remaining)}개 있어 폴더 유지: {stem}")

    n_stale = len(stale_report)
    n_nojson = len(nojson_report)
    print("\n===== 요약 =====")
    print(f"하이라이트 이동: {n_hl}  |  오탐 이동: {n_fp}  |  잔재(미이동): {n_stale}  |  JSON 없는 폴더: {n_nojson}")
    if stale_report:
        print("\n-- 잔재(미이동) 목록 --")
        for s in stale_report:
            print(f"  {s}")
    if nojson_report:
        print("\n-- JSON 없는 폴더(전체 미이동) --")
        for s in nojson_report:
            print(f"  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
