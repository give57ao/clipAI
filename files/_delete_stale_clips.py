# -*- coding: utf-8 -*-
"""ace_clips_hud 잔재(구버전 알고리즘 결과물) 삭제.

_reorg_highlights.py가 이동시키지 않고 남겨둔, 현재 hud_timeline JSON의
ace round와 매칭되지 않는 잔재 클립을 삭제하고 빈 폴더를 정리한다.

사용:
    python -u _delete_stale_clips.py --dry-run
    python -u _delete_stale_clips.py
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
NAME_RE = re.compile(r"^(.*)_R(\d+)_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total_bytes = 0
    total_files = 0

    for stem_dir in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name not in ("_오탐", "_miss_review")):
        stem = stem_dir.name
        jp = TIMELINE_DIR / f"{stem}.json"
        current_idx: set[int] = set()
        if jp.exists():
            data = json.loads(jp.read_text(encoding="utf-8"))
            current_idx = {r["round_index"] for r in data.get("rounds", []) if r.get("ace")}

        for f in stem_dir.glob("*.mp4"):
            m = NAME_RE.match(f.name)
            is_stale = (not m) or (int(m.group(2)) not in current_idx)
            if not is_stale:
                continue
            size = f.stat().st_size
            total_bytes += size
            total_files += 1
            print(f"[DEL] {stem}/{f.name} ({size/1e6:.0f}MB)")
            if not args.dry_run:
                f.unlink()

        remaining = list(stem_dir.glob("*"))
        if not remaining:
            print(f"  (empty) rmdir {stem}")
            if not args.dry_run:
                stem_dir.rmdir()

    print("\n===== 요약 =====")
    print(f"삭제 대상 {total_files}개, {total_bytes/1e9:.2f}GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
