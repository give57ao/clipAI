# -*- coding: utf-8 -*-
"""_reextract_ace_clips.py 1차 실행에서 FAIL난 32건만 재시도."""
from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from detect_ace_hud import ace_clip_window, sec_to_mss
from extract_labeled_clips import run_ffmpeg_extract
from types import SimpleNamespace

TIMELINE_DIR = Path(r"E:\clipai_result\hud_timeline")
OUTPUT_DIR = Path(r"E:\clipai_result\ace_clips_hud")

FAILED = [
    ("2026-05-23 02-48-06", 66), ("2026-05-25 02-06-09", 62), ("2026-05-26 23-45-51", 12),
    ("2026-05-31 02-37-59", 1), ("2026-05-31 02-37-59", 10), ("2026-05-31 03-26-16", 11),
    ("2026-05-31 21-57-14", 26), ("2026-05-31 21-57-14", 78), ("2026-06-02 00-50-15", 1),
    ("2026-06-13 01-46-02", 23), ("2026-06-13 01-46-02", 41), ("2026-06-14 00-55-19", 58),
    ("2026-06-16 00-55-18", 1), ("2026-06-18 23-25-01", 48), ("2026-06-18 23-25-01", 102),
    ("2026-06-19 23-58-29", 53), ("2026-06-22 01-24-45", 2), ("2026-06-22 02-08-51", 16),
    ("2026-06-24 01-55-38", 50), ("2026-06-25 02-10-38", 13), ("2026-06-25 03-10-00", 29),
    ("2026-06-26 23-43-25", 1), ("2026-06-26 23-43-25", 46), ("2026-06-26 23-43-25", 54),
    ("2026-06-27 01-31-51", 16), ("2026-06-28 01-10-51", 5), ("2026-06-28 01-10-51", 25),
    ("2026-06-28 01-42-47", 22), ("2026-06-30 01-17-15", 1), ("2026-06-30 22-24-03", 42),
    ("2026-07-04 01-21-49", 3), ("2026-07-05 02-22-22", 81),
]


def main() -> int:
    ok = fail = 0
    still_failed = []
    for stem, ri in FAILED:
        jp = TIMELINE_DIR / f"{stem}.json"
        data = json.loads(jp.read_text(encoding="utf-8"))
        video_path = Path(data["video_path"])
        r = next(x for x in data["rounds"] if x["round_index"] == ri)
        ns = SimpleNamespace(start_sec=r["start_sec"], end_sec=r["end_sec"],
                              first_kill_sec=r.get("first_kill_sec"), ace_sec=r.get("ace_sec"))
        start, end = ace_clip_window(ns)
        out_dir = OUTPUT_DIR / stem
        label = sec_to_mss(r["end_sec"]).replace(":", "m")
        out_path = out_dir / f"{stem}_R{ri:02d}_{label}s_hud_ace.mp4"
        success = run_ffmpeg_extract(video_path, start, end, out_path)
        if success:
            ok += 1
        else:
            fail += 1
            still_failed.append((stem, ri))
        print(f"{stem} R{ri:02d} -> {'OK' if success else 'FAIL'}")
    print(f"\n[retry] {ok} 성공 / {fail} 재실패")
    if still_failed:
        print("여전히 실패:", still_failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
