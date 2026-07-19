# -*- coding: utf-8 -*-
"""R10 게이트가 새로 만든 FP 3건 — 육안 확인용 클립 추출 (stream copy, 재인코딩 없음)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from detect_ace_hud import ace_clip_window, sec_to_mss
from extract_labeled_clips import run_ffmpeg_extract

ON_DIR = Path(r"E:\clipai_result\_precision_investigate\gate_on")
OUT_DIR = Path(r"E:\clipai_result\_precision_investigate\fp_clips")
OBS_DIR = Path(r"E:\OBS")

CASES = [
    ("2026-03-29 01-01-04", 26),
    ("2026-03-29 03-12-36", 48),
    ("2026-05-17 21-47-39", 16),
]

for stem, ri in CASES:
    data = json.loads((ON_DIR / f"{stem}.json").read_text(encoding="utf-8"))
    r = next(x for x in data["rounds"] if x["round_index"] == ri)
    ns = SimpleNamespace(
        start_sec=r["start_sec"], end_sec=r["end_sec"],
        first_kill_sec=r.get("first_kill_sec"), ace_sec=r.get("ace_sec"),
    )
    start, end = ace_clip_window(ns)
    video_path = OBS_DIR / f"{stem}.mp4"
    label = sec_to_mss(r["end_sec"]).replace(":", "m")
    out_path = OUT_DIR / f"{stem}_R{ri:02d}_{label}s_FP.mp4"
    ok = run_ffmpeg_extract(video_path, start, end, out_path)
    print(f"{stem} R{ri:02d} [{sec_to_mss(start)}-{sec_to_mss(end)}] kt={[sec_to_mss(t) for t in r['kill_times']]} -> {out_path.name} {'OK' if ok else 'FAIL'}")
