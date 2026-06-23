# -*- coding: utf-8 -*-
"""ffprobe/ffmpeg 공통 유틸."""

from __future__ import annotations

import subprocess
from pathlib import Path


def probe_duration_sec(video_path: Path) -> float | None:
    if not video_path.exists():
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    txt = result.stdout.strip()
    if not txt:
        return None
    try:
        return float(txt)
    except ValueError:
        return None
