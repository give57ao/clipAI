# -*- coding: utf-8 -*-
"""_CLIP_MAX_SEC 55s->120s 변경 반영 — 기존 프로덕션 ace 클립 재추출.

영상 재스캔 없음(hud_timeline JSON 그대로 사용) — ffmpeg -c copy만 수행.
"""
from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from detect_ace_hud import RoundTrack, extract_ace_clips

TIMELINE_DIR = Path(r"E:\clipai_result\hud_timeline")
OUTPUT_DIR = Path(r"E:\clipai_result\ace_clips_hud")
_ROUND_FIELDS = {f.name for f in fields(RoundTrack)}


def main() -> int:
    total_videos = 0
    total_clips = 0
    missing_source = []
    for jp in sorted(TIMELINE_DIR.glob("*.json")):
        data = json.loads(jp.read_text(encoding="utf-8"))
        rounds = [r for r in data.get("rounds", []) if r.get("ace")]
        if not rounds:
            continue
        video_path = Path(data["video_path"])
        if not video_path.exists():
            missing_source.append(str(video_path))
            continue
        total_videos += 1
        round_tracks = [RoundTrack(**{k: v for k, v in r.items() if k in _ROUND_FIELDS}) for r in rounds]
        timeline = SimpleNamespace(video_path=data["video_path"], rounds=round_tracks)
        written = extract_ace_clips(video_path, timeline, OUTPUT_DIR)
        total_clips += len(written)

    print(f"\n[reextract] 완료: 영상 {total_videos}개, 클립 {total_clips}개")
    if missing_source:
        print(f"[reextract] 원본 없어서 건너뜀: {len(missing_source)}개")
        for m in missing_source:
            print(" ", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
