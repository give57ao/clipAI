# -*- coding: utf-8 -*-
"""HUD 원시 판독 신호 캐시 빌더 — 영상당 1패스, 이후 판정 실험은 재판독 0회.

캐시 = 프레임별 (t, k, conf, method). 판정 로직(_KTracker/경계/게이트)을 바꿔도
`hud_from_cache.py`로 즉시 재평가 가능. ⚠ 판독 자체(템플릿·ROI·IoU 상수)를 바꾸면
캐시가 낡음 → `--force`로 재생성.

사용:
    python -u hud_sig_cache.py                    # 라벨 10영상 (없는 것만)
    python -u hud_sig_cache.py --force            # 전부 재생성
    python -u hud_sig_cache.py --videos "E:\\OBS\\xxx.mp4" ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from detect_ace_hud import collect_reads

DEFAULT_CACHE_DIR = Path(r"E:\clipai_result\sig_cache_v2")

# 라벨 10영상 (HUD_ACE_HANDOFF.md §3)
DEFAULT_VIDEOS = [
    r"E:\OBS\2026-03-19 23-00-50.mp4",
    r"E:\OBS\2026-03-19 23-13-48.mp4",
    r"E:\OBS\2026-03-21 00-40-56.mp4",
    r"E:\OBS\2026-03-21 02-21-23.mp4",
    r"E:\OBS\2026-03-22 00-44-50.mp4",
    r"E:\OBS\2026-03-22 02-03-10.mp4",
    r"E:\OBS\2026-03-22 03-02-03.mp4",
    r"E:\OBS\2026-03-22 23-51-52.mp4",
    r"E:\OBS\2026-03-24 00-42-33.mp4",
    r"E:\OBS\2026-03-24 02-34-09.mp4",
]

_METHOD_CODE = {"template": "T", "template_miss": "M", "row_miss": "R", "triple_incomplete": "I"}
METHOD_DECODE = {v: k for k, v in _METHOD_CODE.items()}


def build_cache(video_path: Path, cache_dir: Path, scan_fps: float, force: bool) -> bool:
    out = cache_dir / f"{video_path.stem}.json"
    if out.exists() and not force:
        print(f"[cache] skip (존재): {video_path.stem}")
        return True
    if not video_path.exists():
        print(f"[cache] 영상 없음: {video_path}")
        return False
    t0 = time.time()
    reads, duration, err = collect_reads(video_path, scan_fps=scan_fps)
    if err:
        print(f"[cache] 실패 {video_path.stem}: {err}")
        return False
    data = {
        "stem": video_path.stem,
        "scan_fps": scan_fps,
        "duration": duration,
        # [t, k(null 가능), conf(소수3), method 1글자]
        "reads": [
            [round(r.t, 3), r.k, round(r.conf, 3), _METHOD_CODE.get(r.method, "?")]
            for r in reads
        ],
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    hits = sum(1 for r in reads if r.k is not None)
    print(
        f"[cache] {video_path.stem}: reads={len(reads)} hit={hits} "
        f"({duration/60:.0f}min, {time.time()-t0:.0f}s)",
        flush=True,
    )
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="HUD 원시 판독 신호 캐시")
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--scan-fps", type=float, default=4.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    videos = [Path(v) for v in (args.videos or DEFAULT_VIDEOS)]
    cache_dir = Path(args.cache_dir)
    ok = sum(build_cache(v, cache_dir, args.scan_fps, args.force) for v in videos)
    print(f"[cache] 완료 {ok}/{len(videos)} -> {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
