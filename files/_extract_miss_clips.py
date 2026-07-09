# -*- coding: utf-8 -*-
"""GT 정답인데 파이프라인이 못 찾은(미탐) 구간을 원본에서 직접 잘라 _miss_review에 저장.

사용:
    python -u _extract_miss_clips.py --dry-run
    python -u _extract_miss_clips.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
from _compare_hud_gt import GT, mss  # noqa: E402

OBS_DIR = Path(r"E:\OBS")
OUT_DIR = Path(r"E:\clipai_result\ace_clips_hud\_miss_review")
PAD_SEC = 3.0


def det_ranges(stem: str) -> list[tuple[float, float]]:
    import json
    jp = Path(r"E:\clipai_result\hud_timeline") / f"{stem}.json"
    if not jp.exists():
        return []
    data = json.loads(jp.read_text(encoding="utf-8"))
    out = []
    for r in data.get("rounds", []):
        if r.get("ace"):
            d1 = r.get("first_kill_sec") or r["start_sec"]
            d2 = r.get("ace_sec") or r["end_sec"]
            out.append((d1, max(d1, d2)))
    return out


def overlaps(a1, a2, b1, b2, tol=15.0):
    return a1 - tol <= b2 and b1 - tol <= a2


def name_for(sec: float) -> str:
    sec = int(round(sec))
    h, sec = divmod(sec, 3600)
    m, s = divmod(sec, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_done = n_skip = n_missing_src = 0
    for stem, gts in GT.items():
        src = OBS_DIR / f"{stem}.mp4"
        det = det_ranges(stem)
        for g1, g2 in gts:
            hit = any(overlaps(g1, g2, d1, d2) for d1, d2 in det)
            if hit:
                continue  # 이미 탐지된 정탐이면 미탐 클립 불필요
            dest = OUT_DIR / f"{stem}_미탐_{name_for(g1)}-{name_for(g2)}.mp4"
            if dest.exists():
                n_skip += 1
                continue
            if not src.exists():
                print(f"[SRC 없음] {stem} ({mss(g1)}-{mss(g2)}) -> 스킵")
                n_missing_src += 1
                continue
            ss = max(0.0, g1 - PAD_SEC)
            to = g2 + PAD_SEC
            print(f"[EXTRACT] {stem} {mss(g1)}-{mss(g2)} -> {dest.name}")
            if not args.dry_run:
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(ss), "-to", str(to), "-i", str(src),
                     "-c", "copy", str(dest)],
                    check=True, capture_output=True,
                )
            n_done += 1

    print(f"\n===== 요약 ===== 추출 {n_done}  |  이미있음(스킵) {n_skip}  |  원본없음 {n_missing_src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
