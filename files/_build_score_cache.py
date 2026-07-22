# -*- coding: utf-8 -*-
"""T1 후속 — 신규 17영상 score_cache(승수 채널) 생성 (SONNET_TASK_UNDERREAD.md §2 T1).

hud_score_wins.py에는 CLI가 없어(라이브러리 전용) scan_score_timeline()을
직접 호출하는 드라이버. 영상당 1패스(2fps 기본), sig_cache보다 훨씬 가벼움
(K/D/A 4fps 템플릿매칭 대비 "005" 앵커 1회 + 승수 자릿수만 판독).

사용: python -u _build_score_cache.py
"""
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
from hud_score_wins import DEFAULT_SCORE_CACHE_DIR, scan_score_timeline

STEMS = [
    "2026-06-14 00-55-19", "2026-06-16 00-55-18", "2026-06-18 23-25-01",
    "2026-06-19 23-58-29", "2026-06-22 02-08-51", "2026-06-23 01-06-42",
    "2026-06-24 01-55-38", "2026-06-25 03-10-00", "2026-06-26 23-43-25",
    "2026-06-27 01-31-51", "2026-06-28 01-42-47", "2026-06-30 01-17-15",
    "2026-06-30 22-24-03", "2026-06-30 23-36-38", "2026-07-03 21-42-21",
    "2026-07-04 01-21-49", "2026-07-05 02-22-22",
]

OBS_DIR = Path(r"E:\OBS")


def main() -> int:
    ok = skip = fail = 0
    for stem in STEMS:
        out = DEFAULT_SCORE_CACHE_DIR / f"{stem}.json"
        if out.exists():
            print(f"[score] skip (존재): {stem}")
            skip += 1
            continue
        vp = OBS_DIR / f"{stem}.mp4"
        if not vp.exists():
            print(f"[score] 영상 없음: {vp}")
            fail += 1
            continue
        t0 = time.time()
        data = scan_score_timeline(vp)
        dt = time.time() - t0
        if data is None:
            print(f"[score] 앵커 탐지 실패(R6 비활성): {stem} ({dt:.0f}s)")
            fail += 1
        else:
            print(f"[score] {stem}: reads={len(data['reads'])} ({dt:.0f}s)")
            ok += 1
    print(f"\n[score] 완료 {ok} / 스킵 {skip} / 실패 {fail} (총 {len(STEMS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
