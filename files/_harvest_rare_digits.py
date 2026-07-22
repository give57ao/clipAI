# -*- coding: utf-8 -*-
"""희소 클래스(6·9) 전용 완화 수확 — 2026-07-23.

원인 조사: K=6 원시 판독은 4299건이나 되는데(0/5 등과 비슷한 규모), 안정런
기준(_RUN_MIN_HITS=4)을 만족하는 런이 905개 중 182개뿐 — 6은 유독 판독이
자주 끊기는 값(542개가 단발 n=1)이라 기존 하베스터가 대부분 놓침.
n>=2로 완화하면 363개 런 확보(약 2배).

_build_digit_dataset_v2.py의 stable_runs()를 그대로 재사용하되 완화된
_RUN_MIN_HITS로 6/9 값의 런만 추가 수확 (다른 클래스는 건드리지 않음 —
이미 충분하므로 완화 시 노이즈만 늘어날 위험).

사용: python -u _harvest_rare_digits.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
from _build_digit_dataset_v2 import OBS_DIR, OUT_DIR, SIG_DIR, _thin

_RELAXED_RUN_MIN_HITS = 2
_RUN_MAX_GAP = 3.0
_TARGETS = {6, 9}
_CAP_PER_CLASS = 200  # 희소 클래스라 기존 cap(80)보다 넉넉히


def relaxed_runs(reads: list[tuple[float, int | None]]) -> list[tuple[float, float, int]]:
    """stable_runs와 동일 로직, n>=2만 다름(희소 클래스 구제용)."""
    runs = []
    cur_v = None
    t0 = t1 = 0.0
    n = 0

    def flush():
        if cur_v in _TARGETS and n >= _RELAXED_RUN_MIN_HITS:
            runs.append((t0, t1, cur_v))

    for t, k in reads:
        if k is None:
            continue
        if k == cur_v and t - t1 <= _RUN_MAX_GAP:
            t1 = t
            n += 1
        else:
            flush()
            cur_v, t0, t1, n = k, t, t, 1
    flush()
    return runs


def harvest_stem(stem: str) -> dict[str, int]:
    import cv2

    from game_frame import extract_game_crop_bgr
    from hud_kda import locate_kda_glyphs, normalize_glyph

    cache_p = SIG_DIR / f"{stem}.json"
    video_p = OBS_DIR / f"{stem}.mp4"
    if not cache_p.exists() or not video_p.exists():
        return {}

    data = json.loads(cache_p.read_text(encoding="utf-8"))
    k_reads = [(r[0], r[1]) for r in data["reads"]]
    runs = relaxed_runs(k_reads)
    if not runs:
        return {}

    claimed = []
    for r0, r1, v in runs:
        claimed.extend((t, v) for t, k in k_reads if k == v and r0 <= t <= r1)
    claimed = _thin(claimed, key=lambda x: x[0], cap=_CAP_PER_CLASS)

    cap = cv2.VideoCapture(str(video_p))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    saved = {}
    for t, value in claimed:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        game, _box = extract_game_crop_bgr(frame)
        if game is None or game.size == 0:
            continue
        glyphs = locate_kda_glyphs(game)
        if glyphs is None:
            continue
        ds = [int(c) for c in str(value)]
        if len(ds) != len(glyphs.k):
            continue  # 자리수 불일치 — 로케이터 불일치, 폐기
        for gi, p in enumerate(glyphs.k):
            g = normalize_glyph(p)
            if g is None:
                continue
            digit = ds[gi]
            d_out = OUT_DIR / str(digit)
            d_out.mkdir(parents=True, exist_ok=True)
            name = f"{stem}_{int(t)}s_k{gi}_relaxed.png"
            cv2.imwrite(str(d_out / name), g)
            saved[str(digit)] = saved.get(str(digit), 0) + 1
    cap.release()
    if saved:
        print(f"[rare] {stem}: " + ", ".join(f"{k}={v}" for k, v in sorted(saved.items())))
    return saved


def main() -> int:
    stems = sorted(
        p.stem for p in SIG_DIR.glob("*.json")
        if not p.name.endswith(".boundary.json") and (OBS_DIR / f"{p.stem}.mp4").exists()
    )
    total: dict[str, int] = {}
    for s in stems:
        for k, v in harvest_stem(s).items():
            total[k] = total.get(k, 0) + v
    print("\n[rare] 합계:", dict(sorted(total.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
