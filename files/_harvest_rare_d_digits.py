# -*- coding: utf-8 -*-
"""D슬롯 희소 클래스(6·8·9) 전용 완화 수확 — 2026-07-24.

배경: G3(D슬롯 CNN 게이트) 재측정(94.83%, 기준 99%)을 분해해보니 '자신있게
틀린 숫자'는 0/812(0.00%)로 실제 안전 위험은 없었음 — 부족분은 전부 저신뢰
기권(무해, 템플릿 폴백과 동일)이었다. 다만 학습 데이터 자체가
D=8 0개(!) · D=9 1개 · D=6 3개로 극히 희소해 그 값들에서의 실제 안전성은
'측정 불가'였다(`_digit_cnn_v2_gate.py` 클래스별 표 참고).

`_harvest_rare_digits.py`(K슬롯 6/9 전용, 63%→95.1% 개선 검증됨)와 동일한
완화-런(n>=2) 방식을 D 채널(6열 캐시의 5번째 원소)에 적용. 다른 슬롯/클래스는
건드리지 않음(기존 D=0~5,7 학습 데이터는 이미 충분 — `_build_digit_dataset_v2.py`
plan_labels()의 표준 런(n>=4)으로 커버됨).

사용: python -u _harvest_rare_d_digits.py
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
_TARGETS = {6, 8, 9}
_CAP_PER_CLASS = 200  # 희소 클래스라 기존 cap(80)보다 넉넉히


def relaxed_runs(reads: list[tuple[float, int | None]]) -> list[tuple[float, float, int]]:
    """_harvest_rare_digits.relaxed_runs와 동일 로직 (D 채널용 재사용을 위해 복제 —
    원본은 K 전용 모듈이라 임포트 대신 동일 순수함수를 여기 둠)."""
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
    rows = data["reads"]
    if not rows or len(rows[0]) <= 4:
        return {}  # 구 캐시(4열, D 채널 없음) — 스킵
    d_reads = [(r[0], r[4]) for r in rows if r[4] is not None]
    runs = relaxed_runs(d_reads)
    if not runs:
        return {}

    claimed = []
    for r0, r1, v in runs:
        claimed.extend((t, v) for t, d in d_reads if d == v and r0 <= t <= r1)
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
        if len(ds) != len(glyphs.d):
            continue  # 자리수 불일치 — 로케이터 불일치, 폐기
        for gi, p in enumerate(glyphs.d):
            g = normalize_glyph(p)
            if g is None:
                continue
            digit = ds[gi]
            d_out = OUT_DIR / str(digit)
            d_out.mkdir(parents=True, exist_ok=True)
            name = f"{stem}_{int(t)}s_d{gi}_relaxed.png"
            cv2.imwrite(str(d_out / name), g)
            saved[str(digit)] = saved.get(str(digit), 0) + 1
    cap.release()
    if saved:
        print(f"[rare-d] {stem}: " + ", ".join(f"{k}={v}" for k, v in sorted(saved.items())))
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
    print("\n[rare-d] 합계:", dict(sorted(total.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
