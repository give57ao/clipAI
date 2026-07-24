# -*- coding: utf-8 -*-
"""D슬롯 hardneg 하베스터 — CNN-D가 실제로 발동하는 population을 라벨링 (2026-07-24).

배경(사용자 질문 "6,9는 cnn말고 기존 ocr하면 되는거 아니야?" — 정곡을 찔렀음):
`hud_kda._read_digit_group`의 CNN-D 폴백은 `d is None`(OCR/템플릿 미판독)일
때만 발동한다(hud_kda.py 237행 `if _CNN_V2_D and raw_patches is None and d is None`).
그런데 지금까지 D슬롯 CNN 게이트(G3)가 쓰던 데이터(`_build_digit_dataset_v2.py`의
d_claimed)는 전부 "OCR이 이미 성공한 안정런"에서 뽑은 것 — CNN-D가 실전에서
절대 켜지지 않을 population을 재고 있었다(가짜 게이트).

이 스크립트는 K슬롯의 `hardneg_times`(안정런 **내부**의 template_miss 프레임을
그 런의 값으로 라벨)와 동일 원리를 D 채널에 적용한다. D 채널은 별도 method
태그가 없으므로(caches에 d_method 컬럼 없음) d=None 자체가 미스 신호 —
안정 D-런 안쪽(_HARDNEG_PAD 마진)의 d=None 프레임만 그 런의 값으로 라벨링.

기존 K hardneg와 같은 `hardneg_<digit>/` 폴더를 공유(파일명 슬롯 'd'로 구분 —
`_build_digit_dataset_v2.py`의 claimed D도 같은 관례). 이래야
`train_hud_digit_cnn.py`의 hardneg 파싱(`name.split("_",1)[1]`)과 100% 호환.

사용: python -u _harvest_d_hardneg.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
from _build_digit_dataset_v2 import (
    _CAP_PER_CLASS,
    _HARDNEG_PAD,
    OBS_DIR,
    OUT_DIR,
    SIG_DIR,
    _thin,
    stable_runs,
)


def d_hardneg_times(
    all_d_reads: list[tuple[float, int | None]],
    runs: list[tuple[float, float, int]],
) -> list[tuple[float, int]]:
    """안정 D-런 내부(_HARDNEG_PAD 안쪽)의 미판독(d=None) → (t, 정답값).

    K의 `hardneg_times`와 동일 원리 — 여기선 method 태그 대신 d=None 자체가
    미스 신호(D 채널엔 별도 template_miss 구분이 없음, 판독은 이분적)."""
    out: list[tuple[float, int]] = []
    for t, d in all_d_reads:
        if d is not None:
            continue
        for r0, r1, v in runs:
            if r0 + _HARDNEG_PAD <= t <= r1 - _HARDNEG_PAD:
                out.append((t, v))
                break
    return out


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

    all_d = [(r[0], r[4]) for r in rows]
    d_ok = [(t, d) for t, d in all_d if d is not None]
    runs = stable_runs(d_ok)
    if not runs:
        return {}

    hardneg = d_hardneg_times(all_d, runs)
    hardneg = _thin(hardneg, key=lambda x: x[0], cap=_CAP_PER_CLASS)
    if not hardneg:
        return {}

    cap = cv2.VideoCapture(str(video_p))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    saved: dict[str, int] = {}
    for t, value in hardneg:
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
            folder = OUT_DIR / f"hardneg_{digit}"
            folder.mkdir(parents=True, exist_ok=True)
            name = f"{stem}_{int(t)}s_d{gi}.png"
            cv2.imwrite(str(folder / name), g)
            saved[str(digit)] = saved.get(str(digit), 0) + 1
    cap.release()
    if saved:
        print(f"[hardneg-d] {stem}: " + ", ".join(f"{k}={v}" for k, v in sorted(saved.items())))
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
    print("\n[hardneg-d] 합계:", dict(sorted(total.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
