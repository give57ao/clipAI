# -*- coding: utf-8 -*-
"""문맥 증거로 8 글리프 수확 — 순수 시각 유사도(IoU)로는 8이 0과 구분 안 됨(실측:
E:\\clipai_result\\hud_templates_harvest\\cnn_dataset\\eight_candidates_top144.png
상위 유사도 후보조차 태반이 0). 대신 **7 확정 → template_miss(행 발견, 숫자 미매칭)
구간 → 9 확정, 사이 다른 확정값 없음** 이라는 문맥으로 그 구간이 8일 수밖에 없다고
추론(R2 Task 3 팬텀-8과 동일 논리) — 이 구간의 실제 프레임을 영상에서 seek로 재추출.

사용:
    python -u _harvest_eight_contextual.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from game_frame import extract_game_crop_bgr
from hud_digit_match import normalize_glyph
from hud_kda import locate_kda_glyphs

CACHE_DIR = Path(r"E:\clipai_result\sig_cache_v2")
OUT_DIR = Path(r"E:\clipai_result\hud_templates_harvest\eight_ctx")
_MAX_GAP_SEC = 15.0
_MAX_PER_VIDEO_MISS_FRAMES = 8  # 한 갭에서 뽑을 최대 프레임 수 (양끝 치우침 방지 위해 중앙 우선)

VIDEO_MAP = {
    "2026-03-19 23-00-50": r"E:\OBS\2026-03-19 23-00-50.mp4",
    "2026-03-19 23-13-48": r"E:\OBS\2026-03-19 23-13-48.mp4",
    "2026-03-21 00-40-56": r"E:\OBS\2026-03-21 00-40-56.mp4",
    "2026-03-21 02-21-23": r"E:\OBS\2026-03-21 02-21-23.mp4",
    "2026-03-22 00-44-50": r"E:\OBS\2026-03-22 00-44-50.mp4",
    "2026-03-22 02-03-10": r"E:\OBS\2026-03-22 02-03-10.mp4",
    "2026-03-22 03-02-03": r"E:\OBS\2026-03-22 03-02-03.mp4",
    "2026-03-22 23-51-52": r"E:\OBS\2026-03-22 23-51-52.mp4",
    "2026-03-24 00-42-33": r"E:\OBS\2026-03-24 00-42-33.mp4",
    "2026-03-24 02-34-09": r"E:\OBS\2026-03-24 02-34-09.mp4",
}


def find_gaps(reads: list[list]) -> list[tuple[float, float, list[float]]]:
    """(prev_t, cur_t, miss_ts) — 7 확정 → (M만, R 없이) → 9 확정.

    'I'(triple_incomplete, K는 맞았을 수도 있으나 D/A 중 하나가 깨져 트리플가드에
    걸린 프레임)는 체인을 끊지 않음 — 8이 떠 있어도 이 코드로 잡힐 수 있음.
    'R'(row_miss, 행 자체가 없음 = 라운드 경계)만 진짜 리셋 — 경계 넘는 브리징은
    별개 문제(HUD_ACE_HANDOFF.md R3 "cross-round 팬텀 브리징" 과제)라 여기선 제외.
    """
    gaps: list[tuple[float, float, list[float]]] = []
    prev_val = None
    prev_t = None
    miss_run: list[float] = []
    for t, k, conf, method in reads:
        if method == "T" and k is not None:
            if (
                prev_val == 7
                and k == 9
                and miss_run
                and (t - prev_t) <= _MAX_GAP_SEC
            ):
                gaps.append((prev_t, t, list(miss_run)))
            prev_val, prev_t = k, t
            miss_run = []
        elif method == "M":
            if prev_val is not None:
                miss_run.append(t)
        elif method == "R":
            prev_val, prev_t = None, None
            miss_run = []
        # 'I'는 무시 (체인 유지, 증거로도 안 씀 — K단독 신뢰 불가)
    return gaps


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for stem, vpath in VIDEO_MAP.items():
        cache_path = CACHE_DIR / f"{stem}.json"
        if not cache_path.exists():
            print(f"[eight-ctx] 캐시 없음: {stem}")
            continue
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        gaps = find_gaps(data["reads"])
        if not gaps:
            print(f"[eight-ctx] {stem}: 후보 갭 0개")
            continue

        vp = Path(vpath)
        if not vp.exists():
            print(f"[eight-ctx] 영상 없음: {vp}")
            continue
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        saved = 0
        for prev_t, cur_t, miss_ts in gaps:
            # 중앙 우선으로 최대 N개만 (양끝은 이전/다음 값의 잔상일 수 있음)
            mid = len(miss_ts) // 2
            lo = max(0, mid - _MAX_PER_VIDEO_MISS_FRAMES // 2)
            pick = miss_ts[lo : lo + _MAX_PER_VIDEO_MISS_FRAMES]
            for t in pick:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
                ok, frame = cap.read()
                if not ok:
                    continue
                game, _ = extract_game_crop_bgr(frame)
                g = locate_kda_glyphs(game)
                if g is None or not g.k:
                    continue
                glyph = normalize_glyph(g.k[0])
                if glyph is None:
                    continue
                name = f"{stem}_{t:.2f}s_k0.png"
                cv2.imwrite(str(OUT_DIR / name), glyph)
                saved += 1
        cap.release()
        total += saved
        print(f"[eight-ctx] {stem}: 갭 {len(gaps)}개 (예: {[(round(a,1),round(b,1)) for a,b,_ in gaps[:3]]}) -> {saved}장 저장")
    print(f"[eight-ctx] 총 {total}장 -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
