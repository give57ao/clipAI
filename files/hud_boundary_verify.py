# -*- coding: utf-8 -*-
"""row_miss 경계 후보를 전광판 CNN으로 스팟 검증 — 가짜 경계(배너/사망연출 등) 폐기.

SONNET_TASK.md R2 Task 1. 도메인 확인(사용자, 2026-07-07): 라운드 경계 = 전광판.
전광판이 떠 있는 동안만 K/D/A가 미노출되므로, row_miss run이 길어도 전광판이
안 보이면 그건 배너/사망연출 등에 의한 가짜 경계 — GT 올킬 한복판을 쪼개는 원인
(03-02-03 14:17·14:37, 02-34-09 3:04 실측).

각 row_miss run(≥ _BOUNDARY_ROWMISS)의 25/50/75% 지점 프레임 3장을 seek로 추출해
`scoreboard_clf_best.pt`(test acc 95.7%, detect_rounds.py 검증됨)로 분류. 3장 중
하나라도 scoreboard 확률 ≥ SCORE_THRESHOLD면 진짜 경계 유지, 아니면 폐기.
단, run이 LONG_RUN_FRAMES(10s) 이상이면 검증 없이 무조건 유지(확실한 비플레이 구간).

결과는 `{stem}.boundary.json`에 저장 → `timeline_from_reads(boundary_verdicts=...)`가
소비. 영상 재판독 없이(캐시의 reads로 run 위치만 계산 + seek 3장씩) 수 초~수십 초.

사용:
    python -u hud_boundary_verify.py                 # 캐시에 있는 전체 라벨 영상
    python -u hud_boundary_verify.py --videos "2026-03-22 03-02-03"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import torch
from torchvision import transforms

from detect_ace_hud import KRead, rowmiss_runs
from hud_sig_cache import DEFAULT_CACHE_DIR, METHOD_DECODE
from ml_train_common import build_model

SCOREBOARD_MODEL_PATH = Path(r"E:\Highlights\ml_dataset\models\scoreboard_clf_best.pt")
SCORE_THRESHOLD = 0.6
LONG_RUN_FRAMES = 40  # 10s @ 4fps — 확실한 비플레이 구간은 검증 없이 유지


def load_scoreboard_model(device: torch.device):
    ckpt = torch.load(SCOREBOARD_MODEL_PATH, map_location=device)
    model = build_model(len(ckpt["class_to_idx"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    img_size = int(ckpt.get("img_size", 224))
    return model, img_size


def build_eval_transform(img_size: int) -> transforms.Compose:
    # detect_rounds.py의 동일 함수와 일치 — 학습 시 전처리와 어긋나면 안 됨
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@torch.no_grad()
def classify_frame(frame_bgr, model, transform, device: torch.device) -> float:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = transform(rgb).unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0, 1].item()
    return float(prob)


def _load_cached_reads(cache_path: Path) -> tuple[list[KRead], str]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    reads = [
        KRead(t=t, k=k, conf=c, method=METHOD_DECODE.get(m, m))
        for t, k, c, m in data["reads"]
    ]
    return reads, data["stem"]


def verify_video(
    stem: str,
    cache_dir: Path,
    model,
    transform,
    device: torch.device,
    obs_dir: Path,
) -> None:
    cp = cache_dir / f"{stem}.json"
    if not cp.exists():
        print(f"[boundary] 캐시 없음: {stem}")
        return
    reads, _ = _load_cached_reads(cp)
    runs = rowmiss_runs(reads)
    if not runs:
        print(f"[boundary] {stem}: row_miss run 후보 없음")
        return

    vp = obs_dir / f"{stem}.mp4"
    if not vp.exists():
        print(f"[boundary] 영상 없음: {vp}")
        return
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        print(f"[boundary] 영상 열기 실패: {vp}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    verdicts: list[list] = []
    for (start, last, n) in runs:
        if n >= LONG_RUN_FRAMES:
            verdicts.append([start, last, True])
            continue
        best = 0.0
        for frac in (0.25, 0.5, 0.75):
            t = start + (last - start) * frac
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            prob = classify_frame(frame, model, transform, device)
            best = max(best, prob)
        verdicts.append([start, last, best >= SCORE_THRESHOLD])
    cap.release()

    out = cache_dir / f"{stem}.boundary.json"
    out.write_text(json.dumps({"runs": verdicts}, ensure_ascii=False, indent=2), encoding="utf-8")
    n_real = sum(1 for _, _, v in verdicts if v)

    def mss(s: float) -> str:
        return f"{int(s // 60)}:{s % 60:05.2f}"

    print(f"[boundary] {stem}: 후보 {len(verdicts)}개 중 진짜 {n_real} 가짜 {len(verdicts) - n_real}")
    for (s, e, _), (_, _, v) in zip(runs, verdicts):
        tag = "진짜" if v else "가짜(폐기)"
        print(f"    {mss(s)}-{mss(e)} -> {tag}")
    print(f"[boundary]   -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="row_miss 경계 후보 전광판 CNN 스팟 검증")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--obs-dir", default=r"E:\OBS")
    ap.add_argument("--videos", nargs="*", default=None, help="stem 지정 (기본: 캐시 전체)")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    obs_dir = Path(args.obs_dir)

    if args.videos:
        stems = args.videos
    else:
        stems = sorted(
            p.stem for p in cache_dir.glob("*.json") if not p.name.endswith(".boundary.json")
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[boundary] 모델 로드 중 (device={device})...", flush=True)
    model, img_size = load_scoreboard_model(device)
    transform = build_eval_transform(img_size)

    for stem in stems:
        verify_video(stem, cache_dir, model, transform, device, obs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
