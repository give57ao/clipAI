# -*- coding: utf-8 -*-
"""숫자 CNN 학습셋 구축 (R4 Task A) — 재군집화 대신 기존 검증 템플릿으로 고신뢰 자동 라벨.

배경: raw/ 글리프를 IoU 그리디 재군집화하면 경계에서 드리프트(0→6→8 연쇄 오염)가
실측 확인됨(c02 몽타주: 0 다수 + 6/8 혼입). eight/ 폴더(153장)도 6/8/9 혼재로
그대로 쓸 수 없음. 대신:
  1) 이미 사람이 검증·설치한 k_{d}.png 템플릿(0~7,9 — 8 제외)과의 IoU로
     raw/ 글리프를 자동 라벨. margin 엄격 적용(설치용 _K_IOU_MARGIN=0.06보다 훨씬
     엄격한 0.15) → 9개 클래스는 사람 검증 없이 고신뢰 데이터 확보.
  2) 어느 숫자에도 강하게 안 붙는("unclaimed") 글리프만 별도 그리디 재군집화
     (풀이 훨씬 작아 드리프트 위험 낮음) → 몽타주로 8/junk 육안 라벨.
  3) 최종 데이터셋(글리프 경로, 라벨, 소스 영상 stem) JSON 저장 → 학습 스크립트가 로드.

사용:
    python -u _build_digit_dataset.py --claim      # 1) 고신뢰 자동 라벨
    python -u _build_digit_dataset.py --cluster-unclaimed   # 2) unclaimed 재군집화
    (몽타주 확인 후 unclaimed_labels.json 작성: {"u00": 8, "u01": "junk", ...})
    python -u _build_digit_dataset.py --finalize    # 3) 최종 dataset.json 생성
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from hud_digit_match import DEFAULT_TEMPLATE_DIR, _GLYPH_SIZE

HARVEST_DIR = Path(r"E:\clipai_result\hud_templates_harvest")
RAW_DIR = HARVEST_DIR / "raw"
OUT_DIR = HARVEST_DIR / "cnn_dataset"

_CLAIM_MIN = 0.80          # 최고 IoU 최소치 (설치 기준 0.55보다 훨씬 엄격)
_CLAIM_MARGIN = 0.15       # 1등-2등(다른 숫자) IoU 차 — 설치 기준 0.06보다 훨씬 엄격
_UNCLAIMED_CLUSTER_SIM = 0.75


def _sim(a: np.ndarray, b: np.ndarray) -> float:
    ab = a > 127
    bb = b > 127
    union = np.logical_or(ab, bb).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(ab, bb).sum() / union)


def _load_templates() -> dict[int, list[np.ndarray]]:
    d = DEFAULT_TEMPLATE_DIR
    out: dict[int, list[np.ndarray]] = {}
    for p in sorted(d.glob("k_*.png")):
        key = p.stem.split("_", 1)[1]
        digit = int(key.split("_")[0])
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            out.setdefault(digit, []).append(img)
    return out


def claim() -> None:
    templates = _load_templates()
    print(f"[claim] 템플릿 로드: {sorted(templates.keys())}")
    files = sorted(RAW_DIR.glob("*.png"))
    print(f"[claim] raw 글리프 {len(files)}개")

    claimed: dict[int, list[str]] = {d: [] for d in templates}
    unclaimed: list[str] = []

    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (_GLYPH_SIZE[1], _GLYPH_SIZE[0]):
            continue
        by_digit: dict[int, float] = {}
        for digit, tmpls in templates.items():
            best = max(_sim(img, t) for t in tmpls)
            by_digit[digit] = best
        ranked = sorted(by_digit.items(), key=lambda kv: -kv[1])
        top_d, top_s = ranked[0]
        second_s = ranked[1][1] if len(ranked) > 1 else 0.0
        if top_s >= _CLAIM_MIN and (top_s - second_s) >= _CLAIM_MARGIN:
            claimed[top_d].append(p.name)
        else:
            unclaimed.append(p.name)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "claimed.json").write_text(
        json.dumps(claimed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "unclaimed.json").write_text(
        json.dumps(unclaimed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for d, names in sorted(claimed.items()):
        print(f"[claim] digit {d}: {len(names)}개 확보")
    print(f"[claim] unclaimed(8/junk 후보): {len(unclaimed)}개 -> unclaimed.json")


def cluster_unclaimed() -> None:
    unclaimed_path = OUT_DIR / "unclaimed.json"
    if not unclaimed_path.exists():
        print("[cluster-unclaimed] --claim 먼저 실행 필요")
        return
    names = json.loads(unclaimed_path.read_text(encoding="utf-8"))
    glyphs = []
    for name in names:
        img = cv2.imread(str(RAW_DIR / name), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            glyphs.append((name, img))
    print(f"[cluster-unclaimed] {len(glyphs)}개 재군집화 (임계 {_UNCLAIMED_CLUSTER_SIM})")

    reps: list[np.ndarray] = []
    members: list[list[str]] = []
    member_imgs: list[list[np.ndarray]] = []
    for name, g in glyphs:
        best_i, best_s = -1, 0.0
        for i, rep in enumerate(reps):
            s = _sim(g, rep)
            if s > best_s:
                best_i, best_s = i, s
        if best_i >= 0 and best_s >= _UNCLAIMED_CLUSTER_SIM:
            members[best_i].append(name)
            member_imgs[best_i].append(g)
        else:
            reps.append(g)
            members.append([name])
            member_imgs.append([g])

    order = sorted(range(len(reps)), key=lambda i: -len(members[i]))
    udir = OUT_DIR / "unclaimed_clusters"
    if udir.is_dir():
        shutil.rmtree(udir)
    udir.mkdir(parents=True)
    gw, gh = _GLYPH_SIZE
    cols = 12
    meta = {}
    for rank, i in enumerate(order):
        uid = f"u{rank:02d}"
        imgs = member_imgs[i]
        n = len(imgs)
        show = imgs[:96]
        rows_n = (len(show) + cols - 1) // cols
        canvas = np.zeros((rows_n * (gh + 2), cols * (gw + 2)), dtype=np.uint8)
        for j, g in enumerate(show):
            r, c = divmod(j, cols)
            canvas[r * (gh + 2): r * (gh + 2) + gh, c * (gw + 2): c * (gw + 2) + gw] = g
        big = cv2.resize(canvas, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(udir / f"{uid}_montage_n{n}.png"), big)
        meta[uid] = {"size": n, "members": members[i]}
        print(f"[cluster-unclaimed] {uid}: {n}개")
    (udir / "unclaimed_clusters.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[cluster-unclaimed] {len(order)}개 클러스터 -> {udir}")
    print('[cluster-unclaimed] 몽타주 확인 후 unclaimed_labels.json 작성: '
          '{"u00": 8, "u01": "junk", ...} (8 아니고 다른 숫자면 그 숫자로, 애매하면 "junk")')


def finalize() -> None:
    claimed = json.loads((OUT_DIR / "claimed.json").read_text(encoding="utf-8"))
    ulabels_path = OUT_DIR / "unclaimed_labels.json"
    if not ulabels_path.exists():
        print(f"[finalize] {ulabels_path} 없음 — cluster-unclaimed 후 라벨 작성 먼저")
        return
    ulabels = json.loads(ulabels_path.read_text(encoding="utf-8"))
    uclusters = json.loads(
        (OUT_DIR / "unclaimed_clusters" / "unclaimed_clusters.json").read_text(encoding="utf-8")
    )

    dataset: list[dict] = []
    for digit_s, names in claimed.items():
        digit = int(digit_s)
        for name in names:
            dataset.append({"file": name, "label": digit})

    n_eight = 0
    n_junk = 0
    for uid, lab in ulabels.items():
        members = uclusters.get(uid, {}).get("members", [])
        if lab == "junk" or lab is None:
            for name in members:
                dataset.append({"file": name, "label": "junk"})
            n_junk += len(members)
        elif isinstance(lab, int) and 0 <= lab <= 9:
            for name in members:
                dataset.append({"file": name, "label": lab})
            if lab == 8:
                n_eight += len(members)

    # 라벨 안 된(unclaimed_labels.json에 없는) 나머지 unclaimed 클러스터는 junk로 편입
    for uid, info in uclusters.items():
        if uid not in ulabels:
            for name in info["members"]:
                dataset.append({"file": name, "label": "junk"})
            n_junk += len(info["members"])

    by_label: dict = {}
    for row in dataset:
        by_label.setdefault(row["label"], 0)
        by_label[row["label"]] += 1

    (OUT_DIR / "dataset.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[finalize] 총 {len(dataset)}개 글리프 -> dataset.json")
    print(f"[finalize] 라벨 분포: {dict(sorted(by_label.items(), key=lambda kv: str(kv[0])))}")
    print(f"[finalize] 8: {n_eight}개, junk: {n_junk}개")
    if n_eight < 30:
        print(f"[finalize] ⚠ 8 표본이 {n_eight}개뿐 — 부족하면 unclaimed 몽타주에서 "
              f"8로 라벨된 클러스터가 더 있는지, 또는 harvest_hud_digits.py --harvest 로 "
              f"8 표시 구간(예: 79:51~79:53, 54:15~54:20)을 더 수확할지 검토")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claim", action="store_true")
    ap.add_argument("--cluster-unclaimed", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    did = False
    if args.claim:
        claim()
        did = True
    if args.cluster_unclaimed:
        cluster_unclaimed()
        did = True
    if args.finalize:
        finalize()
        did = True
    if not did:
        print("옵션 필요: --claim / --cluster-unclaimed / --finalize")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
