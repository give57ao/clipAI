# -*- coding: utf-8 -*-
"""HUD K/D/A 숫자 템플릿 수확 — 글리프 수집 → 클러스터링 → 육안 라벨 → 설치.

OCR 부트스트랩은 단일 글리프에서 신뢰 불가(실측) → 비지도 클러스터링으로 대체.
폰트가 일정하므로 같은 숫자는 한 클러스터로 뭉침. 몽타주를 보고 라벨만 붙이면 됨.

흐름:
  1. `--harvest`          영상 샘플링 → out/raw/ 글리프 png 수집
  2. `--cluster`          out/clusters/cNN_montage.png + medoid 생성
  3. (사람/에이전트) 몽타주 확인 → out/cluster_labels.json 작성
       예: {"c00": 0, "c01": 1, "c02": "slash", "c03": 3, "c05": "noise"}
       숫자 0~9 외 값("slash","noise" 등)은 무시됨
  4. `--install`          라벨된 클러스터 medoid → hud_templates/k_{d}.png 설치

사용:
    python -u harvest_hud_digits.py --harvest --fps 0.3
    python -u harvest_hud_digits.py --cluster
    python -u harvest_hud_digits.py --install
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np

from game_frame import extract_game_crop_bgr
from hud_digit_match import DEFAULT_TEMPLATE_DIR, _GLYPH_SIZE, normalize_glyph
from hud_kda import locate_kda_glyphs

DEFAULT_OUT = Path(r"E:\clipai_result\hud_templates_harvest")

# 라벨 10영상 (HUD_ACE_HANDOFF.md §3) + 캘리브 영상
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
    r"D:\2026-01-08 02-33-22.mp4",
]

_PER_VIDEO_CAP = 400      # 영상당 글리프 상한
_DEDUP_SIM = 0.92         # 직전 저장 글리프와 IoU 이 이상이면 skip
_CLUSTER_SIM = 0.75       # 클러스터 합류 최소 IoU — 0.62는 0/5/6/8이 연쇄 병합됨(실측)
_MONTAGE_COLS = 12


def _sim(a: np.ndarray, b: np.ndarray) -> float:
    """이진 글리프 IoU — TM_CCOEFF는 이진 마스크에서 변별력 부족(전부 한 클러스터)."""
    ab = a > 127
    bb = b > 127
    union = np.logical_or(ab, bb).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(ab, bb).sum() / union)


def harvest(videos: list[Path], out: Path, fps: float) -> None:
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    total = 0
    for vp in videos:
        vp = Path(vp)
        if not vp.exists():
            print(f"[harvest] 없음: {vp}")
            continue
        cap = cv2.VideoCapture(str(vp))
        vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(vfps / fps)))
        saved = 0
        rows = 0
        last: dict[str, np.ndarray] = {}
        t0 = time.time()
        frame_idx = 0
        while saved < _PER_VIDEO_CAP:
            if frame_idx % step == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                t = frame_idx / vfps
                game, _ = extract_game_crop_bgr(frame)
                g = locate_kda_glyphs(game)
                if g is not None:
                    rows += 1
                    for slot, patches in (("k", g.k), ("d", g.d), ("a", g.a)):
                        for gi, patch in enumerate(patches):
                            glyph = normalize_glyph(patch)
                            if glyph is None:
                                continue
                            key = f"{slot}{gi}"
                            if key in last and _sim(glyph, last[key]) >= _DEDUP_SIM:
                                continue
                            last[key] = glyph
                            name = f"{vp.stem}_{int(t)}s_{key}.png"
                            cv2.imwrite(str(raw / name), glyph)
                            saved += 1
            else:
                if not cap.grab():
                    break
            frame_idx += 1
        cap.release()
        total += saved
        print(f"[harvest] {vp.stem}: rows={rows} saved={saved} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[harvest] 총 {total}개 -> {raw}")


def cluster(out: Path) -> None:
    raw = out / "raw"
    files = sorted(raw.glob("*.png"))
    if not files:
        print("[cluster] raw 글리프 없음 — --harvest 먼저")
        return
    glyphs: list[tuple[str, np.ndarray]] = []
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None and img.shape == (_GLYPH_SIZE[1], _GLYPH_SIZE[0]):
            glyphs.append((p.name, img))
    print(f"[cluster] 글리프 {len(glyphs)}개")

    # 그리디 클러스터링: 기존 클러스터 대표(초기 멤버)와 유사도 >= 임계 → 합류
    reps: list[np.ndarray] = []
    members: list[list[str]] = []
    member_imgs: list[list[np.ndarray]] = []
    for name, g in glyphs:
        best_i, best_s = -1, 0.0
        for i, rep in enumerate(reps):
            s = _sim(g, rep)
            if s > best_s:
                best_i, best_s = i, s
        if best_i >= 0 and best_s >= _CLUSTER_SIM:
            members[best_i].append(name)
            member_imgs[best_i].append(g)
        else:
            reps.append(g)
            members.append([name])
            member_imgs.append([g])

    order = sorted(range(len(reps)), key=lambda i: -len(members[i]))
    cdir = out / "clusters"
    if cdir.is_dir():
        shutil.rmtree(cdir)
    cdir.mkdir(parents=True)
    meta = {}
    gw, gh = _GLYPH_SIZE
    for rank, i in enumerate(order):
        cid = f"c{rank:02d}"
        imgs = member_imgs[i]
        n = len(imgs)
        # medoid (상한 60개로 O(n^2) 제한)
        sub = imgs[:60]
        m = len(sub)
        simm = np.zeros((m, m), dtype=np.float32)
        for a in range(m):
            for b in range(a + 1, m):
                s = _sim(sub[a], sub[b])
                simm[a, b] = simm[b, a] = s
        medoid = int(simm.mean(axis=1).argmax()) if m > 1 else 0
        cv2.imwrite(str(cdir / f"{cid}_medoid.png"), sub[medoid])

        show = imgs[:96]
        rows_n = (len(show) + _MONTAGE_COLS - 1) // _MONTAGE_COLS
        canvas = np.zeros((rows_n * (gh + 2), _MONTAGE_COLS * (gw + 2)), dtype=np.uint8)
        for j, g in enumerate(show):
            r, c = divmod(j, _MONTAGE_COLS)
            canvas[r * (gh + 2) : r * (gh + 2) + gh, c * (gw + 2) : c * (gw + 2) + gw] = g
        big = cv2.resize(canvas, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(cdir / f"{cid}_montage_n{n}.png"), big)
        meta[cid] = {"size": n, "sample": members[i][0]}
        print(f"[cluster] {cid}: {n}개  (예: {members[i][0]})")
    (cdir / "clusters.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[cluster] {len(order)}개 클러스터 -> {cdir}")
    print("[cluster] 몽타주 확인 후 out/cluster_labels.json 작성 (예: {\"c00\": 3, \"c01\": \"slash\"})")


def install(out: Path) -> None:
    labels_path = out / "cluster_labels.json"
    if not labels_path.exists():
        print(f"[install] {labels_path} 없음 — 몽타주 보고 라벨 먼저 작성")
        return
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    cdir = out / "clusters"

    # 숫자별 상위 3개 클러스터 medoid를 변형 템플릿으로 설치 (k_3.png, k_3_b.png, k_3_c.png)
    # — 다중 템플릿이면 안티앨리어싱 변형에도 진짜 숫자의 IoU가 올라가 마진 통과율 상승
    meta = json.loads((cdir / "clusters.json").read_text(encoding="utf-8"))
    by_digit: dict[int, list[tuple[str, int]]] = {}
    for cid, lab in labels.items():
        if not isinstance(lab, int) or not (0 <= lab <= 9):
            continue
        by_digit.setdefault(lab, []).append((cid, meta.get(cid, {}).get("size", 0)))

    tdir = DEFAULT_TEMPLATE_DIR
    if tdir.is_dir():
        backup = tdir.parent / f"hud_templates_backup_{time.strftime('%m%d_%H%M')}"
        shutil.copytree(tdir, backup, dirs_exist_ok=True)
        print(f"[install] 백업 -> {backup}")
        for p in tdir.glob("k_*.png"):
            p.unlink()
    tdir.mkdir(parents=True, exist_ok=True)
    for digit, cands in sorted(by_digit.items()):
        cands.sort(key=lambda cs: -cs[1])
        for i, (cid, size) in enumerate(cands[:3]):
            suffix = "" if i == 0 else f"_{'bc'[i - 1]}"
            shutil.copy2(cdir / f"{cid}_medoid.png", tdir / f"k_{digit}{suffix}.png")
            print(f"[install] k_{digit}{suffix}.png <- {cid} (n={size})")
    missing = [d for d in range(10) if d not in by_digit]
    if missing:
        print(f"[install] ⚠ 미확보 숫자: {missing} — 해당 K값 판독 불가")
    from hud_digit_match import reset_matcher
    reset_matcher()


def main() -> int:
    ap = argparse.ArgumentParser(description="HUD 숫자 템플릿 수확·클러스터·설치")
    ap.add_argument("--harvest", action="store_true")
    ap.add_argument("--cluster", action="store_true")
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--fps", type=float, default=0.3)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out = Path(args.out)
    did = False
    if args.harvest:
        videos = [Path(v) for v in (args.videos or DEFAULT_VIDEOS)]
        harvest(videos, out, args.fps)
        did = True
    if args.cluster:
        cluster(out)
        did = True
    if args.install:
        install(out)
        did = True
    if not did:
        print("옵션 필요: --harvest / --cluster / --install (조합 가능)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
