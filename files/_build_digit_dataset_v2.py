# -*- coding: utf-8 -*-
"""숫자 CNN 학습셋 v2 — 체인-감독(chain-supervised) 자동 라벨 하베스터.

R4 실패의 근본 원인(hud_kda.py _read_digit_group 주석): v1 데이터셋이 "템플릿
IoU로 라벨을 딴" 순환 구조라 (a) 8 표본이 36장뿐이고 (b) 템플릿이 실패하는
프레임(모션블러·전이)의 정답 라벨이 전혀 없었음 → CNN이 블러/0을 고신뢰 8로
오분류해도 학습에서 교정될 기회가 없었다.

v2는 라벨을 템플릿이 아니라 **신호 캐시의 시간적 문맥**에서 딴다:

  1) claimed  : 안정 런(같은 K값 연속 ≥_RUN_MIN_HITS회) 내부의 판독 성공 프레임
                → 런 값의 자리수 분해로 글리프별 라벨. (0~7,9 대량)
  2) eight    : ① 격리된 가짜 0 — _quarantine_zeros가 "직전 양수 K가 리셋 없이
                이어짐"으로 제거한 0 판독 = **화면의 진짜 8** (라벨 확실성이 체인
                문맥에서 나옴 — R4가 가장 목말랐던 표본).
                ② 위상 프로브가 읽은 k==8 프레임 (R8, 안정 노출 포착률 ~98%).
  3) hardneg_v: 안정 런 **내부**의 template_miss(M) 프레임 — 화면엔 v가 떠
                있는데 템플릿이 못 읽은 프레임. 진짜 하드네거티브(블러 상태의
                0/6/9 등). 별도 폴더에 저장 — 학습 전 몽타주 QC 권장.
  4) junk     : 런 밖(전이 구간)의 template_miss — 희소 샘플링.
  5) D 슬롯   : 6열 캐시(d 채널)의 안정 D-런(≥_RUN_MIN_HITS) → D 글리프 라벨.
                D-가드 판독률 개선용 (흰 숫자라 K와 글꼴 다름 — 클래스 공유하되
                파일명에 슬롯 표기, 학습 시 분리/통합은 스크립트 옵션).

출력: E:\\clipai_result\\hud_templates_harvest\\cnn_dataset_v2\\<label>\\
      파일명 "{stem}_{t초}s_{슬롯}{자리}.png" — train_hud_digit_cnn.py의
      stem 기반 val 분리(_STEM_RE)와 호환.

사용:
    python -u _build_digit_dataset_v2.py --stems "<stem>" ...   # 지정 영상만
    python -u _build_digit_dataset_v2.py                        # 원본 있는 캐시 전부
    python -u _build_digit_dataset_v2.py --selftest             # 라벨링 로직 검증
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))

SIG_DIR = Path(r"E:\clipai_result\sig_cache_v2")
OBS_DIR = Path(r"E:\OBS")
OUT_DIR = Path(r"E:\clipai_result\hud_templates_harvest\cnn_dataset_v2")

_RUN_MIN_HITS = 4      # 안정 런 최소 판독 성공 수
_RUN_MAX_GAP = 3.0     # 런 내 인접 성공 판독 최대 간격(초)
_HARDNEG_PAD = 0.75    # 런 경계에서 이만큼 안쪽의 miss만 hardneg (경계 전이 오염 방지)
_CAP_PER_CLASS = 80    # 영상당·클래스당 저장 상한
_MIN_T_GAP = 0.5       # 같은 클래스 연속 샘플 최소 시간 간격


# ---------------------------------------------------------------------------
# 라벨링 로직 (순수 함수 — 영상 접근 없음, selftest 대상)
# ---------------------------------------------------------------------------

def stable_runs(reads: list[tuple[float, int | None]]) -> list[tuple[float, float, int]]:
    """판독 성공 시퀀스에서 안정 런 [(t0, t1, v)] 추출.

    같은 값이 _RUN_MAX_GAP 이내 간격으로 ≥_RUN_MIN_HITS회 연속. 런 도중 다른
    값의 성공 판독이 끼면 런 종료 (오염 방지 — 값이 바뀌는 킬 순간 주변 배제).
    """
    runs: list[tuple[float, float, int]] = []
    cur_v: int | None = None
    t0 = t1 = 0.0
    n = 0

    def flush() -> None:
        if cur_v is not None and n >= _RUN_MIN_HITS:
            runs.append((t0, t1, cur_v))

    for t, k in reads:
        if k is None:
            continue  # miss는 런을 끊지 않음 (그 사이가 hardneg 후보)
        if k == cur_v and t - t1 <= _RUN_MAX_GAP:
            t1 = t
            n += 1
        else:
            flush()
            cur_v, t0, t1, n = k, t, t, 1
    flush()
    return runs


def quarantined_zero_times(reads: list[tuple[float, int | None]],
                           lookahead: float = 120.0) -> list[float]:
    """hud_round_settle._quarantine_zeros와 동일 규칙으로 '가짜 0(=화면의 8)'
    판독의 시각만 골라낸다. (직전 양수 K 존재 + 직후 첫 양수 K ≥ 직전값)"""
    ok = [(t, k) for t, k in reads if k is not None]
    out: list[float] = []
    for i, (t, k) in enumerate(ok):
        if k != 0:
            continue
        prev_nz = None
        for j in range(i - 1, -1, -1):
            t2, k2 = ok[j]
            if t - t2 > lookahead:
                break
            if k2 > 0:
                prev_nz = k2
                break
        if prev_nz is None:
            continue
        next_nz = None
        for j in range(i + 1, len(ok)):
            t2, k2 = ok[j]
            if t2 - t > lookahead:
                break
            if k2 > 0:
                next_nz = k2
                break
        if next_nz is not None and next_nz >= prev_nz:
            out.append(t)
    return out


def hardneg_times(reads: list[tuple[float, int | None, str]],
                  runs: list[tuple[float, float, int]]) -> list[tuple[float, int]]:
    """안정 런 내부(_HARDNEG_PAD 안쪽)의 template_miss(M) → (t, 정답값)."""
    out: list[tuple[float, int]] = []
    for t, k, m in reads:
        if k is not None or m != "M":
            continue
        for r0, r1, v in runs:
            if r0 + _HARDNEG_PAD <= t <= r1 - _HARDNEG_PAD:
                out.append((t, v))
                break
    return out


def plan_labels(cache_reads: list[list]) -> dict[str, list]:
    """캐시 reads(4열/6열) → 슬롯별 (t, 라벨) 수확 계획.

    반환: {"k_claimed": [(t, v)], "k_eight": [t], "k_hardneg": [(t, v)],
           "k_junk": [t], "d_claimed": [(t, v)]}
    """
    k_reads = [(r[0], r[1]) for r in cache_reads]
    k_rm = [(r[0], r[1], r[3]) for r in cache_reads]
    runs = stable_runs(k_reads)

    claimed = []
    for r0, r1, v in runs:
        claimed.extend((t, v) for t, k in k_reads
                       if k == v and r0 <= t <= r1)

    eight = quarantined_zero_times(k_reads)
    eight += [t for t, k in k_reads if k == 8]

    hardneg = hardneg_times(k_rm, runs)

    in_run = lambda t: any(r0 <= t <= r1 for r0, r1, _v in runs)  # noqa: E731
    junk = [t for t, k, m in k_rm if k is None and m == "M" and not in_run(t)]

    d_claimed = []
    if cache_reads and len(cache_reads[0]) > 4:
        d_reads = [(r[0], r[4]) for r in cache_reads if r[4] is not None]
        for r0, r1, v in stable_runs(d_reads):
            d_claimed.extend((t, v) for t, d in d_reads if d == v and r0 <= t <= r1)

    return {"k_claimed": claimed, "k_eight": eight, "k_hardneg": hardneg,
            "k_junk": junk, "d_claimed": d_claimed}


def _thin(items: list, key=lambda x: x, cap: int = _CAP_PER_CLASS) -> list:
    """시간순 정렬 후 _MIN_T_GAP 간격 강제 + cap (균등 솎아내기)."""
    items = sorted(items, key=key)
    kept = []
    last = -1e9
    for it in items:
        t = key(it)
        if t - last >= _MIN_T_GAP:
            kept.append(it)
            last = t
    if len(kept) > cap:
        step = len(kept) / cap
        kept = [kept[int(i * step)] for i in range(cap)]
    return kept


# ---------------------------------------------------------------------------
# 수확 실행 (영상 seek — Sonnet 배치 실행 파트)
# ---------------------------------------------------------------------------

def harvest_stem(stem: str) -> dict[str, int]:
    import cv2

    from game_frame import extract_game_crop_bgr
    from hud_kda import locate_kda_glyphs, normalize_glyph

    cache_p = SIG_DIR / f"{stem}.json"
    video_p = OBS_DIR / f"{stem}.mp4"
    if not cache_p.exists() or not video_p.exists():
        print(f"[skip] 캐시/원본 없음: {stem}")
        return {}

    data = json.loads(cache_p.read_text(encoding="utf-8"))
    plan = plan_labels(data["reads"])

    # (t, 슬롯, 라벨) 작업 목록 — 클래스별 솎아내기
    jobs: list[tuple[float, str, object]] = []
    for t, v in _thin(plan["k_claimed"], key=lambda x: x[0]):
        jobs.append((t, "k", v))
    for t in _thin(plan["k_eight"], key=lambda x: x):
        jobs.append((t, "k", 8))
    for t, v in _thin(plan["k_hardneg"], key=lambda x: x[0]):
        jobs.append((t, "k", ("hardneg", v)))
    for t in _thin(plan["k_junk"], key=lambda x: x, cap=30):
        jobs.append((t, "k", "junk"))
    for t, v in _thin(plan["d_claimed"], key=lambda x: x[0]):
        jobs.append((t, "d", v))
    jobs.sort(key=lambda j: j[0])

    cap = cv2.VideoCapture(str(video_p))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    saved: dict[str, int] = defaultdict(int)
    for t, slot, label in jobs:
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
        patches = glyphs.k if slot == "k" else glyphs.d

        if isinstance(label, tuple):          # hardneg
            digits, folder_of = None, lambda d: f"hardneg_{d}"
            value = label[1]
        elif label == "junk":
            digits, folder_of, value = None, lambda _d: "junk", None
        else:
            digits, folder_of, value = None, str, label

        if value is not None:
            ds = [int(c) for c in str(value)]
            if len(ds) != len(patches):
                continue  # 글리프 수 ≠ 자리수 — 로케이터 불일치, 폐기
            digits = ds

        for gi, p in enumerate(patches):
            g = normalize_glyph(p)
            if g is None:
                continue
            folder = folder_of(digits[gi]) if digits is not None else folder_of(None)
            d_out = OUT_DIR / folder
            d_out.mkdir(parents=True, exist_ok=True)
            name = f"{stem}_{int(t)}s_{slot}{gi}.png"
            cv2.imwrite(str(d_out / name), g)
            saved[folder] += 1
            if digits is None:
                break  # junk/실패 프레임은 첫 글리프만
    cap.release()
    print(f"[harvest] {stem}: " + ", ".join(f"{k}={v}" for k, v in sorted(saved.items())))
    return dict(saved)


def _selftest() -> None:
    # 안정 런: 0×5 → (miss) → 0×2 는 gap 이내면 한 런
    reads = [(i * 1.0, 0) for i in range(5)] + [(5.0, None), (6.0, 0), (7.0, 0)]
    rs = stable_runs([(t, k) for t, k, *_ in [(a, b, None) for a, b in reads]])
    assert rs == [(0.0, 7.0, 0)], rs
    # 값 변화는 런을 끊음
    rs = stable_runs([(0, 3), (1, 3), (2, 3), (3, 3), (4, 4), (5, 4), (6, 4), (7, 4)])
    assert rs == [(0, 3, 3), (4, 7, 4)], rs
    # 가짜 0: 7 안정 → 0 → 10 (리셋 없이 상승) → 그 0은 8
    reads2 = [(0, 7), (1, 7), (2, 0), (3, 10)]
    assert quarantined_zero_times(reads2) == [2], quarantined_zero_times(reads2)
    # 진짜 리셋(하프타임): 7 → 0 → 1 (더 작게 재시작) → 격리 안 됨
    reads3 = [(0, 7), (1, 7), (2, 0), (3, 1)]
    assert quarantined_zero_times(reads3) == [], quarantined_zero_times(reads3)
    # hardneg: 런(0~5, v=6) 내부의 M 프레임만, 패드 밖 제외
    krm = [(0.0, 6, "T"), (1.0, 6, "T"), (2.0, None, "M"), (3.0, 6, "T"),
           (4.9, None, "M"), (5.0, 6, "T"), (5.1, None, "M")]
    runs = stable_runs([(t, k) for t, k, _ in krm])
    hn = hardneg_times(krm, runs)
    assert hn == [(2.0, 6)], hn  # 4.9·5.1은 런 끝(5.0)-패드(0.75) 밖 → 제외
    print("dataset_v2 labeling selftest: OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return 0
    stems = args.stems or sorted(
        p.stem for p in SIG_DIR.glob("*.json")
        if not p.name.endswith(".boundary.json") and (OBS_DIR / f"{p.stem}.mp4").exists()
    )
    total: dict[str, int] = defaultdict(int)
    for s in stems:
        for k, v in harvest_stem(s).items():
            total[k] += v
    print("\n[harvest] 합계: " + ", ".join(f"{k}={v}" for k, v in sorted(total.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
