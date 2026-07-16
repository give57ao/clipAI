# -*- coding: utf-8 -*-
"""한 영상의 실제 파이프라인 경계 후보(row_miss run) 전체와 CNN 판정을 덤프.

R5 CNN 경계검증기가 몇 개를 기각하는지, 기각이 연쇄로 몰리는지 실측 (2026-07-16).
collect_reads(전체 디코드)를 쓰므로 긴 영상은 수십 분 소요.

사용:
    python -u _audit_boundaries.py "2026-03-26 01-26-52"
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
from detect_ace_hud import collect_reads, rowmiss_runs  # noqa: E402
from hud_boundary_verify import get_boundary_verifier, verify_runs_live  # noqa: E402


def mss(sec: float) -> str:
    return f"{int(sec // 60)}:{sec % 60:05.2f}"


def main() -> int:
    stem = sys.argv[1]
    vp = Path(r"E:\OBS") / f"{stem}.mp4"
    print(f"# {stem}  디코드 시작...", flush=True)
    reads, duration, err = collect_reads(vp, scan_fps=4.0)
    print(f"# reads={len(reads)} duration={duration:.1f}s err={err}", flush=True)

    runs = rowmiss_runs(reads)
    print(f"# row_miss run(경계 후보) = {len(runs)}개", flush=True)

    model, transform, device = get_boundary_verifier()
    verdicts = verify_runs_live(vp, reads, model, transform, device)

    n_true = sum(1 for _, _, v in verdicts if v)
    n_false = len(verdicts) - n_true
    print(f"# CNN 판정: 진짜 {n_true} / 가짜(폐기) {n_false}")
    print()
    print(f"# {'구간':>22}  {'프레임수':>6}  판정")
    streak = 0
    max_streak = 0
    for (s, e, n), (_, _, v) in zip(runs, verdicts):
        tag = "진짜" if v else "★가짜(폐기)"
        auto = "  (LONG_RUN 자동인정)" if n >= 40 else ""
        print(f"{mss(s):>10}-{mss(e):<11}  {n:>6}  {tag}{auto}")
        if not v:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    print()
    print(f"# 연속 기각 최대 길이 = {max_streak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
