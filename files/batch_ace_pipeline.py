# -*- coding: utf-8 -*-
"""OBS 폴더 전체 배치: detect_rounds → scoreboard_k_reader → extract_ace_clips.

각 영상마다 라운드 분할 → 본인 K 읽기 → 올킬 클립 추출을 순차 실행한다.
GPU(CNN + EasyOCR)를 공유하므로 영상 단위는 순차(병렬 X)로 돈다.
이미 산출물이 있으면 건너뛴다(resume 가능).

사용:
    python -u batch_ace_pipeline.py                 # OBS 전체
    python -u batch_ace_pipeline.py --limit 1        # 앞 1개만 (검증용)
    python -u batch_ace_pipeline.py --only "2026-03-21 00-40-56"
    python -u batch_ace_pipeline.py --redo           # 기존 산출물 무시하고 재실행
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from video_utils import probe_duration_sec

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

OBS_DIR = Path(r"E:\OBS")
DATASET_ROOT = Path(r"E:\Highlights\ml_dataset")
ROUNDS_ROOT = DATASET_ROOT / "rounds"
KT_DIR = Path(r"E:\clipai_result\kill_timeline")
CLIPS_DIR = Path(r"E:\clipai_result\ace_clips")
LOG_DIR = Path(r"E:\clipai_result\batch_logs")
HERE = Path(__file__).resolve().parent


def run(cmd: list[str], log_handle) -> int:
    """서브프로세스 실행, stdout/err를 로그에 append."""
    log_handle.write(f"\n$ {' '.join(cmd)}\n")
    log_handle.flush()
    proc = subprocess.run(cmd, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def process_video(mp4: Path, *, redo: bool, kreader_only: bool = False,
                  min_duration_sec: float = 0.0) -> dict:
    stem = mp4.stem
    rounds_dir = ROUNDS_ROOT / stem
    sb_csv = rounds_dir / "detected_scoreboards.csv"
    json_out = KT_DIR / f"{stem}.json"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{stem}.log"

    result = {"stem": stem, "ok": False, "aces": [], "k_errors": [], "error": None}
    t0 = time.time()

    # 0) 너무 짧은 영상(클립 조각 등)은 클랜매치가 아님 → 스킵
    if min_duration_sec > 0:
        dur = probe_duration_sec(mp4) or 0.0
        if 0 < dur < min_duration_sec:
            result["error"] = f"too_short({dur:.0f}s<{min_duration_sec:.0f}s)"
            return result

    # kreader_only인데 SB가 아직 없으면(=미분석 영상) detect로 시간 낭비하지 않고 스킵
    if kreader_only and not sb_csv.exists():
        result["error"] = "no_sb_skip(kreader_only)"
        return result

    with log_path.open("w", encoding="utf-8") as log:
        # 1) detect_rounds (kreader_only면 기존 SB 재사용)
        if (redo and not kreader_only) or not sb_csv.exists():
            rc = run(
                [sys.executable, "-u", str(HERE / "detect_rounds.py"), str(mp4),
                 "--dataset-root", str(DATASET_ROOT), "--no-hud"],
                log,
            )
            if rc != 0 or not sb_csv.exists():
                result["error"] = "detect_rounds_failed"
                return result

        # 2) k-reader
        if redo or kreader_only or not json_out.exists():
            rc = run(
                [sys.executable, "-u", str(HERE / "scoreboard_k_reader.py"), str(mp4),
                 "--rounds-dir", str(rounds_dir),
                 "--dataset-root", str(DATASET_ROOT),
                 "--out", str(KT_DIR / f"{stem}.csv"),
                 "--json-out", str(json_out)],
                log,
            )
            if rc != 0 or not json_out.exists():
                result["error"] = f"k_reader_rc{rc}"
                return result

        # 3) clips
        run(
            [sys.executable, "-u", str(HERE / "extract_ace_clips.py"), str(json_out),
             "--output-dir", str(CLIPS_DIR)],
            log,
        )

    try:
        data = json.loads(json_out.read_text(encoding="utf-8"))
        result["aces"] = data.get("ace_rounds", [])
        result["k_errors"] = data.get("k_error_rounds", [])
        result["nick"] = data.get("player_nick", "")
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"json_read:{exc}"

    result["sec"] = round(time.time() - t0, 1)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="OBS 전체 올킬 파이프라인 배치")
    ap.add_argument("--limit", type=int, default=0, help="앞 N개만 (0=전체)")
    ap.add_argument("--only", default=None, help="특정 stem만")
    ap.add_argument("--after", default=None, help="특정 stem 이후부터 처리 (정렬 기준)")
    ap.add_argument("--redo", action="store_true", help="기존 산출물 무시 재실행")
    ap.add_argument("--kreader-only", action="store_true",
                    help="detect_rounds(SB) 재사용, k-reader+클립만 재실행 (닉 ROI 변경 반영용)")
    ap.add_argument("--min-duration-sec", type=float, default=120.0,
                    help="이보다 짧은 영상은 클랜매치 아님으로 스킵 (기본 120초)")
    args = ap.parse_args()

    videos = sorted(OBS_DIR.glob("*.mp4"))
    if args.only:
        videos = [p for p in videos if p.stem == args.only]
    if args.after:
        # 정렬된 stem 기준으로 after(배타) 이후만 남김
        videos = [p for p in videos if p.stem > args.after]
    if args.limit > 0:
        videos = videos[: args.limit]

    print(f"[batch] 대상 {len(videos)}개  (redo={args.redo})", flush=True)
    KT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for i, mp4 in enumerate(videos, 1):
        print(f"\n[batch] ({i}/{len(videos)}) {mp4.name} 처리 중...", flush=True)
        r = process_video(mp4, redo=args.redo, kreader_only=args.kreader_only,
                          min_duration_sec=args.min_duration_sec)
        summary.append(r)
        if r["ok"]:
            print(
                f"[batch] ✓ {mp4.stem}  nick={r.get('nick','')!r} "
                f"ace={r['aces']} k_err={r['k_errors']} ({r.get('sec','?')}s)",
                flush=True,
            )
        else:
            print(f"[batch] ✗ {mp4.stem}  ERROR={r['error']}", flush=True)

    # 최종 요약
    print("\n" + "=" * 60, flush=True)
    ok = [r for r in summary if r["ok"]]
    total_aces = sum(len(r["aces"]) for r in ok)
    print(f"[batch] 완료 {len(ok)}/{len(summary)}  총 올킬 {total_aces}개", flush=True)
    for r in ok:
        if r["aces"]:
            print(f"  {r['stem']}  nick={r.get('nick','')!r}  ace={r['aces']}", flush=True)
    fails = [r for r in summary if not r["ok"]]
    if fails:
        print(f"[batch] 실패 {len(fails)}:", flush=True)
        for r in fails:
            print(f"  {r['stem']}: {r['error']}", flush=True)

    # 요약 JSON 저장
    out = Path(r"E:\clipai_result\batch_summary.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[batch] 요약 -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
