# -*- coding: utf-8
"""OBS 전체 HUD 올킬 배치 — detect_ace_hud (닉·SB 불필요).

사용:
    python -u batch_hud_ace_pipeline.py
    python -u batch_hud_ace_pipeline.py --limit 5
    python -u batch_hud_ace_pipeline.py --only "2026-03-21 00-40-56"
    python -u batch_hud_ace_pipeline.py --redo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from detect_ace_hud import (
    HudAceTimeline,
    RoundTrack,
    extract_ace_clips,
    format_report,
    scan_hud_aces,
)
from video_utils import probe_duration_sec

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

OBS_DIR = Path(r"E:\OBS")
DATASET_ROOT = Path(r"E:\Highlights\ml_dataset")
HUD_JSON_DIR = Path(r"E:\clipai_result\hud_timeline")
HUD_CLIPS_DIR = Path(r"E:\clipai_result\ace_clips_hud")
SUMMARY_PATH = Path(r"E:\clipai_result\batch_hud_summary.json")


def _load_timeline(json_path: Path) -> HudAceTimeline:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    rounds = [RoundTrack(**r) for r in data.get("rounds", [])]
    return HudAceTimeline(
        video_path=data["video_path"],
        scan_fps=data.get("scan_fps", 4.0),
        ace_kills=data.get("ace_kills", 3),
        rounds=rounds,
        ace_rounds=data.get("ace_rounds", []),
        hud_end_count=data.get("hud_end_count", 0),
        k_template_hits=data.get("k_template_hits", 0),
        k_template_miss=data.get("k_template_miss", 0),
        warnings=data.get("warnings", []),
    )


def process_video(
    mp4: Path,
    *,
    redo: bool,
    scan_fps: float,
    min_duration_sec: float,
    extract: bool,
    json_dir: Path = HUD_JSON_DIR,
    clips_dir: Path = HUD_CLIPS_DIR,
    verify_boundary_wins: bool = False,
) -> dict:
    stem = mp4.stem
    json_out = json_dir / f"{stem}.json"
    result: dict = {"stem": stem, "ok": False, "aces": [], "error": None, "clips": 0}

    if min_duration_sec > 0:
        dur = probe_duration_sec(mp4) or 0.0
        if 0 < dur < min_duration_sec:
            result["error"] = f"too_short({dur:.0f}s<{min_duration_sec:.0f}s)"
            return result

    t0 = time.time()
    if not redo and json_out.exists():
        try:
            tl = _load_timeline(json_out)
            result["aces"] = tl.ace_rounds
            result["ok"] = True
            result["cached"] = True
            if extract and tl.ace_rounds:
                clips = extract_ace_clips(mp4, tl, clips_dir)
                result["clips"] = len(clips)
            result["sec"] = round(time.time() - t0, 1)
            return result
        except Exception:
            pass

    try:
        tl = scan_hud_aces(
            mp4, scan_fps=scan_fps, dataset_root=DATASET_ROOT,
            verify_boundary_wins=verify_boundary_wins,
        )
        json_dir.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(asdict(tl), ensure_ascii=False, indent=2), encoding="utf-8")
        print(format_report(tl), flush=True)
        result["aces"] = tl.ace_rounds
        result["ok"] = True
        if extract and tl.ace_rounds:
            clips = extract_ace_clips(mp4, tl, clips_dir)
            result["clips"] = len(clips)
    except Exception as exc:
        result["error"] = str(exc)
        return result

    result["sec"] = round(time.time() - t0, 1)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="OBS HUD 올킬 배치")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default=None)
    ap.add_argument("--after", default=None)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--scan-fps", type=float, default=4.0)
    ap.add_argument("--min-duration-sec", type=float, default=120.0)
    ap.add_argument("--no-extract", action="store_true", help="JSON만, 클립 추출 생략")
    ap.add_argument("--obs-dir", default=str(OBS_DIR))
    ap.add_argument(
        "--output-root",
        default=None,
        help="산출물 루트 재지정 (예: D:\\hud_result) — 미지정시 E:\\clipai_result 기존 경로",
    )
    ap.add_argument(
        "--verify-boundary-wins",
        action="store_true",
        help="R10 승수 교차검증 경계 게이트 활성화 (기본 꺼짐 — detect_ace_hud.py R10 주석 참고)",
    )
    args = ap.parse_args()

    if args.output_root:
        out_root = Path(args.output_root)
        json_dir = out_root / "hud_timeline"
        clips_dir = out_root / "ace_clips_hud"
        summary_path = out_root / "batch_hud_summary.json"
    else:
        json_dir = HUD_JSON_DIR
        clips_dir = HUD_CLIPS_DIR
        summary_path = SUMMARY_PATH

    obs = Path(args.obs_dir)
    videos = sorted(obs.glob("*.mp4"))
    if args.only:
        videos = [p for p in videos if p.stem == args.only]
    if args.after:
        videos = [p for p in videos if p.stem > args.after]
    if args.limit > 0:
        videos = videos[: args.limit]

    print(f"[hud-batch] 대상 {len(videos)}개  redo={args.redo}", flush=True)
    summary = []
    for i, mp4 in enumerate(videos, 1):
        print(f"\n[hud-batch] ({i}/{len(videos)}) {mp4.name}", flush=True)
        r = process_video(
            mp4,
            redo=args.redo,
            scan_fps=args.scan_fps,
            min_duration_sec=args.min_duration_sec,
            extract=not args.no_extract,
            json_dir=json_dir,
            clips_dir=clips_dir,
            verify_boundary_wins=args.verify_boundary_wins,
        )
        summary.append(r)
        if r["ok"]:
            cached = " (cached)" if r.get("cached") else ""
            print(
                f"[hud-batch] OK {mp4.stem} ace={r['aces']} clips={r.get('clips',0)}{cached} "
                f"({r.get('sec','?')}s)",
                flush=True,
            )
        else:
            print(f"[hud-batch] SKIP {mp4.stem}: {r['error']}", flush=True)

    ok = [r for r in summary if r["ok"]]
    total_aces = sum(len(r["aces"]) for r in ok)
    print(f"\n[hud-batch] 완료 {len(ok)}/{len(summary)}  총 올킬 {total_aces}개", flush=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[hud-batch] 요약 -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
