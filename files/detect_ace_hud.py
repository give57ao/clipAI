# -*- coding: utf-8
"""HUD K/D/A 기반 올킬 탐지 — 단일 패스 + ΔK + 템플릿."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import cv2

from game_frame import extract_game_crop_bgr
from hud_digit_match import get_hud_digit_matcher
from hud_kda import read_kda_triple_from_game
from hud_round_end import HudState, analyze_hud_icons

DEFAULT_DATASET_ROOT = Path(r"E:\Highlights\ml_dataset")
DEFAULT_OUTPUT_DIR = Path(r"E:\clipai_result\ace_clips_hud")
DEFAULT_JSON_DIR = Path(r"E:\clipai_result\hud_timeline")
ACE_KILLS = 3
_MIN_ROUND_SEC = 20.0
_MIN_K3_STREAK = 2
_MAX_DELTA_K = 5

# 클립 구간 — end-35 방식 대신 라운드 시작·첫 킬 기준
_CLIP_PRE_ROUND_SEC = 6.0   # 라운드 시작 직전
_CLIP_PRE_KILL_SEC = 14.0   # 첫 킬 이전 (킬 장면이 클립 앞쪽에 오도록)
_CLIP_POST_END_SEC = 1.5    # 라운드 종료 직후 (스코어 잠깐만)
_CLIP_MAX_SEC = 55.0


@dataclass
class RoundTrack:
    round_index: int
    start_sec: float
    end_sec: float = 0.0
    k_base: int | None = None
    max_k: int = 0
    max_delta_k: int = 0
    k_samples: int = 0
    k3_streak: int = 0
    saw_k3_delta: bool = False
    ace: bool = False
    end_reason: str = "hud_elim"
    first_kill_sec: float | None = None
    ace_sec: float | None = None


@dataclass
class HudAceTimeline:
    video_path: str
    scan_fps: float
    ace_kills: int
    rounds: list[RoundTrack] = field(default_factory=list)
    ace_rounds: list[int] = field(default_factory=list)
    hud_end_count: int = 0
    k_template_hits: int = 0
    k_template_miss: int = 0
    warnings: list[str] = field(default_factory=list)


def _finalize_round(cur: RoundTrack, end_sec: float, reason: str) -> None:
    cur.end_sec = end_sec
    cur.end_reason = reason
    dur = cur.end_sec - cur.start_sec
    cur.ace = (
        cur.max_delta_k == ACE_KILLS
        and cur.saw_k3_delta
        and dur >= _MIN_ROUND_SEC
        and cur.k_samples >= 6
    )


def _update_k(cur: RoundTrack, k: int | None, method: str, stats: HudAceTimeline, t: float) -> None:
    if k is None:
        if method == "template_miss":
            stats.k_template_miss += 1
        return
    stats.k_template_hits += 1
    cur.k_samples += 1
    cur.max_k = max(cur.max_k, k)
    if cur.k_base is None:
        cur.k_base = k
    delta = k - cur.k_base
    if not (0 <= delta <= _MAX_DELTA_K):
        return
    if delta >= 1 and cur.first_kill_sec is None:
        cur.first_kill_sec = t
    if delta == ACE_KILLS and cur.ace_sec is None:
        cur.ace_sec = t
    cur.max_delta_k = max(cur.max_delta_k, delta)
    if delta == ACE_KILLS:
        cur.k3_streak += 1
        if cur.k3_streak >= _MIN_K3_STREAK:
            cur.saw_k3_delta = True
    else:
        cur.k3_streak = 0


def _merge_short_rounds(rounds: list[RoundTrack], min_sec: float = 15.0) -> list[RoundTrack]:
    if not rounds:
        return rounds
    merged: list[RoundTrack] = []
    for r in rounds:
        dur = r.end_sec - r.start_sec
        if merged and dur < min_sec:
            prev = merged[-1]
            prev.end_sec = r.end_sec
            prev.max_k = max(prev.max_k, r.max_k)
            prev.max_delta_k = max(prev.max_delta_k, r.max_delta_k)
            prev.k_samples += r.k_samples
            prev.saw_k3_delta = prev.saw_k3_delta or r.saw_k3_delta
            if prev.first_kill_sec is None:
                prev.first_kill_sec = r.first_kill_sec
            elif r.first_kill_sec is not None:
                prev.first_kill_sec = min(prev.first_kill_sec, r.first_kill_sec)
            if prev.ace_sec is None:
                prev.ace_sec = r.ace_sec
            elif r.ace_sec is not None:
                prev.ace_sec = min(prev.ace_sec, r.ace_sec)
            _finalize_round(prev, prev.end_sec, prev.end_reason)
        else:
            merged.append(r)
    for i, r in enumerate(merged):
        r.round_index = i
        _finalize_round(r, r.end_sec, r.end_reason)
    return merged


def scan_hud_aces(
    video_path: Path,
    *,
    scan_fps: float = 4.0,
    ace_kills: int = ACE_KILLS,
    dataset_root: Path | None = None,
) -> HudAceTimeline:
    video_path = Path(video_path)
    timeline = HudAceTimeline(
        video_path=str(video_path),
        scan_fps=scan_fps,
        ace_kills=ace_kills,
    )
    get_hud_digit_matcher()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        timeline.warnings.append("video_open_failed")
        return timeline

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_total / fps if frame_total > 0 else 0.0
    step = max(1, int(round(fps / scan_fps))) if scan_fps > 0 else int(fps)

    rounds: list[RoundTrack] = []
    cur = RoundTrack(round_index=0, start_sec=0.0)

    active_streak = 0
    ended_streak = 0
    ended_state: HudState | None = None
    ended_start = 0.0
    awaiting_next = False
    frame_idx = 0

    while True:
        if frame_idx % step == 0:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            t = frame_idx / fps
            game, _ = extract_game_crop_bgr(frame, dataset_root=dataset_root)
            k, _, _, _, method = read_kda_triple_from_game(game, template_only=True)
            _update_k(cur, k, method, timeline, t)

            state = analyze_hud_icons(frame).state
            if state == HudState.ACTIVE:
                active_streak += 1
                ended_streak = 0
                ended_state = None
                awaiting_next = False
            elif state in (HudState.RED_ELIMINATED, HudState.BLUE_ELIMINATED):
                if ended_state != state:
                    ended_streak = 1
                    ended_state = state
                    ended_start = t
                else:
                    ended_streak += 1
                if not awaiting_next and active_streak >= 3 and ended_streak >= 3:
                    _finalize_round(cur, ended_start, state.value)
                    rounds.append(cur)
                    timeline.hud_end_count += 1
                    cur = RoundTrack(round_index=len(rounds), start_sec=ended_start)
                    awaiting_next = True
                    active_streak = 0
        else:
            if not cap.grab():
                break
            frame_idx += 1
            if frame_idx / fps >= duration:
                break
            continue

        frame_idx += 1
        if frame_idx / fps >= duration:
            break

    cap.release()

    if cur.k_samples > 0 or cur.start_sec < duration:
        _finalize_round(cur, duration, "eof")
        if not rounds or rounds[-1].round_index != cur.round_index:
            rounds.append(cur)

    rounds = _merge_short_rounds(rounds)
    timeline.rounds = rounds
    timeline.ace_rounds = [r.round_index for r in rounds if r.ace]

    if timeline.k_template_hits == 0:
        timeline.warnings.append("k_never_read")

    return timeline


def ace_clip_window(r: RoundTrack) -> tuple[float, float]:
    """올킬 클립 [start, end] — 라운드 시작·첫 킬 기준, 종료 후 tail 최소."""
    start = r.start_sec - _CLIP_PRE_ROUND_SEC
    if r.first_kill_sec is not None:
        start = min(start, r.first_kill_sec - _CLIP_PRE_KILL_SEC)
    start = max(0.0, start)

    end = r.end_sec + _CLIP_POST_END_SEC
    if r.ace_sec is not None:
        end = min(end, max(r.ace_sec + 12.0, r.end_sec + 0.5))

    if end - start > _CLIP_MAX_SEC:
        start = end - _CLIP_MAX_SEC
    if end - start < 12.0:
        start = max(0.0, end - 12.0)
    return start, end


def extract_ace_clips(
    video_path: Path,
    timeline: HudAceTimeline,
    output_dir: Path,
) -> list[Path]:
    from extract_labeled_clips import run_ffmpeg_extract

    out_dir = output_dir / Path(timeline.video_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for r in timeline.rounds:
        if not r.ace:
            continue
        clip_start, clip_end = ace_clip_window(r)
        label = sec_to_mss(r.end_sec).replace(":", "m")
        out_path = out_dir / f"{Path(timeline.video_path).stem}_R{r.round_index:02d}_{label}s_hud_ace.mp4"
        ok = run_ffmpeg_extract(video_path, clip_start, clip_end, out_path)
        tag = "OK" if ok else "FAIL"
        print(
            f"  R{r.round_index} {sec_to_mss(clip_start)}-{sec_to_mss(clip_end)} "
            f"(round {sec_to_mss(r.start_sec)}-{sec_to_mss(r.end_sec)}) -> {out_path.name} {tag}",
            flush=True,
        )
        if ok:
            written.append(out_path)
    return written


def sec_to_mss(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}:{s:02d}"


def format_report(timeline: HudAceTimeline) -> str:
    lines = [
        f"## {Path(timeline.video_path).name}",
        f"scan_fps={timeline.scan_fps}  rounds={len(timeline.rounds)}  "
        f"ace={timeline.ace_rounds}  hud_ends={timeline.hud_end_count}",
        f"k_read: template={timeline.k_template_hits} miss={timeline.k_template_miss}",
    ]
    if timeline.warnings:
        lines.append(f"warnings: {timeline.warnings}")
    for r in timeline.rounds:
        tag = " **ACE**" if r.ace else ""
        lines.append(
            f"  R{r.round_index:02d} {sec_to_mss(r.start_sec)}-{sec_to_mss(r.end_sec)} "
            f"baseK={r.k_base} maxΔK={r.max_delta_k} k3={r.saw_k3_delta} n={r.k_samples}{tag}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HUD K/D/A 올킬 스캔 (닉 불필요)")
    p.add_argument("video_path")
    p.add_argument("--scan-fps", type=float, default=4.0)
    p.add_argument("--ace-kills", type=int, default=ACE_KILLS)
    p.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    p.add_argument("--json-out", default=None)
    p.add_argument("--extract", action="store_true")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"[hud-ace] 영상 없음: {video_path}")
        return 1

    print(f"[hud-ace] 스캔: {video_path.name} (fps={args.scan_fps})", flush=True)
    timeline = scan_hud_aces(
        video_path,
        scan_fps=args.scan_fps,
        ace_kills=args.ace_kills,
        dataset_root=Path(args.dataset_root),
    )
    print(format_report(timeline), flush=True)

    json_out = Path(args.json_out) if args.json_out else (
        DEFAULT_JSON_DIR / f"{video_path.stem}.json"
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(asdict(timeline), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[hud-ace] saved -> {json_out}", flush=True)

    if args.extract and timeline.ace_rounds:
        extract_ace_clips(video_path, timeline, Path(args.output_dir))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
