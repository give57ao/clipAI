# -*- coding: utf-8 -*-
"""B안 검증: scouter_nick.py로 영상당 본인 닉 추출 (랜덤 샘플)."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from scouter_nick import read_scouter


def extract_player_nick_video(
    video_path: Path,
    max_samples: int = 20,
    min_conf: float = 0.3,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": "open_failed"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_total / fps if frame_total > 0 else 0.0
    if duration <= 0:
        cap.release()
        return {"error": "no_duration"}

    t_start = min(60.0, duration * 0.05)
    t_end = max(t_start + 1.0, duration - 30.0)
    sample_times = np.linspace(t_start, t_end, num=max_samples)

    mode_counts: Counter = Counter()
    nick_counts: Counter = Counter()
    conf_by_nick: dict[str, float] = {}
    layout_widths: list[int] = []
    examples: list[dict] = []

    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        r = read_scouter(frame)
        mode_counts[r.mode] += 1
        layout_widths.append(r.game_width)
        if r.player_nick and r.player_conf >= min_conf:
            nick_counts[r.player_nick] += 1
            conf_by_nick[r.player_nick] = max(conf_by_nick.get(r.player_nick, 0.0), r.player_conf)
        if len(examples) < 4 and r.rows:
            examples.append(
                {
                    "sec": round(float(t), 1),
                    "mode": r.mode,
                    "player_nick": r.player_nick,
                    "conf": round(r.player_conf, 3),
                    "rows": r.rows[:4],
                    "game_width": r.game_width,
                }
            )

    cap.release()

    if not nick_counts:
        return {
            "error": "no_nick_hits",
            "duration_sec": round(duration, 1),
            "mode": dict(mode_counts),
            "samples": len(sample_times),
            "game_width_median": int(np.median(layout_widths)) if layout_widths else 0,
            "examples": examples,
        }

    best_nick, votes = nick_counts.most_common(1)[0]
    return {
        "nickname": best_nick,
        "votes": votes,
        "total_hits": sum(nick_counts.values()),
        "samples": len(sample_times),
        "duration_sec": round(duration, 1),
        "confidence": round(conf_by_nick.get(best_nick, 0.0), 3),
        "mode": dict(mode_counts),
        "alternates": nick_counts.most_common(6),
        "game_width_median": int(np.median(layout_widths)) if layout_widths else 0,
        "examples": examples,
    }


def pick_random_videos(obs_dir: Path, count: int, seed: int | None = None) -> list[Path]:
    videos = sorted(obs_dir.glob("*.mp4"))
    if not videos:
        return []
    rng = random.Random(seed)
    if len(videos) <= count:
        return videos
    return rng.sample(videos, count)


def format_report(video_path: Path, result: dict) -> list[str]:
    lines = [f"## {video_path.name}"]
    if "nickname" in result:
        gw = result.get("game_width_median", 0)
        layout = "후원패널형" if gw and gw < 1800 else "풀스크린형"
        lines.append(f"nickname: {result['nickname']}")
        lines.append(
            f"votes: {result['votes']}/{result['total_hits']} (samples={result['samples']})  "
            f"confidence: {result['confidence']}"
        )
        lines.append(f"mode: {result['mode']}  layout: {layout} (game_w={gw})")
        lines.append(f"alternates: {result['alternates']}")
        lines.append("examples:")
        for ex in result.get("examples", []):
            lines.append(
                f"  - sec={ex['sec']} mode={ex['mode']} nick={ex['player_nick']!r} "
                f"conf={ex['conf']} rows={ex['rows']}"
            )
    else:
        lines.append(f"ERROR: {result}")
    lines.append("")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="스카우터 본인 닉 OCR 검증")
    parser.add_argument("--obs-dir", default=r"E:\OBS")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None, help="None이면 매번 다른 랜덤")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--min-conf", type=float, default=0.3)
    parser.add_argument(
        "--output",
        default=r"E:\Highlights\ml_dataset\manifests\ocr_nick_validation.txt",
    )
    parser.add_argument("videos", nargs="*", help="지정 영상 (없으면 랜덤)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.videos:
        videos = [Path(v) for v in args.videos]
        seed_note = "specified"
    else:
        seed = args.seed if args.seed is not None else random.randint(0, 999999)
        videos = pick_random_videos(Path(args.obs_dir), args.count, seed)
        seed_note = str(seed)

    if not videos:
        print("[ocr-nick] OBS mp4 없음")
        return 1

    lines = [
        "# B안 OCR 검증 (scouter_nick.py)",
        "# 스카우터2=맨위행 / 스카우터=점(●)행 / null=진짜닉",
        f"# random_seed={seed_note}  videos={len(videos)}",
        "",
    ]
    for vp in videos:
        print(f"[ocr-nick] {vp.name} ...", flush=True)
        result = extract_player_nick_video(vp, args.max_samples, args.min_conf)
        lines.extend(format_report(vp, result))
        if "nickname" in result:
            print(f"  -> {result['nickname']} ({result['votes']}/{result['total_hits']})", flush=True)
        else:
            print(f"  -> ERROR {result.get('error')}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ocr-nick] saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
