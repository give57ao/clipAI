# -*- coding: utf-8 -*-
"""1b-1 단계: 영상 단위 주인공 닉 확정기 (v2: 스카우터 + 스코어보드 교차검증).

설계 문서: PLAYER_IDENTITY_AND_K_READER.md

소스 A — 스카우터 패널 투표 (scouter2 맨 위 / scouter 점 행)
소스 B — 전체스코어 6행 닉 OCR (detect_rounds CSV 연동)
소스 C — A∩B fuzzy 일치 시 가중치 +2 (교차검증)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from game_frame import get_game_roi_predictor
from nick_fuzzy import cluster_votes, clusters_to_summary, nick_match_text
from scoreboard_layout import (
    collect_scoreboard_nick_votes,
    find_scoreboard_csv,
    load_scoreboard_windows,
)
from scouter_nick import _is_valid_player_nick, read_scouter

DEFAULT_DATASET_ROOT = Path(r"E:\Highlights\ml_dataset")


@dataclass
class PlayerIdentity:
    """영상 1개에 대한 확정 닉 결과."""

    nickname: str
    confidence: float
    mode: str
    sources: dict = field(default_factory=dict)
    game_width_median: int = 0
    samples_total: int = 0
    samples_hit: int = 0


def _iter_sample_times(duration: float, scan_fps: float, max_samples: int) -> np.ndarray:
    t_start = min(60.0, duration * 0.05)
    t_end = max(t_start + 1.0, duration - 30.0)
    interval = 1.0 / scan_fps if scan_fps > 0 else 2.0
    times = np.arange(t_start, t_end, interval)
    if times.size == 0:
        return np.array([(t_start + t_end) / 2.0])
    if times.size > max_samples:
        times = np.linspace(t_start, t_end, max_samples)
    return times


def _collect_scouter_votes(
    cap: cv2.VideoCapture,
    fps: float,
    sample_times: np.ndarray,
    ocr_min_conf: float,
    dataset_root: Path,
) -> tuple[list[dict], Counter, Counter, list[int], int, int]:
    votes: list[dict] = []
    mode_counts: Counter = Counter()
    strong_modes: Counter = Counter()
    layout_widths: list[int] = []
    samples_total = 0
    samples_hit = 0

    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        samples_total += 1
        readout = read_scouter(frame, dataset_root=dataset_root)
        mode_counts[readout.mode] += 1
        if readout.game_width:
            layout_widths.append(readout.game_width)

        if (
            readout.player_nick
            and readout.player_conf >= ocr_min_conf
            and _is_valid_player_nick(readout.player_nick)
        ):
            samples_hit += 1
            strong_modes[readout.mode] += 1
            votes.append(
                {
                    "text": readout.player_nick,
                    "weight": 1.0,
                    "conf": readout.player_conf,
                    "mode": readout.mode,
                    "kind": "strong",
                }
            )
        elif readout.mode == "unknown" and readout.rows:
            top_text, top_conf = readout.rows[0]
            if top_text and top_conf >= ocr_min_conf and _is_valid_player_nick(top_text):
                votes.append(
                    {
                        "text": top_text,
                        "weight": 0.35,
                        "conf": top_conf,
                        "mode": "unknown",
                        "kind": "weak",
                    }
                )
    return votes, mode_counts, strong_modes, layout_widths, samples_total, samples_hit


def _build_cross_votes(scouter_votes: list[dict], frame_nicks: list[dict]) -> list[dict]:
    """스카우터 닉이 스코어보드 6행 중 하나와 일치하면 +2 가중치."""
    cross_votes: list[dict] = []
    for vote in scouter_votes:
        if vote.get("kind") not in ("strong", "weak"):
            continue
        for frame in frame_nicks:
            if any(nick_match_text(vote["text"], nick) for nick in frame["nicks"]):
                cross_votes.append(
                    {
                        "text": vote["text"],
                        "weight": 2.0,
                        "conf": vote["conf"],
                        "mode": vote.get("mode", ""),
                        "kind": "cross",
                        "sec": frame.get("sec"),
                    }
                )
                break
    return cross_votes


def _cross_source_bonus(scouter_votes: list[dict], cross_votes: list[dict]) -> float:
    strong = [v for v in scouter_votes if v.get("kind") in ("strong", "weak")]
    if not strong or not cross_votes:
        return 0.0
    confirmed = 0
    for vote in strong:
        if any(nick_match_text(vote["text"], cv["text"]) for cv in cross_votes):
            confirmed += 1
    return min(1.0, confirmed / len(strong))


def resolve_player_identity(
    video_path: Path,
    *,
    scan_fps: float = 0.5,
    scoreboard_csv: Path | None = None,
    rounds_dir: Path | None = None,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    max_samples: int = 150,
    max_scoreboards: int = 20,
    min_votes: int = 3,
    min_conf: float = 0.25,
    ocr_min_conf: float = 0.3,
) -> PlayerIdentity:
    """영상을 스캔해 주인공 닉을 확정한다 (v2: 소스 A+B+C)."""
    video_path = Path(video_path)
    if not video_path.exists():
        return PlayerIdentity("", 0.0, "unknown", {"error": "not_found"})

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return PlayerIdentity("", 0.0, "unknown", {"error": "open_failed"})

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_total / fps if frame_total > 0 else 0.0
    if duration <= 0:
        cap.release()
        return PlayerIdentity("", 0.0, "unknown", {"error": "no_duration"})

    sample_times = _iter_sample_times(duration, scan_fps, max_samples)
    scouter_votes, mode_counts, strong_modes, layout_widths, samples_total, samples_hit = (
        _collect_scouter_votes(cap, fps, sample_times, ocr_min_conf, dataset_root)
    )

    sb_csv = scoreboard_csv or find_scoreboard_csv(video_path, rounds_dir, dataset_root)
    scoreboard_votes: list[dict] = []
    frame_nicks: list[dict] = []
    scoreboard_windows = 0
    if sb_csv:
        windows = load_scoreboard_windows(sb_csv)
        scoreboard_windows = len(windows)
        scoreboard_votes, frame_nicks = collect_scoreboard_nick_votes(
            cap,
            fps,
            windows,
            ocr_min_conf=max(ocr_min_conf, 0.35),
            max_windows=max_scoreboards,
            dataset_root=dataset_root,
        )

    cap.release()

    cross_votes = _build_cross_votes(scouter_votes, frame_nicks)
    all_votes = scouter_votes + scoreboard_votes + cross_votes

    game_width_median = int(np.median(layout_widths)) if layout_widths else 0
    clusters = cluster_votes(all_votes)

    sources: dict = {
        "mode_counts": dict(mode_counts),
        "strong_modes": dict(strong_modes),
        "game_roi_neural": get_game_roi_predictor(dataset_root).uses_neural,
        "scoreboard_csv": str(sb_csv) if sb_csv else "",
        "scoreboard_windows": scoreboard_windows,
        "scoreboard_samples": len(scoreboard_votes),
        "cross_votes": len(cross_votes),
        "scouter_votes": len(scouter_votes),
        "clusters": clusters_to_summary(clusters),
        "scoreboard_frames": frame_nicks[:6],
    }

    if not clusters:
        sources["accepted"] = False
        sources["reason"] = "no_votes"
        return PlayerIdentity(
            "", 0.0, "unknown", sources, game_width_median, samples_total, samples_hit
        )

    top = clusters[0]
    total_weight = sum(c["weight"] for c in clusters)
    vote_ratio = top["weight"] / total_weight if total_weight > 0 else 0.0

    if strong_modes:
        dominant_mode, dominant_n = strong_modes.most_common(1)[0]
        mode_consistency = dominant_n / sum(strong_modes.values())
    elif scoreboard_votes:
        dominant_mode = "scoreboard"
        mode_consistency = top["samples"] / max(1, len(scoreboard_votes))
    else:
        dominant_mode, mode_consistency = "unknown", 0.0

    cross_bonus = _cross_source_bonus(scouter_votes, cross_votes)
    confidence = min(
        1.0,
        vote_ratio * 0.5 + cross_bonus * 0.3 + mode_consistency * 0.2,
    )

    accepted = top["samples"] >= min_votes and confidence >= min_conf
    sources.update(
        {
            "vote_ratio": round(vote_ratio, 3),
            "mode_consistency": round(mode_consistency, 3),
            "cross_source_bonus": round(cross_bonus, 3),
            "accepted": accepted,
        }
    )

    if not accepted:
        sources["reason"] = "low_votes" if top["samples"] < min_votes else "low_confidence"
        return PlayerIdentity(
            "", round(confidence, 3), dominant_mode, sources,
            game_width_median, samples_total, samples_hit,
        )

    return PlayerIdentity(
        nickname=top["canonical"],
        confidence=round(confidence, 3),
        mode=dominant_mode,
        sources=sources,
        game_width_median=game_width_median,
        samples_total=samples_total,
        samples_hit=samples_hit,
    )


def format_report(video_path: Path, identity: PlayerIdentity) -> str:
    layout = "후원패널형" if identity.game_width_median and identity.game_width_median < 1800 else "풀스크린형"
    src = identity.sources
    lines = [
        f"## {video_path.name}",
        f"nickname: {identity.nickname!r}  confidence: {identity.confidence}  mode: {identity.mode}",
        f"samples: scouter_hit={identity.samples_hit}/{identity.samples_total}  "
        f"layout: {layout} (game_w={identity.game_width_median})",
        f"scoreboard: windows={src.get('scoreboard_windows', 0)} "
        f"votes={src.get('scoreboard_samples', 0)} cross={src.get('cross_votes', 0)}",
        f"mode_counts: {src.get('mode_counts', {})}",
        "clusters:",
    ]
    for cluster in src.get("clusters", []):
        lines.append(
            f"  - {cluster['canonical']!r} weight={cluster['weight']} "
            f"samples={cluster['samples']} conf={cluster['best_conf']} "
            f"variants={cluster['variants']}"
        )
    if not src.get("accepted", False):
        lines.append(f"  [!] 확정 실패 (reason={src.get('reason')}) → 파이프라인 SKIP 권장")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="영상 단위 주인공 닉 확정기 (v2)")
    parser.add_argument("video_path", help="입력 mp4/mkv")
    parser.add_argument("--scan-fps", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=150)
    parser.add_argument("--max-scoreboards", type=int, default=20)
    parser.add_argument("--min-votes", type=int, default=3)
    parser.add_argument("--min-conf", type=float, default=0.25)
    parser.add_argument("--ocr-min-conf", type=float, default=0.3)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--rounds-dir", default=None, help="detect_rounds 출력 폴더")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video_path)
    rounds_dir = Path(args.rounds_dir) if args.rounds_dir else None

    print(f"[identity] {video_path.name} 스캔 중 (v2) ...", flush=True)
    identity = resolve_player_identity(
        video_path,
        scan_fps=args.scan_fps,
        rounds_dir=rounds_dir,
        dataset_root=Path(args.dataset_root),
        max_samples=args.max_samples,
        max_scoreboards=args.max_scoreboards,
        min_votes=args.min_votes,
        min_conf=args.min_conf,
        ocr_min_conf=args.ocr_min_conf,
    )

    print(format_report(video_path, identity), flush=True)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(identity)
        payload["video_path"] = str(video_path)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[identity] saved -> {out_path}", flush=True)

    return 0 if identity.nickname else 2


if __name__ == "__main__":
    raise SystemExit(main())
