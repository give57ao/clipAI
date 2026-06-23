# -*- coding: utf-8 -*-
"""OBS 녹화본에서 하이라이트 구간 탐지 및 클립 추출 (2단계 모델)."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from game_roi import GameRoiPredictor
from labeling_constants import HIGHLIGHT_LABELS
from ml_train_common import forward_multi_frame, build_model
from video_utils import probe_duration_sec

BINARY_IDX_TO_LABEL = {0: "background", 1: "highlight"}
TYPE_IDX_TO_LABEL = {idx: label for idx, label in enumerate(HIGHLIGHT_LABELS)}


@dataclass
class HighlightSegment:
    start_sec: float
    end_sec: float
    highlight_score: float
    highlight_type: str
    type_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="녹화본 하이라이트 추론")
    parser.add_argument("video_path", help="입력 mp4/mkv 경로")
    parser.add_argument("--dataset-root", default=r"E:\Highlights\ml_dataset")
    parser.add_argument("--output-dir", default=None, help="클립 저장 폴더 (미지정 시 dataset-root/inferred)")
    parser.add_argument("--window-sec", type=float, default=12.0)
    parser.add_argument("--stride-sec", type=float, default=6.0)
    parser.add_argument("--binary-threshold", type=float, default=0.55)
    parser.add_argument("--type-threshold", type=float, default=0.35)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--merge-gap-sec", type=float, default=3.0)
    parser.add_argument("--no-game-roi", action="store_true", help="게임 ROI 크롭 비활성화")
    parser.add_argument("--binary-only", action="store_true", help="1단계만 사용 (타입 분류 생략)")
    parser.add_argument("--dry-run", action="store_true", help="클립 추출 없이 구간만 출력")
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, int]:
    ckpt = torch.load(path, map_location=device)
    num_classes = len(ckpt["label_to_idx"])
    model = build_model(num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    num_frames = int(ckpt.get("num_frames", 4))
    return model, num_frames


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def read_frame_at_sec(
    video_path: Path,
    time_sec: float,
    roi_predictor: GameRoiPredictor | None = None,
) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = max(0, int(time_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if roi_predictor is not None:
        return roi_predictor.crop_rgb(rgb)
    return rgb


def sample_window_tensors(
    video_path: Path,
    start_sec: float,
    window_sec: float,
    num_frames: int,
    transform: transforms.Compose,
    roi_predictor: GameRoiPredictor | None = None,
) -> torch.Tensor | None:
    if num_frames <= 1:
        offsets = [window_sec * 0.5]
    else:
        step = window_sec / num_frames
        offsets = [step * (i + 0.5) for i in range(num_frames)]

    frames: list[torch.Tensor] = []
    for offset in offsets:
        rgb = read_frame_at_sec(video_path, start_sec + offset, roi_predictor)
        if rgb is None:
            return None
        frames.append(transform(rgb))
    return torch.stack(frames, dim=0)


@torch.no_grad()
def predict_binary(
    model: torch.nn.Module,
    window_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[float, int]:
    batch = window_tensor.unsqueeze(0).to(device)
    logits = forward_multi_frame(model, batch)
    probs = torch.softmax(logits, dim=1)[0]
    score = float(probs[1].item())
    pred = int(probs.argmax().item())
    return score, pred


@torch.no_grad()
def predict_type(
    model: torch.nn.Module,
    window_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[str, float]:
    batch = window_tensor.unsqueeze(0).to(device)
    logits = forward_multi_frame(model, batch)
    probs = torch.softmax(logits, dim=1)[0]
    idx = int(probs.argmax().item())
    return TYPE_IDX_TO_LABEL[idx], float(probs[idx].item())


def scan_video(
    video_path: Path,
    duration_sec: float,
    binary_model: torch.nn.Module,
    type_model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    transform: transforms.Compose,
    roi_predictor: GameRoiPredictor | None = None,
) -> list[HighlightSegment]:
    segments: list[HighlightSegment] = []
    start = 0.0
    while start + args.window_sec <= duration_sec + 0.01:
        end = min(start + args.window_sec, duration_sec)
        window_tensor = sample_window_tensors(
            video_path, start, args.window_sec, args.num_frames, transform, roi_predictor
        )
        if window_tensor is None:
            start += args.stride_sec
            continue

        hi_score, hi_pred = predict_binary(binary_model, window_tensor, device)
        if hi_pred == 1 and hi_score >= args.binary_threshold:
            if args.binary_only:
                segments.append(
                    HighlightSegment(
                        start_sec=start,
                        end_sec=end,
                        highlight_score=hi_score,
                        highlight_type="highlight",
                        type_score=1.0,
                    )
                )
            else:
                hi_type, type_score = predict_type(type_model, window_tensor, device)
                if type_score >= args.type_threshold:
                    segments.append(
                        HighlightSegment(
                            start_sec=start,
                            end_sec=end,
                            highlight_score=hi_score,
                            highlight_type=hi_type,
                            type_score=type_score,
                        )
                    )
        start += args.stride_sec
    return segments


def merge_segments(segments: list[HighlightSegment], gap_sec: float) -> list[HighlightSegment]:
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: s.start_sec)
    merged: list[HighlightSegment] = [segments[0]]

    for seg in segments[1:]:
        prev = merged[-1]
        if seg.start_sec <= prev.end_sec + gap_sec and seg.highlight_type == prev.highlight_type:
            best = seg if seg.highlight_score > prev.highlight_score else prev
            merged[-1] = HighlightSegment(
                start_sec=prev.start_sec,
                end_sec=max(prev.end_sec, seg.end_sec),
                highlight_score=max(prev.highlight_score, seg.highlight_score),
                highlight_type=best.highlight_type,
                type_score=max(prev.type_score, seg.type_score),
            )
        else:
            merged.append(seg)
    return merged


def extract_clip(video_path: Path, seg: HighlightSegment, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, seg.end_sec - seg.start_sec)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{seg.start_sec:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-c", "copy",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.returncode == 0


def write_results_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"[infer] 영상 없음: {video_path}")
        return 1

    duration_sec = probe_duration_sec(video_path)
    if duration_sec is None or duration_sec <= 0:
        print(f"[infer] duration 확인 실패: {video_path}")
        return 1

    dataset_root = Path(args.dataset_root)
    models_dir = dataset_root / "models"
    binary_path = models_dir / "highlight_binary_best.pt"
    types_path = models_dir / "highlight_types_best.pt"
    if not binary_path.exists():
        print("[infer] binary 모델 없음. train_binary.py 먼저 실행.")
        return 1
    if not args.binary_only and not types_path.exists():
        print("[infer] types 모델 없음. train_highlight_types.py 또는 --binary-only 사용.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    binary_model, _ = load_checkpoint(binary_path, device)
    type_model = None
    ckpt_frames = 4
    if args.binary_only:
        print("[infer] mode=binary-only (1단계만)", flush=True)
    else:
        type_model, ckpt_frames = load_checkpoint(types_path, device)
    if args.num_frames != ckpt_frames:
        args.num_frames = ckpt_frames

    transform = build_transform()
    roi_predictor = None
    if not args.no_game_roi:
        roi_predictor = GameRoiPredictor.from_dataset_root(dataset_root, device=device)
        mode = "neural" if roi_predictor.uses_neural else "teacher-fallback"
        print(f"[infer] game_roi={mode}", flush=True)

    print(f"[infer] video={video_path.name} duration={duration_sec:.1f}s device={device}")

    raw_segments = scan_video(
        video_path, duration_sec, binary_model, type_model, device, args, transform, roi_predictor
    )
    segments = merge_segments(raw_segments, args.merge_gap_sec)
    print(f"[infer] windows_hit={len(raw_segments)} merged={len(segments)}")

    output_dir = Path(args.output_dir) if args.output_dir else dataset_root / "inferred" / video_path.stem
    name_prefix = f"{video_path.stem}_하이라이트"
    rows: list[dict] = []

    for i, seg in enumerate(segments, start=1):
        if args.binary_only:
            clip_name = f"{name_prefix}_{i:03d}.mp4"
            out_path = output_dir / clip_name
        else:
            clip_name = f"{seg.highlight_type}_{i:03d}_{int(seg.start_sec)}s.mp4"
            out_path = output_dir / seg.highlight_type / clip_name
        row = {
            "video_path": str(video_path),
            "start_sec": f"{seg.start_sec:.2f}",
            "end_sec": f"{seg.end_sec:.2f}",
            "highlight_type": seg.highlight_type,
            "highlight_score": f"{seg.highlight_score:.4f}",
            "type_score": f"{seg.type_score:.4f}",
            "clip_path": str(out_path),
        }
        rows.append(row)
        print(
            f"  [{i}] {seg.highlight_type} {seg.start_sec:.1f}-{seg.end_sec:.1f}s "
            f"hi={seg.highlight_score:.2f} type={seg.type_score:.2f}"
        )
        if not args.dry_run:
            ok = extract_clip(video_path, seg, out_path)
            if not ok:
                print(f"    ffmpeg 실패: {out_path}")

    csv_path = output_dir / "inferred_segments.csv"
    if not args.dry_run:
        write_results_csv(csv_path, rows)
        print(f"[infer] saved -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
