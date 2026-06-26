# -*- coding: utf-8 -*-
"""1a 단계: scoreboard 분류기로 영상을 스캔해 라운드 자동 분할.

- scoreboard_clf_best.pt 로 일정 간격 프레임 분류 (학습된 CNN)
- 연속 scoreboard 감지를 병합 → 스코어보드 윈도우
- "진짜 전체스코어는 ~4초 지속" 규칙: min-duration 미만은 폐기 (키 눌러 잠깐 본 스코어 제거)
- 보조: 상단 HUD 인원 아이콘 CV 휴리스틱으로 라운드 종료 직전 시점 탐지 (ML 아님)
- 스코어보드 사이 구간 = 라운드

출력:
  <output-dir>/detected_scoreboards.csv
  <output-dir>/detected_hud_round_ends.csv
  <output-dir>/rounds.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from ml_train_common import build_model
from hud_round_end import HudRoundEnd, find_hud_end_before_scoreboard, scan_hud_round_ends
from video_utils import probe_duration_sec

IDX_TO_CLASS = {0: "other", 1: "scoreboard"}
ASSETS_DIR = Path(r"C:\Users\give5\.cursor\projects\c-clipAI\assets")


@dataclass
class ScoreWindow:
    start_sec: float
    end_sec: float
    mean_prob: float
    n_frames: int

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class Round:
    round_id: int
    start_sec: float
    end_sec: float

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="scoreboard 기반 라운드 자동 분할")
    parser.add_argument("video_path", help="입력 mp4/mkv")
    parser.add_argument("--dataset-root", default=r"E:\Highlights\ml_dataset")
    parser.add_argument("--output-dir", default=None, help="결과 폴더 (기본 dataset-root/rounds/<stem>)")
    parser.add_argument("--scan-fps", type=float, default=2.0, help="초당 스캔 프레임 수")
    parser.add_argument("--threshold", type=float, default=0.6, help="scoreboard 확률 임계값")
    parser.add_argument("--merge-gap-sec", type=float, default=1.5, help="감지 병합 허용 간격")
    parser.add_argument(
        "--min-scoreboard-sec", type=float, default=3.5,
        help="진짜 전체스코어 최소 지속(초). 미만은 폐기 (실제 노출 ~4초, 안전 마진으로 살짝 아래)",
    )
    parser.add_argument(
        "--require-hud-round-end",
        action="store_true",
        help="(보조) 스코어보드 직전 HUD 라운드 종료(한쪽 인원 전멸)가 있을 때만 인정",
    )
    parser.add_argument(
        "--hud-lookback-sec",
        type=float,
        default=8.0,
        help="스코어보드 시작 이전 HUD 종료 탐색 범위(초)",
    )
    parser.add_argument(
        "--hud-scan-fps",
        type=float,
        default=4.0,
        help="HUD 인원 아이콘 스캔 fps (ML 없음, OpenCV 색상)",
    )
    parser.add_argument(
        "--require-win-defeat",
        action="store_true",
        help="(실험) WIN/DEFEAT 직후에만 scoreboard 인정 (템플릿 기반)",
    )
    parser.add_argument(
        "--win-lookback-sec",
        type=float,
        default=3.0,
        help="WIN/DEFEAT 탐지 시 scoreboard 시작 이전 탐색 범위(초)",
    )
    parser.add_argument(
        "--win-template-threshold",
        type=float,
        default=0.55,
        help="WIN 템플릿 매칭 임계값(0~1). 높을수록 엄격",
    )
    parser.add_argument(
        "--win-model-threshold",
        type=float,
        default=0.75,
        help="WIN/DEFEAT 모델 확률 임계값(0~1). 높을수록 엄격",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true", help="CSV 저장 안 함")
    return parser.parse_args()


def load_scoreboard_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model = build_model(len(ckpt["class_to_idx"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    img_size = int(ckpt.get("img_size", 224))
    return model, img_size


def load_win_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model = build_model(len(ckpt["class_to_idx"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    img_size = int(ckpt.get("img_size", 224))
    return model, img_size


def build_eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _load_win_template() -> np.ndarray | None:
    """사용자 제공 이미지에서 WIN 텍스트 영역을 템플릿으로 추출."""
    candidates = [
        ASSETS_DIR / "c__Users_give5_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_3-e8c549f3-c630-4fb3-a0fd-d5904d1f75eb.png",
        ASSETS_DIR / "c__Users_give5_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_2-9ce4094c-f5f5-4c1f-b58f-1fba9075cf49.png",
    ]
    for path in candidates:
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        # WIN 글자가 화면 중앙에 크게 존재하는 구간을 대략 crop
        y1, y2 = int(h * 0.38), int(h * 0.72)
        x1, x2 = int(w * 0.26), int(w * 0.74)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return gray
    return None


def _win_present_near(
    cap: cv2.VideoCapture,
    fps: float,
    scoreboard_start_sec: float,
    lookback_sec: float,
    template_gray: np.ndarray,
    threshold: float,
) -> bool:
    """scoreboard 직전 구간에서 WIN(또는 유사) 화면이 있는지 근사."""
    if template_gray is None:
        return True

    t0 = max(0.0, scoreboard_start_sec - lookback_sec)
    t1 = max(0.0, scoreboard_start_sec)
    step = 0.5  # 0.5초 간격 확인
    best = 0.0

    t = t0
    while t <= t1 + 1e-6:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            y1, y2 = int(h * 0.32), int(h * 0.78)
            x1, x2 = int(w * 0.18), int(w * 0.82)
            roi = frame[y1:y2, x1:x2]
            if roi.size:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                if gray.shape[0] >= template_gray.shape[0] and gray.shape[1] >= template_gray.shape[1]:
                    res = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    best = max(best, float(max_val))
                    if best >= threshold:
                        return True
        t += step

    return best >= threshold


@torch.no_grad()
def _win_model_present_near(
    cap: cv2.VideoCapture,
    fps: float,
    scoreboard_start_sec: float,
    lookback_sec: float,
    model,
    img_size: int,
    device: torch.device,
    threshold: float,
) -> bool:
    """scoreboard 직전 lookback 구간에서 win 모델 확률이 threshold 이상인지."""
    transform = build_eval_transform(img_size)
    t0 = max(0.0, scoreboard_start_sec - lookback_sec)
    t1 = max(0.0, scoreboard_start_sec)
    step = 0.5
    t = t0
    best = 0.0
    batch: list[torch.Tensor] = []
    while t <= t1 + 1e-6:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch.append(transform(rgb))
        t += step

    if not batch:
        return False
    images = torch.stack(batch).to(device)
    logits = model(images)
    probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    best = float(np.max(probs)) if probs.size else 0.0
    return best >= threshold


@torch.no_grad()
def scan_scoreboard_probs(
    video_path: Path,
    model,
    img_size: int,
    device: torch.device,
    scan_fps: float,
    batch_size: int,
) -> list[tuple[float, float]]:
    """(time_sec, prob_scoreboard) 리스트 반환.

    긴 영상에서는 프레임별 seek가 극도로 느리므로 순차 디코드 + N프레임마다 샘플.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    transform = build_eval_transform(img_size)
    frame_step = max(1, int(round(fps / scan_fps))) if scan_fps > 0 else max(1, int(fps))

    results: list[tuple[float, float]] = []
    batch_tensors: list[torch.Tensor] = []
    batch_times: list[float] = []

    def flush():
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors).to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
        for t, p in zip(batch_times, probs):
            results.append((t, float(p)))
        batch_tensors.clear()
        batch_times.clear()

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_idx % frame_step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_tensors.append(transform(rgb))
            batch_times.append(frame_idx / fps)
            if len(batch_tensors) >= batch_size:
                flush()
        frame_idx += 1

    flush()
    cap.release()
    return results


def merge_to_windows(
    probs: list[tuple[float, float]],
    threshold: float,
    merge_gap_sec: float,
    scan_step: float,
) -> list[ScoreWindow]:
    """임계값 초과 프레임을 연속 윈도우로 병합."""
    hits = [(t, p) for t, p in probs if p >= threshold]
    if not hits:
        return []

    windows: list[ScoreWindow] = []
    cur_start = hits[0][0]
    cur_end = hits[0][0]
    cur_probs = [hits[0][1]]

    for t, p in hits[1:]:
        if t - cur_end <= merge_gap_sec:
            cur_end = t
            cur_probs.append(p)
        else:
            windows.append(
                ScoreWindow(cur_start, cur_end + scan_step, float(np.mean(cur_probs)), len(cur_probs))
            )
            cur_start = t
            cur_end = t
            cur_probs = [p]

    windows.append(
        ScoreWindow(cur_start, cur_end + scan_step, float(np.mean(cur_probs)), len(cur_probs))
    )
    return windows


def rounds_from_windows(
    windows: list[ScoreWindow], duration: float, min_round_sec: float = 5.0
) -> list[Round]:
    """스코어보드 윈도우 사이 구간을 라운드로."""
    boundaries = sorted(windows, key=lambda w: w.start_sec)
    rounds: list[Round] = []
    prev_end = 0.0
    rid = 1
    for w in boundaries:
        seg_start = prev_end
        seg_end = w.start_sec
        if seg_end - seg_start >= min_round_sec:
            rounds.append(Round(rid, seg_start, seg_end))
            rid += 1
        prev_end = w.end_sec
    # 마지막 스코어보드 이후 잔여 구간
    if duration - prev_end >= min_round_sec:
        rounds.append(Round(rid, prev_end, duration))
    return rounds


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"[rounds] 영상 없음: {video_path}")
        return 1

    dataset_root = Path(args.dataset_root)
    model_path = dataset_root / "models" / "scoreboard_clf_best.pt"
    if not model_path.exists():
        print(f"[rounds] 모델 없음: {model_path}")
        print("  train_scoreboard_clf.py 먼저 실행.")
        return 1
    win_model_path = dataset_root / "models" / "win_clf_best.pt"

    duration = probe_duration_sec(video_path) or 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, img_size = load_scoreboard_model(model_path, device)
    print(f"[rounds] video={video_path.name} dur={duration:.0f}s device={device} scan_fps={args.scan_fps}")

    probs = scan_scoreboard_probs(
        video_path, model, img_size, device, args.scan_fps, args.batch_size
    )
    if not probs:
        print("[rounds] 스캔 실패")
        return 1

    scan_step = 1.0 / args.scan_fps if args.scan_fps > 0 else 1.0
    raw_windows = merge_to_windows(probs, args.threshold, args.merge_gap_sec, scan_step)
    # 4초 지속 규칙: 짧은 윈도우(키 눌러 잠깐 본 스코어) 폐기
    windows = [w for w in raw_windows if w.duration >= args.min_scoreboard_sec]
    dropped = len(raw_windows) - len(windows)

    if args.require_win_defeat:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        before = len(windows)

        if win_model_path.exists():
            win_model, win_img_size = load_win_model(win_model_path, device)
            windows = [
                w
                for w in windows
                if _win_model_present_near(
                    cap,
                    fps,
                    w.start_sec,
                    args.win_lookback_sec,
                    win_model,
                    win_img_size,
                    device,
                    args.win_model_threshold,
                )
            ]
            print(f"[rounds] require_win_defeat(win_model): kept={len(windows)}/{before}", flush=True)
        else:
            template = _load_win_template()
            if template is None:
                print("[rounds] require_win_defeat: WIN 모델/템플릿 없음 → 필터 스킵", flush=True)
            else:
                windows = [
                    w
                    for w in windows
                    if _win_present_near(
                        cap,
                        fps,
                        w.start_sec,
                        args.win_lookback_sec,
                        template,
                        args.win_template_threshold,
                    )
                ]
                print(f"[rounds] require_win_defeat(template): kept={len(windows)}/{before}", flush=True)

        cap.release()

    rounds = rounds_from_windows(windows, duration)

    hud_ends = scan_hud_round_ends(video_path, scan_fps=args.hud_scan_fps)
    print(f"[rounds] hud round_ends={len(hud_ends)} (CV heuristic, not ML)")
    for event in hud_ends:
        print(
            f"  HUD end {event.time_sec:7.1f}s eliminated={event.eliminated_team} "
            f"(R={event.red_icons} B={event.blue_icons})"
        )

    if args.require_hud_round_end:
        before = len(windows)
        windows = [
            w
            for w in windows
            if find_hud_end_before_scoreboard(hud_ends, w.start_sec, args.hud_lookback_sec) is not None
        ]
        rounds = rounds_from_windows(windows, duration)
        print(f"[rounds] require_hud_round_end: kept={len(windows)}/{before}", flush=True)

    sb_rows: list[list] = []
    hud_matched = 0
    for w in windows:
        match = find_hud_end_before_scoreboard(hud_ends, w.start_sec, args.hud_lookback_sec)
        if match:
            hud_matched += 1
        sb_rows.append(
            [
                f"{w.start_sec:.2f}",
                f"{w.end_sec:.2f}",
                f"{w.duration:.2f}",
                f"{w.mean_prob:.4f}",
                w.n_frames,
                "" if match is None else f"{match.time_sec:.2f}",
                "" if match is None else match.eliminated_team,
            ]
        )

    print(f"[rounds] scoreboard windows: raw={len(raw_windows)} kept={len(windows)} dropped_short={dropped}")
    print(f"[rounds] scoreboard+hud matched={hud_matched}/{len(sb_rows)}")
    for w in windows:
        print(f"  SB {w.start_sec:7.1f}-{w.end_sec:7.1f}s ({w.duration:.1f}s, p={w.mean_prob:.2f})")
    print(f"[rounds] rounds={len(rounds)}")
    for r in rounds:
        print(f"  R{r.round_id:02d} {r.start_sec:7.1f}-{r.end_sec:7.1f}s ({r.duration:.0f}s)")

    if not args.dry_run:
        out_dir = Path(args.output_dir) if args.output_dir else dataset_root / "rounds" / video_path.stem
        write_csv(
            out_dir / "detected_scoreboards.csv",
            [
                "start_sec",
                "end_sec",
                "duration_sec",
                "mean_prob",
                "n_frames",
                "hud_round_end_sec",
                "hud_eliminated_team",
            ],
            sb_rows,
        )
        write_csv(
            out_dir / "detected_hud_round_ends.csv",
            ["time_sec", "eliminated_team", "red_icons", "blue_icons"],
            [
                [
                    f"{e.time_sec:.2f}",
                    e.eliminated_team,
                    e.red_icons,
                    e.blue_icons,
                ]
                for e in hud_ends
            ],
        )
        write_csv(
            out_dir / "rounds.csv",
            ["round_id", "video_path", "start_sec", "end_sec", "duration_sec"],
            [[r.round_id, str(video_path), f"{r.start_sec:.2f}", f"{r.end_sec:.2f}", f"{r.duration:.2f}"] for r in rounds],
        )
        print(f"[rounds] saved -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
