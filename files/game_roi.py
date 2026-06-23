# -*- coding: utf-8 -*-
"""게임 화면 ROI: 신경망 추론 + (학습용) 적응형 pseudo-label."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

ROI_INPUT_SIZE = 224


@dataclass
class RoiBox:
    """픽셀 좌표 (x1, y1, x2, y2) — x2/y2는 exclusive."""

    x1: int
    y1: int
    x2: int
    y2: int

    def clamp(self, width: int, height: int) -> RoiBox:
        x1 = int(np.clip(self.x1, 0, max(0, width - 1)))
        y1 = int(np.clip(self.y1, 0, max(0, height - 1)))
        x2 = int(np.clip(self.x2, x1 + 1, width))
        y2 = int(np.clip(self.y2, y1 + 1, height))
        return RoiBox(x1, y1, x2, y2)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def to_normalized(self, width: int, height: int) -> tuple[float, float, float, float]:
        return (
            self.x1 / max(1, width),
            self.y1 / max(1, height),
            self.x2 / max(1, width),
            self.y2 / max(1, height),
        )

    @classmethod
    def from_normalized(
        cls,
        nx1: float,
        ny1: float,
        nx2: float,
        ny2: float,
        width: int,
        height: int,
    ) -> RoiBox:
        box = cls(
            int(nx1 * width),
            int(ny1 * height),
            int(nx2 * width),
            int(ny2 * height),
        )
        return box.clamp(width, height)


def build_roi_model() -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 4)
    return model


def build_roi_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((ROI_INPUT_SIZE, ROI_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def detect_game_roi_teacher(rgb: np.ndarray) -> RoiBox:
    """프레임마다 내용 영역 추정 (고정 비율 없음). ROI 모델 학습용 pseudo-label."""
    height, width = rgb.shape[:2]
    if height < 2 or width < 2:
        return RoiBox(0, 0, width, height)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    target_w = min(640, width)
    target_h = max(1, int(height * (target_w / width)))
    small = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)

    blur = cv2.GaussianBlur(small, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    col_mean = small.mean(axis=0)
    col_std = small.std(axis=0)
    row_mean = small.mean(axis=1)
    row_std = small.std(axis=1)

    mean_lo, mean_hi = float(col_mean.min()), float(col_mean.max())
    std_lo, std_hi = float(col_std.min()), float(col_std.max())
    mean_cut = mean_lo + (mean_hi - mean_lo) * 0.12
    std_cut = std_lo + (std_hi - std_lo) * 0.25

    col_active = (col_mean >= mean_cut) | (col_std >= std_cut)
    row_active = (row_mean >= mean_cut) | (row_std >= std_cut)

    if col_active.any():
        xs = np.where(col_active)[0]
        x1s, x2s = int(xs[0]), int(xs[-1]) + 1
    else:
        x1s, x2s = 0, target_w

    if row_active.any():
        ys = np.where(row_active)[0]
        y1s, y2s = int(ys[0]), int(ys[-1]) + 1
    else:
        y1s, y2s = 0, target_h

    mask_roi = mask[y1s:y2s, x1s:x2s]
    if mask_roi.size > 0:
        coords = cv2.findNonZero(mask_roi)
        if coords is not None:
            rx, ry, rw, rh = cv2.boundingRect(coords)
            x1s = x1s + rx
            y1s = y1s + ry
            x2s = x1s + rw
            y2s = y1s + rh

    scale_x = width / target_w
    scale_y = height / target_h
    box = RoiBox(
        int(x1s * scale_x),
        int(y1s * scale_y),
        int(x2s * scale_x),
        int(y2s * scale_y),
    ).clamp(width, height)

    area_ratio = (box.x2 - box.x1) * (box.y2 - box.y1) / max(1, width * height)
    if area_ratio < 0.2:
        return RoiBox(0, 0, width, height)
    return box


def crop_game_rgb(rgb: np.ndarray, box: RoiBox) -> np.ndarray:
    h, w = rgb.shape[:2]
    box = box.clamp(w, h)
    cropped = rgb[box.y1 : box.y2, box.x1 : box.x2]
    if cropped.size == 0:
        return rgb
    return cropped


@torch.no_grad()
def predict_roi_box(
    rgb: np.ndarray,
    model: nn.Module,
    device: torch.device,
    transform: transforms.Compose | None = None,
) -> RoiBox:
    height, width = rgb.shape[:2]
    transform = transform or build_roi_transform()
    tensor = transform(rgb).unsqueeze(0).to(device)
    model.eval()
    preds = torch.sigmoid(model(tensor))[0].cpu().tolist()
    return RoiBox.from_normalized(preds[0], preds[1], preds[2], preds[3], width, height)


def load_roi_checkpoint(path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(path, map_location=device)
    model = build_roi_model()
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


class GameRoiPredictor:
    """게임 화면 ROI 추론. 모델 없으면 teacher fallback."""

    def __init__(self, model_path: Path | None, device: torch.device | None = None):
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = build_roi_transform()
        self.model: nn.Module | None = None
        self._warned_fallback = False

        if model_path and model_path.exists():
            self.model = load_roi_checkpoint(model_path, self.device)

    @classmethod
    def from_dataset_root(cls, dataset_root: Path, device: torch.device | None = None) -> GameRoiPredictor:
        return cls(dataset_root / "models" / "game_roi_best.pt", device=device)

    @property
    def uses_neural(self) -> bool:
        return self.model is not None

    def predict_box(self, rgb: np.ndarray) -> RoiBox:
        if self.model is not None:
            return predict_roi_box(rgb, self.model, self.device, self.transform)
        if not self._warned_fallback:
            print("[game_roi] 모델 없음 → teacher fallback (train_game_roi.py 실행 권장)", flush=True)
            self._warned_fallback = True
        return detect_game_roi_teacher(rgb)

    def crop_rgb(self, rgb: np.ndarray) -> np.ndarray:
        return crop_game_rgb(rgb, self.predict_box(rgb))


class GameRoiBatchProcessor:
    """학습용: ROI 추론 + 크롭 + 분류 전처리를 GPU 배치로 처리."""

    def __init__(
        self,
        dataset_root: Path,
        device: torch.device,
        train: bool,
    ):
        self.device = device
        self.predictor = GameRoiPredictor.from_dataset_root(dataset_root, device=device)
        aug = [transforms.RandomHorizontalFlip(p=0.5)] if train else []
        self.frame_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                *aug,
                transforms.Resize((ROI_INPUT_SIZE, ROI_INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.no_grad()
    def __call__(self, batch_frames: list[list[np.ndarray]]) -> torch.Tensor:
        flat_frames: list[np.ndarray] = [frame for sample in batch_frames for frame in sample]
        if not flat_frames:
            return torch.zeros(0)

        batch_size = len(batch_frames)
        frames_per_clip = len(batch_frames[0])

        if self.predictor.model is not None:
            roi_inputs = torch.stack(
                [self.predictor.transform(rgb) for rgb in flat_frames]
            ).to(self.device, non_blocking=True)
            roi_preds = torch.sigmoid(self.predictor.model(roi_inputs)).cpu().numpy()
        else:
            roi_preds = None

        tensors: list[torch.Tensor] = []
        for i, rgb in enumerate(flat_frames):
            h, w = rgb.shape[:2]
            if roi_preds is not None:
                box = RoiBox.from_normalized(
                    float(roi_preds[i][0]),
                    float(roi_preds[i][1]),
                    float(roi_preds[i][2]),
                    float(roi_preds[i][3]),
                    w,
                    h,
                )
            else:
                box = detect_game_roi_teacher(rgb)
            cropped = crop_game_rgb(rgb, box)
            tensors.append(self.frame_transform(cropped))

        return torch.stack(tensors).view(batch_size, frames_per_clip, 3, ROI_INPUT_SIZE, ROI_INPUT_SIZE)
