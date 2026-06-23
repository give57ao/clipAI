# -*- coding: utf-8 -*-
"""학습 공통: Dataset, FocalLoss, 메트릭, EfficientNet."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from labeling_constants import ALL_CLIP_LABELS, BACKGROUND_LABEL, HIGHLIGHT_LABELS

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm"}


@dataclass
class ClipRow:
    clip_path: str
    label: str
    split: str


def load_clip_rows(index_path: Path) -> list[ClipRow]:
    rows: list[ClipRow] = []
    with index_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            label = (row.get("label") or "").strip()
            split = (row.get("split") or "train").strip()
            clip_path = (row.get("clip_path") or "").strip()
            if label not in ALL_CLIP_LABELS or not clip_path:
                continue
            if not Path(clip_path).exists():
                continue
            rows.append(ClipRow(clip_path=clip_path, label=label, split=split))
    return rows


def build_train_rows_binary(
    highlights: list[ClipRow],
    backgrounds: list[ClipRow],
    bg_max: int,
    highlight_repeat: int,
    rng: random.Random,
) -> list[ClipRow]:
    hi = highlights * max(1, highlight_repeat)
    bg_pool = backgrounds
    if bg_max > 0 and len(bg_pool) > bg_max:
        bg_pool = rng.sample(bg_pool, bg_max)
    return hi + bg_pool


def build_train_rows_types(
    highlights: list[ClipRow],
    highlight_repeat: int,
) -> list[ClipRow]:
    return highlights * max(1, highlight_repeat)


class ClipFrameDataset(Dataset):
    def __init__(
        self,
        rows: list[ClipRow],
        label_to_idx: dict[str, int],
        train: bool,
        num_frames: int = 4,
        binary: bool = False,
    ):
        self.rows = rows
        self.label_to_idx = label_to_idx
        self.num_frames = max(1, num_frames)
        self.binary = binary
        self.train = train
        aug = [transforms.RandomHorizontalFlip(p=0.5)] if train else []
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                *aug,
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _label_idx(self, label: str) -> int:
        if self.binary:
            return 0 if label == BACKGROUND_LABEL else 1
        return self.label_to_idx[label]

    def _read_random_frame(self, path: str) -> np.ndarray | None:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return None
        idx = random.randint(0, max(0, frame_count - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int):
        row = self.rows[index]
        label_idx = self._label_idx(row.label)
        frames: list[torch.Tensor] = []

        for _ in range(self.num_frames):
            tensor = None
            for _try in range(4):
                frame = self._read_random_frame(row.clip_path)
                if frame is not None:
                    tensor = self.transform(frame)
                    break
            if tensor is None:
                blank = np.zeros((224, 224, 3), dtype=np.uint8)
                tensor = self.transform(blank)
            frames.append(tensor)

        return torch.stack(frames, dim=0), label_idx


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def build_model(num_classes: int) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def forward_multi_frame(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """images: [B, K, C, H, W] → logits [B, num_classes] (프레임 평균)."""
    b, k, c, h, w = images.shape
    logits = model(images.view(b * k, c, h, w))
    logits = logits.view(b, k, -1).mean(dim=1)
    return logits


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    idx_to_label: dict[int, str],
    positive_labels: set[str] | None = None,
) -> tuple[float, dict[str, float], float]:
    model.eval()
    correct = 0
    total = 0
    per_label_correct: dict[str, int] = {name: 0 for name in idx_to_label.values()}
    per_label_total: dict[str, int] = {name: 0 for name in idx_to_label.values()}

    tp = fp = fn = 0
    use_binary_recall = positive_labels is not None

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = forward_multi_frame(model, images)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for pred, label in zip(preds.cpu().tolist(), labels.cpu().tolist()):
            name = idx_to_label[label]
            per_label_total[name] = per_label_total.get(name, 0) + 1
            if pred == label:
                per_label_correct[name] = per_label_correct.get(name, 0) + 1

            if use_binary_recall:
                pred_pos = pred == 1
                true_pos = label == 1
                if pred_pos and true_pos:
                    tp += 1
                elif pred_pos and not true_pos:
                    fp += 1
                elif not pred_pos and true_pos:
                    fn += 1

    acc = correct / max(1, total)
    per_label_recall = {
        name: per_label_correct[name] / max(1, per_label_total[name])
        for name in per_label_total
        if per_label_total[name] > 0
    }
    highlight_recall = tp / max(1, tp + fn) if use_binary_recall else 0.0
    return acc, per_label_recall, highlight_recall


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_multi_frame(model, images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    return running_loss / max(1, total)
