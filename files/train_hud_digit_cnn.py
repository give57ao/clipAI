# -*- coding: utf-8 -*-
"""HUD K 숫자 CNN 학습 (R4 Task A) — 24x32 이진 글리프, 11클래스(0~9 + junk).

작은 커스텀 conv net(2~3층)을 처음부터 학습 — `ml_train_common.build_model`의
EfficientNet-B0는 224x224 실사진(전광판/WIN 분류)용이라 24x32 이진 글리프에는
과대적합 위험만 크고 부적합(스펙 문서의 "detect_rounds.py 패턴 재사용" 서술은
검증 결과 부정확 — scoreboard_clf도 실제로는 EfficientNet-B0 사용. 이 특성상
작은 글리프엔 안 맞아 커스텀 아키텍처로 대체).

데이터 소스:
  - 0,1,2,3,4,5,6,7,9: `_build_digit_dataset.py --claim` 결과(claimed.json) —
    기존 사람이 검증·설치한 k_{d}.png 템플릿과 엄격 IoU(>=0.80, margin>=0.15)로
    자동 매칭된 raw/ 글리프. 영상 단위로 train/val 분리.
  - 8: 재군집화 시도 결과 0/6/9와 시각적으로 거의 분리 불가(실측 확인) → 대신
    사용자 확인 프레임(00-40-56 42:00~42:04, 7→8 전환 구간)에서 직접 추출한
    33장(육안 검증 완료, `eight_targeted/_eight_zoom.png`) + 과거 검증된
    `_removed_8/k_8*.png` medoid 3장 = 총 36장. **전량 train, val 분리 없음**
    (표본 부족 — 단일 영상 출처라 held-out 분리 시 val=0). 실제 검증은
    사용자 확인 프레임(79:51, 54:37 등)에서의 정성적 스팟체크로 대체.
  - junk: unclaimed 재군집화 결과 중 육안 확인된 잡음 클러스터(u02, 135장).

사용:
    python -u train_hud_digit_cnn.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

RAW_DIR = Path(r"E:\clipai_result\hud_templates_harvest\raw")
CNN_DATASET_DIR = Path(r"E:\clipai_result\hud_templates_harvest\cnn_dataset")
CNN_DATASET_V2_DIR = Path(r"E:\clipai_result\hud_templates_harvest\cnn_dataset_v2")
EIGHT_TARGETED_DIR = Path(r"E:\clipai_result\hud_templates_harvest\eight_targeted")
REMOVED_8_DIR = Path(__file__).parent / "hud_templates" / "_removed_8"
MODEL_DIR = Path(r"E:\Highlights\ml_dataset\models")
# R15(2026-07-23): v2(체인-감독 재학습)는 별도 파일로 저장 — v1은 롤백용으로 보존.
# SONNET_TASK_DIGIT_CNN.md 게이트 통과 전엔 hud_digit_match._CNN_MODEL_PATH를
# 이 파일로 전환하지 말 것.
MODEL_PATH = MODEL_DIR / "hud_digit_clf_v2_best.pt"
META_PATH = MODEL_DIR / "hud_digit_clf_v2_meta.json"

CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "junk"]
LABEL_TO_IDX = {lab: i for i, lab in enumerate(CLASSES)}

# v1 val은 3월 2영상뿐 — v2는 날짜 스프레드를 넓혀 게이트 표본 신뢰도 확보
# (2026-07-23 Sonnet: 4영상(n=67)에서 G1/G3 게이트가 통계적으로 불안정 —
# 10영상으로 확장해 재측정. 배경/해상도 드리프트 검증 겸함)
VAL_STEMS = {
    "2026-03-22 03-02-03", "2026-03-24 02-34-09",
    "2026-03-26 01-26-52", "2026-03-30 02-55-36",
    "2026-04-10 00-08-17", "2026-04-26 01-29-40",
    "2026-05-12 00-23-40", "2026-05-23 02-14-52",
    "2026-06-01 00-26-52", "2026-06-24 01-55-38",
    "2026-07-03 21-42-21", "2026-07-12 00-11-57",
}
_STEM_RE = re.compile(r"^(.*)_\d+s_[a-z]\d+\.png$")


def stem_of(name: str) -> str | None:
    m = _STEM_RE.match(name)
    return m.group(1) if m else None


def load_image(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape != (32, 24):
        return None
    return img


def build_samples() -> tuple[list[tuple[Path, int, str, str]], list[tuple[Path, int, str, str]]]:
    """반환: (train_samples, val_samples) — 각 원소 (경로, label_idx, video_stem, origin).

    origin: 'v1_claimed'|'v1_junk'|'v1_eight_seed' (기존 템플릿-순환 라벨) |
            'v2_claimed'|'v2_hardneg'|'v2_eight'|'v2_junk' (신규 체인-감독 라벨,
            R15 — `_build_digit_dataset_v2.py`). T3 게이트 리포트에서 origin별로
            분리 집계해 v2(특히 hardneg/eight)가 실제로 기여하는지 확인할 것.
    """
    train: list[tuple[Path, int, str, str]] = []
    val: list[tuple[Path, int, str, str]] = []

    claimed = json.loads((CNN_DATASET_DIR / "claimed.json").read_text(encoding="utf-8"))
    for digit_s, names in claimed.items():
        digit = int(digit_s)
        idx = LABEL_TO_IDX[digit]
        for name in names:
            stem = stem_of(name) or "?"
            p = RAW_DIR / name
            row = (p, idx, stem, "v1_claimed")
            (val if stem in VAL_STEMS else train).append(row)

    uclusters = json.loads(
        (CNN_DATASET_DIR / "unclaimed_clusters" / "unclaimed_clusters.json").read_text(
            encoding="utf-8"
        )
    )
    junk_idx = LABEL_TO_IDX["junk"]
    for name in uclusters["u02"]["members"]:
        stem = stem_of(name) or "?"
        p = RAW_DIR / name
        row = (p, junk_idx, stem, "v1_junk")
        (val if stem in VAL_STEMS else train).append(row)

    eight_idx = LABEL_TO_IDX[8]
    for p in sorted(EIGHT_TARGETED_DIR.glob("0040_4200_*.png"))[2:35]:
        train.append((p, eight_idx, "2026-03-21 00-40-56", "v1_eight_seed"))  # 전량 train (표본 부족)
    for p in REMOVED_8_DIR.glob("k_8*.png"):
        train.append((p, eight_idx, "_seed", "v1_eight_seed"))

    # R15 v2 — cnn_dataset_v2/<label>/<stem>_<t>s_<slot><gi>.png
    # <label>은 "0".."9"(claimed, K/D 슬롯 통합) | "hardneg_0".."hardneg_9" | "junk"
    if CNN_DATASET_V2_DIR.is_dir():
        for d in sorted(CNN_DATASET_V2_DIR.iterdir()):
            if not d.is_dir():
                continue
            name = d.name
            if name == "junk":
                # QC 결과(2026-07-23 Sonnet) — 이 폴더는 "안정런 밖 template_miss"
                # 기준이라 실제로는 멀쩡히 읽히는 숫자(빠른 킬스트릭 중 값이 순식간에
                # 바뀌어 4프레임 안정을 못 채운 정상 프레임)를 대량 오염시킴. 파일
                # 단위 정리로 해결 불가(폴더 전체가 구조적으로 잘못됨) — 하베스터
                # 로직 재설계 필요(Fable 보고, SONNET_TASK_DIGIT_CNN.md 진행기록
                # 참고). 재설계 전까지 전량 스킵, v1 junk만 사용.
                continue
            elif name.startswith("hardneg_"):
                digit = int(name.split("_", 1)[1])
                idx, origin = LABEL_TO_IDX[digit], "v2_hardneg"
            else:
                digit = int(name)
                idx = LABEL_TO_IDX[digit]
                origin = "v2_eight" if digit == 8 else "v2_claimed"
            for p in d.glob("*.png"):
                stem = stem_of(p.name) or "?"
                row = (p, idx, stem, origin)
                (val if stem in VAL_STEMS else train).append(row)

    return train, val


class GlyphDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int, str, str]], augment: bool):
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        p, label, _stem, _origin = self.samples[i]
        img = load_image(p)
        if img is None:
            img = np.zeros((32, 24), dtype=np.uint8)
        x = img.astype(np.float32) / 255.0
        if self.augment:
            dx = random.randint(-1, 1)
            dy = random.randint(-1, 1)
            x = np.roll(x, (dy, dx), axis=(0, 1))
            if random.random() < 0.15:
                x = x + np.random.normal(0, 0.05, x.shape).astype(np.float32)
                x = np.clip(x, 0.0, 1.0)
        t = torch.from_numpy(x).unsqueeze(0)
        return t, label


class TinyDigitCNN(nn.Module):
    def __init__(self, num_classes: int = 11):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 8 * 6, 128)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # 16x16x12
        x = self.pool(F.relu(self.conv2(x)))   # 32x8x6
        x = F.relu(self.conv3(x))              # 64x8x6
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


def evaluate(model: nn.Module, samples, device) -> tuple[float, dict[str, tuple[int, int]]]:
    if not samples:
        return 0.0, {}
    ds = GlyphDataset(samples, augment=False)
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    per_class: dict[int, list[int]] = {}  # idx -> [correct, total]
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            for yi, pi in zip(y.tolist(), pred.tolist()):
                c = per_class.setdefault(yi, [0, 0])
                c[1] += 1
                if yi == pi:
                    c[0] += 1
    by_label = {str(CLASSES[k]): tuple(v) for k, v in sorted(per_class.items())}
    return correct / total, by_label


def evaluate_by_origin(model: nn.Module, samples, device) -> dict[str, tuple[int, int]]:
    """origin별(v1_*/v2_claimed/v2_hardneg/v2_eight/v2_junk) 정확도 — T2 보고 요건.

    hardneg/eight는 R4를 죽인 바로 그 표본군이라 이 숫자가 핵심 증거다.
    """
    if not samples:
        return {}
    ds = GlyphDataset(samples, augment=False)
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    origins = [s[3] for s in samples]
    model.eval()
    per_origin: dict[str, list[int]] = {}
    i = 0
    with torch.no_grad():
        for x, y in loader:
            bs = x.size(0)
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            for j in range(bs):
                o = origins[i + j]
                c = per_origin.setdefault(o, [0, 0])
                c[1] += 1
                if pred[j].item() == y[j].item():
                    c[0] += 1
            i += bs
    return {k: tuple(v) for k, v in sorted(per_origin.items())}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_samples, val_samples = build_samples()
    print(f"[train] train={len(train_samples)} val={len(val_samples)}")

    counts: dict[int, int] = {}
    for _p, lab, _s, _o in train_samples:
        counts[lab] = counts.get(lab, 0) + 1
    print("[train] train class counts:", {str(CLASSES[k]): v for k, v in sorted(counts.items())})

    origin_counts: dict[str, int] = {}
    for _p, _lab, _s, o in train_samples + val_samples:
        origin_counts[o] = origin_counts.get(o, 0) + 1
    print("[train] origin counts (train+val):", dict(sorted(origin_counts.items())))

    # 클래스 불균형 보정 — inverse-frequency 가중치 (8·6·9처럼 표본 적은 클래스 보호)
    weights = torch.tensor(
        [1.0 / max(1, counts.get(i, 1)) for i in range(len(CLASSES))], dtype=torch.float32
    )
    weights = weights / weights.sum() * len(CLASSES)
    weights = weights.to(device)

    train_ds = GlyphDataset(train_samples, augment=True)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = TinyDigitCNN(num_classes=len(CLASSES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val = -1.0
    best_state = None

    epochs = 40
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=weights)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        train_acc, _ = evaluate(model, train_samples, device)
        val_acc, val_by_label = evaluate(model, val_samples, device)
        score = val_acc if val_samples else train_acc
        if score > best_val:
            best_val = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"[train] epoch {epoch:02d} loss={total_loss/len(train_samples):.4f} "
                f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
            )

    model.load_state_dict(best_state)
    train_acc, train_by_label = evaluate(model, train_samples, device)
    val_acc, val_by_label = evaluate(model, val_samples, device)
    print(f"[train] BEST train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
    print("[train] val per-class:", val_by_label)
    print("[train] train per-class:", train_by_label)

    val_by_origin = evaluate_by_origin(model, val_samples, device)
    train_by_origin = evaluate_by_origin(model, train_samples, device)
    print("[train] val per-origin (T2 게이트 근거 — hardneg/eight 필수 확인):", val_by_origin)
    print("[train] train per-origin:", train_by_origin)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    META_PATH.write_text(
        json.dumps(
            {
                "classes": [str(c) for c in CLASSES],
                "input_size": [32, 24],
                "val_acc": val_acc,
                "train_acc": train_acc,
                "val_stems": sorted(VAL_STEMS),
                "val_by_origin": {k: list(v) for k, v in val_by_origin.items()},
                "note": "R15 v2 — 체인-감독 라벨(v2_claimed/hardneg/eight/junk) + v1 유지. "
                        "게이트 통과 전 hud_digit_match._CNN_MODEL_PATH 전환 금지 "
                        "(SONNET_TASK_DIGIT_CNN.md §3 오프라인 게이트).",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[train] saved -> {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
