# -*- coding: utf-8 -*-
"""T3 오프라인 게이트 (SONNET_TASK_DIGIT_CNN.md §T3) — held-out(val stem) 크롭에
v2 모델을 직접 돌려 R4 킬러(0→8 고신뢰 오분류)를 재현하는지 측정.

파일명에서 슬롯(k/d)·stem을 직접 파싱해 build_samples()의 origin 태그보다
정밀하게 분리한다 (origin='v2_claimed'는 K/D 슬롯이 섞여 있어 D 단독 게이트에
못 씀).

게이트:
  1) 0→8 오발률 < 0.5%  (claimed-0, held-out stem, model p>=0.92로 8 예측)
  2) 8 포착률            (label='8' 폴더 전체, held-out stem)
  3) D 슬롯 정확도 >= 99% (claimed 폴더의 '_d' 파일만, held-out stem, p>=0.95)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import torch

from train_hud_digit_cnn import CLASSES, MODEL_PATH, VAL_STEMS, TinyDigitCNN

DATA_DIR = Path(r"E:\clipai_result\hud_templates_harvest\cnn_dataset_v2")
_NAME_RE = re.compile(r"^(?P<stem>.+)_(?P<t>\d+)s_(?P<slot>[a-z])(?P<gi>\d+)\.png$")


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDigitCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device


def load_img(p: Path) -> np.ndarray | None:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape != (32, 24):
        return None
    return img


def predict_batch(model, device, paths: list[Path]) -> list[tuple[int, float]]:
    """반환: [(pred_label_or_-1(junk), prob)] — CLASSES 순서 그대로."""
    imgs = []
    for p in paths:
        im = load_img(p)
        imgs.append(im if im is not None else np.zeros((32, 24), dtype=np.uint8))
    x = np.stack(imgs).astype(np.float32) / 255.0
    t = torch.from_numpy(x).unsqueeze(1).to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(t), 512):
            batch = t[i : i + 512]
            probs = torch.softmax(model(batch), dim=1)
            top_p, top_i = probs.max(dim=1)
            for pi, pp in zip(top_i.tolist(), top_p.tolist()):
                out.append((CLASSES[pi], pp))
    return out


def held_out_files(folder: Path, slot: str | None = None) -> list[Path]:
    out = []
    if not folder.is_dir():
        return out
    for p in folder.glob("*.png"):
        m = _NAME_RE.match(p.name)
        if not m or m.group("stem") not in VAL_STEMS:
            continue
        if slot is not None and m.group("slot") != slot:
            continue
        out.append(p)
    return out


def main() -> int:
    model, device = load_model()

    # --- 게이트 1: claimed-0 → 8 오발률 (held-out) ---
    zero_files = held_out_files(DATA_DIR / "0")
    preds = predict_batch(model, device, zero_files)
    n = len(preds)
    false8 = sum(1 for lab, p in preds if lab == 8 and p >= 0.92)
    rate1 = false8 / n if n else float("nan")
    print(f"[G1] claimed-0 held-out n={n}  0→8오발(p>=0.92)={false8}  "
          f"오발률={rate1:.3%}  {'PASS' if rate1 < 0.005 else 'FAIL'} (기준 <0.5%)")

    # --- 게이트 2: 8 포착률 (held-out, '8' 폴더 전체 — quarantine 8 + 직접 8-read 혼합) ---
    eight_files = held_out_files(DATA_DIR / "8")
    preds8 = predict_batch(model, device, eight_files)
    n8 = len(preds8)
    hit8 = sum(1 for lab, p in preds8 if lab == 8)
    hit8_conf = sum(1 for lab, p in preds8 if lab == 8 and p >= 0.85)
    rate2 = hit8 / n8 if n8 else float("nan")
    print(f"[G2] 8 held-out n={n8}  포착(argmax=8)={hit8}({rate2:.1%})  "
          f"고신뢰(p>=0.85)포착={hit8_conf}({(hit8_conf/n8 if n8 else 0):.1%})  "
          f"(참고용 — 통과기준 없음, R4 대비 개선폭 지표)")

    # --- 게이트 3: D 슬롯 정확도 (claimed 폴더의 '_d' 파일만, held-out) ---
    d_files, d_labels = [], []
    for lbl_dir in sorted(DATA_DIR.iterdir()):
        if not lbl_dir.is_dir() or lbl_dir.name in ("junk",) or lbl_dir.name.startswith("hardneg_"):
            continue
        try:
            digit = int(lbl_dir.name)
        except ValueError:
            continue
        for p in held_out_files(lbl_dir, slot="d"):
            d_files.append(p)
            d_labels.append(digit)
    predsD = predict_batch(model, device, d_files)
    nD = len(predsD)
    correctD = sum(1 for (lab, p), y in zip(predsD, d_labels) if lab == y and p >= 0.95)
    rateD = correctD / nD if nD else float("nan")
    print(f"[G3] D슬롯 held-out n={nD}  정확(p>=0.95)={correctD}  "
          f"정확도={rateD:.2%}  {'PASS' if rateD >= 0.99 else 'FAIL'} (기준 >=99%)")

    print()
    print("종합:", "T3 게이트 전부 통과 — T4(파이프라인 전환) 진행 가능"
          if (rate1 < 0.005 and rateD >= 0.99) else "T3 게이트 미달 — T4 진행 보류, 데이터/모델 재검토 필요")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
