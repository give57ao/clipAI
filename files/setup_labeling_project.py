# -*- coding: utf-8 -*-
"""라벨링/학습 프로젝트 초기 세팅 (4종 하이라이트)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from labeling_constants import ALL_CLIP_LABELS, HIGHLIGHT_LABEL_KO, HIGHLIGHT_LABELS


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv_if_missing(path: Path, header: list[str]) -> bool:
    if path.exists():
        return False
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
    return True


def write_text_if_missing(path: Path, content: str) -> bool:
    if path.exists():
        return False
    path.write_text(content, encoding="utf-8")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="라벨링 프로젝트 초기 세팅")
    parser.add_argument(
        "--dataset-root",
        default=r"E:\Highlights\ml_dataset",
        help="라벨링/학습 데이터셋 루트 경로",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_root = Path(args.dataset_root)

    manifests_dir = dataset_root / "manifests"
    clips_dir = dataset_root / "clips"
    models_dir = dataset_root / "models"
    logs_dir = dataset_root / "logs"

    dirs = [dataset_root, manifests_dir, clips_dir, models_dir, logs_dir]
    for label in ALL_CLIP_LABELS:
        dirs.append(clips_dir / label)
    dirs.append(clips_dir / "review")

    for d in dirs:
        ensure_directory(d)

    created: list[str] = []
    skipped: list[str] = []

    known_highlights_path = manifests_dir / "known_highlights.csv"
    if write_csv_if_missing(
        known_highlights_path,
        [
            "video_path",
            "timestamp_sec",
            "label",
            "window_before_sec",
            "window_after_sec",
            "notes",
        ],
    ):
        created.append(str(known_highlights_path))
    else:
        skipped.append(str(known_highlights_path))

    label_segments_path = manifests_dir / "label_segments.csv"
    if write_csv_if_missing(
        label_segments_path,
        [
            "segment_id",
            "video_path",
            "start_sec",
            "end_sec",
            "label",
            "split",
            "source",
            "review_status",
            "notes",
        ],
    ):
        created.append(str(label_segments_path))
    else:
        skipped.append(str(label_segments_path))

    background_videos_path = manifests_dir / "background_videos.txt"
    background_template = (
        "# (선택) 일반 플레이 원본 영상 — 학습용 음성 샘플 추출\n"
        "# 하이라이트 4종과 섞여 있어도 됩니다.\n"
        "# 예시:\n"
        "# D:\\2026-01-11 00-33-10.mp4\n"
    )
    if write_text_if_missing(background_videos_path, background_template):
        created.append(str(background_videos_path))
    else:
        skipped.append(str(background_videos_path))

    label_lines = "\n".join(
        f"- {label} ({HIGHLIGHT_LABEL_KO[label]})" for label in HIGHLIGHT_LABELS
    )
    readme_content = (
        "라벨링 세팅 (4종 하이라이트)\n"
        "\n"
        "클래스:\n"
        f"{label_lines}\n"
        "\n"
        "1) known_highlights.csv\n"
        "- 아는 하이라이트 시점 + label 입력\n"
        "- label: doublekill | multikill | save | allkill\n"
        "- 한글도 가능: 더블킬, 멀티킬, 세이브, 올킬\n"
        "\n"
        "2) background_videos.txt (선택)\n"
        "- 일반 플레이 영상 목록 → background 음성 샘플 자동 생성\n"
        "\n"
        "3) label_segments.csv\n"
        "- build_label_manifest.py로 자동 생성\n"
    )
    project_readme_path = dataset_root / "README_LABELING_DATASET.txt"
    if write_text_if_missing(project_readme_path, readme_content):
        created.append(str(project_readme_path))
    else:
        skipped.append(str(project_readme_path))

    print("[setup] dataset root:", dataset_root)
    print("[setup] highlight labels:", ", ".join(HIGHLIGHT_LABELS))
    print("[setup] created:")
    for p in created:
        print("  +", p)
    print("[setup] existing (kept):")
    for p in skipped:
        print("  =", p)

    print("\n다음 단계:")
    print(
        "  python build_label_manifest.py "
        f"--dataset-root \"{dataset_root}\" "
        "--background-per-video 8 --allow-overwrite"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
