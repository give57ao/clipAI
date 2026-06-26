# -*- coding: utf-8 -*-
"""라운드 타임라인 txt → round_segments.csv 변환.

manifests/round_timelines/*.txt 를 읽어 구간 CSV를 만듭니다.
- "START - END  라벨"  → 구간
- "TIME 라벨"          → 순간 이벤트(기본 INSTANT_SEC 구간)
한글 라벨은 내부 라벨로 매핑됩니다.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

OBS_DIR = Path(r"E:\OBS")
INSTANT_SEC = 3.0  # 순간 이벤트 기본 구간 길이

# 키워드 포함 매칭 (위에서부터 우선순위 — 더 구체적인 것 먼저)
# 자유 표현("대기실 재입장", "게임종료로딩 및 매치대기화면", "움직" 오타 등) 대응
ROUND_LABEL_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("단일킬", "single_kill"),
    ("single", "single_kill"),
    ("전체스코어", "scoreboard"),
    ("스코어", "scoreboard"),
    ("scoreboard", "scoreboard"),
    ("win", "win"),
    ("defeat", "win"),
    ("로딩", "loading"),
    ("loading", "loading"),
    ("대기실", "lobby"),
    ("대기화면", "lobby"),
    ("lobby", "lobby"),
    ("방송종료", "stream_end"),
    ("팀원", "teammate"),
    ("teammate", "teammate"),
    ("사망", "death"),
    ("죽", "death"),
    ("death", "death"),
    ("움직", "movement"),
    ("이동", "movement"),
    ("movement", "movement"),
)

ROUND_LABELS: tuple[str, ...] = (
    "movement",
    "death",
    "teammate",
    "scoreboard",
    "single_kill",
    "win",
    "loading",
    "lobby",
    "stream_end",
)

VIDEO_COMMENT_RE = re.compile(r"#\s*영상\s*:\s*(.+)$")
RANGE_RE = re.compile(
    r"^\s*([0-9:]+)\s*[-~]\s*([0-9:]+)\s+(.+?)\s*$"
)
INSTANT_RE = re.compile(r"^\s*([0-9:]+)\s+(.+?)\s*$")


@dataclass
class RoundSegment:
    video_path: str
    start_sec: float
    end_sec: float
    label: str
    label_ko: str
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="라운드 타임라인 → round_segments.csv")
    parser.add_argument("--dataset-root", default=r"E:\Highlights\ml_dataset")
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="기존 round_segments.csv 덮어쓰기",
    )
    return parser.parse_args()


def parse_time_to_sec(text: str) -> float | None:
    parts = text.strip().split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    if len(nums) == 1:
        return float(nums[0])
    return None


def normalize_round_label(raw: str) -> tuple[str, str] | None:
    # "게임대기실, 방송종료" → 첫 토막 우선
    raw_main = raw.split(",")[0].strip()
    text = raw_main.lower()
    for keyword, label in ROUND_LABEL_KEYWORDS:
        if keyword.lower() in text:
            return label, raw_main
    return None


def resolve_video_path(txt_path: Path, lines: list[str]) -> str:
    for line in lines:
        m = VIDEO_COMMENT_RE.search(line)
        if m:
            return m.group(1).strip()
    # 주석 없으면 파일명 stem 기반 추론
    return str(OBS_DIR / f"{txt_path.stem}.mp4")


def parse_timeline_file(txt_path: Path) -> tuple[list[RoundSegment], list[str]]:
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    video_path = resolve_video_path(txt_path, lines)
    segments: list[RoundSegment] = []
    warnings: list[str] = []

    for lineno, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        m = RANGE_RE.match(line)
        if m:
            start = parse_time_to_sec(m.group(1))
            end = parse_time_to_sec(m.group(2))
            label_info = normalize_round_label(m.group(3))
            if start is None or end is None or label_info is None:
                warnings.append(f"{txt_path.name}:{lineno} 파싱 실패: {line}")
                continue
            if end <= start:
                warnings.append(f"{txt_path.name}:{lineno} end<=start: {line}")
                continue
            label, label_ko = label_info
            segments.append(
                RoundSegment(video_path, float(start), float(end), label, label_ko, "manual")
            )
            continue

        m = INSTANT_RE.match(line)
        if m:
            t = parse_time_to_sec(m.group(1))
            label_info = normalize_round_label(m.group(2))
            if t is None or label_info is None:
                warnings.append(f"{txt_path.name}:{lineno} 파싱 실패: {line}")
                continue
            label, label_ko = label_info
            segments.append(
                RoundSegment(
                    video_path,
                    float(t),
                    float(t) + INSTANT_SEC,
                    label,
                    label_ko,
                    "manual_instant",
                )
            )
            continue

        warnings.append(f"{txt_path.name}:{lineno} 형식 불명: {line}")

    return segments, warnings


def write_csv(path: Path, segments: list[RoundSegment], allow_overwrite: bool) -> None:
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"이미 존재: {path} (--allow-overwrite 사용)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["video_path", "start_sec", "end_sec", "label", "label_ko", "source"]
        )
        for s in segments:
            writer.writerow(
                [s.video_path, f"{s.start_sec:.2f}", f"{s.end_sec:.2f}", s.label, s.label_ko, s.source]
            )


def summarize_rounds(segments: list[RoundSegment]) -> None:
    """scoreboard 기준으로 라운드 경계 개수 출력."""
    by_video: dict[str, list[RoundSegment]] = {}
    for s in segments:
        by_video.setdefault(s.video_path, []).append(s)

    for video, segs in by_video.items():
        scoreboard = [s for s in segs if s.label == "scoreboard"]
        boundaries = sorted(s.start_sec for s in scoreboard)
        print(f"\n[round] {Path(video).name}")
        print(f"  scoreboard(라운드 경계)={len(boundaries)}개")
        if boundaries:
            shown = ", ".join(f"{b:.0f}s" for b in boundaries[:12])
            print(f"  경계 시각: {shown}{' ...' if len(boundaries) > 12 else ''}")


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    timelines_dir = dataset_root / "manifests" / "round_timelines"
    out_path = dataset_root / "manifests" / "round_segments.csv"

    if not timelines_dir.exists():
        print(f"[round] 타임라인 폴더 없음: {timelines_dir}")
        return 1

    txt_files = sorted(timelines_dir.glob("*.txt"))
    if not txt_files:
        print(f"[round] *.txt 없음: {timelines_dir}")
        return 1

    all_segments: list[RoundSegment] = []
    all_warnings: list[str] = []
    for txt_path in txt_files:
        segs, warns = parse_timeline_file(txt_path)
        all_segments.extend(segs)
        all_warnings.extend(warns)

    if not all_segments:
        print("[round] 세그먼트 없음")
        for w in all_warnings[:20]:
            print("  -", w)
        return 1

    write_csv(out_path, all_segments, args.allow_overwrite)

    counts: dict[str, int] = {label: 0 for label in ROUND_LABELS}
    for s in all_segments:
        counts[s.label] = counts.get(s.label, 0) + 1

    print(f"[round] saved: {out_path}")
    print(f"[round] total={len(all_segments)} files={len(txt_files)}")
    print("[round] counts:", ", ".join(f"{k}={v}" for k, v in counts.items() if v))
    summarize_rounds(all_segments)

    if all_warnings:
        print(f"\n[round] warnings ({len(all_warnings)}):")
        for w in all_warnings[:20]:
            print("  -", w)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
