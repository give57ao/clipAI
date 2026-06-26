# -*- coding: utf-8 -*-
"""파일럿 라벨링 템플릿 CSV 생성."""

from __future__ import annotations

import csv
from pathlib import Path

KNOWN_PATH = Path(r"E:\Highlights\ml_dataset\manifests\known_highlights.csv")
OUT_PATH = Path(r"E:\Highlights\ml_dataset\manifests\pilot_labeling_template.csv")
GUIDE_PATH = Path(r"E:\Highlights\ml_dataset\manifests\pilot_labeling_guide.txt")
OBS_DIR = Path(r"E:\OBS")

PILOT_VIDEOS = [
    "2026-03-19 23-00-50.mp4",
    "2026-03-19 23-13-48.mp4",
    "2026-03-21 00-40-56.mp4",
]

HEADER = [
    "row_id",
    "video_file",
    "video_path",
    "time_display",
    "timestamp_sec",
    "highlight_type",
    "use_as",
    "negative_reason",
    "window_before_sec",
    "window_after_sec",
    "notes",
]


def fmt_time(sec: int) -> str:
    hours, rem = divmod(int(sec), 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def main() -> int:
    rows: list[list] = []
    with KNOWN_PATH.open(encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            video_path = row["video_path"]
            ts = int(float(row["timestamp_sec"]))
            rows.append(
                [
                    f"K{i:03d}",
                    Path(video_path).name,
                    video_path,
                    fmt_time(ts),
                    ts,
                    "",
                    "",
                    "",
                    "4",
                    "4",
                    "기존 킬 시각 — 분류 입력 필요",
                ]
            )

    extras: list[list] = []
    for vi, fname in enumerate(PILOT_VIDEOS, start=1):
        video_path = str(OBS_DIR / fname)
        for j in range(1, 6):
            extras.append(
                [
                    f"N{vi:01d}{j:02d}",
                    fname,
                    video_path,
                    "",
                    "",
                    "none",
                    "negative",
                    "",
                    "4",
                    "4",
                    "새 negative 시각 (선택)",
                ]
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)
        writer.writerows(extras)

    GUIDE_PATH.write_text(
        """파일럿 3개 OBS 라벨링 가이드
========================

파일: pilot_labeling_template.csv (Excel로 열기)

■ 기존 90행 (K001~K090)
  보내주신 킬 시각입니다. 영상 보면서 아래만 채우세요.

  highlight_type (4종 하이라이트일 때만)
    doublekill  더블킬
    multikill   멀티킬 (3연속 이상)
    save        세이브
    allkill     올킬
    none        4종 해당 없음

  use_as
    positive    학습용 하이라이트 (4종만)
    negative    학습용 배경 (하이라이트 아님)
    skip        애매함 / 제외
    review      나중에 다시 볼 것

  negative_reason (use_as=negative 일 때)
    single_kill     단일킬만
    death           내가 죽음
    idle_move       그냥 이동/대기
    teammate_only   팀원만 움직임
    respawn         리스폰/라운드 대기
    other           기타

  time_display 예: 2:09, 10:57, 1:20:08

■ 추가 행 (N... 로 시작, 15개)
  기존 90개에 없는 negative 시각을 새로 적을 때 사용.
  time_display (M:SS 또는 H:MM:SS) 와 timestamp_sec 중 하나만 적어도 됨.

■ 판단 기준 (짧게)
  - 킬 배너/UI가 4종(더블·멀티·세이브·올킬)에 해당 → positive + highlight_type
  - 킬은 났지만 단일킬 → negative + single_kill
  - 죽음/이동/팀원 플레이만 → negative + 해당 reason
  - 확실하지 않으면 skip 또는 review

■ 최소 목표 (전수 라벨링 아님)
  - 4종 positive: 가능한 만큼 정확히
  - negative: 단일킬 10~20개 + 죽음/이동 각 5~10개 (영상당)
  - 나머지 일반 플레이는 스크립트가 자동 샘플

■ 채운 뒤
  채팅에 "라벨링 완료" 하시면 known_highlights.csv 반영 + 재학습 진행.

■ 영상 경로
  E:\\OBS\\2026-03-19 23-00-50.mp4
  E:\\OBS\\2026-03-19 23-13-48.mp4
  E:\\OBS\\2026-03-21 00-40-56.mp4
""",
        encoding="utf-8",
    )

    print(f"[template] kill_rows={len(rows)} extra_rows={len(extras)} -> {OUT_PATH}")
    print(f"[template] guide -> {GUIDE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
