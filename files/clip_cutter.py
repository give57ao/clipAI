# -*- coding: utf-8 -*-
"""수동 구간 자르기 exe — PyInstaller로 clipAI_clipper.exe 빌드 대상
(본인 PC 전용, docs/superpowers/specs/2026-08-01-clip-cutter-design.md 참고).

"영상 경로 + 자를 구간들"이 한 줄씩 적힌 .txt 파일을 이 exe(또는
`python clip_cutter.py <txt...>`) 위로 드래그하면 시스템 ffmpeg로 스트림
복사(-c copy)해 구간들을 잘라낸다. 표준 라이브러리 + ffmpeg만 사용 —
batch_hud_ace_pipeline.py나 torch/opencv/easyocr는 전혀 거치지 않는다.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def parse_timecode(text: str) -> float:
    """'13:45' -> 825.0 (분:초), '1:8:55' -> 4135.0 (시:분:초).
    콜론 2개면 분:초, 3개면 시:분:초로 판단. 그 외 형식이거나 숫자가
    아니면 ValueError."""
    parts = text.strip().split(":")
    if len(parts) == 2:
        h_str, m_str, s_str = "0", parts[0], parts[1]
    elif len(parts) == 3:
        h_str, m_str, s_str = parts
    else:
        raise ValueError(f"잘못된 시간 형식: {text!r}")
    try:
        h, m, s = int(h_str), int(m_str), int(s_str)
    except ValueError as exc:
        raise ValueError(f"잘못된 시간 형식: {text!r}") from exc
    return float(h * 3600 + m * 60 + s)


def split_path_and_ranges(line: str) -> tuple[str, str] | None:
    """줄에서 '.mp4'가 처음 나오는 위치까지를 경로로, 그 뒤를 구간
    문자열로 분리한다(경로 자체에 공백이 있어도 '.mp4' 확장자로 경계를
    확정할 수 있음). '.mp4'가 없으면 None."""
    idx = line.lower().find(".mp4")
    if idx == -1:
        return None
    path_str = line[: idx + 4].strip()
    ranges_str = line[idx + 4 :].strip()
    return path_str, ranges_str


def parse_ranges(
    ranges_str: str,
) -> list[tuple[str, float | None, float | None, str | None]]:
    """구간 문자열을 (원문 토큰, 시작초, 종료초, 에러사유) 목록으로 파싱.
    '없음'이거나 빈 문자열이면 빈 목록을 반환한다(그 영상은 통째로
    스킵 — 에러가 아니라 사용자의 의도된 선택). 개별 토큰이 잘못됐으면
    error에 사유를 채워서 반환한다 — 그 구간만 나중에 건너뛰고 같은
    줄의 나머지 구간은 계속 처리하기 위함."""
    text = ranges_str.strip()
    if text == "" or text == "없음":
        return []

    results: list[tuple[str, float | None, float | None, str | None]] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" not in token:
            results.append((token, None, None, "형식 오류"))
            continue
        start_str, _, end_str = token.partition("-")
        try:
            start = parse_timecode(start_str)
            end = parse_timecode(end_str)
        except ValueError:
            results.append((token, None, None, "형식 오류"))
            continue
        if end <= start:
            results.append((token, start, end, "시작>=종료"))
            continue
        results.append((token, start, end, None))
    return results


PRE_ROLL_SECONDS = 2.0


def format_start_label(seconds: float) -> str:
    """825.0 -> '13m45s', 4135.0 -> '1h08m55s' (파일명용 시작시각 표기)."""
    total = max(int(round(seconds)), 0)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"


def build_output_path(output_dir: Path, source_stem: str, start_seconds: float) -> Path:
    label = format_start_label(start_seconds)
    return output_dir / f"{source_stem}_{label}.mp4"


def find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def build_ffmpeg_command(
    ffmpeg_path: str, source: Path, start: float, end: float, output: Path
) -> list[str]:
    """스트림 복사(-c copy)로 [start-PRE_ROLL_SECONDS, end] 구간을 잘라내는
    ffmpeg 커맨드를 조립한다. 시작을 앞당겨 -ss로 넘겨 키프레임 스냅으로
    시작 부분이 잘려나가는 것을 막는다(0초 미만이면 0으로 클램프). 끝은
    요청한 시각 그대로 유지한다(뒷부분에 여유가 더 붙는 것은 허용)."""
    adjusted_start = max(start - PRE_ROLL_SECONDS, 0.0)
    duration = end - adjusted_start
    return [
        ffmpeg_path,
        "-y",
        "-ss",
        f"{adjusted_start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(output),
    ]
