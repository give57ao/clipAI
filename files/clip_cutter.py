# -*- coding: utf-8 -*-
"""수동 구간 자르기 exe — PyInstaller로 clipAI_clipper.exe 빌드 대상
(본인 PC 전용, docs/superpowers/specs/2026-08-01-clip-cutter-design.md 참고).

"영상 경로 + 자를 구간들"이 한 줄씩 적힌 .txt 파일을 이 exe(또는
`python clip_cutter.py <txt...>`) 위로 드래그하면 시스템 ffmpeg로 스트림
복사(-c copy)해 구간들을 잘라낸다. 표준 라이브러리 + ffmpeg만 사용 —
batch_hud_ace_pipeline.py나 torch/opencv/easyocr는 전혀 거치지 않는다.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


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


OUTPUT_DIR = Path(r"E:\clipai_result\manual_clips")


def _wait_for_key() -> None:
    print("\n아무 키나 누르면 닫힙니다...")
    try:
        import msvcrt

        msvcrt.getch()
    except ImportError:
        input()


def process_txt_file(txt_path: Path, output_dir: Path, ffmpeg_path: str) -> tuple[int, int]:
    """txt_path의 각 줄을 처리해 (성공한 구간 수, 건너뛴 구간/줄 수)를
    반환한다. '없음'으로 명시적으로 스킵된 줄은 에러가 아니므로 건너뜀
    집계에 포함하지 않는다."""
    success = 0
    skipped = 0
    try:
        lines = txt_path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError:
        print(f"[clip] SKIP {txt_path.name} (텍스트 인코딩을 읽을 수 없음 — UTF-8로 저장해주세요)")
        return success, skipped + 1

    for line in lines:
        if not line.strip():
            continue

        split = split_path_and_ranges(line)
        if split is None:
            print(f"[clip] SKIP {line.strip()} (.mp4 경로를 찾을 수 없음)")
            skipped += 1
            continue

        path_str, ranges_str = split
        source = Path(path_str)
        if not source.exists():
            print(f"[clip] SKIP {source.name} (파일 없음)")
            skipped += 1
            continue

        ranges = parse_ranges(ranges_str)
        if not ranges:
            print(f"[clip] SKIP {source.name} (없음)")
            continue

        for raw, start, end, error in ranges:
            if error is not None:
                print(f"[clip] SKIP {source.name} 구간 '{raw}' ({error})")
                skipped += 1
                continue

            output_path = build_output_path(output_dir, source.stem, start)
            cmd = build_ffmpeg_command(ffmpeg_path, source, start, end, output_path)
            proc = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            if proc.returncode != 0:
                print(f"[clip] SKIP {source.name} 구간 '{raw}' (ffmpeg 실패)")
                print(proc.stderr)
                skipped += 1
                continue

            print(f"[clip] OK {source.name} {raw} -> {output_path.name}")
            success += 1

    return success, skipped


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("잘라낼 구간이 적힌 .txt 파일을 이 exe 위로 드래그하세요.")
        _wait_for_key()
        return 0

    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path is None:
        print("오류: ffmpeg를 찾을 수 없습니다 (PATH 확인).")
        _wait_for_key()
        return 1

    txt_files: list[Path] = []
    for raw in args:
        p = Path(raw).resolve()
        if p.suffix.lower() != ".txt":
            print(f"무시: {p.name} (txt 아님)")
            continue
        if not p.exists():
            print(f"무시: {p.name} (파일 없음)")
            continue
        txt_files.append(p)

    if not txt_files:
        print("처리할 .txt 파일이 없습니다.")
        _wait_for_key()
        return 0

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"오류: 결과 폴더를 만들 수 없습니다 ({OUTPUT_DIR}): {exc}")
        _wait_for_key()
        return 1

    total_success = 0
    total_skipped = 0
    for txt in txt_files:
        success, skipped = process_txt_file(txt, OUTPUT_DIR, ffmpeg_path)
        total_success += success
        total_skipped += skipped

    print(f"\n성공 {total_success}개 / 건너뜀 {total_skipped}개")

    if total_success == 0:
        print("처리할 구간이 없습니다.")
        _wait_for_key()
        return 0

    print(f"\n결과 폴더: {OUTPUT_DIR}")
    answer = input("여시겠습니까? (Y/N): ").strip().lower()
    if answer == "y":
        try:
            os.startfile(OUTPUT_DIR)
        except OSError as exc:
            print(f"폴더를 열 수 없습니다: {OUTPUT_DIR} ({exc})")

    _wait_for_key()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - last-resort guard so the console window never vanishes silently
        print(f"\n예상치 못한 오류가 발생했습니다: {exc}")
        _wait_for_key()
        raise
