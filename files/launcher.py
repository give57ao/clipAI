# -*- coding: utf-8 -*-
"""드래그 앤 드롭 exe 실행기 — PyInstaller로 clipAI_launcher.exe 빌드 대상
(본인 PC 전용, docs/superpowers/specs/2026-07-31-exe-launcher-design.md 참고).

영상 파일을 이 exe(또는 `python launcher.py <파일...>`) 위로 드래그하면
이 PC의 전역 파이썬(torch/opencv/easyocr 설치됨)으로 batch_hud_ace_pipeline.py를
실행한다. 표준 라이브러리만 사용하여 재포장하지 않는다.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

VENV_PYTHON = Path(r"C:\Users\give5\AppData\Local\Programs\Python\Python311\python.exe")
PIPELINE_SCRIPT = Path(r"C:\clipAI\files\batch_hud_ace_pipeline.py")
RESULT_CLIPS_DIR = Path(r"E:\clipai_result\ace_clips_hud")


def _wait_for_key() -> None:
    print("\n아무 키나 누르면 닫힙니다...")
    try:
        import msvcrt

        msvcrt.getch()
    except ImportError:
        input()


def filter_video_paths(args: list[str]) -> tuple[list[Path], list[str]]:
    """드롭된 인자들을 (유효한 mp4 절대경로 목록, 스킵 사유 메시지 목록)으로 분류."""
    valid: list[Path] = []
    skipped: list[str] = []
    for raw in args:
        p = Path(raw).resolve()
        if p.suffix.lower() != ".mp4":
            skipped.append(f"건너뜀: {p.name} (영상 파일 아님)")
            continue
        if not p.exists():
            skipped.append(f"건너뜀: {p.name} (파일 없음)")
            continue
        valid.append(p)
    return valid, skipped


def run_pipeline(video_paths: list[Path]) -> int:
    """임시 files-list txt를 만들어 배치 파이프라인을 subprocess로 실행, 종료코드 반환."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", encoding="utf-8", delete=False
    ) as f:
        for p in video_paths:
            f.write(f"{p}\n")
        list_path = f.name

    try:
        proc = subprocess.run(
            [str(VENV_PYTHON), "-u", str(PIPELINE_SCRIPT), "--files-list", list_path],
            cwd=str(PIPELINE_SCRIPT.parent),
        )
        return proc.returncode
    finally:
        Path(list_path).unlink(missing_ok=True)


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("영상 파일을 이 exe 위로 드래그하세요.")
        _wait_for_key()
        return 0

    if not VENV_PYTHON.exists():
        print(f"오류: 파이썬 환경을 찾을 수 없습니다 ({VENV_PYTHON})")
        _wait_for_key()
        return 1

    valid, skipped = filter_video_paths(args)
    for msg in skipped:
        print(msg)

    if not valid:
        print("처리할 영상이 없습니다.")
        _wait_for_key()
        return 0

    rc = run_pipeline(valid)
    if rc != 0:
        print(f"\n파이프라인 실행 중 오류가 발생했습니다 (종료 코드 {rc}).")
        _wait_for_key()
        return rc

    print(f"\n결과 폴더: {RESULT_CLIPS_DIR}")
    answer = input("여시겠습니까? (Y/N): ").strip().lower()
    if answer == "y":
        os.startfile(RESULT_CLIPS_DIR)

    _wait_for_key()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
