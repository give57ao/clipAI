# 수동 구간 자르기 exe(clipAI_clipper) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** "영상 경로 + 자를 구간들"이 한 줄씩 적힌 `.txt` 파일을 드래그하면 ffmpeg 스트림 복사로 그 구간들을 잘라주는 독립 실행기(`clipAI_clipper.exe`)를 만든다.

**Architecture:** 표준 라이브러리 + 시스템 `ffmpeg`만 쓰는 단일 모듈 `files/clip_cutter.py`를 새로 만든다. 파싱(시간 표기, 줄 분리, 구간 목록) → 자르기 준비(출력 파일명, ffmpeg 커맨드 조립) → 오케스트레이션(txt 파일 순회, ffmpeg 실행, 콘솔 UX) 순서로 3개 태스크에 걸쳐 쌓아 올리고, 마지막 태스크에서 PyInstaller onefile로 exe화한다. 파이썬 파이프라인이나 torch/opencv/easyocr는 전혀 거치지 않는다 — `clipAI_launcher.exe`보다도 가벼운 구조.

**Tech Stack:** Python 3.11 (표준 라이브러리만: `subprocess`, `shutil`, `pathlib`, `os`, `sys`, `msvcrt`), 시스템 `ffmpeg`(subprocess로 직접 호출), pytest, PyInstaller(빌드 전용).

## Global Constraints

- 배포 대상은 본인 PC 전용 — ffmpeg가 PATH에 있다는 전제를 그대로 사용해도 됨(README에 이미 문서화됨).
- 입력은 `.txt` 파일을 exe에 드래그하는 방식. `.txt`가 아닌 파일을 드래그하면 "무시: <파일명> (txt 아님)"만 출력하고 처리하지 않음. 여러 `.txt`를 동시에 드래그하면 각각 독립적으로 순서대로 처리.
- 한 줄의 경로/구간 경계는 그 줄에서 `.mp4`가 처음 나오는 위치까지로 판단(대소문자 무관).
- 시간 표기는 콜론 개수로 자동 판별: 2개면 `분:초`, 3개면 `시:분:초`. 자릿수 패딩 여부는 무관(`1:8:55`, `01:08:55` 동일 결과).
- 구간은 콤마로 여러 개 나열, 트레일링 콤마(빈 토큰)는 무시.
- 구간 자리가 정확히 `없음`이거나 완전히 빈 문자열이면 그 줄의 영상은 통째로 스킵 — 이건 에러가 아니라 사용자의 의도된 선택이므로 "건너뜀" 집계에 포함하지 않는다.
- 자르기는 스트림 복사(`-c copy`, 재인코딩 없음)로 확정. 키프레임 스냅 보정을 위해 시작 시각에서 2초를 앞당겨 `-ss`로 넘기고(0초 미만이면 0으로 클램프), 끝은 요청한 시각 그대로 유지(뒷부분에 여유가 더 붙는 것은 허용). `-avoid_negative_ts make_zero`를 반드시 포함.
- 출력 위치는 `E:\clipai_result\manual_clips\`, 파일명은 `<원본파일명(확장자 제외)>_<시작시각>.mp4`(분초만 있으면 `13m45s`, 시가 있으면 `1h08m55s`, 분·초는 2자리 패딩).
- 문제 있는 줄/구간(파일 없음, 형식 오류, 시작≥종료, ffmpeg 실패)은 그 항목만 건너뛰고 나머지는 계속 처리. ffmpeg 자체가 PATH에 없으면 프로그램 시작 시점에 바로 에러 후 종료.
- 콘솔 UX는 `clipAI_launcher.exe`(`files/launcher.py`)와 동일한 패턴: 진행 로그 → 완료 요약 → 결과 폴더 열기 Y/N → 키 입력 대기 후 종료. `os.startfile()` 호출은 반드시 `try/except OSError`로 감쌀 것(1차 exe 최종 리뷰에서 발견된 실수 — 결과 폴더가 없을 때 콘솔이 안내 없이 크래시하는 문제를 처음부터 방지).
- 스펙 문서: `docs/superpowers/specs/2026-08-01-clip-cutter-design.md`

---

### Task 1: 시간/줄 파싱 유틸리티

**Files:**
- Create: `C:\clipAI\files\clip_cutter.py`
- Test: `C:\clipAI\files\tests\test_clip_cutter.py` (신규)

**Interfaces:**
- Produces: `parse_timecode(text: str) -> float` — `"13:45"` → `825.0`, `"1:8:55"` → `4135.0`. 형식이 아니면 `ValueError`.
- Produces: `split_path_and_ranges(line: str) -> tuple[str, str] | None` — 줄에서 `.mp4`까지를 경로 문자열로, 그 뒤를 구간 문자열로 분리. `.mp4`가 없으면 `None`.
- Produces: `parse_ranges(ranges_str: str) -> list[tuple[str, float | None, float | None, str | None]]` — `(원문 토큰, 시작초, 종료초, 에러사유)` 목록. `없음`/빈 문자열이면 빈 목록. 에러가 있으면 `error`가 채워짐(그 구간만 나중에 스킵하기 위함).

- [ ] **Step 1: 실패하는 테스트 작성**

`C:\clipAI\files\tests\test_clip_cutter.py` 신규 생성:

```python
# -*- coding: utf-8 -*-
"""clip_cutter.py(수동 구간 자르기 exe) 단위 테스트."""

from __future__ import annotations

import pytest

import clip_cutter


def test_parse_timecode_minutes_seconds():
    assert clip_cutter.parse_timecode("13:45") == 825.0


def test_parse_timecode_hours_minutes_seconds():
    assert clip_cutter.parse_timecode("1:8:55") == 4135.0


def test_parse_timecode_invalid_format_raises():
    with pytest.raises(ValueError):
        clip_cutter.parse_timecode("abc")


def test_parse_timecode_too_many_parts_raises():
    with pytest.raises(ValueError):
        clip_cutter.parse_timecode("1:2:3:4")


def test_split_path_and_ranges_basic():
    line = r"E:\OBS\2026-06-30 22-24-03.mp4        13:45 - 14:20, 25:18 - 25:35"

    result = clip_cutter.split_path_and_ranges(line)

    assert result == (
        r"E:\OBS\2026-06-30 22-24-03.mp4",
        "13:45 - 14:20, 25:18 - 25:35",
    )


def test_split_path_and_ranges_no_mp4_returns_none():
    assert clip_cutter.split_path_and_ranges("이건 그냥 텍스트") is None


def test_parse_ranges_multiple_with_trailing_comma():
    result = clip_cutter.parse_ranges("13:45 - 14:20, 25:18 - 25:35, ")

    assert result == [
        ("13:45 - 14:20", 825.0, 860.0, None),
        ("25:18 - 25:35", 1518.0, 1535.0, None),
    ]


def test_parse_ranges_none_keyword_returns_empty():
    assert clip_cutter.parse_ranges("없음") == []


def test_parse_ranges_empty_string_returns_empty():
    assert clip_cutter.parse_ranges("") == []


def test_parse_ranges_invalid_token_marks_error():
    result = clip_cutter.parse_ranges("abc")

    assert result == [("abc", None, None, "형식 오류")]


def test_parse_ranges_end_before_start_marks_error():
    result = clip_cutter.parse_ranges("14:20 - 13:45")

    assert result == [("14:20 - 13:45", 860.0, 825.0, "시작>=종료")]
```

- [ ] **Step 2: 테스트 실패 확인**

Run (in `C:\clipAI\files`): `python -m pytest tests/test_clip_cutter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'clip_cutter'`

- [ ] **Step 3: `clip_cutter.py` 생성 — 파싱 함수**

`C:\clipAI\files\clip_cutter.py` 신규 생성:

```python
# -*- coding: utf-8 -*-
"""수동 구간 자르기 exe — PyInstaller로 clipAI_clipper.exe 빌드 대상
(본인 PC 전용, docs/superpowers/specs/2026-08-01-clip-cutter-design.md 참고).

"영상 경로 + 자를 구간들"이 한 줄씩 적힌 .txt 파일을 이 exe(또는
`python clip_cutter.py <txt...>`) 위로 드래그하면 시스템 ffmpeg로 스트림
복사(-c copy)해 구간들을 잘라낸다. 표준 라이브러리 + ffmpeg만 사용 —
batch_hud_ace_pipeline.py나 torch/opencv/easyocr는 전혀 거치지 않는다.
"""

from __future__ import annotations

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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_clip_cutter.py -v`
Expected: PASS (11 passed)

- [ ] **Step 5: 커밋**

```bash
git add files/clip_cutter.py files/tests/test_clip_cutter.py
git commit -m "feat: clip_cutter.py 시간/줄 파싱 유틸리티 추가

parse_timecode/split_path_and_ranges/parse_ranges — 수동 구간 자르기
exe(clipAI_clipper)의 입력 파싱 담당. 자르기 로직은 다음 태스크."
```

---

### Task 2: 출력 파일명 규칙 + ffmpeg 커맨드 조립

**Files:**
- Modify: `C:\clipAI\files\clip_cutter.py`
- Modify: `C:\clipAI\files\tests\test_clip_cutter.py`

**Interfaces:**
- Consumes: 없음(Task 1의 파싱 결과는 다음 태스크에서 연결됨).
- Produces: `format_start_label(seconds: float) -> str` — `825.0` → `"13m45s"`, `4135.0` → `"1h08m55s"`.
- Produces: `build_output_path(output_dir: Path, source_stem: str, start_seconds: float) -> Path`.
- Produces: `find_ffmpeg() -> str | None` — `shutil.which("ffmpeg")` 래퍼.
- Produces: `build_ffmpeg_command(ffmpeg_path: str, source: Path, start: float, end: float, output: Path) -> list[str]`.
- Produces: 모듈 상수 `PRE_ROLL_SECONDS: float = 2.0`.

- [ ] **Step 1: 실패하는 테스트 작성**

`C:\clipAI\files\tests\test_clip_cutter.py` 상단 import 블록을 아래로 교체(`Path` 추가):

```python
from __future__ import annotations

from pathlib import Path

import pytest

import clip_cutter
```

파일 맨 끝에 추가:

```python
def test_format_start_label_minutes_seconds():
    assert clip_cutter.format_start_label(825.0) == "13m45s"


def test_format_start_label_with_hours():
    assert clip_cutter.format_start_label(4135.0) == "1h08m55s"


def test_build_output_path():
    result = clip_cutter.build_output_path(
        Path("E:/clipai_result/manual_clips"), "video", 825.0
    )

    assert result == Path("E:/clipai_result/manual_clips/video_13m45s.mp4")


def test_build_ffmpeg_command_applies_preroll_and_duration():
    cmd = clip_cutter.build_ffmpeg_command(
        "ffmpeg.exe", Path("in.mp4"), 100.0, 130.0, Path("out.mp4")
    )

    assert cmd[0] == "ffmpeg.exe"
    assert cmd[cmd.index("-ss") + 1] == "98.000"
    assert cmd[cmd.index("-i") + 1] == "in.mp4"
    assert cmd[cmd.index("-t") + 1] == "32.000"
    assert cmd[cmd.index("-c") + 1] == "copy"
    assert "-avoid_negative_ts" in cmd
    assert cmd[-1] == "out.mp4"


def test_build_ffmpeg_command_clamps_negative_start():
    cmd = clip_cutter.build_ffmpeg_command(
        "ffmpeg.exe", Path("in.mp4"), 1.0, 10.0, Path("out.mp4")
    )

    assert cmd[cmd.index("-ss") + 1] == "0.000"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_clip_cutter.py -v`
Expected: 이전 11개는 PASS, 신규 5개는 FAIL — `AttributeError: module 'clip_cutter' has no attribute 'format_start_label'` 등

- [ ] **Step 3: 구현 추가**

`C:\clipAI\files\clip_cutter.py` 상단 import 블록을 아래로 교체(`shutil` 추가):

```python
from __future__ import annotations

import shutil
import sys
from pathlib import Path
```

`parse_ranges` 함수 바로 뒤(파일 끝)에 추가:

```python
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
    duration = (end - start) + PRE_ROLL_SECONDS
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_clip_cutter.py -v`
Expected: PASS (16 passed)

- [ ] **Step 5: 커밋**

```bash
git add files/clip_cutter.py files/tests/test_clip_cutter.py
git commit -m "feat: clip_cutter.py 출력 파일명 규칙 + ffmpeg 커맨드 조립 추가

format_start_label/build_output_path/find_ffmpeg/build_ffmpeg_command.
2초 프리롤 안전막 포함. 오케스트레이션(main)은 다음 태스크."
```

---

### Task 3: 오케스트레이션(`main`) + 콘솔 UX

**Files:**
- Modify: `C:\clipAI\files\clip_cutter.py`
- Modify: `C:\clipAI\files\tests\test_clip_cutter.py`

**Interfaces:**
- Consumes: `split_path_and_ranges`, `parse_ranges` (Task 1), `build_output_path`, `build_ffmpeg_command`, `find_ffmpeg` (Task 2) — 전부 같은 모듈이므로 import 불필요.
- Produces: `process_txt_file(txt_path: Path, output_dir: Path, ffmpeg_path: str) -> tuple[int, int]` — `(성공한 구간 수, 건너뛴 구간/줄 수)`.
- Produces: `main() -> int`, `_wait_for_key() -> None`, 모듈 상수 `OUTPUT_DIR: Path`.

- [ ] **Step 1: 실패하는 테스트 작성**

`C:\clipAI\files\tests\test_clip_cutter.py` 파일 맨 끝에 추가:

```python
def test_process_txt_file_skips_missing_source_file(tmp_path):
    txt = tmp_path / "req.txt"
    txt.write_text(r"E:\OBS\gone.mp4   13:45 - 14:20" + "\n", encoding="utf-8")

    success, skipped = clip_cutter.process_txt_file(txt, tmp_path, "ffmpeg.exe")

    assert success == 0
    assert skipped == 1


def test_process_txt_file_skips_none_marker(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    txt = tmp_path / "req.txt"
    txt.write_text(f"{video}   없음\n", encoding="utf-8")

    success, skipped = clip_cutter.process_txt_file(txt, tmp_path, "ffmpeg.exe")

    assert success == 0
    assert skipped == 0


def test_process_txt_file_runs_ffmpeg_for_valid_range(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    txt = tmp_path / "req.txt"
    txt.write_text(f"{video}   13:45 - 14:20\n", encoding="utf-8")

    captured = {}

    class FakeCompleted:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeCompleted()

    monkeypatch.setattr(clip_cutter.subprocess, "run", fake_run)

    success, skipped = clip_cutter.process_txt_file(txt, tmp_path, "ffmpeg.exe")

    assert success == 1
    assert skipped == 0
    assert captured["cmd"][0] == "ffmpeg.exe"
    assert str(video) in captured["cmd"]


def test_process_txt_file_skips_ffmpeg_failure(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    txt = tmp_path / "req.txt"
    txt.write_text(f"{video}   13:45 - 14:20\n", encoding="utf-8")

    class FakeCompleted:
        returncode = 1
        stderr = "boom"

    monkeypatch.setattr(clip_cutter.subprocess, "run", lambda cmd, **kwargs: FakeCompleted())

    success, skipped = clip_cutter.process_txt_file(txt, tmp_path, "ffmpeg.exe")

    assert success == 0
    assert skipped == 1


def test_main_no_args_prints_usage(monkeypatch, capsys):
    monkeypatch.setattr(clip_cutter.sys, "argv", ["clip_cutter.py"])
    monkeypatch.setattr(clip_cutter, "_wait_for_key", lambda: None)

    rc = clip_cutter.main()

    out = capsys.readouterr().out
    assert rc == 0
    assert "드래그하세요" in out


def test_main_skips_non_txt_files(tmp_path, monkeypatch, capsys):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    monkeypatch.setattr(clip_cutter.sys, "argv", ["clip_cutter.py", str(video)])
    monkeypatch.setattr(clip_cutter, "find_ffmpeg", lambda: "ffmpeg.exe")
    monkeypatch.setattr(clip_cutter, "_wait_for_key", lambda: None)

    rc = clip_cutter.main()

    out = capsys.readouterr().out
    assert rc == 0
    assert "txt 아님" in out
    assert "처리할 .txt 파일이 없습니다." in out


def test_main_shows_folder_prompt_only_when_success_gt_zero(tmp_path, monkeypatch, capsys):
    txt = tmp_path / "req.txt"
    txt.write_text("dummy\n", encoding="utf-8")
    monkeypatch.setattr(clip_cutter.sys, "argv", ["clip_cutter.py", str(txt)])
    monkeypatch.setattr(clip_cutter, "find_ffmpeg", lambda: "ffmpeg.exe")
    monkeypatch.setattr(clip_cutter, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(clip_cutter, "process_txt_file", lambda *a, **k: (0, 1))
    monkeypatch.setattr(clip_cutter, "_wait_for_key", lambda: None)

    rc = clip_cutter.main()

    out = capsys.readouterr().out
    assert rc == 0
    assert "결과 폴더" not in out
    assert "처리할 구간이 없습니다." in out


def test_main_folder_open_guarded_against_oserror(tmp_path, monkeypatch, capsys):
    txt = tmp_path / "req.txt"
    txt.write_text("dummy\n", encoding="utf-8")
    monkeypatch.setattr(clip_cutter.sys, "argv", ["clip_cutter.py", str(txt)])
    monkeypatch.setattr(clip_cutter, "find_ffmpeg", lambda: "ffmpeg.exe")
    monkeypatch.setattr(clip_cutter, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(clip_cutter, "process_txt_file", lambda *a, **k: (1, 0))
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    def boom(path):
        raise OSError("no such folder")

    monkeypatch.setattr(clip_cutter.os, "startfile", boom, raising=False)
    monkeypatch.setattr(clip_cutter, "_wait_for_key", lambda: None)

    rc = clip_cutter.main()

    out = capsys.readouterr().out
    assert rc == 0
    assert "폴더를 열 수 없습니다" in out
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_clip_cutter.py -v`
Expected: 이전 16개는 PASS, 신규 8개는 FAIL — `AttributeError: module 'clip_cutter' has no attribute 'process_txt_file'` 등

- [ ] **Step 3: 구현 추가**

`C:\clipAI\files\clip_cutter.py` 상단 import 블록을 아래로 교체(`os`, `subprocess` 추가):

```python
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
```

파일 끝(`build_ffmpeg_command` 바로 뒤)에 추가:

```python
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
    for line in txt_path.read_text(encoding="utf-8").splitlines():
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    raise SystemExit(main())
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_clip_cutter.py -v`
Expected: PASS (24 passed)

- [ ] **Step 5: 전체 테스트 스위트 회귀 확인**

Run: `python -m pytest tests/ -v`
Expected: 모든 테스트 PASS (기존 40개 + 신규 24개 = 64 passed)

- [ ] **Step 6: 커밋**

```bash
git add files/clip_cutter.py files/tests/test_clip_cutter.py
git commit -m "feat: clip_cutter.py main 오케스트레이션 구현

txt 파일 순회 -> ffmpeg 실행 -> 콘솔 UX(진행 로그/요약/결과폴더 Y-N).
os.startfile은 OSError 가드 포함(1차 exe 최종 리뷰 교훈 반영).
PyInstaller 빌드는 다음 태스크."
```

---

### Task 4: PyInstaller 빌드 + 수동 스모크 테스트 + 문서화

**Files:**
- Modify: `C:\clipAI\.gitignore`
- Modify: `C:\clipAI\README.md`
- Create (빌드 산출물, git 추적 안 함): `C:\clipAI\dist\clipAI_clipper.exe`

**Interfaces:**
- Consumes: `C:\clipAI\files\clip_cutter.py` (Task 1-3).
- Produces: 없음(최종 산출물 — exe 자체).

- [ ] **Step 1: PyInstaller 빌드 산출물 gitignore 처리**

`C:\clipAI\.gitignore`에서 기존 항목:

```
# PyInstaller build artifacts (launcher.py -> clipAI_launcher.exe)
/dist/
/build/
/clipAI_launcher.spec
```

을 아래로 교체(clipAI_clipper.spec 추가, `/dist/`·`/build/`는 두 exe가 공유하므로 그대로 유지):

```
# PyInstaller build artifacts (launcher.py -> clipAI_launcher.exe, clip_cutter.py -> clipAI_clipper.exe)
/dist/
/build/
/clipAI_launcher.spec
/clipAI_clipper.spec
```

- [ ] **Step 2: gitignore 커밋**

```bash
git add .gitignore
git commit -m "chore: clipAI_clipper.spec을 gitignore에 추가"
```

- [ ] **Step 3: exe 빌드**

Run (in `C:\clipAI`):
```powershell
C:\Users\give5\AppData\Local\Programs\Python\Python311\python.exe -m PyInstaller --onefile --console --distpath dist --workpath build --name clipAI_clipper files\clip_cutter.py
```
Expected: 마지막 줄 `completed successfully`, `C:\clipAI\dist\clipAI_clipper.exe` 생성됨.

확인: `Get-ChildItem C:\clipAI\dist\clipAI_clipper.exe` — 파일 존재 확인. 표준 라이브러리만 쓰므로 크기는 십수 MB 이내여야 함(`clipAI_launcher.exe`와 비슷하거나 더 작은 수준 — 수 GB면 뭔가 잘못 잡힌 것).

- [ ] **Step 4: 인자 없이 실행 — 안내 메시지 스모크 테스트**

`clipAI_launcher.exe`와 동일하게 `_wait_for_key()`가 `msvcrt.getch()`를 쓰므로 파이프(`echo |`)로는 마지막 키 대기를 자동 통과시킬 수 없다(파이프 stdin에 반응하지 않음 — `docs/superpowers/specs/2026-07-31-exe-launcher-design.md`에 기록된 것과 동일한 제약). 더블클릭하거나 터미널에서 직접 실행해 눈으로 확인:

Run: `C:\clipAI\dist\clipAI_clipper.exe` (인자 없이)
Expected 출력: `잘라낼 구간이 적힌 .txt 파일을 이 exe 위로 드래그하세요.` 그리고 `아무 키나 누르면 닫힙니다...` — 아무 키나 눌러서 종료 확인.

- [ ] **Step 5: txt 아닌 파일 — 스킵 경로 스모크 테스트**

`E:\OBS`의 아무 `.mp4` 파일 경로를 그대로 인자로 실행:

Run: `C:\clipAI\dist\clipAI_clipper.exe "E:\OBS\<아무 영상>.mp4"`
Expected 출력에 포함: `무시: <파일명> (txt 아님)` 그리고 `처리할 .txt 파일이 없습니다.`

- [ ] **Step 6: 실제 영상으로 end-to-end 스모크 테스트**

메모장 등으로 요청 텍스트 파일을 하나 만든다(`E:\OBS`에 있는 아무 영상 하나 선택, 영상 길이보다 짧은 범위로 구간 지정 — 예: 영상 시작 부분 `0:01 - 0:05`):

```
E:\OBS\<선택한 영상 파일명>.mp4    0:01 - 0:05
```

이 파일을 `smoke_request.txt`로 저장한 뒤:

Run: `C:\clipAI\dist\clipAI_clipper.exe "C:\clipAI\smoke_request.txt"` (또는 탐색기에서 실제로 드래그)
Expected 출력에 포함: `[clip] OK <영상파일명> 0:01 - 0:05 -> <영상파일명>_00m01s.mp4`(시작 2초 안전막 때문에 실제로는 0초부터 잘리지만 파일명 라벨은 요청한 시작시각 `0:01` 기준), `성공 1개 / 건너뜀 0개`, `결과 폴더: E:\clipai_result\manual_clips`, `여시겠습니까? (Y/N):` 프롬프트.

확인: `E:\clipai_result\manual_clips\<영상파일명>_00m01s.mp4` 파일이 실제로 생성되었고 재생 가능한지 확인(`ffprobe`로 duration이 대략 5초 안팎인지 확인 가능 — 이 예시는 시작 시각 0:01이 2초 프리롤보다 작아 클램프가 걸리는 케이스라, `-t`는 클램프된 시작 기준 `end - adjusted_start`로 계산되어 요청한 종료 시각(0:05)을 넘기지 않는다. 클램프 이전 계획 단계의 "(종료-시작)+2초" 공식이었다면 약 6초가 나왔겠지만, 리뷰에서 그 공식이 클램프 시 종료 시각을 최대 2초까지 초과시키는 버그로 지적되어 실제 코드는 다르게 구현됐다 — 자세한 내용은 `docs/superpowers/specs/2026-08-01-clip-cutter-design.md`의 "구현 중 수정 사항" 참고).

작업 후 `smoke_request.txt`와 생성된 스모크 테스트 클립은 삭제해도 무방(테스트용 산출물).

- [ ] **Step 7: README에 사용법 추가**

`C:\clipAI\README.md`의 "### exe 드래그앤드롭 실행기" 섹션이 끝나는 지점 바로 뒤에 아래 서브섹션 추가:

```markdown
### 수동 구간 자르기 exe

이미 골라둔 "몇분부터 몇분까지" 구간들을 텍스트로 적어두면 그대로
잘라주는 두 번째 실행기(`clipAI_clipper.exe`, 본인 PC 전용 —
`clip_cutter.py`가 시스템 PATH의 `ffmpeg`를 그대로 호출한다).
HUD 자동탐지나 파이썬 파이프라인은 전혀 거치지 않는다.

메모장 등으로 아래 형식의 `.txt` 파일을 만들어 `dist\clipAI_clipper.exe`에
드래그한다:

```
E:\OBS\2026-06-30 22-24-03.mp4        13:45 - 14:20, 25:18 - 25:35
E:\OBS\2026-07-03 20-07-56.mp4     없음
```

- 시간은 `분:초`(`13:45`) 또는 `시:분:초`(`1:8:55`) 자유롭게. 구간은
  콤마로 여러 개 나열, `없음`이면 그 영상은 통째로 건너뜀.
- 스트림 복사라 빠르지만 키프레임 스냅 때문에 시작 2초 정도는 미리
  당겨서 잘린다(끝은 요청한 그대로 — 뒤에 살짝 여유가 붙는 정도).
- 결과는 `E:\clipai_result\manual_clips\`에 `<원본파일명>_<시작시각>.mp4`
  형식으로 저장.

재빌드가 필요하면:

```powershell
cd C:\clipAI
C:\Users\give5\AppData\Local\Programs\Python\Python311\python.exe -m PyInstaller --onefile --console --distpath dist --workpath build --name clipAI_clipper files\clip_cutter.py
```
```

- [ ] **Step 8: 커밋**

```bash
git add README.md
git commit -m "docs: README에 수동 구간 자르기 exe(clipAI_clipper) 사용법 추가"
```

---

## Self-Review 결과 (계획 작성자 기준)

- **스펙 커버리지**: 입력 형식(경로/구간 경계, 시간표기, 콤마목록, `없음`) → Task 1. 자르기 정확도(스트림 복사 + 2초 프리롤 + 클램프 + `avoid_negative_ts`) → Task 2. 출력 위치/파일명 규칙 → Task 2(`build_output_path`). 에러 처리 표의 7개 케이스(파일없음/형식오류/시작≥종료/ffmpeg실패/ffmpeg PATH없음/txt아님/유효구간0개) 전부 Task 3 구현·테스트에 반영. 콘솔 UX(진행로그→요약→결과폴더 Y/N→키대기) → Task 3. PyInstaller 빌드 → Task 4. `os.startfile` 예외 가드는 1차 exe에서 사후에 발견된 결함이었으므로 이번엔 Task 3에서 처음부터 구현 + 전용 회귀 테스트(`test_main_folder_open_guarded_against_oserror`) 포함.
- **플레이스홀더 스캔**: "TBD"/"나중에" 류 없음. 모든 스텝에 실행 가능한 코드·명령 포함.
- **타입/시그니처 일관성**: `parse_timecode`/`split_path_and_ranges`/`parse_ranges`(Task 1)와 `format_start_label`/`build_output_path`/`find_ffmpeg`/`build_ffmpeg_command`(Task 2), `process_txt_file`/`main`/`_wait_for_key`/`OUTPUT_DIR`(Task 3) 이름과 시그니처가 Task 3의 구현·테스트에서 동일하게 사용됨(`process_txt_file`이 Task 1·2 함수들을 그대로 호출).
- **`없음` 집계 정책**: 명시적으로 "에러가 아니므로 건너뜀 카운트에 미포함"이라고 Global Constraints와 `process_txt_file` 독스트링에 동일하게 명시 — 리뷰 시 혼동 방지.
