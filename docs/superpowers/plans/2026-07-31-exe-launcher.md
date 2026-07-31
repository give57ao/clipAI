# 드래그 앤 드롭 exe 실행기 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 영상 파일을 드래그하면 기존 `.venv`를 그대로 호출해 HUD 올킬 파이프라인을 돌려주는 초경량 exe 실행기(`clipAI_launcher.exe`)를 만든다.

**Architecture:** `batch_hud_ace_pipeline.py`에 임의 절대경로 목록을 받는 `--files-list` 옵션을 추가하고, 표준 라이브러리만 쓰는 `launcher.py`를 새로 만들어 드롭된 파일을 임시 목록 파일로 저장한 뒤 `.venv\Scripts\python.exe`로 파이프라인을 subprocess 실행한다. `launcher.py`만 PyInstaller로 exe화한다 — torch/opencv/easyocr는 재포장하지 않는다.

**Tech Stack:** Python 3.11 (표준 라이브러리만: `subprocess`, `tempfile`, `pathlib`, `os`, `sys`, `msvcrt`), pytest, PyInstaller(빌드 전용).

## Global Constraints

- 배포 대상은 본인 PC 전용 — `.venv` 경로(`C:\clipAI\.venv\Scripts\python.exe`)를 하드코딩해도 됨.
- 처리 대상은 `.mp4` 파일만 (기존 `batch_hud_ace_pipeline.py`의 `obs.glob("*.mp4")` 기준과 동일).
- 기존 CLI 옵션(`--only`, `--after`, `--limit`, `--stems-file`, `--redo` 등)은 `--files-list` 추가로 인해 동작이 바뀌면 안 됨(가산적 변경).
- exe는 파일을 드래그했을 때만 파이프라인을 실행한다. 인자 없이 더블클릭하면 안내 메시지만 표시하고 종료(전체 배치 스캔 자동 실행 금지).
- 파이프라인 subprocess가 에러로 끝나도 콘솔 창은 즉시 닫히지 않고 키 입력을 대기한 뒤 닫힌다.
- 스펙 문서: `docs/superpowers/specs/2026-07-31-exe-launcher-design.md`

---

### Task 1: `batch_hud_ace_pipeline.py` — `--files-list` 옵션

**Files:**
- Modify: `C:\clipAI\files\batch_hud_ace_pipeline.py`
- Test: `C:\clipAI\files\tests\test_batch_files_list.py` (신규)

**Interfaces:**
- Produces: `_load_files_list(path: Path) -> list[Path]` — 한 줄에 절대경로 하나씩 적힌 텍스트 파일을 읽어 `Path` 리스트로 반환(빈 줄 무시). `_load_stems_filter`와 동일한 스타일.
- Produces: CLI 옵션 `--files-list <path>` — 지정 시 `--obs-dir` 글롭을 건너뛰고 그 파일에 적힌 경로들만 대상으로 사용. `.mp4`가 아니거나 존재하지 않는 경로는 `[hud-batch] SKIP <파일명>: not_mp4` / `[hud-batch] SKIP <파일명>: not_found` 로 출력하고 제외.

- [ ] **Step 1: 실패하는 테스트 작성**

`C:\clipAI\files\tests\test_batch_files_list.py` 신규 생성:

```python
# -*- coding: utf-8 -*-
"""batch_hud_ace_pipeline --files-list 옵션 회귀 테스트."""

from __future__ import annotations

from pathlib import Path

import batch_hud_ace_pipeline as bhp


def test_load_files_list_reads_absolute_paths(tmp_path):
    v1 = tmp_path / "a.mp4"
    v2 = tmp_path / "b.mp4"
    v1.write_bytes(b"")
    v2.write_bytes(b"")
    list_file = tmp_path / "files.txt"
    list_file.write_text(f"{v1}\n\n{v2}\n  \n", encoding="utf-8")

    result = bhp._load_files_list(list_file)

    assert result == [v1, v2]


def test_load_files_list_empty_file_returns_empty_list(tmp_path):
    list_file = tmp_path / "empty.txt"
    list_file.write_text("", encoding="utf-8")

    assert bhp._load_files_list(list_file) == []
```

- [ ] **Step 2: 테스트 실패 확인**

Run (in `C:\clipAI\files`): `python -m pytest tests/test_batch_files_list.py -v`
Expected: FAIL — `AttributeError: module 'batch_hud_ace_pipeline' has no attribute '_load_files_list'`

- [ ] **Step 3: `_load_files_list` 구현**

`C:\clipAI\files\batch_hud_ace_pipeline.py`에서 `_load_stems_filter` 함수 바로 아래에 추가:

```python
def _load_files_list(path: Path) -> list[Path]:
    """한 줄에 절대경로 하나씩 적힌 파일 → Path 리스트. 빈 줄은 무시.

    launcher.py(드래그앤드롭 exe 실행기)가 임의 위치의 드롭 파일을
    OBS_DIR 글롭 없이 그대로 넘기기 위해 사용."""
    return [
        Path(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_batch_files_list.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: CLI 옵션 배선**

`C:\clipAI\files\batch_hud_ace_pipeline.py`의 `main()`에서 `--stems-file` 인자 정의 바로 뒤에 추가:

```python
    ap.add_argument(
        "--files-list",
        default=None,
        help="한 줄에 절대경로 하나씩 적힌 파일로 처리 대상 지정 — 지정 시 --obs-dir 글롭을 "
             "건너뛰고 이 목록만 사용(임의 위치 파일 허용). launcher.py(exe 실행기) 전용.",
    )
```

같은 함수에서 기존 비디오 목록 구성 블록:

```python
    obs = Path(args.obs_dir)
    videos = sorted(obs.glob("*.mp4"))
    if args.stems_file:
        wanted = _load_stems_filter(Path(args.stems_file))
        videos = [p for p in videos if p.stem in wanted]
    if args.only:
        videos = [p for p in videos if p.stem == args.only]
    if args.after:
        videos = [p for p in videos if p.stem > args.after]
    if args.limit > 0:
        videos = videos[: args.limit]
```

를 아래로 교체:

```python
    obs = Path(args.obs_dir)
    if args.files_list:
        raw_paths = _load_files_list(Path(args.files_list))
        videos = []
        for p in raw_paths:
            if p.suffix.lower() != ".mp4":
                print(f"[hud-batch] SKIP {p.name}: not_mp4", flush=True)
                continue
            if not p.exists():
                print(f"[hud-batch] SKIP {p.name}: not_found", flush=True)
                continue
            videos.append(p)
        videos.sort()
    else:
        videos = sorted(obs.glob("*.mp4"))
        if args.stems_file:
            wanted = _load_stems_filter(Path(args.stems_file))
            videos = [p for p in videos if p.stem in wanted]
        if args.only:
            videos = [p for p in videos if p.stem == args.only]
        if args.after:
            videos = [p for p in videos if p.stem > args.after]
    if args.limit > 0:
        videos = videos[: args.limit]
```

- [ ] **Step 6: 기존 회귀 테스트 + 신규 테스트 모두 통과 확인**

Run: `python -m pytest tests/test_scan_lock.py tests/test_batch_files_list.py -v`
Expected: 9 passed (기존 7 + 신규 2)

- [ ] **Step 7: 커밋**

```bash
git add files/batch_hud_ace_pipeline.py files/tests/test_batch_files_list.py
git commit -m "feat: batch_hud_ace_pipeline에 --files-list 옵션 추가

임의 위치의 절대경로 목록으로 처리 대상을 지정할 수 있게 함 - OBS_DIR
글롭을 건너뛴다. exe 드래그앤드롭 실행기(launcher.py)에서 사용 예정."
```

---

### Task 2: `launcher.py` — 드래그앤드롭 실행 로직

**Files:**
- Create: `C:\clipAI\files\launcher.py`
- Test: `C:\clipAI\files\tests\test_launcher.py` (신규)

**Interfaces:**
- Consumes: `batch_hud_ace_pipeline.py`를 **subprocess로만** 호출(`--files-list` 옵션, Task 1에서 구현됨) — 직접 import하지 않음.
- Produces: `filter_video_paths(args: list[str]) -> tuple[list[Path], list[str]]`, `run_pipeline(video_paths: list[Path]) -> int`, `main() -> int`, 모듈 상수 `VENV_PYTHON: Path`, `PIPELINE_SCRIPT: Path`, `RESULT_CLIPS_DIR: Path`.

- [ ] **Step 1: 실패하는 테스트 작성**

`C:\clipAI\files\tests\test_launcher.py` 신규 생성:

```python
# -*- coding: utf-8 -*-
"""launcher.py(드래그앤드롭 exe 실행기) 단위 테스트."""

from __future__ import annotations

from pathlib import Path

import launcher


def test_filter_video_paths_accepts_existing_mp4(tmp_path):
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"")

    valid, skipped = launcher.filter_video_paths([str(v)])

    assert valid == [v.resolve()]
    assert skipped == []


def test_filter_video_paths_skips_non_mp4(tmp_path):
    t = tmp_path / "notes.txt"
    t.write_bytes(b"")

    valid, skipped = launcher.filter_video_paths([str(t)])

    assert valid == []
    assert len(skipped) == 1
    assert "notes.txt" in skipped[0]
    assert "영상 파일 아님" in skipped[0]


def test_filter_video_paths_skips_missing_file(tmp_path):
    missing = tmp_path / "gone.mp4"

    valid, skipped = launcher.filter_video_paths([str(missing)])

    assert valid == []
    assert len(skipped) == 1
    assert "파일 없음" in skipped[0]


def test_filter_video_paths_mixed_inputs(tmp_path):
    good = tmp_path / "good.mp4"
    good.write_bytes(b"")
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"")

    valid, skipped = launcher.filter_video_paths([str(good), str(bad)])

    assert valid == [good.resolve()]
    assert len(skipped) == 1


def test_run_pipeline_invokes_venv_python_with_files_list(tmp_path, monkeypatch):
    v1 = tmp_path / "a.mp4"
    v1.write_bytes(b"")
    captured = {}

    class FakeCompleted:
        returncode = 0

    def fake_run(cmd, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        # subprocess.run에 넘겨진 --files-list 인자의 임시파일 내용을 검증
        list_path = Path(cmd[cmd.index("--files-list") + 1])
        captured["list_contents"] = list_path.read_text(encoding="utf-8")
        return FakeCompleted()

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    rc = launcher.run_pipeline([v1])

    assert rc == 0
    cmd = captured["cmd"]
    assert cmd[0] == str(launcher.VENV_PYTHON)
    assert str(launcher.PIPELINE_SCRIPT) in cmd
    assert "--files-list" in cmd
    assert str(v1) in captured["list_contents"]
    # 임시 목록 파일은 실행 후 정리돼야 함
    list_path = Path(cmd[cmd.index("--files-list") + 1])
    assert not list_path.exists()
```

- [ ] **Step 2: 테스트 실패 확인**

Run (in `C:\clipAI\files`): `python -m pytest tests/test_launcher.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'launcher'`

- [ ] **Step 3: `launcher.py` 구현**

`C:\clipAI\files\launcher.py` 신규 생성:

```python
# -*- coding: utf-8 -*-
"""드래그 앤 드롭 exe 실행기 — PyInstaller로 clipAI_launcher.exe 빌드 대상
(본인 PC 전용, docs/superpowers/specs/2026-07-31-exe-launcher-design.md 참고).

영상 파일을 이 exe(또는 `python launcher.py <파일...>`) 위로 드래그하면
기존 .venv를 그대로 호출해 batch_hud_ace_pipeline.py를 실행한다. torch/
opencv/easyocr는 재포장하지 않는다 — 표준 라이브러리만 사용.
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

VENV_PYTHON = Path(r"C:\clipAI\.venv\Scripts\python.exe")
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

    run_pipeline(valid)

    print(f"\n결과 폴더: {RESULT_CLIPS_DIR}")
    answer = input("여시겠습니까? (Y/N): ").strip().lower()
    if answer == "y":
        os.startfile(RESULT_CLIPS_DIR)

    _wait_for_key()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_launcher.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: 전체 테스트 스위트 회귀 확인**

Run: `python -m pytest tests/ -v`
Expected: 모든 테스트 PASS (기존 스위트 + Task 1, 2 신규 테스트 전부 포함)

- [ ] **Step 6: 커밋**

```bash
git add files/launcher.py files/tests/test_launcher.py
git commit -m "feat: 드래그앤드롭 실행기 launcher.py 추가

영상 파일을 드롭하면 기존 .venv로 batch_hud_ace_pipeline.py를
subprocess 실행. PyInstaller onefile 빌드 대상(다음 태스크)."
```

---

### Task 3: PyInstaller 빌드 + 수동 스모크 테스트 + 문서화

**Files:**
- Modify: `C:\clipAI\.gitignore`
- Modify: `C:\clipAI\README.md`
- Create (빌드 산출물, git 추적 안 함): `C:\clipAI\dist\clipAI_launcher.exe`

**Interfaces:**
- Consumes: `C:\clipAI\files\launcher.py` (Task 2), `VENV_PYTHON`/`PIPELINE_SCRIPT`/`RESULT_CLIPS_DIR` 상수.
- Produces: 없음(최종 산출물 — exe 자체).

- [ ] **Step 1: PyInstaller 빌드 산출물 gitignore 처리**

`C:\clipAI\.gitignore`의 `# Python` 섹션 바로 아래에 추가:

```
# PyInstaller build artifacts (launcher.py -> clipAI_launcher.exe)
/dist/
/build/
*.spec
```

- [ ] **Step 2: gitignore 커밋**

```bash
git add .gitignore
git commit -m "chore: PyInstaller 빌드 산출물(dist/build/*.spec) gitignore 추가"
```

- [ ] **Step 3: PyInstaller 설치**

Run (아무 위치에서): `C:\clipAI\.venv\Scripts\pip.exe install pyinstaller`
Expected: `Successfully installed pyinstaller-...` (이미 설치돼 있으면 `Requirement already satisfied`)

- [ ] **Step 4: exe 빌드**

Run (in `C:\clipAI`): `.venv\Scripts\pyinstaller.exe --onefile --console --distpath dist --workpath build --name clipAI_launcher files\launcher.py`
Expected: 마지막 줄 `completed successfully`, `C:\clipAI\dist\clipAI_launcher.exe` 생성됨.

확인: `Get-ChildItem C:\clipAI\dist\clipAI_launcher.exe` (PowerShell) 또는 `ls -la /c/clipAI/dist/clipAI_launcher.exe` (bash) — 파일 존재 확인. torch/opencv를 import하지 않으므로 크기는 수십 MB 이내여야 함(수 GB면 뭔가 잘못 잡힌 것).

- [ ] **Step 5: 인자 없이 실행 — 안내 메시지 스모크 테스트**

Run: `echo | C:\clipAI\dist\clipAI_launcher.exe`
(파이프로 빈 입력을 줘서 키 입력 대기를 자동 통과시킴 — 자동화 환경에서의 검증용)
Expected 출력에 포함: `영상 파일을 이 exe 위로 드래그하세요.` 그리고 `아무 키나 누르면 닫힙니다...`

- [ ] **Step 6: 존재하지 않는 파일 인자 — 스킵 경로 스모크 테스트**

Run: `echo | C:\clipAI\dist\clipAI_launcher.exe "C:\clipAI\존재안함.mp4"`
Expected 출력에 포함: `건너뜀: 존재안함.mp4 (파일 없음)` 과 `처리할 영상이 없습니다.`

- [ ] **Step 7: 실제 영상 하나로 end-to-end 스모크 테스트**

`E:\OBS`에서 이미 스캔 완료된(캐시 있는) 영상 하나를 골라(예: 아무 `.mp4` 하나) 아래 실행 — 캐시가 있으면 재스캔 없이 즉시 `(cached)`로 끝나므로 빠르게 전체 경로(subprocess 호출 → 결과 폴더 안내 → Y/N)를 검증할 수 있음:

Run: `echo n | C:\clipAI\dist\clipAI_launcher.exe "E:\OBS\<이미 스캔된 영상 파일명>.mp4"`
Expected 출력에 포함: `[hud-batch] OK ... (cached)` 로그 한 줄, 그리고 `결과 폴더: E:\clipai_result\ace_clips_hud`, `여시겠습니까? (Y/N):` 프롬프트. `n` 입력이라 탐색기는 안 열림.

- [ ] **Step 8: README에 사용법 추가**

`C:\clipAI\README.md`의 "## HUD 올킬 파이프라인 (현재 주력)" 섹션이 끝나는 지점(옵션 표 바로 뒤, "### R10" 섹션 시작 전)에 아래 서브섹션 추가:

```markdown
### exe 드래그앤드롭 실행기

`dist\clipAI_launcher.exe`에 영상 파일 여러 개를 드래그하면 위 배치
파이프라인을 그 파일들만 대상으로 돌려준다(본인 PC 전용 — `.venv` 경로가
`launcher.py`에 하드코딩돼 있음). 재빌드가 필요하면:

```powershell
cd C:\clipAI
.venv\Scripts\pip.exe install pyinstaller
.venv\Scripts\pyinstaller.exe --onefile --console --distpath dist --workpath build --name clipAI_launcher files\launcher.py
```

인자 없이 더블클릭하면 안내 메시지만 뜨고 종료된다(실수로 전체 배치가
돌아가는 것을 막기 위함). 전체 배치를 돌리려면 지금처럼
`python batch_hud_ace_pipeline.py`를 직접 실행할 것.
```

- [ ] **Step 9: 커밋**

```bash
git add README.md
git commit -m "docs: README에 exe 드래그앤드롭 실행기 사용법 추가"
```

---

## Self-Review 결과 (계획 작성자 기준)

- **스펙 커버리지**: `--files-list`(스펙 §구성요소 1) → Task 1. `launcher.py` 5단계 로직(스펙 §구성요소 2) → Task 2. PyInstaller 빌드(스펙 §구성요소 3) → Task 3. 에러 처리 표의 5개 케이스(영상 아닌 파일/유효 영상 없음/venv 없음/subprocess 에러/더블클릭만) 모두 `launcher.py` 구현과 테스트에 반영됨. venv 없음 케이스는 단위테스트 대상이 아니라 Task 3 Step 5-7 수동 스모크로 대체 커버(경로 하드코딩값이라 단위테스트로 만들면 실제 파일시스템 상태에 의존하게 되어 오히려 깨지기 쉬움).
- **플레이스홀더 스캔**: "TBD"/"나중에" 류 없음, 모든 스텝에 실행 가능한 코드·명령 포함.
- **타입/시그니처 일관성**: `filter_video_paths`/`run_pipeline`/`VENV_PYTHON`/`PIPELINE_SCRIPT`/`RESULT_CLIPS_DIR` 이름이 Task 2 구현과 테스트 전체에서 동일하게 사용됨. `_load_files_list`도 Task 1 테스트·구현에서 일치.
