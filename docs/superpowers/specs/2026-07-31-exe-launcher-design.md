# 드래그 앤 드롭 exe 실행기 설계 (2026-07-31)

## 배경

HUD 올킬 파이프라인(`batch_hud_ace_pipeline.py`)은 지금 터미널에서
`python -u batch_hud_ace_pipeline.py` 형태로 수동 실행한다. 본인 PC에서
새로 녹화된 영상 몇 개를 편하게 돌리기 위한 exe 실행기가 필요하다.

- **배포 대상**: 본인 PC 전용 (다른 사람 PC 배포 아님 — GPU/CUDA/ffmpeg/venv
  이미 세팅 완료 상태를 그대로 전제해도 됨)
- **입력 방식**: 영상 파일 여러 개를 exe 아이콘에 드래그 앤 드롭. 파일은
  `E:\OBS`뿐 아니라 임의 위치(복사본, 임시 폴더 등)에서 와도 된다.
- **UI**: 검은 콘솔 창. 기존 `batch_hud_ace_pipeline.py`가 이미 출력하는
  진행 로그(`[hud-batch] OK ...` / `SKIP ...`)를 그대로 스크롤 출력하고,
  끝나면 결과 폴더 경로를 보여준 뒤 열지 여부(Y/N)를 묻고 키 입력 대기 후 종료.
- **드래그 없이 더블클릭만 한 경우**: 아무 작업도 하지 않고 "영상 파일을
  이 exe 위로 드래그하세요" 안내만 표시하고 종료.

## 아키텍처

```
[영상 파일 여러 개를 드래그]
          │
          ▼
  clipAI_launcher.exe  (PyInstaller로 만든 초경량 실행기, 표준 라이브러리만 사용)
          │  드롭된 절대경로들을 임시 txt로 저장
          ▼
  C:\clipAI\.venv\Scripts\python.exe -u batch_hud_ace_pipeline.py --files-list <tmp.txt>
          │  (콘솔 창에 진행 로그 실시간 스트리밍 — subprocess stdout 그대로 중계)
          ▼
  기존 파이프라인 그대로 동작 → E:\clipai_result\ace_clips_hud 에 결과 저장
          │
          ▼
  "결과 폴더: ... (여시겠습니까? Y/N)" → 아무 키나 누르면 종료
```

핵심 설계 원칙: **exe는 얇은 실행기 역할만 한다.** torch·opencv·easyocr
같은 무거운 의존성은 PyInstaller로 재포장하지 않고, 이미 세팅되어 있는
`.venv`를 subprocess로 그대로 호출해서 재사용한다. 이렇게 하면:

- exe 빌드가 표준 라이브러리만 대상이라 빠르고 안정적(수십 MB 수준).
- torch/opencv/easyocr의 PyInstaller hidden-import·CUDA DLL·모델 경로
  문제를 원천적으로 피한다.
- 파이프라인 코드 변경 시 exe를 다시 빌드할 필요 없음(launcher와
  파이프라인이 완전히 분리되어 있으므로).

## 구성 요소

### 1. `batch_hud_ace_pipeline.py` — `--files-list` 옵션 추가

현재 CLI는 `E:\OBS` 폴더 글롭 + `--only`/`--after`/`--stems-file`(모두
OBS_DIR 내부 stem 이름 기준)만 지원한다. 드롭된 파일이 OBS 폴더 밖에
있어도 되어야 하므로, 절대경로 목록을 한 줄에 하나씩 담은 텍스트 파일을
받아 그 경로들을 `videos` 리스트로 직접 사용하는 옵션을 추가한다.

- 새 인자: `--files-list <path>` — 지정 시 `--obs-dir` 글롭을 건너뛰고
  파일에 적힌 절대경로들을 그대로 `videos`로 사용.
- 기존 `--only`/`--after`/`--limit`/`--stems-file`/`--redo` 등은 전부
  기존 동작 그대로 유지(가산적 변경, 회귀 없음).
- 존재하지 않는 경로나 `.mp4`가 아닌 파일은 이 시점에서 걸러내고
  스킵 사유를 출력.

### 2. 신규 `launcher.py`

- `sys.argv[1:]`로 드롭된 파일 경로를 받는다.
- 인자가 없으면 안내 메시지만 출력하고 키 입력 대기 후 종료(파이프라인
  미실행).
- 인자가 있으면:
  1. 각 경로를 절대경로로 정규화.
  2. 확장자가 `.mp4`가 아닌 파일은 걸러내고 "건너뜀: <파일명> (영상 파일
     아님)" 출력 — 기존 `batch_hud_ace_pipeline.py`가 `obs.glob("*.mp4")`로
     `.mp4`만 대상으로 삼는 것과 동일한 기준.
  3. 남은 경로들을 임시 txt 파일(스크래치 경로)에 한 줄씩 기록.
  4. `C:\clipAI\.venv\Scripts\python.exe -u batch_hud_ace_pipeline.py
     --files-list <tmp.txt>` 를 subprocess로 실행, stdout/stderr를
     실시간으로 그대로 콘솔에 중계.
  5. 종료 코드와 무관하게(성공/실패 모두) 마지막에 결과 폴더 경로
     (`E:\clipai_result\ace_clips_hud`)를 보여주고 여시겠습니까(Y/N) 질문.
     Y면 `os.startfile()`로 탐색기 오픈.
  6. 마지막에 "아무 키나 누르면 닫힙니다" 로 대기 후 종료 — cmd 창이
     결과를 볼 새도 없이 바로 꺼지는 것을 방지.

### 3. PyInstaller 빌드

- `launcher.py`만 대상으로 `pyinstaller --onefile launcher.py` 빌드.
- 표준 라이브러리(`subprocess`, `pathlib`, `tempfile`, `os`, `sys`)만
  사용하므로 hidden-import 이슈 없이 그대로 빌드됨.
- 산출물 `launcher.exe`를 `clipAI_launcher.exe`로 이름 변경해 바탕화면
  등 편한 곳에 배치(드래그 타겟).

## 에러 처리

| 상황 | 처리 |
|------|------|
| 드롭 파일 중 영상 아닌 파일 포함 | 확장자로 필터링, 스킵 사유 출력 후 나머지 계속 진행 |
| 유효한 영상 파일이 하나도 없음 | "처리할 영상이 없습니다" 출력 후 키 입력 대기, 파이프라인 미실행 |
| `.venv\Scripts\python.exe` 없음 | 명확한 에러 메시지 출력(경로 포함) 후 키 입력 대기 |
| 파이프라인 subprocess가 예외/에러 종료 | 에러 메시지를 그대로 콘솔에 중계, 창 자동으로 안 닫힘(키 입력 대기까지 유지) |
| 더블클릭만(드래그 없음) | 안내 메시지만 출력, 파이프라인 미실행 |

## 테스트 계획

- 이미 처리된 영상 + 신규 영상 섞어서 드래그 → 신규만 처리되는지 확인
  (기존 `process_video()`의 캐시 재사용 로직 그대로 적용됨).
- 영상 아닌 파일(예: `.txt`) 하나 섞어서 드래그 → 스킵 로그 확인.
- `E:\OBS` 밖의 임의 폴더에 있는 영상 드래그 → 정상 처리 확인.
- 드래그 없이 exe 더블클릭 → 안내 메시지만 뜨는지 확인.
- 처리 도중 강제로 에러 유발(예: 존재하지 않는 파일을 txt에 심어보는 등) →
  창이 즉시 닫히지 않고 에러가 보이는지 확인.
- 결과 폴더 열기 Y/N 프롬프트 양쪽 다 확인.

## 범위 밖 (이번에 하지 않는 것)

- 다른 PC 배포용 풀 번들링(PyInstaller onefile에 torch/opencv/ffmpeg까지
  전부 포함) — 본인 PC 전용이므로 불필요.
- GUI(Tkinter 등) 창 — 콘솔 창으로 충분하다고 결정.
- 드래그 없이 더블클릭 시 전체 `E:\OBS` 배치 스캔 자동 실행 — 실수로
  전체 스캔이 걸리는 걸 방지하기 위해 안내 메시지만 표시하는 것으로 결정.
