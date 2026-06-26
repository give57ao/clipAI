# clipAI — 서든어택 하이라이트 자동 추출

OBS 녹화에서 **더블킬·멀티킬·세이브·올킬** 하이라이트를 ML로 탐지합니다.

## 설치

```powershell
git clone https://github.com/give57ao/clipAI.git
cd clipAI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

ffmpeg, ffprobe가 PATH에 있어야 합니다.

**4종 하이라이트 라벨링 + 학습** 방식으로 진행합니다.

| label | 한글 |
|-------|------|
| `doublekill` | 더블킬 |
| `multikill` | 멀티킬 |
| `save` | 세이브 |
| `allkill` | 올킬 |

## 라운드 기반 파이프라인 (권장, 2026-06-26~)

```powershell
python -u detect_rounds.py "E:\OBS\<영상>.mp4"
python -u scoreboard_k_reader.py "E:\OBS\<영상>.mp4" --rounds-dir "E:\Highlights\ml_dataset\rounds\<stem>" --json-out "E:\clipai_result\kill_timeline\<stem>.json"
```

상세: [HANDOFF.md](HANDOFF.md), [PLAYER_IDENTITY_AND_K_READER.md](PLAYER_IDENTITY_AND_K_READER.md)

## 가장 쉬운 방법 (클립 라벨링)

이미 잘라둔 mp4를 폴더에 넣기만 하면 됩니다.

```powershell
cd C:\clipAI\files
python setup_labeling_project.py
# E:\Highlights\ml_dataset\clips\{doublekill,multikill,save,allkill}\ 에 mp4 복사
python scan_clip_folders.py --allow-overwrite
```

상세: [LABELING_SETUP.md](LABELING_SETUP.md)

## 데이터셋 경로

`E:\Highlights\ml_dataset\`

## 스크립트

| 파일 | 역할 |
|------|------|
| `setup_labeling_project.py` | 폴더 초기 세팅 |
| `scan_clip_folders.py` | 클립 폴더 스캔 → `clips_index.csv` |
| `slice_background.py` | background 풀영상 → 12초 청크 분할 |
| `train_game_roi.py` | **0단계** 게임 화면 ROI 추론 모델 |
| `game_roi.py` | ROI 추론/크롭 공통 모듈 |
| `train_binary.py` | **1단계** 하이라이트 vs background |
| `train_highlight_types.py` | **2단계** 4종 타입 분류 |
| `infer_highlights.py` | 녹화본 슬라이딩 윈도우 추론 + 클립 추출 |
| `batch_infer_obs.py` | OBS 폴더 배치 추론 (`MAX_VIDEOS`로 파일럿 제한 가능) |
| `build_label_manifest.py` | (선택) 원본 영상+시각 → 구간 생성 |
| `extract_labeled_clips.py` | (선택) 구간 클립 추출 |
| `labeling_constants.py` | 4종 라벨 정의 |

## 학습 (3단계)

```powershell
cd C:\clipAI\files
python -u train_game_roi.py
python -u train_binary.py
python -u train_highlight_types.py
```

- **0단계**: 게임 화면 ROI (`game_roi_best.pt`) — 방송/전체화면 레이아웃 자동 크롭
- **1단계**: 하이라이트 vs background (`highlight_binary_best.pt`)
- **2단계**: 4종 타입 분류 (`highlight_types_best.pt`)

## 추론 (OBS 녹화본)

```powershell
cd C:\clipAI\files
# 구간만 확인 (클립 추출 없음)
python -u infer_highlights.py "D:\OBS\recording.mp4" --dry-run
# 클립 추출
python -u infer_highlights.py "D:\OBS\recording.mp4" --output-dir "E:\Highlights\inferred\run1"
```

옵션: `--window-sec 12`, `--stride-sec 6`, `--binary-threshold 0.55`, `--type-threshold 0.35`
