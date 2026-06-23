# 라벨링 세팅 (4종 하이라이트)

| label | 한글 | 설명 |
|---|---|---|
| `doublekill` | 더블킬 | 2연속 킬 |
| `multikill` | 멀티킬 | 3연속 이상 킬 |
| `save` | 세이브 | 클러치/세이브 |
| `allkill` | 올킬 | 상대 전원 격파 |
| `background` | (선택) | 일반 플레이 음성 |

---

## 방법 A — 클립 mp4 직접 제공 (권장)

이미 잘라둔 mp4를 **폴더별로** 넣으면 됩니다. 원본 영상·타임스탬프·추출 과정 **불필요**.

### 1) 폴더 생성

```powershell
cd C:\clipAI\files
python setup_labeling_project.py
```

### 2) 클립 넣기

```
E:\Highlights\ml_dataset\clips\
├── doublekill\   ← 더블킬 mp4
├── multikill\    ← 멀티킬 mp4
├── save\         ← 세이브 mp4
├── allkill\      ← 올킬 mp4
└── background\   ← (선택) 일반 플레이 mp4
```

폴더명은 영문(`multikill`) 또는 한글(`멀티킬`) 둘 다 됩니다.

### 3) background 풀영상 분할 (긴 녹화본만)

`background`에 넣은 **풀 녹화본**(수 GB)은 학습 전에 잘라야 합니다.

```powershell
cd C:\clipAI\files
# 계획만 확인
python slice_background.py --dry-run
# 실제 분할 (시간 오래 걸릴 수 있음)
python slice_background.py --skip-existing
```

출력: `clips/background/_chunks/{이름}_part_0001.mp4` (12초)

### 4) 인덱스 생성

```powershell
cd C:\clipAI\files
python scan_clip_folders.py --dataset-root "E:\Highlights\ml_dataset" --allow-overwrite
```

→ `manifests/clips_index.csv` 생성 (학습용 목록)

---

## 방법 B — 원본 영상 + 시각 (선택)

긴 녹화본에서 자동으로 구간을 잘라야 할 때만 사용.

1. `known_highlights.csv`에 `video_path`, `timestamp_sec`, `label` 입력
2. `python build_label_manifest.py --allow-overwrite`
3. `python extract_labeled_clips.py`

---

## 권장 최소 데이터

| 클래스 | 권장 |
|---|---|
| doublekill | 30+ |
| multikill | 30+ |
| save | 20+ |
| allkill | 15+ |
| background | 150+ (선택) |

---

## 6) 학습 (2단계)

```powershell
cd C:\clipAI\files
python train_binary.py
python train_highlight_types.py
```

- **1단계**: 하이라이트 vs background → `models/highlight_binary_best.pt`
- **2단계**: 4종 타입 분류 → `models/highlight_types_best.pt`

하이라이트 클립 정리: [HIGHLIGHT_REVIEW.md](../HIGHLIGHT_REVIEW.md)
