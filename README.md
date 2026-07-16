# clipAI — 서든어택 하이라이트 자동 추출

OBS 녹화에서 **올킬** 하이라이트를 HUD 판독으로 탐지합니다.
(더블킬·멀티킬·세이브 4종 ML 분류는 아래 "레거시" 절 참고 — 현재 주력은 HUD 올킬 파이프라인입니다.)

## 설치

```powershell
git clone https://github.com/give57ao/clipAI.git
cd clipAI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

ffmpeg, ffprobe가 PATH에 있어야 합니다.

## ⚠ 배치 재스캔은 병렬 금지

**단일 프로세스, 최대 2-way까지만.** 6-way 병렬 재스캔 시 여러 디코더가 E: 드라이브를
동시에 때려 ffmpeg `Stream timeout`이 발생하고, read 실패가 조용히 잘못된 판독으로
이어져 **측정 결과 전체가 오염**됩니다 (2026-07-16 실측). 속도보다 측정 신뢰가 우선입니다.

## HUD 올킬 파이프라인 (현재 주력)

닉네임·스코어보드 판독 없이 HUD만으로 올킬을 탐지합니다.

```powershell
cd C:\clipAI\files

# E:\OBS 전체 배치
python -u batch_hud_ace_pipeline.py

# 일부만 / 특정 영상만
python -u batch_hud_ace_pipeline.py --limit 5
python -u batch_hud_ace_pipeline.py --only "2026-03-21 00-40-56"

# 재처리 + JSON만 (클립 추출 생략)
python -u batch_hud_ace_pipeline.py --only "<영상>" --redo --no-extract

# 산출물을 별도 루트로 (평가·실험용)
python -u batch_hud_ace_pipeline.py --output-root E:\clipai_result\_r10_eval
```

| 옵션 | 기본값 | 역할 |
|------|--------|------|
| `--limit N` | 0 (전체) | 처리할 영상 수 제한 |
| `--only <stem>` | – | 특정 영상만 |
| `--after <stem>` | – | 해당 영상 이후만 |
| `--redo` | off | 기존 산출물 무시하고 재처리 |
| `--no-extract` | off | JSON만 생성, 클립 추출 생략 |
| `--scan-fps` | 4.0 | HUD 스캔 프레임레이트 |
| `--min-duration-sec` | 120.0 | 이보다 짧은 영상 건너뜀 |
| `--obs-dir` | `E:\OBS` | 입력 폴더 |
| `--output-root` | `E:\clipai_result` | 산출물 루트 재지정 |
| `--verify-boundary-wins` | **off** | R10 승수 교차검증 게이트 (아래 참고) |

### R10 — 승수 교차검증 게이트 (기본 비활성)

라운드 경계를 승수(score) 변화로 교차검증하는 게이트입니다. **기본 꺼져 있으며,
효과는 아직 측정 중**입니다. 상세는 `detect_ace_hud.py`의 R10 주석 참고.

### 현재 성능 (2026-07-16 클린 베이스라인)

GT 107건 기준. 베이스라인: `files/_tp_baselines/r10_cleanbase.json`

| 지표 | 값 |
|------|-----|
| recall | 72.0% (77/107) |
| precision | 80.2% (77/96) |

성능 비교는 `_tp_diff.py`로:

```powershell
python -u _tp_diff.py --compare-to r10_cleanbase
```

## 데이터 경로

| 경로 | 내용 |
|------|------|
| `E:\OBS` | 원본 녹화 (배치 입력) |
| `E:\clipai_result` | 산출물 루트 (hud_timeline, kill_timeline, ace_clips 등) |
| `E:\Highlights\ml_dataset` | ML 데이터셋 |

## 주요 스크립트

### HUD 파이프라인

| 파일 | 역할 |
|------|------|
| `batch_hud_ace_pipeline.py` | **배치 진입점** — OBS 폴더 전체 HUD 올킬 탐지 |
| `detect_ace_hud.py` | 올킬 탐지 코어 (HUD 스캔 → 라운드 트래킹 → 클립 추출) |
| `hud_kda.py` | HUD K/D/A 판독 |
| `hud_score_wins.py` | 승수 판독 |
| `hud_round_end.py` / `hud_round_settle.py` | 라운드 종료·정산 판정 |
| `hud_boundary_verify.py` | CNN 라운드 경계 검증기 |
| `hud_digit_match.py` / `train_hud_digit_cnn.py` | 숫자 판독 / 숫자 CNN 학습 |
| `hud_sig_cache.py` / `hud_from_cache.py` | 시그니처 캐시 |
| `extract_ace_clips.py` | 올킬 구간 클립 추출 |

### 검증·진단 도구 (`_` 접두)

| 파일 | 역할 |
|------|------|
| `_tp_diff.py` | 베이스라인 대비 TP 획득/상실 비교 |
| `_compare_hud_gt.py` | GT 대조 |
| `_miss_diag.py` | 미탐 원인 분류 |
| `_audit_boundaries.py` | 라운드 경계 감사 |
| `_ace_verifier.py` / `_verify_hud_aces.py` | 올킬 판정 검증 |
| `_extract_miss_clips.py` | 미탐 구간 클립 추출 |

## 레거시 — 라운드 기반 스코어보드 판독

닉네임·스코어보드 OCR 기반. HUD 파이프라인 이전 방식입니다.

```powershell
python -u detect_rounds.py "E:\OBS\<영상>.mp4"
python -u scoreboard_k_reader.py "E:\OBS\<영상>.mp4" --rounds-dir "E:\Highlights\ml_dataset\rounds\<stem>" --json-out "E:\clipai_result\kill_timeline\<stem>.json"
```

## 레거시 — 4종 ML 분류 파이프라인

| label | 한글 |
|-------|------|
| `doublekill` | 더블킬 |
| `multikill` | 멀티킬 |
| `save` | 세이브 |
| `allkill` | 올킬 |

> 세이브 감지는 종결됨(2026-07-15 결정) — 올킬이 아닌 세이브는 실질적으로 거의 없음.

**라벨링**

```powershell
cd C:\clipAI\files
python setup_labeling_project.py
# E:\Highlights\ml_dataset\clips\{doublekill,multikill,save,allkill}\ 에 mp4 복사
python scan_clip_folders.py --allow-overwrite
```

**학습 (3단계)**

```powershell
python -u train_game_roi.py          # 0단계: 게임 화면 ROI (game_roi_best.pt)
python -u train_binary.py            # 1단계: 하이라이트 vs background
python -u train_highlight_types.py   # 2단계: 4종 타입 분류
```

**추론**

```powershell
python -u infer_highlights.py "E:\OBS\recording.mp4" --dry-run
python -u infer_highlights.py "E:\OBS\recording.mp4" --output-dir "E:\Highlights\inferred\run1"
```

옵션: `--window-sec 12`, `--stride-sec 6`, `--binary-threshold 0.55`, `--type-threshold 0.35`

## 문서

| 파일 | 내용 |
|------|------|
| [HANDOFF.md](HANDOFF.md) | **세션 핸드오프 — 작업 재개 시 먼저 읽을 것** |
| [HUD_ACE_HANDOFF.md](HUD_ACE_HANDOFF.md) | HUD 올킬 파이프라인 상세 |
| [PLAYER_IDENTITY_AND_K_READER.md](PLAYER_IDENTITY_AND_K_READER.md) | 닉네임 식별 / K 판독 (레거시) |
| [LABELING_SETUP.md](LABELING_SETUP.md) | 클립 라벨링 세팅 |
| [MISS_FEEDBACK_FORM.md](MISS_FEEDBACK_FORM.md) | 미탐 피드백 양식 |
