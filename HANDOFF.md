# clipAI 세션 핸드오프

> 마지막 업데이트: 2026-06-23 (KST 밤)  
> 다음 채팅에서 이 파일을 먼저 읽고 이어서 작업하세요.

## 사용자 결정 (중요)

1. **오늘은 파일럿만**: OBS 82개 전체가 아니라 **상위 3개 mp4만** 배치 추론 후 수동 검토.
2. **2단계 타입 분류는 보류**: `train_highlight_types.py` / 타입 추론은 당분간 사용 안 함. **`--binary-only` 추론만**.
3. **전체 배치는 피드백 후**: 3개 결과 품질 확인 → threshold/stride 조정 → `MAX_VIDEOS=0`으로 82개 실행.

## 파이프라인 현황

| 단계 | 스크립트 | 상태 | 산출물 |
|------|----------|------|--------|
| 0 ROI | `train_game_roi.py` | **완료** | `game_roi_best.pt`, `game_roi_meta.json` |
| 1 binary | `train_binary.py` | **ROI GPU-batch 재학습 중** (epoch 10/10 진행) | `highlight_binary_best.pt` (갱신 중) |
| 2 types | `train_highlight_types.py` | **보류** (체인에 묶여 있으나 사용자 미관심) | — |
| OBS 추론 | `batch_infer_obs.py` | **대기 중** (학습 완료 후 3개만 실행) | `E:\clipai_result\` |

### binary 재학습 로그 (ROI crop, batch-size 24)

- 로그: `E:\Highlights\ml_dataset\models\train_binary_roi.log`
- 설정: `--epochs 10 --batch-size 24 --num-workers 4`, `game_roi=gpu-batch`
- 데이터: highlights=188, background_pool=3925
- epoch 9/10: loss=0.0049, val_acc=95.4%, highlight_recall=100%
- epoch 10/10 진행 중 (로그 마지막 줄 확인)

### 배치 추론 (파일럿 3개)

- 스크립트: `C:\clipAI\files\batch_infer_obs.py`
- `MAX_VIDEOS = 3`, `PARALLEL = 3` → **1배치(3개 동시) 후 종료**
- `train_binary.py` 프로세스가 없어질 때까지 30초 간격 대기 후 시작
- 처리 대상 (파일명 정렬 상위 3개):
  1. `2026-03-19 23-00-50.mp4`
  2. `2026-03-19 23-13-48.mp4`
  3. `2026-03-21 00-40-56.mp4`
- 출력: `E:\clipai_result\{stem}_하이라이트\`
- 로그: `E:\clipai_result\_logs\{stem}.log`, 마스터: `_batch_master.log`

**재시작 필요**: 기존 `batch_infer_obs.py`는 82개 전체 실행 코드였음.  
`MAX_VIDEOS=3` 반영 후 프로세스를 **kill → 재시작** 해야 함.

## 경로 요약

| 용도 | 경로 |
|------|------|
| 코드 | `C:\clipAI\files\` |
| 데이터셋 | `E:\Highlights\ml_dataset\` |
| 모델 | `E:\Highlights\ml_dataset\models\` |
| OBS 원본 (82 mp4) | `E:\OBS\` |
| 추론 결과 | `E:\clipai_result\` |

## 명령어

### binary 학습 (ROI, GPU batch) — 현재 실행 중이면 생략

```powershell
cd C:\clipAI\files
python -u train_binary.py --epochs 10 --batch-size 24 --num-workers 4 2>&1 `
  | Tee-Object -FilePath E:\Highlights\ml_dataset\models\train_binary_roi.log
```

### 파일럿 배치 추론 (3개, 학습 완료 대기 포함)

```powershell
cd C:\clipAI\files
python -u batch_infer_obs.py 2>&1 | Tee-Object -FilePath E:\clipai_result\_batch_master.log
```

### 단일 영상 수동 추론 (binary-only)

```powershell
cd C:\clipAI\files
python -u infer_highlights.py "E:\OBS\2026-03-19 23-00-50.mp4" `
  --dataset-root E:\Highlights\ml_dataset `
  --output-dir E:\clipai_result\2026-03-19` 23-00-50_하이라이트 `
  --binary-only --stride-sec 8 --binary-threshold 0.55
```

### 전체 82개 실행 (피드백 후)

`batch_infer_obs.py`에서 `MAX_VIDEOS = 0`으로 변경 후 동일 명령 실행.

## 알려진 이슈

- `batch_infer_obs.ps1`: PowerShell 인코딩/문자열 파싱 오류로 실패 → **Python 스크립트 사용 권장**
- `batch_infer_obs.ps1`에도 `-MaxVideos 3` 파라미터 추가됨 (인코딩 수정 전까지 py 우선)
- GPU VRAM 8GB: `PARALLEL=3` 유지. OOM 시 `PARALLEL=2`로 낮추기
- binary 체인 터미널이 types 학습까지 이어질 수 있음 — 사용자는 types 무시해도 됨

## 다음 채팅 TODO

1. `train_binary_roi.log` tail — epoch 10/10 완료 및 `highlight_binary_best.pt` 갱신 확인
2. `batch_infer_obs.py` 실행 중이면 `_batch_master.log` / `_logs\*.log` 모니터링
3. 3개 결과 폴더(`E:\clipai_result\*_하이라이트`) 수동 검토 — 하이라이트 품질, false positive
4. 필요 시 `--binary-threshold`, `--stride-sec` 조정 후 재추론
5. 만족 시 `MAX_VIDEOS=0`으로 전체 OBS 배치
6. (선택) README에 ROI 단계·배치 스크립트 문서화

## 워크플로 (확정)

```
[0] train_game_roi.py     → 완료
[1] train_binary.py       → 재학습 완료 대기 (~epoch 10/10)
[2] batch_infer_obs.py    → 3개 파일럿만 (binary-only)
[3] 사용자 피드백 루프
[4] MAX_VIDEOS=0 전체 82개 (별도 세션)
```

types(2단계)는 현재 범위 밖.
