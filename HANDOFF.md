# clipAI 세션 핸드오프

> 마지막 업데이트: 2026-07-10 (KST)  
> 다음 채팅에서 이 파일을 먼저 읽고 이어서 작업하세요.

## ★ HUD 올킬 파이프라인 (2026-07-10 R5 완료 — recall 54.6%→66.7%)

**닉·스코어보드 없이** HUD K/D/A로 올킬 탐지. 사용자 육안 검수로 GT를
**60영상/96건**으로 확장하고, R5에서 근본 원인 3개를 규명·수정.

- **현재 성능**: **recall 66.7% (64/96), precision 71.1%** — 라벨 10영상 튜닝
  수치(88.5%)가 일반화 안 된다는 착시를 GT 확장으로 규명하고, 진짜 baseline
  54.6%에서 +12.1%p 끌어올림. 회귀 0(`_tp_diff` r5_final = 64/96).
- **R5 핵심 수정 3건** (상세: [`HUD_ACE_HANDOFF.md`](HUD_ACE_HANDOFF.md) §0):
  1. **전광판 CNN 경계검증을 실배치에 연결** — R2 개선이 `scan_hud_aces`에
     한 번도 안 붙어있던 공백. MULTI KILL 연출이 라운드를 쪼개던 문제 해소.
  2. **"스퓨리어스 0" = 화면의 8 오독 규명** — `hud_round_settle._quarantine_zeros`
     (K 단조성 도메인 규칙으로 가짜 0 격리). 만성 K체인 파괴 원인.
  3. **D/A 채널 기록 + D-가드 + gap 킬 시각 보정** (사망/관전 오독, GT창 이탈 방지).
- **코드**: `detect_ace_hud.py`, `hud_round_settle.py`(정산 디코더), `hud_boundary_verify.py`,
  `hud_from_cache.py`, `_compare_hud_gt.py`(GT 단일 진처, 96건), `_dump_reads_window.py`(구간 판독 덤프),
  `_reorg_highlights.py`/`_delete_stale_clips.py`/`_extract_miss_clips.py`(클립 관리)
- **하위 AI 작업 명세**: [`SONNET_TASK.md`](SONNET_TASK.md) R5 절 (덤프 진단→재캐시→재스캔→게이트)
- **클립 폴더**: `E:\clipai_result\ace_clips_hud\` 평탄화(`<날짜>_하이라이트(n).mp4`),
  오탐 `_오탐\`, 미탐 검수용 `_miss_review\`(43건)
- **다음 (남은 숙제)**:
  1. **E:\OBS 재처리 미완** — R5 수정 이전 stale JSON **75개** 재스캔 중단 상태
     (todo 리스트: `scratchpad/obs_todo.txt`). `--redo`로 이어서 돌리면 됨.
  2. **폭0 FP 분리** — "같은 순간 3킬"은 진짜/가짜 K채널 단독 분리 불가(재확인).
     D채널 활용이 다음 카드(캐시에 D 저장됨).
  3. **경계 넘는 미관측 첫 킬** (02-55-36 11→14 유형), 세이브 감지, D드라이브 162개.

## 보고·도메인 규칙 (2026-06-26 확정)

### 시각 표기
- **초(s) 단독 사용 금지** — 보고·로그·문서는 **`M:SS`** (예: `3:37`, `29:01`)
- 코드 내부·CSV는 `start_sec` float 유지, **사람에게 보여줄 때만** 분:초 변환

### ΔK (라운드 간 킬 증가량) — 절대 규칙
- 서든어택 한 라운드에서 본인이 낼 수 있는 킬은 **최대 3** (올킬)
- 따라서 **정상 ΔK ∈ {0, 1, 2, 3}**
- **ΔK ≥ 4 → OCR/K 읽기 오류 또는 팀원 K 오귀속** — 올킬로 취급하면 안 됨
- ~~현재 `compute_delta_kills`는 `delta >= 3`으로 ace 판정~~ → **2026-06-26 수정 완료**: `delta == 3`만 올킬, `delta >= 4`는 `k_read_error` 플래그
- 올킬 판정은 **`ΔK == 3`만** (또는 도메인상 3킬 확정 시)
- **추론(팀폴백 `[T]`) 라운드는 올킬 판정에서 제외** — 서로 다른 팀원 K를 비교해 거짓 ΔK 생성 (R03 오탐 원인)

### ★ K 컬럼 OCR 버그 (2026-06-26 발견·수정) — 가장 중요

- **증상**: 괴명 실제 `K=0 D=3 A=0`인데 K를 **3(데스값)으로 오독** → 거짓 올킬(R52 등)
- **원인**: `_RED_K=(0.400,0.452)` ROI가 **K와 D 두 숫자를 모두 포함**, `_ocr_best`가 신뢰도 높은 D 숫자를 반환
- **수정**:
  - K 단독 ROI로 재보정: `_RED_K=(0.407,0.430)`, `_BLUE_K=(0.724,0.748)` (픽셀 측정 검증)
  - 단일 숫자(특히 0) 인식용 `_ocr_digit()` 추가 — Otsu 이진화 + allowlist (`scoreboard_layout.py`)
  - 검증: 괴명/희령/massacre/길증경/chal 전원 K conf 1.00 정확
- ⚠️ 처음엔 "WIN 화면이 스코어보드로 오분류" 로 오진했으나, 윈도우엔 진짜 스코어보드가 들어있었고 **K컬럼 위치 오류**가 진짜 원인이었음

## 1b 구현 현황 (2026-06-26)

### 완료 모듈

| 모듈 | 역할 | 상태 |
|------|------|------|
| `player_identity.py` | 영상 단위 닉 확정 (스카우터 + 스코어보드 투표) | ✅ v1 |
| `scoreboard_layout.py` | Clan Match 6행 ROI, 닉/K OCR 전처리 | ✅ |
| `nick_fuzzy.py` | 닉 정규화·fuzzy 매칭·클러스터 | ✅ |
| `scoreboard_k_reader.py` | 확정 닉으로 K 읽기, ΔK, 올킬 후보 | ✅ v1 |
| `detect_rounds.py` | 스코어보드 CNN 스캔 → 라운드 분할 | ✅ (순차 디코드) |
| `game_frame.py` | game_roi crop | ✅ |

### 핵심 개선 (이번 세션)

1. **닉 OCR**: `_ocr_nick` — 반전 이미지 우선 + CLAHE 폴백 (`scoreboard_layout.py`)
2. **fuzzy 점수**: `nick_match_score` = `1 - dist/max_len` (정규화 편집거리)
3. **fuzzy 허용**: 6글자 이상 닉 편집거리 3까지 (`nick_fuzzy.py`)
4. **팀 안전장치**: 2-pass pre-pass로 전역 `team_lock` 선수집 → 실패 라운드만 `[T]` 폴백 (`inferred_by_team` 플래그)
5. **K cross-frame**: 닉 매칭 프레임과 K 읽힌 프레임이 달라도 동일 `row_index`에서 K 보정
6. **detect_rounds**: 긴 영상 seek 병목 → **순차 디코드 + N프레임 샘플**로 교체

### 파일럿 결과

#### `2026-03-19 23-00-50` (~12분, 닉 `overclock`)

| 항목 | v2 (이전) | v5 (현재) |
|------|-----------|-----------|
| 직접 매칭 | 2/8 | **6/8** |
| 안전장치 `[T]` | 4~5 | **1** (R01) |
| `ace_rounds` | 잘못된 양성 | **[]** (이 구간 올킬 없음) |

출력: `E:\clipai_result\kill_timeline\2026-03-19 23-00-50_v5.json`

#### `2026-03-21 00-40-56` (~81분, 닉 `괴명`) — 장시간 파일럿

| 항목 | v1 (구) | **v3 (현재, K컬럼 수정)** |
|------|---------|--------------------------|
| `ace_rounds` | 12개 (대부분 오탐) | **2개** (R36, R73) |
| `k_error_rounds` | 0 | **3** (R30 ΔK=10, R38 ΔK=20, R75 ΔK=90) |

확정 올킬 (직접매칭 conf 1.00, 킬피드 육안 검증 완료):

| 라운드 | 시각(M:SS) | ΔK | 검증 |
|--------|-----------|-----|------|
| R36 | 38:02 | 3 (0→3) | 킬피드 `괴명→맘스터처…` ✓ WIN |
| R73 | 72:44 | 3 (2→5) | 킬피드 `괴명→길종경`(권총) ✓ WIN |

**제거된 오탐**: R52(53:08)는 실제 `K=0 D=3 A=0`인데 D를 K로 오독해 거짓 올킬이었음 → 수정 후 K=0, ΔK=0. R03도 팀폴백 오귀속으로 거짓 ΔK=3 → 수정 후 ΔK=1.

출력: `E:\clipai_result\kill_timeline\2026-03-21 00-40-56_v5.json` (v5 = 결합 OCR + 인접성 규칙)
클립: `E:\clipai_result\ace_clips\2026-03-21 00-40-56\` (R36, R73 — 라운드 플레이 35초 + 확인 스코어보드)
ROI 검증 샘플: `E:\clipai_result\roi_check\` (랜덤 R05/R11/R74 — K셀 크롭 + 샘플 클립)

#### `2026-03-19 23-00-50` (~12분, 닉 `overclock`) — v7

`ace_rounds=[]` (이 구간 올킬 없음, 정상)

### 알려진 이슈 (다음 작업)

1. ~~**ΔK 상한 미적용**~~ → ✅ `==3` ace, `>=4` k_read_error
2. ~~**K 컬럼 OCR 오독**~~ → ✅ K 단독 ROI + Otsu 이진화
3. ~~**안전장치 오귀속**~~ → ✅ 추론(`[T]`) 라운드는 ace 판정 제외
4. ~~**1c 미구현**~~ → ✅ `extract_ace_clips.py` (라운드 인지형 클립)
5. ~~**리포트 시각 M:SS**~~ → ✅ `sec_to_mss`
6. ~~**2자리 K 오독**~~ → ✅ 원인=**빨간 킬바**가 하이라이트(1위/본인) 행 숫자에 겹침 + K 누적 2자리. `_ocr_digit`을 **Otsu + 흰글씨분리(HSV) 결합**으로, ROI 2자리 폭(`0.406~0.432`)으로 수정. 괴명 0/3/7/10 전부 conf 1.00
7. ~~**라운드 건너뛴 거짓 ace**~~ → ✅ **인접성 규칙**: ace는 직전 유효 K가 "바로 앞 라운드"일 때만 (중간 읽기실패 시 ΔK가 여러 라운드 합산 → R55 거짓 제거)
8. **연속 라운드 읽기 실패** — 일부 구간 닉 매칭 실패 (인코딩/조명). 상대(길증경 등) 하이라이트 행 K는 간헐 None (본인 K는 견고)
9. **detect_rounds 인코딩** — stdout UTF-8 reconfigure 미적용 (한글 깨짐, 기능엔 무영향)
10. **결합 OCR 속도** — K셀당 OCR 2회로 긴 영상 느림. 델타 로직만 바꿀 땐 기존 JSON 재계산으로 OCR 생략 가능

### 명령어 (1b 파이프라인)

```powershell
cd C:\clipAI\files

# 1a 라운드 분할 (긴 영상 ~30분+)
python -u detect_rounds.py "E:\OBS\<영상>.mp4" --dataset-root "E:\Highlights\ml_dataset"

# 1b 닉 확정 + K 읽기
python -u scoreboard_k_reader.py "E:\OBS\<영상>.mp4" `
  --rounds-dir "E:\Highlights\ml_dataset\rounds\<stem>" `
  --dataset-root "E:\Highlights\ml_dataset" `
  --json-out "E:\clipai_result\kill_timeline\<stem>_v1.json"
```

디버그: `_debug_roi.py`, `_visualize_roi.py`, `_debug_preprocess.py`, `_debug_k_col.py`

---

- **1차**: 전체스코어 기준 라운드에서 **올킬** 잡기 (라운드 분할 = 완료, 킬 집계 = 진행)
- **2차**: 잡은 올킬 내 더블킬/멀티킬/세이브 콤보 분류 (clips types 모델 보유)
- **3차**: 킬 자체 학습도 올려 더블킬 직접 탐지

## B안: 스카우터 패널 닉네임/킬 추적 (메인, 2026-06-25)

> 닉네임은 수시로 바뀌므로 B안(스카우터 패널 본인 행 추적)을 **메인**으로 한다.

### scouter_nick.py — 검증 완료 (영상 4개)

방송 레이아웃 2종 + 스카우터 모드 2종을 처리:

1. **레이아웃 감지** `detect_game_width()`
   - 후원패널형: 우측 ~27% 검은 영역 → 게임 폭 ≈ 73% (예 2026-03-26: 1405px)
   - 풀스크린형: 게임이 전체 (예 뮤크퀵, 05-18, 06-14)
   - **이전 OCR 실패 원인 = 이 레이아웃 무시하고 전체폭 고정좌표로 crop → 후원패널(검은영역) 긁음**
2. **패널 crop** = 게임 폭 기준 우하단 (x 0.78~0.995, y 0.61~0.80)
3. **헤더 판별**
   - `스카우터2 (L)` → **맨 위 행 = 본인**
   - `스카우터 (L)` → **점(●) 행 = 본인** (닉이 `null`인 경우도 있음 — 널값 아님, 진짜 닉이름)
4. 특수문자 닉(◇¤깜띸겅쥬¤◇) 가능 → OCR 약점

### 검증 결과

| 영상 | 정답 | 레이아웃 | 모드판별 | 닉추출 |
|------|------|----------|----------|--------|
| 뮤크퀵 | 스2 / 단호 | 풀 | ✓ | **단호 conf1.00** ✓ |
| 2026-03-26 | 스2 / ◇¤깜띸겅쥬¤◇ | 후원 | ✓ | 특수문자 OCR 실패 |
| 2026-05-18 | 스1 / 닉 `null` | 풀 | ✓ scouter | 점 행 = 본인 |
| 2026-06-14 | 스1 / 닉 `null` | 풀 | ✓ scouter | 점 행 = 본인 |

- **레이아웃·패널·모드 판별 = 4/4 성공**
- **스카우터2 일반 닉 = 성공** (단호)
- 한계: ① 특수문자 닉 OCR ② 영문 짧은 닉(`null`)은 OCR과 코드 null 혼동 주의

### 핵심 인사이트 (킬 집계 관점) — **2026-06-26 확정**

- **스카우터2 패널**: 맨 위 행 = 본인 (플레이 중) → **신원(닉) 확보용**
- **Clan Match 전체스코어**: **킬 순으로 행 재정렬** (사용자 확인) → 맨 위 ≠ 본인
- 따라서 킬 집계 흐름:
  1. 라운드 중 스카우터2 **맨 위** 또는 스카우터 **점(●) 행**에서 본인 닉 확보
  2. 전체스코어 프레임에서 **닉 매칭으로 본인 행 찾기** (퍼지 매칭)
  3. 해당 행 **K 숫자** 읽기 → 라운드 간 **ΔK**
- 닉 OCR은 "보조"가 아니라 **전체스코어에서 행을 찾는 데 필수** (위치 고정 불가)
- 특수문자 닉 / 짧은 영문 닉(`null`) OCR 주의

### 다음 (상세 설계 → `PLAYER_IDENTITY_AND_K_READER.md`)

1. ~~**`player_identity.py`**~~ — ✅ 구현
2. ~~**`scoreboard_k_reader.py`**~~ — ✅ v1 (ΔK 상한·시각 표기 보완 필요)
3. **`1c` 클립 추출** — ace_rounds → ffmpeg
4. `scouter_nick.py` 헤더 fuzzy 보완 (스무무터 등)
5. 발로란트 등 타게임 영상 사전 필터(2026-04-09)

> OCR 단독 한계 확인됨 (20샘플 ~50% 실패). 단발 OCR이 아니라 **영상 단위 닉 확정** 방향으로 전환.

## 도메인 규칙 (서든어택, 사용자 제공 2026-06-25) — 중요

### 탐지 방식 구분 (ML vs 규칙/CV)

| 단계 | 방식 | 설명 |
|------|------|------|
| 풀 스코어보드 | **ML (CNN)** | `train_scoreboard_clf.py`로 학습, 화면 전체 분류 |
| WIN/DEFEAT | **ML (CNN)** 또는 템플릿 | `train_win_clf.py` / OpenCV matchTemplate |
| HUD 인원 아이콘 | **규칙/CV (ML 아님)** | `hud_round_end.py` — HSV 색상 + blob 개수 |
| 킬 수 OCR | **OCR** | `scoreboard_k_reader.py` + `scoreboard_layout.py` |
| 하이라이트 types | **ML (CNN)** | clips/ 4종 분류 (`train_highlight_types.py`) |

- HUD·지속시간(3.5초) 규칙은 **학습 없이** 도메인 지식 + OpenCV
- 스코어보드/WIN/types만 **라벨 데이터로 학습**

### 라운드 정의
- **라운드 = 풀 스코어보드("전체스코어") 사이 구간**
- 상단 중앙 **라이브 HUD** (항상 표시, 라운드 경계 아님)
  - 형식: `레드팀점수 | 라운드번호 | 블루팀점수` (예 `001|005|002`)
  - 각 점수 아래에 **팀 인원 아이콘**(빨강/파랑 실루엣) 표시
  - **한쪽 팀 아이콘이 전부 사라지면 = 해당 라운드 종료**(엘리미네이션)
    - WIN/DEFEAT 문구가 없어도 이 상태면 라운드가 끝난 것
    - 이후 킬로그 → 풀 스코어보드(~4초) 순서로 이어짐
  - **전반전/후반전에 점수·라운드번호 초기화** → 킬 증가량 계산 시 전/후반 경계에서 리셋 고려
- 풀 스코어보드는 라운드 종료 시퀀스에서 노출: **(HUD 한쪽 전멸) → 킬로그 → 풀 스코어보드(약 4초)**
  - WIN/DEFEAT는 자주 보이지만 **타임라인에 WIN 구간을 따로 적을 필요 없음**
- 사용자가 키 눌러 잠깐 보는 스코어 ≠ 진짜 경계 → **지속시간으로 구별**
  - `detect_rounds.py --min-scoreboard-sec 3.5` (실제 ~4초, 안전 마진으로 살짝 아래)

### 킬 집계 / 하이라이트 1차 규칙
- **ΔK 최대 3** — 한 라운드에서 본인 킬 증가는 0~3만 정상. **4 이상 = 읽기 오류**
- **올킬 = ΔK == 3** (3킬 = 상대 전원 격파)
- 라운드 내 **단일킬 3개 → 올킬**로 1차 처리 → 하이라이트 후보 합격 (2차 분류는 나중)
- `전체스코어 → 움직임/사망/팀원 → 전체스코어` (킬 없음) → 자연 폐기
- 더 정확한 방법(사용자 제안): **본인 닉네임 추적 + 라운드 간 킬 증가량**
  - **스카우터** 패널: 점(●) 행 = 본인 (닉 문자열이 `null`일 수 있음)
  - **스카우터2** 패널: 맨 위 행 = 본인
  - **전체스코어(Clan Match)** 는 **킬 순 정렬** → 본인 행은 닉 매칭으로 찾기

### 죽음
1. 팀원 생존 → 팀원 플레이(관전) 노출
2. 본인이 마지막 → 곧바로 전체스코어

### 리조인 (검정화면 오래 지속)
- 리조인/게임 나감 시 **본인 킬만 초기화** (예 2/3/4 → 0/0/0)
- 전체스코어 숫자가 갑자기 낮아지면 리조인/검정화면 여부 검토 후 수계산·킬로그로 보정
- 리조인이어도 멀티킬/올킬/세이브 가능 → 오류로 취급 금지

### 세이브
- "[닉네임]님이 세이브찬스를 획득하였습니다" (흰 글씨) → 성공 시 "세이브에 성공하였습니다"
- 세이브 성공 시 **멀티킬/더블킬 로고 동반** (노이즈 가능)
- 세이브 = 기본 올킬급

### 참고 이미지
- `assets/...image-e22d9108...png`: 풀 스코어보드 + 상단 라이브 HUD(빨간 박스 구간)
- `assets/...image-a44f92ba...png`: 라이브 HUD 클로즈업 — 좌/우 팀 점수 + 하단 인원 아이콘
- `assets/...images_3...png`: WIN + Killed 배너 + 스카우터2(본인=맨 위)
- `assets/...images_2...png`: MULTI KILL 로고 + 세이브 성공 + 킬로그(전원) + K/D/A

## 새 방향: 라운드 기반 파이프라인 (2026-06-25)

기존 sliding-window binary는 precision이 낮음(일반 플레이/단일킬 과검출).
→ **라운드 단위로 자르고, 킬 많은 라운드만 2단계에 넘기는** 구조로 전환.

```
[1a] 라운드 분할 — "전체스코어 화면" 감지 (라운드 경계)
     보조 신호(미구현): 상단 HUD 한쪽 팀 인원 아이콘 전멸 → 라운드 종료 직전
[1b] 라운드별 내 킬 수 — OCR 또는 킬 이벤트 카운트
[1c] KEEP — 킬 >= 2 (더블킬 이상) 라운드만
[2]  KEEP 구간에만 types 4종 분류 적용
```

### 진행 (1a 스코어 화면 감지 데이터)

- 라운드 타임라인 raw: `manifests/round_timelines/2026-03-19 23-00-50.txt`
- 파서: `parse_round_timeline.py` → `manifests/round_segments.csv`
  - 42세그먼트, scoreboard(라운드 경계) **13개**, movement14/death6/teammate5/loading2/lobby2
- 프레임 추출: `extract_scoreboard_frames.py --clean`
  - `scoreboard_frames/scoreboard/` **102장** (positive)
  - `scoreboard_frames/other/` **175장** (negative)
  - 샘플 육안 검증: positive=Clan Match 스코어 오버레이, negative=일반 플레이 → 명확히 구분됨

### 핵심 발견 (2026-06-25)

- 이전 90개 "킬 시각" 중 `23-13-48`의 9개가 **전부 단일킬**로 확인됨
  (0:27, 2:26, 2:33, 3:01, 3:25, 4:24, 7:05, 9:11, 13:14 = 사용자 단일킬 목록과 일치)
- → 1단계 binary가 단일킬도 하이라이트로 학습한 직접 원인. `known_highlights.csv` 재정의 필요.

### 스코어 분류기 (train_scoreboard_clf.py) — 영상 2개 학습 완료

- 라운드 타임라인 2개: `23-00-50.txt`, `23-13-48.txt`
- 파서 키워드 매칭으로 개선 ("대기실 재입장", "게임종료로딩…", "움직" 오타, "단일킬" 처리)
- `round_segments.csv`: 105세그먼트, scoreboard 26(영상당 13), single_kill 9
- 프레임: scoreboard 206 / other 405 (single_kill도 negative)
- **학습 결과: test acc 95.7%, P 91.7%, R 91.7%, F1 91.7%** (1개 영상 P63.6% → 대폭 개선)
- 모델: `models/scoreboard_clf_best.pt`

### 라운드 자동 분할 (detect_rounds.py) — 작동 확인

- scoreboard 분류기로 영상 스캔 → 연속 감지 병합 → **4초 지속 규칙**으로 가짜 스코어 폐기 → 라운드 분할
- `23-00-50` 검증: known 스코어보드 **13/13 전부 검출**, FP 1개(사망 구간), 짧은 가짜 9개 자동 제거 → **15라운드 분할**
- `00-40-56` (~81분): 스코어보드 **81**, 라운드 **82** (~31분, 순차 디코드)
- 출력: `rounds/<stem>/detected_scoreboards.csv`, `rounds.csv`
- 실행: `python -u detect_rounds.py "E:\OBS\<영상>.mp4" --scan-fps 2`
- ~~cv2 seek 병목~~ → **2026-06-26 순차 디코드로 해결** (긴 영상 필수)

### 다음

1. ~~**라운드별 킬 수 집계**~~ → v1 완료, **ΔK==3 엄격화**·오류 플래그 필요
2. KEEP 구간에만 2단계 types 적용
3. `1c` ace_rounds → 클립 ffmpeg 추출
4. ~~detect_rounds seek 병목~~ → 완료
5. `known_highlights.csv`에서 단일킬 제외 → 1단계 재정의(또는 라운드 방식으로 대체)

## 사용자 결정 (중요)

1. **2단계 types**: `clips/` 4종 클립 학습 **진행 중** (저녁 백그라운드)
2. **1단계 binary**: window 모델 완료 — precision 낮음, 라벨링 후 재학습 예정
3. **전체 배치는 피드백 후**: `MAX_VIDEOS=0`으로 84개 실행

## 파이프라인 현황

| 단계 | 스크립트 | 상태 |
|------|----------|------|
| 0 ROI | `train_game_roi.py` | **완료** |
| 1a 라운드 | `detect_rounds.py` | **완료** (순차 스캔) |
| 1b 닉+K | `player_identity.py`, `scoreboard_k_reader.py` | **v1 완료** (ΔK 엄격화·클립 추출 남음) |
| 1c KEEP+클립 | — | **미구현** |
| 1 binary (window) | `train_binary.py --mode window` | **완료** (precision 낮음, 레거시) |
| 2 types | `train_highlight_types.py` | **완료** (clips 4종, val macro_recall 74%) |
| OBS 파일럿 추론 | `batch_infer_obs.py` | **완료** (90/90 recall, FP 많음) |

### 2단계 types 학습 (clips 4종, 완료)

- 데이터: 하이라이트 188개 (doublekill 38, multikill 32, save 72, allkill 46)
- best: epoch 12, val macro_recall **74%**, val_acc 76.2%
- val 타입별: doublekill 100%, multikill **25%**, save 87.5%, allkill 83.3%
- test: acc 56.2% — multikill·save 약함, 데이터 부족 영향
- 로그: `train_types_clips.log`, 모델: `highlight_types_best.pt`
- **multikill 샘플 32개로 부족** — 클립 추가 또는 라벨링 후 재학습 권장


- 데이터: `label_segments.csv` 204개 (highlight 90 + background 114)
  - hard negative 90 (`obs_hard_negative`) + background_videos 24
- 클립 추출: `extract_labeled_clips.py --overwrite` → 204개 12초 mp4
- 학습 로그: `E:\Highlights\ml_dataset\models\train_binary_window_run.log`
- best 모델: epoch 1 (highlight_recall 100%)

### 파일럿 추론 + recall (2026-06-24)

| 영상 | known | predictions(merged) | recall |
|------|-------|---------------------|--------|
| 2026-03-19 23-00-50 | 10 | 14 | 100% |
| 2026-03-19 23-13-48 | 9 | 14 | 100% |
| 2026-03-21 00-40-56 | 71 | 67 | 100% |
| **합계** | **90** | **95** | **100%** |

- 이전(클립 모델): `windows_hit=0` → 현재: 전 킬 검출
- **주의**: recall 100%이지만 **false positive 많음** (짧은 영상도 14구간 검출). threshold 상향 또는 negative 추가 학습 필요.

평가 스크립트: `python eval_pilot_recall.py`

## 경로 요약

| 용도 | 경로 |
|------|------|
| 코드 | `C:\clipAI\files\` |
| 킬 라벨 | `E:\Highlights\ml_dataset\manifests\known_highlights.csv` |
| 세그먼트 | `E:\Highlights\ml_dataset\manifests\label_segments.csv` |
| window 클립 | `E:\Highlights\ml_dataset\clips\{label}\` |
| 모델 | `E:\Highlights\ml_dataset\models\` |
| OBS (84 mp4) | `E:\OBS\` |
| 추론 결과 | `E:\clipai_result\` |
| 킬 타임라인 | `E:\clipai_result\kill_timeline\` |

## 다음 TODO

1. **ΔK 엄격화**: ace = `ΔK==3` only, `ΔK>=4` → `k_read_error` (올킬 아님)
2. **리포트 시각** `M:SS` 형식 (`format_report`)
3. **1c 클립 추출** — 검증된 ace_rounds만 ffmpeg
4. 안전장치 오귀속 줄이기 (직접매칭 OCR 개선)
5. precision 개선 (레거시 binary) 또는 라운드 방식으로 대체

## 명령어

```powershell
cd C:\clipAI\files

# 세그먼트 재생성
python build_label_manifest.py --allow-overwrite --obs-negative-per-video 30

# window 클립 추출 + 학습
python extract_labeled_clips.py --overwrite
python -u train_binary.py --mode window --epochs 10 --batch-size 24 --num-workers 0

# 파일럿 추론 + recall
python -u batch_infer_obs.py
python eval_pilot_recall.py
```

## 워크플로

```
[0] ROI                    → 완료
[1a] detect_rounds        → 완료 (순차 스캔)
[1b] identity + K reader  → v1 완료 (ΔK 엄격화 필요)
[1c] ace → 클립           → 다음
[2] types 분류            → 모델 보유, KEEP 연동 대기
[레거시] window binary    → recall 100% / FP 많음
```
