# 플레이어 닉 확정 + 스코어판 K 읽기

> 작성: 2026-06-26 | **구현 v1 완료: 2026-06-26**  
> 상세 핸드오프: `HANDOFF.md`  
> 선행 코드: `scouter_nick.py`, `detect_rounds.py`, `train_scoreboard_clf.py`

## 구현 상태 (2026-06-26)

| 모듈 | 상태 | 비고 |
|------|------|------|
| `player_identity.py` | ✅ v1 | 스카우터 + 스코어보드 투표, fuzzy 클러스터 |
| `scoreboard_layout.py` | ✅ | 6행 ROI, CLAHE·반전 닉 OCR |
| `nick_fuzzy.py` | ✅ | 정규화 편집거리, 6자+ 닉 dist≤3 |
| `scoreboard_k_reader.py` | ✅ v1 | 2-pass team_lock, `[T]` 안전장치 |
| `detect_rounds.py` | ✅ | 순차 디코드 스캔 (긴 영상) |
| 1c 클립 추출 | ⬜ | ace_rounds → ffmpeg 미연결 |

### 도메인 규칙 (재확인)

- **ΔK 최대 3** — 올킬 = `ΔK == 3`. **`ΔK ≥ 4`는 오류** (현 코드는 `>=3` ace → 수정 필요)
- **시각 보고**: `M:SS` (초 단독 금지)
- **안전장치 `[T]`**: 메인 아님. `inferred_by_team=true` → 재검증 필수

### 파일럿

- `2026-03-19 23-00-50`: 직접매칭 6/8, 안전장치 1회
- `2026-03-21 00-40-56`: 81 스코어보드, 닉 `괴명`, ace 12건(다수 오탐 — ΔK>3)

---

### 현재 한계 (`scouter_nick.py` 단독)

- **한 프레임 · 한 위치 · OCR 한 번** → 실패 시 영상 전체 실패
- 랜덤 20샘플 검증(`validate_scouter_nick_ocr.py`)에서 **성공률 약 50%**
- 주요 실패 원인:
  1. 헤더 OCR 깨짐 (`스카우터2` → `스카무터2`) → `mode=unknown` → 닉 버림
  2. 짧은 닉 오인식 (`null` → `Jyu`, `nulI`, `mull`)
  3. 비게임 구간 (로딩/대기실/패널 없음)
  4. 스카우터 점(●) 탐지 실패

### 사용자 요구 (다양한 관점)

| 관점 | 내용 |
|------|------|
| 스카우터 | 플레이 중 본인 행에서 닉 확보 |
| 전체스코어(Clan Match) | 라운드 끝 6행에도 닉·K 존재 |
| 시간축 | 영상이 쌓이면 **주인공 닉을 확정**해야 함 — 매 프레임 재추측 아님 |
| 교차 검증 | 스카우터 닉 ≈ 스코어판 닉이면 신뢰도 상승 |

**결론:** OCR 자체를 없앨 수는 없지만, **영상 단위 다소스 투표 + 확정 후 고정**이 필요하다.

---

## 파이프라인 위치

```
[1a] detect_rounds.py          — 라운드 경계 (스코어보드 CNN)     ✅ 완료
[1b-1] player_identity.py      — 영상 주인공 닉 확정              ✅ v1
[1b-2] scoreboard_k_reader.py — 확정 닉으로 K 읽기, ΔK 계산     ✅ v1 (ΔK==3 엄격화 남음)
[1c] 올킬 KEEP + 클립          — ΔK == 3 만                       ⬜
[2]  train_highlight_types     — KEEP 구간 types 분류             모델 보유
```

---

## 1. `player_identity.py` — 영상 단위 닉 확정기

### 목적

영상 전체를 스캔해 **이 녹화의 주인공 닉**을 확정한다.  
한 프레임 OCR 실패는 허용; **누적 증거**로 최종 닉을 고정한다.

### 입력 / 출력

```python
@dataclass
class PlayerIdentity:
    nickname: str              # 확정 닉 (예: "null", "단호")
    confidence: float          # 0~1
    mode: str                  # "scouter2" | "scouter" | "mixed" | "unknown"
    sources: dict              # 소스별 투표 상세
    game_width_median: int     # 레이아웃 (후원패널 vs 풀스크린)
    samples_total: int
    samples_hit: int

def resolve_player_identity(
    video_path: Path,
    *,
    scan_fps: float = 0.5,           # 스카우터 샘플 간격 (초)
    scoreboard_csv: Path | None = None,  # detect_rounds 출력 연동
    min_votes: int = 3,
    min_conf: float = 0.25,
) -> PlayerIdentity:
    ...
```

### 데이터 소스 (3종)

#### 소스 A — 스카우터 패널 (`scouter_nick.read_scouter`)

| 모드 | 본인 행 규칙 |
|------|-------------|
| 스카우터2 | 맨 위 행 |
| 스카우터 | 점(●) 행 |

- 샘플: 영상 `t=60s` ~ `duration-30s` 구간, `scan_fps` 간격 (기본 0.5fps ≈ 2초마다)
- `mode=unknown`이어도 **rows에 닉 후보가 있으면** 보조 투표에 포함 (헤더 실패 완화)
- 기존 `validate_scouter_nick_ocr.py`의 투표 로직을 **모듈화**해 이식

#### 소스 B — 전체스코어 프레임 (`detect_rounds` 연동)

- `detected_scoreboards.csv`의 각 스코어보드 구간 **중앙 시각** 1프레임
- Clan Match 오버레이에서 **팀별 3행 × 2팀 = 6행** 닉+K crop 후 OCR
- 행 순서: **킬 순 정렬** (맨 위 ≠ 본인) → 위치로 본인 찾기 불가, **닉 매칭만** 가능
- 스코어보드 crop 좌표: `game_width` 기준 (후원패널형 보정 필수) — **구현 시 캘리브레이션 필요**

#### 소스 C — 교차 검증 (투표 가중치)

- 소스 A에서 나온 닉이 소스 B 6행 중 하나와 **fuzzy match** → 해당 후보에 **가중치 +2**
- 스카우터2 맨 위와 스코어판 동일 닉 동시 출현 → confidence 상한 boost

### Fuzzy 닉 클러스터링

OCR 오인식을 하나의 닉으로 묶는다.

```python
# 예시 규칙 (구현 시 tunable)
- 대소문자 무시
- Levenshtein 거리 ≤ 2 (짧은 닉: ≤ 1)
- 알려진 null 변형: {"null", "nill", "nulI", "mull", "Jyu", "nul", "nu11"}
- 특수문자 제거 후 비교 (◇¤깜띸겅쥬¤◇ ↔ 부분 매칭)
```

클러스터별 **가중 투표 합**으로 최종 닉 선택.

### 헤더 fuzzy (빠른 개선 — `scouter_nick.py` 보완)

`scouter_nick._is_header` / `_header_is_scouter2` 확장:

- `스무무터`, `스카무터`, `스카우터]2` 등 → 스카우터2
- 헤더 실패 시: data_lines 첫 행을 scouter2 후보로 **약한 투표** (패널 구조상 맨 위가 본인인 경우 많음)

### 신뢰도 산출 (초안)

```
confidence = min(1.0,
    vote_ratio * 0.5 +           # 최다 득표 / 전체 유효 샘플
    cross_source_bonus * 0.3 +   # A∩B 일치
    mode_consistency * 0.2       # scouter vs scouter2 일관
)
```

- `confidence < 0.5` → 경고 로그, 수동 검토 플래그
- 확정 실패 시: `nickname=""`, 파이프라인은 해당 영상 **SKIP** (올킬 판정 불가)

### 의존성

| 모듈 | 용도 |
|------|------|
| `scouter_nick.py` | 패널 crop, OCR, dot 행 |
| `detect_rounds.py` 출력 | 스코어보드 시각 목록 |
| `easyocr` | 텍스트 (기존과 동일) |
| (선택) `rapidfuzz` | 닉 fuzzy — 없으면 자체 Levenshtein |

### CLI (초안)

```powershell
python player_identity.py "E:\OBS\2026-03-19 23-00-50.mp4" `
  --rounds-dir "E:\clipai_result\rounds\2026-03-19 23-00-50" `
  --out identity.json
```

### 검증 영상 (회귀 테스트)

| 영상 | 기대 닉 | 모드 | 비고 |
|------|---------|------|------|
| `D:\뮤크퀵.mp4` | 단호 | scouter2 | 풀스크린 |
| `2026-03-26 01-26-52.mp4` | ◇¤깜띸겅쥬¤◇ | scouter2 | 후원패널, OCR 어려움 |
| `2026-05-18`, `2026-06-14` | `null` | scouter | 점 행, **문자열 null** |
| `2026-04-09` | — | — | 발로란트, **제외** |

### 알려진 한계

- 특수문자 닉: fuzzy만으로 부족할 수 있음 → 스카우터2 맨 위 단일 소스 다수결에 의존
- 닉 변경(영상 중간): 드묾, v1에서는 **단일 닉 가정**. 리조인 시 킬만 0 리셋, 닉 유지

---

## 2. `scoreboard_k_reader.py` — K 읽기 + ΔK

### 목적

`player_identity.py`가 확정한 닉으로, **각 라운드 스코어보드**에서 본인 **K(킬 수)** 를 읽고  
라운드 간 **ΔK**를 계산해 올킬 후보를 판별한다.

### 전제

1. `detect_rounds.py` → `detected_scoreboards.csv` (라운드별 스코어보드 구간)
2. `player_identity.py` → `PlayerIdentity.nickname` 확정
3. Clan Match 스코어는 **킬 순 정렬** — 행 인덱스 고정 불가

### 입력 / 출력

```python
@dataclass
class RoundKillReadout:
    round_index: int
    scoreboard_start_sec: float
    scoreboard_end_sec: float
    kills: int | None          # K 읽기 실패 시 None
    nick_matched: str          # 실제 매칭된 OCR 문자열
    match_score: float         # fuzzy match 점수
    row_index: int | None      # 0~5 (6행 중)

@dataclass
class VideoKillTimeline:
    video_path: str
    player_nick: str
    rounds: list[RoundKillReadout]
    delta_kills: list[int]     # rounds[i].kills - rounds[i-1].kills (리조인 보정 후)
    ace_rounds: list[int]      # ΔK >= 3 인 라운드 인덱스

def read_kills_per_round(
    video_path: Path,
    identity: PlayerIdentity,
    scoreboards: list[ScoreboardWindow],  # detect_rounds 구조체
    *,
    nick_match_threshold: float = 0.75,
) -> VideoKillTimeline:
    ...
```

### 스코어보드 프레임 처리 흐름

```
각 scoreboard 윈도우 (지속 ≥ 3.5초)
  1. 중앙 시각 프레임 추출 (또는 구간 내 2~3프레임 다수결)
  2. detect_game_width() → game ROI
  3. Clan Match 패널 crop (좌표: 구현 시 캘리브레이션 — HANDOFF assets 참고)
  4. 6행 분할 (팀당 3행) → 각 행: [닉 영역 | K 영역 | D | A ...]
  5. 닉 OCR → identity.nickname과 fuzzy match → 본인 행 선택
  6. 해당 행 K 영역 OCR (숫자만) 또는 digit crop CNN (후순위)
  7. kills 값 기록
```

### K OCR 전략

| 단계 | 방식 | 비고 |
|------|------|------|
| v1 | EasyOCR + 숫자만 필터 | 빠르게 동작 확인 |
| v2 | K 열만 crop → 소형 digit CNN | OCR보다 안정적, 라벨 적게 필요 |

- K는 닉보다 **문자 집합이 작음** (0~9) → OCR 성공률 상대적으로 높을 것으로 기대
- 여러 프레임 다수결: 같은 라운드에서 K 값 2/3 일치 시 채택

### ΔK 계산 + 올킬 규칙

```python
# 기본
delta = kills[i] - kills[i-1]

# 리조인 감지: kills[i] < kills[i-1] 이고 급격한 하락
# → delta 무시 또는 kills[i]를 기준점으로 리셋 (HANDOFF 리조인 규칙)
# → "본인 킬만 0 리셋" — 전체스코어 전원 0은 아님

# 올킬 1차 (도메인 규칙, 사용자 확인)
KEEP if delta == 3   # 라운드 내 정확히 3킬 = 올킬
# delta >= 4 → k_read_error (OCR/오귀속), 올킬 아님
```

- 전반/후반전 점수 초기화: 스코어보드 **라운드 번호** 또는 팀 점수 리셋 감지 시 ΔK 리셋
- `전체스코어 → 움직임 → 전체스코어` (킬 없음): ΔK ≈ 0 → 자연 폐기

### 실패 처리

| 상황 | 처리 |
|------|------|
| 닉 매칭 실패 (6행 중 없음) | `kills=None`, 해당 라운드 스킵 |
| K OCR 실패 | 구간 내 다른 프레임 재시도 (최대 3회) |
| 연속 N라운드 실패 | 영상 품질 경고, ace 판정 신뢰도 하락 |

### CLI (초안)

```powershell
python scoreboard_k_reader.py `
  "E:\OBS\2026-03-19 23-00-50.mp4" `
  --identity identity.json `
  --rounds-dir "E:\clipai_result\rounds\2026-03-19 23-00-50" `
  --out kill_timeline.csv
```

출력 예 (`kill_timeline.csv`):

```csv
round_index,start_sec,end_sec,kills,delta_k,ace
0,125.2,129.1,2,,0
1,248.5,252.0,5,3,1
...
```

### 스코어보드 crop 좌표 (TODO — 구현 시 확정)

- 참고 이미지: `HANDOFF.md` → `assets/...image-e22d9108...png` (풀 스코어보드 + HUD)
- `game_width` 기준 상대 좌표로 정의 (후원패널형 1405px vs 풀 1920px)
- **구현 첫 작업:** `23-00-50` 스코어보드 positive 프레임 1장으로 6행 ROI 수동 캘리브레이션 → 비율 고정

### 의존성

| 모듈 | 용도 |
|------|------|
| `player_identity.py` | 확정 닉 |
| `scouter_nick.detect_game_width` | 레이아웃 |
| `detect_rounds` | 스코어보드 윈도우 |
| `easyocr` | 닉·K OCR (v1) |

---

## 구현 순서 (다음 세션 권장)

1. **`scouter_nick.py` 헤더 fuzzy** — 즉시 성공률 소폭 상승
2. **`player_identity.py` v1** — 소스 A만 (스카우터 투표), `validate_scouter_nick_ocr` 로직 이전
3. **스코어보드 6행 ROI 캘리브레이션** — positive 프레임 1~3장
4. **`player_identity.py` v2** — 소스 B 추가 + 교차 검증
5. **`scoreboard_k_reader.py` v1** — 닉 매칭 + K OCR + ΔK
6. **회귀:** `23-00-50` known 올킬 라운드와 대조

---

## 관련 파일

| 파일 | 상태 |
|------|------|
| `files/scouter_nick.py` | ✅ 기존 (단프레임) |
| `files/validate_scouter_nick_ocr.py` | ✅ 검증 스크립트 → identity v1 참고 |
| `files/detect_rounds.py` | ✅ 라운드 경계 |
| `files/player_identity.py` | ✅ v1 |
| `files/scoreboard_k_reader.py` | ✅ v1 |
| `files/scoreboard_layout.py` | ✅ |
| `files/nick_fuzzy.py` | ✅ |
| `HANDOFF.md` | 도메인 규칙·파일럿·TODO |

---

## 설계 원칙 (재확인)

1. **단발 OCR ≠ 답** — 시간축·다소스가 핵심
2. **스카우터 = 신원 확보**, **스코어판 = K 집계** — 역할 분리
3. **확정 후 고정** — 매 라운드 닉 재추측하지 않음 (fuzzy match는 행 찾기용)
4. **실패는 숨기지 않음** — confidence·None·SKIP 명시
5. **ML은 필요한 곳만** — 스코어보드 화면 감지(CNN), digit K(v2); 닉은 OCR+투표로 v1
