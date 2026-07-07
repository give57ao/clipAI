# SONNET 작업 명세 — HUD 올킬 A안 강화: "증거창 확정 + 오독 FP 가드"

> 작성: 2026-07-06 Fable. 배경·용어는 `HUD_ACE_HANDOFF.md` §0 필독.
> 원 목표: recall ≥ 85% AND precision ≥ 80% — 정답 타임라인 밖 올킬은 전부 오답.
> **▶▶▶ 최신 작업 = 맨 아래 "R4 (2026-07-07): 숫자 CNN + 라운드 타이머" — 거기부터.**
> R3 완료 (2026-07-07 Sonnet): recall 88.5%(23/26) / precision 65.7%(23/35, 퇴보).
> 상세는 `HUD_ACE_HANDOFF.md` §0 "R3 통합 결과" 절. 베이스라인 `r3_final` 저장 완료.
> 아래 "R2"/원 설계 명세는 과거 기록(참고용).
>
> **★★ R2 완료 (2026-07-07 Sonnet)**: `SONNET_TASK.md` R2 Task 0~5 실행 완료.
> GT **26건** 기준 **recall 88.5% (23/26) ✓**, **precision 74.2% (23/31) ✗**.
> 상세 per-video·미탐 3건·코드 변경은 **`HUD_ACE_HANDOFF.md` §0 "2026-07-07 Sonnet R2"** 절.
> 이전 8 템플릿 제거(74.1%)는 그 아래 절 참고.
> **새 도구**: `files/_tp_diff.py` — 실험 전후 GT별 획득/상실을 diff로 보여줌
> (두더지잡기 방지). 베이스라인 3종은 `files/_tp_baselines/`에 고정 저장됨.
> 모든 실험은 `hud_from_cache.py && _compare_hud_gt.py && _tp_diff.py` 3종 세트로 측정할 것.
>
> (2026-07-06 Sonnet, 참고): Phase 0+1 자체는 recall 그대로/precision 56.0%→66.7%.
> 시도했으나 되돌린 3가지 실험·구조적 시사점(전역 순차 상태 리플렉트 문제)도 위 절에 기록.
> 아래는 원 설계 명세(참고용, 이미 구현된 §3 Phase 1 부분은 완료 표시).

---

## 0. 절대 규칙 (위반 금지)

1. ace = 라운드 내 킬 합 **정확히 3** (`kills == 3`). ≥3 금지.
2. **사용자 정답 27건이 유일한 기준** — 정답 밖 탐지는 "실제 올킬일지도"라고 재해석하지 말 것. 전부 FP다.
3. 시각 보고는 M:SS.
4. 라운드 경계는 row_miss 방식 유지 (detect_rounds CNN으로 전면 교체 금지 — 단 §4의 "스팟 검증 게이트"는 허용: HUD 파이프라인의 단일패스 메리트를 유지하는 보조 장치임).
5. `hud_kda.py`의 행 탐지/IoU 매칭 자체는 건드리지 말 것 (검증 완료 영역).

## 1. 문제의 뿌리 — "판독 지연 점프" (17건 중 7건)

라운드 종료 무렵 HUD가 깜빡여 중간 K값(7·8)이 "연속 2회" 확정을 못 받으면,
확정값이 뒤처지다가 **6→9(+3)로 뭉쳐** 가짜 올킬(+진짜 올킬 왜곡)이 됨.

**중요한 사전 발견**: 기존 측정(77.8%)의 캐시는 **숫자 8 템플릿이 없던 시점**에 생성됨.
현재는 `k_8/_b/_c` 설치 완료 → 42:00 사례에서 8이 conf 0.96~0.98로 잘 읽힘을 확인.
**따라서 재캐시만으로도 6→9 계열 상당수가 해소될 가능성이 높다. 반드시 Phase 0부터.**

## 2. Phase 0 — 재캐시 + 기준 측정 (필수 선행, 다른 작업 전에)

인프라는 준비 완료 (Fable이 리팩토링함):

- `detect_ace_hud.py`: `collect_reads()`(영상→원시판독) + `timeline_from_reads()`(판정) 분리.
  **판정 로직의 단일 진입점** — 실스캔과 캐시 재계산이 같은 함수를 씀.
- `hud_sig_cache.py`: 원시 판독(t, K, conf, method) 캐시 빌더. 영상당 1패스.
- `hud_from_cache.py`: 캐시→판정 재계산(3초). `--dump`로 구간 원시판독 열람.

```powershell
cd C:\clipAI\files
# 1) 재캐시 (~70분, 백그라운드 권장. 8 템플릿 포함 판독으로 새로 생성됨)
python -u hud_sig_cache.py
# 2) 기준 측정
python -u hud_from_cache.py
python -u _compare_hud_gt.py
```

결과(재캐시 후 recall/precision)를 기록하고 §5 대조표를 갱신할 것.
이 수치가 이미 목표를 넘으면 Phase 1은 마진만 소폭 실험하고 종료해도 됨.

## 3. Phase 1 — `_KTracker` v2: 증거창(evidence window) 확정

`detect_ace_hud.py`의 `_KTracker`를 아래로 교체. **"연속 2회"의 취약점(깜빡임에 끊김)을
"시간창 내 증거 2회"로 바꾸되, 오독 FP를 막는 가드 4개를 함께 구현.**

### 3-1. 자료구조

```python
_EV_WINDOW = 4.0        # 증거 수집 창(초)
_EV_CONFIRM_HI = 2      # conf>=0.75 판독 포함 시 확정 증거 수
_EV_CONFIRM_LO = 3      # 전부 저신뢰(conf<0.75)면 3회 요구  ← 가드 G2
_EV_REBASE = 5          # 하향/+4↑ 리베이스 증거 수 (기존 유지)
_ROLLBACK_WINDOW = 4.0  # 확정 직후 되돌림 감시 창          ← 가드 G4
_EV_ROLLBACK = 3

class _KTracker:
    confirmed: int | None
    ev: dict[int, list[tuple[float, float]]]   # k값 → [(t, conf), ...] (창 내)
    kills / resets: 기존과 동일
    last_kill: (t_confirmed, from_k, to_k) | None   # 되돌림 감시용
```

### 3-2. update(t, k, conf) 의사코드

```
k None → return
ev 전체에서 t - _EV_WINDOW 보다 오래된 항목 제거
ev[k].append((t, conf))

confirmed None:
    strong(ev[k]) 이면 confirmed=k, ev에서 k 이하 제거. return

k == confirmed:
    가드 G4(되돌림): last_kill 있고, t - last_kill.t <= _ROLLBACK_WINDOW 이고,
      k == last_kill.from_k 이고 len(ev[k]) >= _EV_ROLLBACK 이고
      last_kill 확정 후 to_k 증거가 더 안 쌓였으면:
        kills에서 last_kill 제거(pop), confirmed = from_k, last_kill = None
        # 순간 오독 2프레임이 킬로 확정된 경우 원복하는 안전핀
    return

# 상향 체인: confirmed+1 부터 오름차순으로 확정 시도 (핵심!)
loop:
    nxt = min { v in ev : confirmed < v <= confirmed+3 and strong(ev[v]) }
    없으면 break
    kills.append(KillEvent(first_t(ev[nxt]), confirmed, nxt))
    last_kill = (t, confirmed, nxt); confirmed = nxt
    ev에서 confirmed 이하 항목 제거

# 리베이스 (하향 or confirmed+3 초과) — 가드 G1
v_reb = value with len(ev[v]) >= _EV_REBASE and (v < confirmed or v > confirmed + 3)
있으면 resets.append(...), confirmed = v_reb, ev.clear(), last_kill=None

def strong(entries):  # 가드 G2: 신뢰도 연동 증거 기준
    return (len(entries) >= _EV_CONFIRM_HI and max(conf) >= 0.75) \
        or len(entries) >= _EV_CONFIRM_LO
```

### 3-3. 왜 이 설계가 지연점프를 없애는가 (실사례: 02-03-10 41:52~42:02)

```
판독:  8(0.98) · · · 8(0.96) · · 9(0.93) 9(0.94)     (· = 판독실패)
현행:  8이 "연속 2회"를 못 채움(사이가 끊겨도 pend는 유지되지만 이 케이스에선
       중간 confirmed값 재등장·다른 후보가 pend를 리셋) → 6→9 +3 뭉침 = 가짜 올킬
v2:    ev[8] = 2회(고신뢰) → 창 내 증거로 8 확정 → 6→7? (없음, 7은 이전에 확정됐거나
       증거 없으면 8 확정 시 6→8 +2로 이전 시점 first_t(8)에 기록)
       이후 ev[9] 2회 → 8→9 +1.  라운드 경계가 8확정과 9확정 사이면
       킬이 올바른 라운드에 각각 귀속 → 가짜 +3 소멸
```

핵심 차이: **오름차순 체인**이라 9에 증거가 먼저 쌓여도 7·8 증거가 있으면
7→8→9 순서로 확정되고, 각 킬은 **그 값이 처음 관측된 시각**에 기록됨.

### 3-4. 가드 요약 (A안의 오독 FP 방어)

| 가드 | 내용 | 막는 것 |
|---|---|---|
| G1 | confirmed+3 초과·하향은 _EV_REBASE(5) 증거 필요 | 큰 점프 오독(3→9 등) |
| G2 | 저신뢰(conf<0.75)뿐이면 증거 3회 요구 | 흐릿한 프레임 연쇄 오독 |
| G3 | (기존 유지) 트리플 가드 — K/D/A 셋 다 파싱된 프레임만 | 배너/페이드 행 오염 |
| G4 | 확정 직후 이전 값이 되쌓이면 킬 롤백 | 순간 오독 2프레임 확정 |

## 4. Phase 2 — 가짜 경계 스팟 검증 (Phase 1 후에도 recall < 85%면)

> **2026-07-06 Sonnet 판단: 착수 보류**. Phase 1 완료 후 남은 미탐 13건을 개별 진단한
> 결과, 명확히 "가짜 경계가 쪼갬"으로 특정된 건 51:40-52:10(02-34-09) 1건뿐이고, 그마저
> **경계는 진짜**(5.25초짜리 실제 스코어보드)였다 — 문제는 경계가 아니라 그 경계를
> 넘나드는 킬체인 중간값이 `_EV_WINDOW`(4초)보다 드물게 관측되는 것. 즉 남은 recall
> 손실의 지배적 원인은 **경계 오검출이 아니라 트래커의 증거창 크기**로 확인됐다.
> 이 CNN 경계검증은 "가짜 경계 자체를 없애는" 접근이라 이번에 발견된 지배적 원인에는
> 안 맞음 — `HUD_ACE_HANDOFF.md` "다음 세션 우선순위" 절(라운드별 독립 트래커 재설계
> 또는 confirmed+1 한정 무제한 누적)이 더 높은 우선순위. 착수 전 그 절을 먼저 읽을 것.

미탐 2건(03-02-03 14:15, 02-34-09 2:49)은 킬 합이 정확히 3인데 **가짜 row_miss 경계**가
중간을 쪼갬. row_miss 경계 후보(런 ≥ `_BOUNDARY_ROWMISS`)마다 **런 중앙 프레임 1~3장만**
학습된 스코어보드 CNN(`E:\Highlights\ml_dataset\models\scoreboard_clf_best.pt`,
`detect_rounds.py` 참고 — test acc 95.7%, 13/13 검출 검증)으로 분류:

- 스코어보드 확률 ≥ 0.6 프레임이 있으면 → 진짜 경계 유지
- 아니면 → 가짜 경계로 폐기 (단, 런이 40프레임(10s)↑면 무조건 유지 — 확실한 비플레이 구간)

구현: 별도 스크립트 `hud_boundary_verify.py` — 캐시에서 경계 후보 시각을 뽑아
영상을 **seek로 그 프레임만** 읽어 분류 → `sig_cache_v2/{stem}.boundary.json`에 verdict 저장
→ `timeline_from_reads`가 있으면 반영. 영상당 후보 수십 개 × 3프레임 = 수초.
(전면 CNN 스캔이 아니므로 HUD 파이프라인의 단일패스 메리트 유지)

## 5. 측정·수용 기준

```powershell
python -u hud_from_cache.py && python -u _compare_hud_gt.py   # 3초 루프
```

- 수용: **recall ≥ 85% AND precision ≥ 80%** (정답 27건 기준)
- 각 Phase 후 per-video 표를 `HUD_ACE_HANDOFF.md` §0에 갱신
- 완료 시: 실스캔 일치 확인 1회(`detect_ace_hud.py`를 짧은 영상 1개에 실행, 캐시 결과와 ace 동일한지)
  → 커밋 → 배치 37~113 재개는 사용자 승인 후

### 진단 팁

- 특정 구간 원시 판독: `python -u hud_from_cache.py --dump "2026-03-22 02-03-10" 41:50 42:10`
- 미탐/오탐별 킬 이벤트는 hud_timeline JSON의 `kill_events`/`reset_events`에 M:SS로 다 있음
- 판독 자체가 이상하면(캐시에 없는 값) → 템플릿 문제 → `harvest_hud_digits.py` (단, 재캐시 필요)

## 6. 금지·주의

- `_REQ_CONFIRM` 을 3으로 올리는 방식 금지 (막판 킬 유실 실측됨)
- 경계를 detect_rounds 전면 스캔으로 교체 금지 (§0-4)
- `_EV_WINDOW` 를 6초 이상으로 넓히지 말 것 — 이전 라운드 잔존 증거가 다음 라운드로 새는 창이 됨
- 캐시(sig_cache_v2)는 **판독 로직/템플릿 변경 시에만** `--force` 재생성. 판정 로직 변경은 재생성 불필요
- 구 `sig_cache`(v1, 킬 이벤트 저장 방식)는 폐기 대상 — 쓰지 말 것

---

# R2 (2026-07-07): 미탐 7건 소탕 명세 — 사용자 육안 피드백 반영 (Fable 설계)

> 배경: 현재 recall 74.1%(20/27), precision 76.9%(20/26). 사용자가 미탐 7건을
> 스크린샷으로 직접 확인해 준 결과를 반영한 정밀 타격 명세.
> **순서대로 진행, 태스크마다 3종 세트(`hud_from_cache` → `_compare_hud_gt` → `_tp_diff`) 측정.**

## 사용자 확인 도메인 사실 (2026-07-07, 절대 신뢰)

- **마지막 킬 후 최소 2초간 K/D/A가 그대로 노출**된 뒤에야 라운드 전광판이 뜬다.
- **전광판이 떠 있는 동안 K/D/A는 미노출** (= row_miss 경계 논리의 도메인 근거 확정).
- WIN/DEFEAT 배너 → **전광판 약 3초 유지** → 다음 라운드 이동.
- 상단 점수(000|005|000→001|005|000)는 라운드 승리 시 즉시 갱신되지만 이걸로
  감지하지 말 것(상단 스코어 OCR은 예전에 실패 판정, 사용자도 "이걸로 감지는 안될 것").
- 02-21-23 79:51 케이스: **3번째 킬(7→8)은 실제 있었고 K/D/A=8이 ~2초 표시됨**
  (스크린샷 확인). 우리가 8을 EXCLUDE_DIGITS로 뺐기 때문에 못 읽은 것 → Task 3.

## Task 0 — GT 수정 (오답 1건 제거) + 베이스라인 재저장

사용자 확인: **02-34-09 38:14–38:35는 3번째 킬이 실제로 없었음(오답 제공, 제거 지시)**.
- `_compare_hud_gt.py`의 GT dict에서 `"2026-03-24 02-34-09"` 항목 중
  `(_s("38:14"), _s("38:35"))` 튜플 삭제 → **GT 총 26건**.
- 즉시 재측정 후 `python -u _tp_diff.py --save-baseline c2_gt26` 저장.
  이후 모든 diff는 `--compare-to c2_gt26`.
- 예상: recall 20/26 = 76.9% (분모만 감소), precision 변화 없음.
- 목표 재정의: **recall ≥ 85% = 23/26 이상**, precision ≥ 80%.

## Task 1 — 경계 검증 게이트 (`hud_boundary_verify.py` 신규) ★사용자 요청 확정

사용자 질문 "win/전광판 함수 반영돼 있나?" → **현재 미반영** (grep 확인: HUD 파이프라인
4개 파일 어디에도 scoreboard_clf/win_clf 참조 없음. 경계는 오직 row_miss run 길이).
모델은 둘 다 존재: `E:\Highlights\ml_dataset\models\scoreboard_clf_best.pt`(test acc
95.7%, 13/13 검증)와 `win_clf_best.pt`. 이번에 연결한다.

구현 (§4 원안 + 사용자 확인 도메인 시퀀스 반영):
1. 각 캐시 영상의 row_miss run(≥`_BOUNDARY_ROWMISS`) 후보마다, 영상 seek로
   run의 25%/50%/75% 지점 프레임 3장 추출.
2. `scoreboard_clf_best.pt`로 분류 (로드·전처리는 `detect_rounds.py`의
   `build_model`/transforms 코드 재사용 — 새로 만들지 말 것).
3. 3장 중 하나라도 scoreboard prob ≥ 0.6 → **진짜 경계 유지**. 아니면 **폐기**.
   (도메인상 라운드 경계 = 전광판. 전광판 없는 row_miss run은 배너 가림/사망 연출 등
   가짜 — 실측: 03-02-03 14:17·14:37 경계가 GT 올킬 한가운데를 쪼개고 있음.)
4. verdict를 `E:\clipai_result\sig_cache_v2\{stem}.boundary.json`에 저장
   (`{"run_center_sec": true/false}` 맵) — 이후 재평가는 다시 3초 루프 유지.
5. `hud_from_cache.py`/`timeline_from_reads`에 연결: boundary.json 있으면
   폐기 판정 run은 경계로 쓰지 않음. (실스캔 경로는 나중에 — 우선 캐시 경로만.)

기대 효과: 03-02-03 14:15–14:57 (+1), 02-34-09 2:49–3:10 (+1).
⚠ 가짜 경계 제거로 라운드가 길어지며 이웃 킬이 합산돼 기존 TP가 kills>3으로 깨질 수
있음 — `_tp_diff`에서 "상실" 발생 시 해당 케이스 개별 덤프 후 보고.

## Task 2 — 킬 귀속 그레이스: "경계 직후 첫 판독이 곧 증가값이면 이전 라운드 킬"

도메인 근거(사용자 확인): 마지막 킬 후 ≥2초 K/D/A 유지 → 전광판(미노출) → 다음
라운드는 스폰이라 시작 직후 킬 불가. 따라서 **경계 직후 K가 이전 확정값의 재관측 없이
곧바로 +1/+2 올라가 있으면, 그 킬은 전광판 직전(이전 라운드 막판)에 일어난 것**.

구현 (`_assign_events`):
- kill event `e`에 대해 직전 경계 `B`를 찾고, `(B, e.t)` 구간에 `from_k`의 성공 판독이
  **하나도 없으며** `e.t - B ≤ 10s`이면 → `e`를 B 이전 라운드로 귀속.
- 캐시의 원시 판독 접근이 필요하므로 `timeline_from_reads`에서 k_read 시각·값 리스트를
  `_assign_events`에 넘길 것 (이미 `k_read_times`를 넘기는 구조 있음 — 값 포함으로 확장).

기대 효과: 23-51-52 16:23–16:43 (+1) — 9→10이 경계(16:46) 직후 16:49에 첫 관측,
사이에 9 판독 없음 → 이전 라운드 귀속 → kills 3.
⚠ 부작용 감시: 다음 라운드 초반 진짜 킬을 뺏어올 위험은 "(B, e.t)에 from_k 판독 없음"
조건이 방어하지만, `_tp_diff` 상실 0 확인 필수.

## Task 3 — 팬텀-8 보정: 7→9(+2) 이벤트 분리 배치

8은 EXCLUDE_DIGITS(판독 제외)라서 화면에 "8"이 떠도 template_miss(행 발견, 숫자
미매칭)로 기록됨. 02-21-23 79:51의 3번째 킬(7→8)이 정확히 이 케이스(스크린샷 확인).

규칙 (from_k==7, to_k==9 인 kill event 한정 — 8만 판독 불가 숫자이므로 남용 금지):
- 마지막 "7" 성공 판독 시각 t7 ~ "9" 첫 관측 시각 t9 사이에 (a) K슬롯 template_miss
  프레임이 존재하고 (b) 경계 B가 끼어 있으면 →
  `7→8` 킬을 **B 이전 마지막 template_miss 구간 시작 시각**에, `8→9` 킬을 t9에 분리 배치.
- 경계가 안 끼어 있으면(같은 라운드 내 +2) 현행 유지 — 어차피 합산 결과 동일.

기대 효과: 02-21-23 79:51–80:10 (+1).

## Task 4 — G1 오염 (02-21-23 54:20 완전 미검출) — 실험 2건, 각각 3초 측정

이미 진단 완료된 케이스: 52:26 `7→6`, 52:32 `6→0` **오독 리베이스 연쇄**가 confirmed를
0으로 오염 → 이후 진짜 킬 체인(→9→10→11)이 증발. 과거 "G1 v==0 제한" 실험은 전역
리플렉트로 순손실이었으므로(핸드오프 참고) **그대로 재시도 금지**. 대신:
- 실험 4a: **하향(v≠0) 리베이스만** 증거 요건 `_EV_REBASE` 5→8 상향 (0-리셋은 5 유지).
- 실험 4b: **리베이스 롤백**(G4의 리셋판) — 하향 리베이스 직후 `_ROLLBACK_WINDOW` 내에
  구 confirmed 값 증거가 다시 강해지면(strong) 리셋 취소·원복.
- 각각 독립 실험 → `_tp_diff`로 상실 0 확인. 둘 다 순손실이면 포기하고 결과만 기록
  (이 케이스는 HMM 전역 디코딩 카드로 이월).

## Task 5 — 00-40-56 41:54–42:39 완전 미검출 — 진단부터

`python -u hud_from_cache.py --dump "2026-03-21 00-40-56" 41:40 42:50` 으로 원시 판독
확인 → 원인 분류(행 미발견? 판독 실패? 리셋 오염?) 후 위 태스크들의 규칙으로 커버되는지
판단. 커버 안 되면 원인만 핸드오프에 기록하고 멈출 것(새 메커니즘 발명 금지).

## 측정 규율·기대치

- 태스크마다: `python -u hud_from_cache.py && python -u _compare_hud_gt.py &&
  python -u _tp_diff.py --compare-to c2_gt26`
- 기대 누적(이론치): 20/26 → T1 +2 → T2 +1 → T3 +1 → T4 +1 = **25/26 (96%)**.
  절반만 먹혀도 23/26 = 88.5% ≥ 목표 85%. precision은 T1이 가짜 경계발 FP도 일부
  제거할 것으로 기대(현 FP 6건 중 단발성 +3 점프형 확인 필요).
- 완료 시: 실스캔 1회 일치 확인(짧은 영상) → 핸드오프 갱신 → 커밋. 배치 37~113
  재개는 사용자 승인 후.

---

# R3 (2026-07-07): 라운드별 독립 정산 디코더로 판정 전면 교체 (Fable 설계·핵심 구현 완료)

> 배경: R2 완료 시점 recall 88.5% (23/26), precision 74.2% (23/31).
> 사용자가 미탐 3건을 영상으로 재확인해 준 결과 + 캐시 덤프 대조로 **세 건 모두
> "판독(캐시)에는 정답이 이미 있는데 전역 상태머신이 버리는" 사례**로 확정.
> 핵심 디코더는 Fable이 구현·자가검증 완료(`files/hud_round_settle.py`, selftest 6/6).
> **이 R3의 작업 = 그 모듈을 `detect_ace_hud.py`에 통합하고 3종 세트로 측정·튜닝.**

## 사용자 육안 확인 (2026-07-07, 절대 신뢰) + 캐시 대조 결과

| GT | 사용자 확인 | 캐시 원시 판독 (덤프 확인 완료) | 미탐 진짜 원인 |
|---|---|---|---|
| 00-40-56 41:54–42:39 | 42:35–42:37에 K/D/A **9/4/0** 노출 | `9` 6회 판독(conf ≤0.97) 멀쩡히 존재. 그런데 42:24.75·42:32.75~33.75에 **스퓨리어스 `0` 5회(conf 0.91~0.95!)** | v==0 무기한 누적 예외가 산발 0 오독 5회를 모아 `_EV_REBASE=5` 충족 → **가짜 리셋 6→0, 0→9 연쇄** → 킬 전멸 |
| 02-21-23 54:20–54:41 | 54:37–54:40에 K/D/A **11/8/0** 노출 | `9`×3, `10`×3, `11`×1(conf 0.90) 존재. 54:28.75에 또 스퓨리어스 `0`(0.92) | 52:32 `7→0` 오독 리베이스로 confirmed 오염(전역 이월) → 9/10/11이 rebase 문턱(5회)을 못 넘어 증발 |
| 03-02-03 14:15–14:57 | 14:53–14:56에 K/D/A **3/0/0** 노출 | 직전 세그먼트 `0` 안정 판독 → 긴 row_miss(14:12.5~14:41) → `1`×다수, `2`×3, `3`×6 | 0→1 킬이 경계에 걸쳐 라운드 귀속 실패(킬 1+2 분산). **k_base=0 이월 + 라운드 내 1→2→3이면 ΔK=3 정확** |

공통 패턴 2가지 (이번에 처음 정량 확인):
1. **스퓨리어스 고신뢰 `0` 오독**이 여러 영상에서 반복 실측(conf 0.9+ — 킬 연출 중 슬래시
   분리가 밀려 A슬롯 0을 K로 읽는 것으로 추정). v==0 무기한 누적 예외와 결합하면 치명적.
2. 라운드 끝값은 사용자 도메인 확정("마지막 킬 후 ≥2초 노출")대로 **거의 항상 안정 판독됨**
   → 라운드 단위 정산(끝값-시작값)이 킬 체인 추적보다 근본적으로 강건.

## 설계 요지 (`files/hud_round_settle.py` — 구현·selftest 완료, 수정 금지 영역 아님이지만 로직 변경 전 Fable 근거 주석 필독)

- `settle_rounds(reads, boundaries, duration) -> list[SettledRound]` — 순수 함수.
  세그먼트(경계 사이)마다 독립으로:
  1. **지지 필터**: 세그먼트 내 관측 <2회이고 conf<0.88인 값 제거 (42:00 `7` 0.73 단발 등)
  2. **DP 최적 체인**: 비감소(상승폭≤3) 경로 중 conf 합 최대. **0-리셋 간선은 페널티(2.0)**
     → 산발 0 오독은 뒤따르는 정상 체인에 점수로 지고, 진짜 하프타임(이후 저값 지속)은 이김
  3. **k_base 이월**: 직전 세그먼트 k_end를 기준으로 `kills = k_end - k_base`.
     경계 직후(≤10s) 첫 관측이 곧 +1/+2면 직전 라운드 귀속(그레이스, R2 Task 2 의미 동일)
- 전역 순차 상태가 없으므로 "국소 수정 → 원거리 회귀"(핸드오프 '구조적 시사점') 구조적 소멸.
- ace 판정(kills==3·reset 제외·span·G5)은 기존대로 `detect_ace_hud.py` 소관.

## Task 1 — `timeline_from_reads` 통합 (판정 경로 교체)

`detect_ace_hud.py`에서:
1. `_KTracker` 경로(`tracker.update` 루프, `_postprocess_kills`, `_kill_round_index`,
   `_assign_events`의 킬 귀속부)를 `hud_round_settle.settle_rounds` 호출로 교체.
   - reads → `[(r.t, r.k, r.conf)]` 매핑, boundaries는 기존 CNN 검증(boundary_verdicts)
     반영 후의 것을 그대로 전달. **경계 계산·CNN 검증 로직은 변경 금지.**
   - `SettledRound` → `RoundTrack` 매핑: kills/kill_times/k_samples 그대로,
     `reset=True`→`resets=1`, first_kill_sec=min(kill_times), ace_sec=3번째 kill_time.
   - ace 판정식(G5 `_MIN_ROUND_K_SAMPLES`·`_MAX_ACE_SPAN_SEC`·`kills==3`·resets==0)은
     기존 `_assign_events` 말미 형태 유지.
   - JSON `kill_events`: 라운드별 k_base에서 누적 재구성
     (`KillEvent(t=kill_times[i], from_k=k_base+i, to_k=k_base+i+1)`) — 스키마 유지.
2. 구 코드는 삭제하지 말고 일단 남겨둠 (`_KTracker` 등) — diff 검증 끝나고 Task 4에서 정리.
3. 합성 스모크: `python -u hud_round_settle.py` (6/6) + `hud_from_cache.py` 1영상 실행 확인.

## Task 2 — 3종 세트 측정 + TP diff 정밀 대조

```powershell
python -u hud_from_cache.py && python -u _compare_hud_gt.py && python -u _tp_diff.py --compare-to r2_final
```

- **기대: +2 획득** (00-40-56 41:54, 03-02-03 14:15), 02-21-23 54:20은 **미탐 유지가 정상**
  (base 8이 EXCLUDE_DIGITS로 미판독 → ΔK 확정 불가. selftest ④에 명시. HMM 카드로 이월).
  이론치 recall 25/26 = 96.2%.
- **상실 0 필수.** 특히 확인할 기존 TP: 02-21-23 79:51(팬텀-8 케이스 — 정산 방식에선
  그레이스가 7→9 gap 2킬을 직전 라운드로 귀속해 커버할 것으로 설계됨. `_tp_diff`에서
  상실로 나오면 해당 구간 덤프 후 보고), 23-51-52 16:23(그레이스), 02-34-09 2:49(경계 검증).
- 상실 발생 시: 해당 GT 구간 `--dump` → 원인 분류 → **`hud_round_settle.py`의 상수 스윕**
  (`_SUPPORT_SINGLE_CONF` 0.84~0.95, `_RESET_PENALTY` 1.0~4.0, `_GRACE_SEC` 6~12)으로
  해결 시도. 로직 자체 변경은 selftest 6종 + 상실 0을 모두 유지할 때만.

## Task 3 — precision 정리 (목표 ≥80%)

R2 시점 FP 8건(00-40-56 3, 02-34-09 2, 02-03-10 1, 23-51-52 1, 02-21-23 1).
정산 방식은 (a) 스퓨리어스-0 리셋 오탐 (b) 오염 이월발 가짜 +3을 구조적으로 제거하므로
FP 감소 기대. 남는 FP는 각각 덤프로 분류하고, 위 상수 + `_MIN_ROUND_K_SAMPLES` 스윕.
**recall을 깎는 트레이드는 diff 표로 명시하고 중단** — 총점 개선이라도 TP 상실이 있으면 보고.

## Task 4 — 마무리

1. 실스캔 일치 확인 1회 (짧은 영상 1개 `detect_ace_hud.py` 직접 실행 vs 캐시 재계산).
2. 구 `_KTracker`·`_postprocess_kills`(팬텀-8)·`_kill_round_index` 등 대체된 코드 정리
   (완전 삭제 대신 파일 하단 주석 블록 or git 히스토리 의존 — 판단 맡김, 단 import 오류 없게).
3. `python -u _tp_diff.py --save-baseline r3_final` 저장.
4. `HUD_ACE_HANDOFF.md` §0에 R3 절 추가(수치 표·per-video·남은 미탐), 이 파일 헤더 갱신, 커밋.
5. 배치 37~113 재개는 **사용자 승인 후** (변경 금지).

## 금지·주의 (R3 추가분)

- `hud_round_settle.py`의 DP·이월 로직 수정 시 반드시 selftest 6종 유지 + 수정 근거를
  모듈 docstring에 추가. **실측 픽스처(①~⑥)는 삭제 금지.**
- 8 템플릿 재설치 금지 (EXCLUDE_DIGITS 유지 — 근거는 핸드오프 ★★★★ 절).
- 캐시(sig_cache_v2·boundary.json)는 그대로 재사용 — 판독 로직을 안 건드리므로 재캐시 불필요.
- 절대 규칙(§0) 전부 유효: ace=정확히 3, GT 26건이 유일 기준, M:SS 보고.

---

# R4 (2026-07-07): 숫자 CNN + 라운드 타이머 경계 — 근본 처방 (Fable 설계)

> 사용자 피드백(스크린샷 2장, 절대 신뢰):
> 1. **숫자를 못 읽는 건 치명적** — 8 제외(EXCLUDE_DIGITS)는 임시방편이었고, 남은
>    미탐 3건 중 2건(54:20 base=8, 79:51 3번째킬=7→8)의 뿌리가 그것. 전광판/WIN엔
>    학습 CNN을 쓰면서 숫자만 비학습(템플릿 IoU)으로 남긴 게 근본 문제.
> 2. **라운드 타이머(미니맵 아래, 좌상단)를 쓰라**: 라운드 진행 중 카운트다운 →
>    종료 시 `00:00` → WIN/DEFEAT 배너 → 킬 전광판 → 다음 라운드 시작 시 **02:20으로
>    리셋**. "10초 그레이스" 같은 보조장치는 절대 중요하지 않음 — 타이머 리셋이 경계다.
> 3. **클립 범위 완화**: 사용자는 킬 순간이 잘리는 걸 극도로 싫어함. 퍼스트킬 **4초 전**
>    부터 시작, 멀티킬 뒤가 잘리지 않도록 끝 +n초는 감당 가능.
>
> R4 완료 시 기대: 미탐 3건 전부 해소 (26/26). 폭0 FP(R3 정밀 퇴보의 주범 —
> "6에서 곧장 9로 점프")도 상당수가 8 미판독 때문일 가능성이 높아 precision 동반 개선 기대.

## 절대 규칙 (기존 + R4)

- ace = 킬 합 정확히 3. GT 26건이 유일 기준. M:SS 보고.
- **EXCLUDE_DIGITS(8 제외)는 R4에서 폐지가 목표** — 단 반드시 CNN 판독기로 교체 후.
  IoU 매칭에 8 템플릿을 다시 넣는 방식은 여전히 금지(★★★★ 절에서 순손실 실측됨).
- 측정은 3종 세트 + `--compare-to r3_final`. 태스크마다 상실 0 확인.

## Task A — 숫자 판독기를 CNN으로 교체 (8 부활)

**데이터 (전부 이미 존재)**:
- `E:\clipai_result\hud_templates_harvest\raw\` — 수확 글리프 3789장
- `E:\clipai_result\hud_templates_harvest\cluster_labels.json` — 클러스터→숫자 라벨
  (0~7·9 각 3클러스터). 라벨은 클러스터 단위이므로 raw 글리프를 클러스터에 재배정
  (harvest_hud_digits.py의 IoU 클러스터링 재사용)해서 글리프 단위 라벨 생성.
- `E:\clipai_result\hud_templates_harvest\eight\` — 8 전용 재수확분 (라벨=8).
- **junk 클래스**: 수확 중 클러스터 라벨이 "noise"/"mixed"였던 것 + template_miss
  프레임에서 잘린 글리프 일부 → "숫자 아님" 클래스로 포함 (오탐 방어).

**모델·학습**:
- 입력 = `hud_digit_match.normalize_glyph` letterbox 출력(기존 전처리 재사용 — 판독
  파이프와 완전 동일해야 함). 클래스 11개(0~9 + junk).
- 아키텍처는 `detect_rounds.py`의 scoreboard_clf 패턴(작은 conv 2~3층)을 축소 재사용.
  글리프가 작으므로 수 분 내 학습 완료. `E:\Highlights\ml_dataset\models\hud_digit_clf_best.pt`.
- 학습/검증 분리: **영상 단위 split**(같은 영상 글리프가 train/val 양쪽에 가면 누수).

**통합 (`hud_digit_match.py`)**:
```
match_glyph_cnn(glyph): p = softmax(cnn(normalize_glyph(glyph)))
  top1, top2 = 상위 2클래스
  if top1 == junk or p[top1] < _CNN_MIN_P(초기 0.85): return None(판독실패)
  return top1, p[top1], (top2, p[top2])   # top-2는 캐시에 저장 (HMM 대비)
```
- IoU 매칭은 폴백으로 남기되 기본 경로는 CNN. `EXCLUDE_DIGITS` 제거.
- 트리플 가드(G3)는 그대로 유지.

**수용 기준 (재캐시 전에 반드시)**:
- held-out 정확도: 0~7·9 ≥ 99%, 8 ≥ 95%, junk 거짓양성 ≤ 1%.
- 실프레임 스팟 A/B: (a) 79:51.0~79:53 / 54:15~54:20 부근 프레임에서 **8이 읽히는지**,
  (b) 기존에 잘 읽히던 0/6/9 프레임 20장에서 판독률이 IoU 대비 안 떨어지는지.

## Task B — 라운드 타이머 판독 → 경계 신호 교체

**위치**: 미니맵 바로 아래, K/D/A 행(`_KDA_BAND_Y=(0.20,0.34)`) **바로 위** —
대략 gy 0.13~0.21, gx 0.0~0.15 부근의 큰 흰 숫자 `MM:SS`(콜론 흐릿할 수 있음).
정확 좌표는 캘리브 스크립트로 확정(스크린샷 확인된 프레임: 라운드 중 `02:19`,
종료 시 `00:00`). 4:3/16:9 둘 다 확인할 것.

**판독**: 흰색 채널 이진화 → 글리프 분리 → Task A의 숫자 CNN 재사용(색만 다름,
전처리에서 그레이 정규화하면 동일 모델로 가능한지 먼저 스팟 확인; 안 되면 타이머
글리프 소량 수확해 fine-tune). 초 단위 정밀도 불필요 — **분 자리만 정확해도 충분**.

**핵심 로직 (경계 = 타이머 리셋)**:
```
timer_sec(t) 시퀀스에서:
  리셋 = 직전 유효 판독 대비 +60초 이상 점프 (예: 00:xx → 02:20)
  → 경계 시각 = 리셋이 처음 관측된 t (다음 라운드 시작)
  → 직전 라운드 세그먼트는 [이전 리셋, 이번 리셋) — 00:00 구간·배너·전광판이
    전부 "끝나는 라운드"에 자연 귀속 (그레이스 하드코딩 불필요해짐)
검증: 타이머 경계 vs 기존 row_miss+CNN 경계를 라벨 영상에서 대조 —
  일치율 확인 후 타이머를 1순위, row_miss는 타이머 미판독 구간의 폴백으로 강등
```

**캐시 스키마 확장**: `hud_sig_cache.py` reads 항목에 timer 필드 추가
(`[t, k, conf, method, timer_sec|null]`). `hud_from_cache.py` 로더 호환 유지
(길이 4 항목도 읽히게 — 구 캐시 폴백).

## Task C — 재캐시 + 측정 (A·B 합쳐서 1회)

```powershell
python -u hud_sig_cache.py --force   # ~70분, 백그라운드
python -u hud_boundary_verify.py     # 경계 검증 재생성 (참고용 유지)
python -u hud_from_cache.py && python -u _compare_hud_gt.py && python -u _tp_diff.py --compare-to r3_final
```
- 기대: +54:20(base 8 판독), +79:51(7→8 킬 판독), +14:15(타이머 경계) = **26/26**.
- 폭0 FP 6건 재검토: 8이 읽히면 "6→9 점프"들이 6→7→8→9로 풀리거나 오독으로 판명될 것.
  잔존 FP는 개별 덤프 → 분류 → 보고.
- `hud_round_settle.py`는 로직 변경 없이 그대로 (판독·경계 품질만 올라감).
  selftest 6/6 유지 확인.

## Task D — 클립 규칙 (사용자 확정 사양)

`detect_ace_hud.py` `ace_clip_window()`:
- 시작 = **첫 킬 − 4초** (`_CLIP_PRE_KILL_SEC` 14→4).
- 끝 = 3번째 킬 + 여유(잘림 절대 금지 — 라운드 종료(타이머 리셋)까지 포함해도 됨,
  `_CLIP_POST_END_SEC` 유지하되 ace_sec+12 캡이 킬을 자르지 않는지 확인).
- `_CLIP_MAX_SEC` 초과 시 뒤가 아니라 **앞을 잘라야** 함(킬은 항상 보존) — 현행
  `start = end - MAX` 방식이 이미 그렇게 동작하는지 확인만.

## 완료 시

실스캔 1회 일치 확인 → `r4_final` 베이스라인 저장 → 핸드오프 §0에 R4 절 +
이 파일 헤더 갱신 → 커밋. 배치 37~113 재개는 사용자 승인 후.
