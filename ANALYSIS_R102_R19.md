# R102 오탐 원인분석 + R19 후보 (2026-07-23 Opus)

## R19 구현 완료 (2026-07-24)

`detect_ace_hud.py`에 `_r19_enemy_win_in_gap()` 함수 + 게이트로 구현(R14와 같은
detect 레벨, score_win_events 있을 때만 활성). `_R19_OWN_WIN_MARGIN=3.0`.

**전체 GT 코퍼스 재확인 결과 (hud_from_cache.py 재계산 + _compare_hud_gt.py, 재스캔 불필요):**
- 이전: GT 174건 | 탐지 104 (recall 59.8%) | 검출 122건 중 TP 104 (precision 85.2%)
- 이후: GT 174건 | 탐지 104 (recall 59.8%) | 검출 **121**건 중 TP 104 (precision **86.0%**)
- 서브셋(원본보유 68영상): precision 94.2% → **95.3%**
- **TP 손실 0, FP 정확히 1건(R102) 제거** — 설계대로 동작.

R102 실측: `end_reason`이 `hud_elim`→`enemy_win_in_gap`으로 바뀌며 `ace=False`.

회귀 고정: `files/tests/test_r19_gate.py` (합성 픽스처, R102 지문 + TP보호 3케이스).
`hud_round_settle.py`가 아닌 `detect_ace_hud.py` 레벨 구현이라 그 selftest는
변경 없음(22/22 유지) — score_win_events는 settle_rounds 밖 신호이므로 R10/R14와
같은 계층에 두는 것이 기존 구조와 일관적.

## Task 1 결과 — 2026-06-24 R102 (4118-4140) 오탐 원인 확정

**결론: boundary_merge FP (진 라운드 2킬 + 이긴 라운드 1킬을 3킬로 병합). CNN 무관 baseline FP.**

### 근거 (원시 판독 + 승수 채널 대조)
- R102 검출: kills=3, kt=[4118.125, 4118.125, 4135.0], ace=True
- 실제 K 진행: 5(~4105s) → [blind gap] → 7(4128s) → 8(4135s). 8은 위상프로브(conf 0.8, CNN 아님)
- 앞 2킬(5→7)은 관측공백 **[4105.5, 4128.5]**에서 발생 → 라운드 시작(4118)에 gap-stamp
- **승수 이벤트**: 4112.5s **적팀(B) 라운드 승리**, 4135.0s 우리팀(R) 승리
- 즉 blind gap [4105.5, 4128.5] 안에 적팀 승리(4112.5)가 있음 → 앞 2킬은 "우리가 진
  라운드"를 가로지른 것. 진짜 우리 라운드 킬은 마지막 1개(7→8, 4135=우리팀 승리)뿐.
- D채널: 4137.75s에 사망(d 2→3) — ace_sec(4135) 이후라 D-가드 무발동(무관)

### R14가 못 잡은 이유
R14 경계병합 게이트는 "킬 폭 > 25s"를 조건으로 씀. R102 폭 = 4135-4118 = **16.9s**로,
진짜 TP(06-22 R8, 폭 20.9s)보다도 짧아 폭 기준으로 구분 불가.

## R19 후보 신호 (폭 무관, 더 날카로움)

> **gap-stamp 킬의 blind gap [carry_t, first_obs_t] 안에 '적팀 win'(ace_sec 근방
> 우리팀 win과 반대 side)이 있으면, 그 gap 킬들은 진 라운드를 가로지른 것 → 이
> 라운드 올킬에서 제외.**

- 원리: 진짜 올킬 = 적 전멸 = 우리팀 라운드 승리. gap 킬이 적팀 승리를 건너뛰었다면
  그 킬은 다른(진) 라운드 몫.
- R14 폭게이트의 사각지대(폭<25s boundary_merge)를 정확히 커버.
- side 판별: ace_sec ±3s의 win side를 '우리팀'으로 잡고, blind gap 내 반대 side win을
  '적팀 승리'로 판정. ace_sec에 우리팀 win이 없으면 신호 적용 불가(스킵) — 보수적.

### 검증 필요 (구현 전)
`scratchpad/r19_enemy_win_check.py`로 현행 전체 ace를 스캔 — R19 신호 걸린 것 중
**GT-TP가 하나라도 있으면 그 케이스 정밀검토 필요**(R19가 진짜 올킬을 깰 위험).
TP 0건이면 안전하게 FP만 제거. (Sonnet의 hud_timeline 재생성 완료 후 실행 예정.)

### 구현 시 주의 (settle 핵심 로직)
- R11 DARK_GAP_GUARD, R18 스트립과 상호작용 확인 — 셋 다 gap 귀속을 다룸
- score_win_events 없는 영상은 신호 미적용(무해)
- selftest 케이스 추가 필수(R102 지문 고정), 회귀셋 무손실 확인
