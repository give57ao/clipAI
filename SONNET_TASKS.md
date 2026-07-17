# Sonnet 실행 작업서 (2026-07-16)

> `IMPROVEMENT_REPORT.md`의 항목 중 **핵심 로직이 이미 준비된 것들의 배선·마무리 작업**.
> 각 작업은 독립 실행 가능. 항목 번호(§A-1 등)는 보고서 참조.
>
> ## 공통 규칙 (모든 작업에 적용 — 어길 시 측정 오염)
> 1. **[평가 종료 후] 표시 작업은 R10 평가가 끝나기 전 절대 시작 금지.**
>    종료 확인법: `Get-Process python`에 batch_hud_ace_pipeline 없음 +
>    `E:\clipai_result\_r10_eval\hud_timeline`의 파일 수가 10분간 불변.
>    (평가는 영상마다 새 파이썬 프로세스를 띄우므로, 평가가 import하는 파일을
>    고치면 다음 영상부터 수정된 코드로 돌아 결과가 오염된다.)
> 2. 판독 경로를 건드린 뒤에는 `python -u files/_tp_diff.py --compare-to r10_cleanbase`로
>    무변화(TP 77/107) 증명.
> 3. 배치·캐시 구축은 **단일 프로세스**(병렬 금지 — HANDOFF 07-16 I/O 오염 사건).
> 4. 커밋은 main에 직접, 작업 단위별 1커밋.

---

## T1. [지금 가능] requirements.txt 보수 (§E-3)

**목표**: 새 환경에서 `pip install -r requirements.txt`만으로 현행 파이프라인이 돌게.
**단계**:
1. `files/` 추적 파이썬 전체의 외부 import 전수 조사 (`easyocr`는 현행 `hud_kda.py`가 사용 — 확정 누락).
2. `average_precision_score` 사용처가 현행인지 레거시인지 판별해 scikit-learn 포함 여부 결정.
3. requirements.txt 갱신 — 레거시 전용 의존성은 `# legacy:` 주석 절로 분리.
**수용 기준**: 깨끗한 venv에서 `python -c "import detect_ace_hud, hud_kda, hud_boundary_verify, hud_cache_io"` 성공.

## T2. [지금 가능] GT를 데이터 파일로 이관 (§B-1)

**목표**: `files/_compare_hud_gt.py`의 하드코딩 GT dict → `files/gt_aces.json`.
**준비된 것**: `files/gt_source_audit.json`에 61영상의 원본 가용성 감사 결과 존재
(2026-07-16 실측: 소실 10개·구간 26건, 소실분 중 9개는 sig_cache 보유).
**단계**:
1. GT dict를 JSON으로 추출: `{stem: {"spans": [[start_s, end_s], ...], "source_available": bool}}` —
   `source_available`은 gt_source_audit.json에서 병합.
2. `_compare_hud_gt.py`를 gt_aces.json 로더로 수정(±0.01s 이내 값 동일 검증 스크립트 필수 —
   dict 61영상/107구간과 JSON의 전수 일치를 자동 확인 후 dict 삭제).
3. 리포트 출력에 "재스캔 가능 GT(51영상) 서브셋" recall 지표 추가.
4. `HUD_ACE_HANDOFF.md` §3 표 위에 "원본은 files/gt_aces.json" 안내 1줄 추가(표 삭제는 하지 않음).
**수용 기준**: 이관 전후 `_compare_hud_gt.py` 출력의 recall/precision 숫자 완전 동일.
**금지**: GT 값 자체의 수정·추가.

## T3. [지금 가능] pytest 도입 + 회귀 테스트 (§B-3)

**목표**: 최소 테스트 3종. `files/tests/` 신설, `requirements-dev.txt`(pytest) 추가.
**단계**:
1. `test_cache_io.py`: `hud_cache_io.save_scan_cache`→`load_scan_cache` 왕복 + `hud_from_cache.load_reads`
   하위호환. (검증된 인라인 테스트가 이미 있음 — 이 파일 커밋 메시지의 세션 기록 참고, KRead 2건/verdicts/score 왕복.)
2. `test_boundary_failopen.py`: fail-open 계약(07-16 수정, `hud_boundary_verify.py:115-116`) —
   read 실패 주입 시 기각(False)이 나오면 안 됨. 프레임 3장 전부 판독됐을 때만 기각 허용.
   cv2.VideoCapture를 몽키패치해 read 실패를 시뮬레이션(실영상 불필요).
3. `test_timeline_golden.py`: `E:\clipai_result\sig_cache`의 기존 캐시 1~2개를 픽스처로 복사해
   `timeline_from_reads` 출력(rounds 수·ace 판정)을 스냅샷 고정. E: 미존재 시 skip 처리.
**수용 기준**: `pytest files/tests -q` 전체 통과. 기존 코드 수정 없음(테스트만 추가).

## T4. [평가 종료 후] 캐시를 스캔 경로에 배선 (§A-1) — **최우선 가치** — **완료**

**결과**: `scan_hud_aces`에 `cache_dir: Path | None = None` 파라미터 추가,
`timeline_from_reads` 호출 직전 `hud_cache_io.save_scan_cache` 호출(저장 실패는
`cache_warning`으로 경고만 남기고 진행 — 기존 `boundary_warning` 패턴 그대로).
`batch_hud_ace_pipeline.py`는 `cache_dir=<output_root>/sig_cache_v2` 전달
(**주의**: 이 절 원문의 `sig_cache`는 오기 — 실제 소비측(`hud_from_cache.py`,
`hud_boundary_verify.py`)이 이미 `sig_cache_v2`를 기본값으로 쓰고 있고, 구
`sig_cache`(접미사 없음)는 HUD_ACE_HANDOFF.md에 "v1, 킬 이벤트 저장 방식 —
폐기"로 명시된 완전히 다른 스키마라 그대로 썼다면 데이터 충돌이었음).
추가로 `hud_from_cache.py`에 `load_inline_extras()`를 신설해 `save_scan_cache`가
같은 파일에 인라인 저장한 `boundary_verdicts`/`score_win_events`를 우선 사용하고
없으면(구 캐시) 기존 별도 `.boundary.json` 로더로 폴백 — 이게 없으면 스모크 4의
동일성 자체가 성립 안 함(구 로더는 인라인 키를 모름).
**스모크 검증**: GT 영상 `2026-03-30 02-14-48`(원본 있음, 73.6s) 스캔 → scratch
`--output-root`에 캐시 생성 확인(`version:2`, `boundary_verdicts` 인라인 존재) →
`hud_from_cache.py` 재생성 JSON과 원본 스캔 JSON `diff` 완전 동일 확인.
`_tp_diff --compare-to r10_cleanbase`: 77/107 Δ0 (변경 전후 동일 — 기존 9개
`sig_cache_v2` 캐시는 인라인 키가 없어 폴백 경로로 기존과 100% 동일 동작).
`pytest files/tests -q`: 8 passed.
**후속(사용자 승인 후)**: 원본 있는 GT 51영상 캐시 일괄 구축(단일 프로세스, 영상당 20-30분).
드라이브 용량 부족으로 원본 추가 삭제 예정이므로 **삭제 전 캐시 구축이 데이터 보존의 마지막 기회** —
`files/_gt_source_audit.py` 재실행으로 대상 목록 갱신.

## T5. [평가 종료 후] 배치 동시 실행 락 (§B-4) — **완료**

**결과**: `batch_hud_ace_pipeline.py`에 `scan_lock(lock_dir, force=...)` 컨텍스트매니저 신설 —
`<out_root>/.scan_locks/<pid>.lock` 파일로 동시 실행 추적. `_pid_alive`는 Windows에서
`os.kill(pid, 0)`을 쓰지 않음(그 신호값이 `TerminateProcess`로 오인돼 대상을 실제로
죽이는 위험한 동작이라 `ctypes.OpenProcess`로 대체). 진입 시 stale(죽은 PID) 락 자동
청소 → 생존 락 `_MAX_CONCURRENT_SCANS=2`개 이상이면 `ScanLockError`로 즉시 거부(안내
메시지 출력 후 exit 1). `--force-parallel`로 우회 가능. `main()`의 스캔 루프+요약 저장
전체를 `with scan_lock(...):` 안에 두어 종료·예외 시 `finally`로 자기 락 확실히 해제.
**검증**: `files/tests/test_scan_lock.py` 6건(단일 실행 락 생성/해제, 2개 활성 시 3번째
거부, 2개까지는 허용, stale 청소, `--force-parallel` 우회, 예외 시 해제) — `_pid_alive`
몽키패치로 실제 프로세스 없이 결정론적 검증. CLI 스모크(`--only` 존재하지 않는 stem)로
단일 실행 시 락 생성 후 정상 해제 확인. `pytest files/tests -q`: 14 passed.

## T6. [지금 가능] 검수 대장 도입 (§D-3) — **완료**

**결과**: `files/_build_review_ledger.py` 신규 — `E:\clipai_result\ace_clips*\**\*.mp4` 326건
전수 스캔 → `E:\clipai_result\review_ledger.csv`(`stem, round, t, verdict, note, path`) 생성.
**주의**: 이 절 원문의 "기존 오답 29건"은 오기 — `IMPROVEMENT_REPORT.md:196` 원본 실측 자체가
33건이고, 실제 스캔도 33건으로 확인(재검증하며 바로잡음).
파일명 컨벤션이 도구별로 5종 이상 혼재(ace-clip 후보 `_R{n}_{M}m{S}s_(hud_ace|ace)[_오답_설명]`,
`하이라이트(n)` 수동추출, `miss_`/`탐색_`/`확인_R`/`R{n:03d}_킬` 진단 클립 등) — round/시각 개념이
없는 147건은 파싱 불가로 verdict=unreviewed·note에 원인 기록 후 대장에 포함(행 수 불변 유지,
조용히 스킵하면 수용 기준이 깨짐). gt_aces.json(T2) 대조로 tp 59건 자동 기입, 사람이 `오답`
태깅했는데 GT 구간과도 겹치는 5건은 자동 재분류 없이 `[GT_CONFLICT]`로만 플래그(사람 판정 유지,
후속 검수 대상으로 표시).
실행 결과: 스캔 326건 tp=59 fp=33 unreviewed=234(미파싱 147 포함), GT_CONFLICT 5건. 읽기 전용
(클립 파일 미변경). `files/tests/test_build_review_ledger.py` 8건(파싱 4종·GT매칭 허용오차·
행수불변·fp-GT충돌 플래그).
**수용 기준**: ledger 행 수(326) = 스캔된 mp4 수(326) ✓. `오답` 33건 fp로 들어감(원문 29 → 33 정정) ✓.

## T7. [지금 가능] 저장소 위생 (§E-1, §E-2, §B-5-1단계)

**단계**:
1. `HANDOFF.md` 최상단에 "현재 유효 사실 요약(≤30줄)" 블록 신설, 뒤집힌 절(07-15 R10 결론)에
   `[폐기됨 → 07-16 절 참고]` 머리표.
2. 레거시 파일 docstring 첫 줄에 `[LEGACY-ML]`/`[LEGACY-SB]` 태그 (IMPROVEMENT_REPORT §C-1 표 기준,
   **§C-2의 살아있는 의존성 5개는 태그 금지**).
3. 미추적 `files/_*.py` 32개 분류표 작성(커밋 권장/attic/삭제 후보) — **실제 삭제·이동은
   사용자 승인 대기**, 표만 산출.
**수용 기준**: 태그는 주석 1줄 추가만(코드 변화 0) — `git diff --stat`으로 확인.

---

## 산출물 현황 (이 작업서의 전제)

| 파일 | 상태 |
|------|------|
| `IMPROVEMENT_REPORT.md` | 평가 보고서 (근거·우선순위) |
| `files/hud_cache_io.py` | **완성** — 확장 캐시 I/O, 왕복+하위호환 검증 통과 |
| `files/_gt_source_audit.py` | **완성** — GT 자산 감사, 실행 검증 완료 |
| `files/gt_source_audit.json` | 감사 스냅샷 (61영상: 소실 10 / OBS 50 / D: 1) |
| `files/gt_aces.json` | **완성**(T2 산출물) — GT 61영상/107구간 + source_available |
| T1 (requirements) | **완료** (`9094d60`) — easyocr 추가, 전수조사로 발견 |
| T2 (GT 이관) | **완료** (`a95cb1e`) — recall/precision 이관 전후 완전 동일 검증 |
| T3 (pytest 도입) | **완료** — `files/tests/`(cache_io 왕복·boundary fail-open·timeline 골든 8건), `requirements-dev.txt`. 기존 코드 무수정 |
| T4 (캐시 배선) | **완료** — 스모크(2026-03-30 02-14-48) diff 동일, `_tp_diff` Δ0, `hud_from_cache.py` 인라인 폴백 추가 |
| T5 (배치 동시실행 락) | **완료** — `scan_lock`, `_pid_alive`(OpenProcess), 테스트 6건 |
| T6 (검수 대장) | **완료** — `review_ledger.csv` 326행(tp59/fp33/unreviewed234), 오답 카운트 29→33 정정 |
| T7 | 미착수 — 이 문서가 명세 |
