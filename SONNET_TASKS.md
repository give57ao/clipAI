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

## T4. [평가 종료 후] 캐시를 스캔 경로에 배선 (§A-1) — **최우선 가치**

**목표**: 스캔 1회 = 캐시 1개. 이후 판정 실험은 영상 재디코딩 없이 수행.
**준비된 것**: `files/hud_cache_io.py` (저장/로드 완성·왕복 검증 완료). 배선만 남음.
**단계**:
1. `detect_ace_hud.scan_hud_aces`(`detect_ace_hud.py:592` `timeline_from_reads` 호출 직전)에서
   `hud_cache_io.save_scan_cache(stem, reads, duration, scan_fps, cache_dir,
   boundary_verdicts=..., score_win_events=...)` 호출. `cache_dir`는 새 파라미터
   `cache_dir: Path | None = None`으로 받아 None이면 저장 생략(기존 동작 보존).
2. `batch_hud_ace_pipeline.py`에서 `cache_dir=<output_root>/sig_cache`로 전달
   (`--output-root` 미지정 시 `E:\clipai_result\sig_cache`).
3. 저장 실패는 경고 후 진행(스캔 결과를 죽이지 않음 — boundary_warning 패턴 답습,
   `detect_ace_hud.py:577` 참고).
4. 스모크: GT 영상 1개를 `--only`로 스캔 → 캐시 생성 확인 → `hud_from_cache.py`로 재생성한
   JSON이 스캔 JSON과 동일한지 diff.
**수용 기준**: 위 4의 동일성 + `_tp_diff --compare-to r10_cleanbase` 무변화.
**후속(사용자 승인 후)**: 원본 있는 GT 51영상 캐시 일괄 구축(단일 프로세스, 영상당 20-30분).
드라이브 용량 부족으로 원본 추가 삭제 예정이므로 **삭제 전 캐시 구축이 데이터 보존의 마지막 기회** —
`files/_gt_source_audit.py` 재실행으로 대상 목록 갱신.

## T5. [평가 종료 후] 배치 동시 실행 락 (§B-4)

**목표**: 병렬 재스캔 사고(HANDOFF 07-16)의 코드 수준 재발 방지.
**단계**: `batch_hud_ace_pipeline.py` main 시작부에 `<output_root>/.scan_locks/` 디렉터리 검사 —
자기 PID 파일 생성, 활성 락(PID 생존 확인) 2개 초과면 안내 후 종료. `--force-parallel` 우회 플래그.
종료·예외 시 자기 락 제거(try/finally). stale 락(죽은 PID)은 자동 청소.
**수용 기준**: 동시 3개 기동 시 3번째가 거부됨. 단일 실행은 영향 없음.

## T6. [지금 가능] 검수 대장 도입 (§D-3)

**목표**: "태그 없음 = 정답인지 미검수인지 모름" 해소.
**단계**:
1. `files/_build_review_ledger.py` 신규: `E:\clipai_result` 하위 `ace_clips*` mp4 전수 스캔 →
   파일명에서 stem/라운드/시각/`오답_*` 태그 파싱 → `E:\clipai_result\review_ledger.csv`
   (`stem, round, t, verdict{fp|unreviewed}, note, path`) 생성. `오답` → fp, 그 외 unreviewed.
2. gt_aces.json(T2)과 대조해 GT 구간과 매칭되는 클립은 verdict=tp 자동 기입.
3. 읽기 전용 — 클립 파일 리네임·이동·삭제 금지.
**수용 기준**: ledger 행 수 = 스캔된 mp4 수. 기존 `오답` 29건이 fp로 들어감.

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
| T1~T7 | 미착수 — 이 문서가 명세 |
