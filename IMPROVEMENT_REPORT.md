# clipAI 구조 평가 보고서 (2026-07-16)

> **목적**: 후속 에이전트(Sonnet급)가 개별 항목을 독립 작업으로 집어 실행할 수 있도록
> 근거(파일:줄)·제안·수용 기준을 명시한 평가 보고서. **이 문서 자체는 수정을 수행하지 않는다.**
>
> **전제 조건 — 반드시 읽을 것**:
> 1. 어떤 코드 수정이든 **진행 중인 R10 평가(61영상, `E:\clipai_result\_r10_eval`)가 끝난 뒤에** 시작한다.
> 2. 판독 경로(디코딩·ROI·판정)를 건드리는 변경은 **반드시 `_tp_diff.py --compare-to r10_cleanbase`로
>    성능 무변화(TP 77/107, precision 80.2%)를 증명한 뒤** 채택한다. 이 프로젝트의 최우선 원칙은
>    "속도·구조보다 측정 신뢰"다 (HANDOFF.md 2026-07-16 절, 병렬 재스캔 오염 사건 참고).
> 3. 배치 재스캔은 **병렬 금지(최대 2-way)** — I/O 경합이 결과를 조용히 오염시킨 실측 사례 있음.

---

## 0. 요약 스코어카드

| 영역 | 평가 | 핵심 근거 |
|------|------|-----------|
| 탐지 성능 | recall 72.0% / precision 80.2% (GT 107) | `files/_tp_baselines/r10_cleanbase.json` |
| 캐시 활용 | **심각한 미활용** — 설계는 있으나 배치에 미연결 | §A-1 |
| 하드웨어 활용 | CNN은 GPU, 디코딩은 CPU 전담(NVDEC 0%) | §A-2 |
| 테스트 | **0개** | §B-3 |
| GT 관리 | 코드 하드코딩 + 문서 이중 관리 | §B-1 |
| 레거시 코드 | 3계보 혼재, 일부는 살아있는 의존성 보유 | §C |
| 데이터 리스크 | 원본 일부 소실, 백업 없음, 라벨 음성-전용 | §D |

---

## A. 만들어놓고 활용하지 못하는 코드

### A-1. 신호 캐시(sig_cache)가 배치 파이프라인에 연결돼 있지 않음 — **최대 낭비**

**현상**: `detect_ace_hud.py:617-620`의 `collect_reads()` docstring이 스스로 말한다:
"이 결과(신호 캐시)만 있으면 판정 로직은 영상 재판독 없이 무한 재실험 가능."
그런데 `batch_hud_ace_pipeline.py`에는 `sig_cache`/`build_cache` 참조가 **0회** —
배치 스캔은 매번 수 시간짜리 판독(reads)을 만들고 그대로 버린다.

**실측 피해**:
- 캐시 보유: GT 61영상 중 **9개뿐** (`E:\clipai_result\sig_cache\`).
- 오늘(07-16) R10 평가는 GT 61영상을 **통째로 재디코딩** 중 — 영상당 20-30분, CPU 100%.
  R10은 판정 게이트(`timeline_from_reads`의 `boundary_score_gate`, `detect_ace_hud.py:600`)일 뿐
  원시 판독은 동일하므로, 캐시가 있었다면 `hud_from_cache.py`로 초 단위 재평가가 가능했다
  (`hud_from_cache.py:1-12` docstring: "판정 로직 수정 → 3초 재평가").

**제안 작업**:
1. `batch_hud_ace_pipeline.py`의 스캔 성공 시 reads를 `hud_sig_cache.build_cache`와 동일 포맷으로
   `--output-root` 하위 `sig_cache/`에 저장. (`collect_reads` 반환값을 이미 들고 있으므로 재판독 불필요 —
   `hud_sig_cache.py:48-80`의 직렬화 코드를 함수로 분리해 공유하면 됨.)
2. `boundary_verdicts`(`detect_ace_hud.py:576`)와 score 타임라인도 함께 캐시.
   주의: 경계 검증(`verify_runs_live`)은 영상을 다시 열어 스팟 프레임을 읽으므로 reads 캐시만으로는
   R10류 재평가가 완결되지 않는다 — verdicts까지 저장해야 "영상 없이 재실험"이 성립.
   (score는 이미 `load_score_timeline`이 캐시 우선 조회: `detect_ace_hud.py:585-587`.)
3. 완료 후 GT 61영상 전체 캐시를 1회 구축(단일 프로세스)해 두면 이후 모든 게이트/판정 실험이
   재스캔 없이 가능해진다.

**수용 기준**: 동일 영상에 대해 (a) 배치 스캔 산출 JSON과 (b) 저장된 캐시로 `hud_from_cache.py`가
재생성한 JSON이 동일해야 한다.

### A-2. GPU 디코딩(NVDEC) 미사용 — CPU 100%의 직접 원인

**현상**: 소스는 HEVC 1080p60 20Mbps. `detect_ace_hud.py:623` `cv2.VideoCapture`가 전량 CPU 디코딩.
`scan_fps=4` → step=15, 즉 15프레임 중 1장만 분석하고 14장은 `cap.grab()`(`:652`)으로 디코딩 후 폐기 —
**디코딩 작업의 93%가 버려진다**. 실측(07-16): 파이썬 2개가 CPU 71%, NVDEC 엔진 0%, RTX 4060 Ti 유휴.
ffmpeg에 `hevc_cuvid`/`-hwaccel cuda` 사용 가능 확인 완료. pip OpenCV(4.13)는 CUDA 미포함 빌드라
`cv2.cudacodec` 경로는 불가 — **ffmpeg 서브프로세스 파이프로 교체**가 현실적 경로.

**제안 작업**: `collect_reads`의 프레임 공급부를 ffmpeg NVDEC 파이프
(`ffmpeg -hwaccel cuda -c:v hevc_cuvid -i <in> -vf fps=4 -f rawvideo -pix_fmt bgr24 -`)로 교체하는
**옵트인 플래그**(`--decoder nvdec`, 기본 cv2 유지) 구현.

**수용 기준(엄격)**: GT 61영상에서 cv2 경로와 NVDEC 경로의 hud_timeline JSON이 **동일**해야 기본값 승격.
디코더가 다르면 프레임 타임스탬프·픽셀이 미세하게 달라질 수 있고 이 파이프라인은 픽셀 단위 판독이므로,
1건이라도 판정이 달라지면 원인 규명 전 채택 금지. A-1(캐시)이 먼저 구현되면 이 검증 비용도 급감한다.

### A-3. `hud_from_cache.py` 측정 루프가 사실상 사장됨

A-1의 결과. 캐시 커버리지가 9/61이라 "3초 재평가" 루프를 GT 전체에 못 쓴다.
A-1 해결 시 자동으로 살아난다. 별도 작업 불필요 — 연결 관계만 기록.

---

## B. 구조적 개선 필요

### B-1. GT(정답 데이터)가 코드에 하드코딩 + 문서와 이중 관리

**현상**: 정답 올킬 구간이 `files/_compare_hud_gt.py:31~` 파이썬 dict 리터럴로 하드코딩
(61영상, 구간 ~107건). 같은 정보가 `HUD_ACE_HANDOFF.md` §3에 표로 또 존재. GT 수정 시 두 곳을
사람이 동기화해야 하며, 실제로 07-15 "GT 충돌 6건" 사건이 이 구조에서 발생했다.

**제안 작업**: GT를 `files/gt_aces.json`(스키마: `{stem: [[start_s, end_s], ...]}`)으로 추출,
`_compare_hud_gt.py`는 로더로 축소, HANDOFF §3은 "원본은 gt_aces.json" 포인터로 대체.
**주의**: 추출 시 dict 리터럴과 JSON의 구간 수·값이 1:1 일치함을 스크립트로 검증할 것.
문서상 GT 107건 vs 코드 파싱 ~108건의 ±1 불일치 가능성이 있으니 정확한 건수를 이 작업에서 확정·기록.

### B-2. 경로 상수가 배치 스크립트마다 중복 하드코딩

**현상**: `E:\OBS`, `E:\Highlights\ml_dataset`, `E:\clipai_result` 계열 상수가
`batch_hud_ace_pipeline.py:33-37`, `batch_ace_pipeline.py:30-35`, `batch_infer_obs.py:16-19`,
`hud_boundary_verify.py`, `detect_ace_hud.py:23` 등에 각자 정의. `--output-root`는 이 중
hud 배치에만 후처리로 연결돼 있음.

**제안 작업**: `files/paths.py` 단일 모듈(환경변수 `CLIPAI_*` 오버라이드 허용)로 통합.
기계적 치환이므로 위험도 낮음 — 단 실행 중인 평가와 충돌하지 않게 평가 종료 후 진행.

### B-3. 테스트가 0개 — 회귀를 사람이 GT 재측정으로만 잡는 구조

**현상**: `test_*.py` 0개. 07-16 fail-closed 버그도 일회성 결함주입 수동 테스트로 검증했고
그 테스트 코드는 리포에 없다(HANDOFF 서술만 존재). 판정 로직(`timeline_from_reads`,
`hud_round_settle.py` 583줄, `_KTracker` 후속)은 순수 함수형이라 테스트 작성이 쉬운데도 없음.

**제안 작업(우선순위 순)**:
1. **fail-closed 회귀 테스트 복원**: `hud_boundary_verify.verify_runs_live`에 read 실패 주입 →
   "3장 전부 판독됐을 때만 기각" 계약(`hud_boundary_verify.py`, 07-16 수정분) 검증.
2. **캐시 기반 골든 테스트**: sig_cache 9개를 픽스처로 `timeline_from_reads` 출력 JSON 스냅샷 고정.
   판정 로직 수정 시 의도된 변화만 diff로 드러남.
3. `hud_digit_match`, `hud_round_settle` 단위 테스트.
pytest 도입, `requirements-dev.txt` 분리.

### B-4. "병렬 금지" 교훈이 문서에만 있고 코드로 강제되지 않음

**현상**: 6-way 병렬 재스캔이 측정을 오염시킨 사건(07-16) 이후 HANDOFF·README에 경고는 있으나,
`batch_hud_ace_pipeline.py`는 동시 실행을 감지·차단하지 않는다. 미래 세션이 문서를 안 읽고
`xargs -P 6`을 다시 돌리면 같은 사고가 재발한다.

**제안 작업**: 배치 시작 시 락파일(예: `<output-root>/.scan_lock`, PID 기록) 검사 —
활성 락 2개 이상이면 경고 후 종료(`--force`로 우회 가능). 실행 중 프로세스가 크래시한 stale 락 처리 포함.

### B-5. `files/` 평면 구조에 65개(+미추적 32개) 파이썬 파일 혼재

**현상**: 현행 HUD 파이프라인(~12개), 레거시 2계보(§C), 진단 도구(`_` 접두 20개),
미추적 실험 스크립트 32개가 한 폴더에. 신규 세션(사람이든 에이전트든)이 "어느 파일이 살아있는가"를
매번 재조사한다 — 이번 보고서 작성에도 import 그래프 분석이 필요했다.

**제안 작업**: 물리적 재배치는 import 경로를 깨므로 **2단계**로:
1. (즉시, 무위험) README의 스크립트 표를 유지하는 선에서, 각 레거시 파일 상단 docstring 첫 줄에
   `[LEGACY-ML]` / `[LEGACY-SB]` 태그 추가.
2. (평가 한산기) `files/legacy_ml/`, `files/legacy_scoreboard/` 이동 + import 수정 + 스모크 실행.
미추적 32개는 §E-2에서 별도 처리.

---

## C. 더 이상 필요하지 않은 것 (단, 살아있는 의존성 주의)

세 계보가 확인됨. **일괄 삭제는 금지** — 아래 의존성 표를 먼저 볼 것.

### C-1. 레거시 계보 목록

| 계보 | 파일 | 상태 |
|------|------|------|
| **ML 4종 분류** (doublekill/multikill/save/allkill) | `train_binary.py`, `train_highlight_types.py`, `infer_highlights.py`, `batch_infer_obs.py`, `ml_train_common.py`, `slice_background.py`, `scan_clip_folders.py`, `setup_labeling_project.py`, `build_label_manifest.py`, `eval_pilot_recall.py`, `make_pilot_labeling_template.py`, `labeling_constants.py` | 세이브 감지 종결(07-15 사용자 결정), 올킬은 HUD 파이프라인이 대체. **목적 상실** — 단 더블킬/멀티킬을 미래에 다룰 경우 유일한 자산 |
| **스코어보드 OCR** | `detect_rounds.py`, `scoreboard_k_reader.py`(1040줄, 리포 최대 파일), `player_identity.py`, `scouter_nick.py`, `nick_fuzzy.py`, `scoreboard_layout.py`, `batch_ace_pipeline.py`, `parse_round_timeline.py`, `analyze_nick_misses.py`, `validate_scouter_nick_ocr.py`, `extract_scoreboard_frames.py` | HUD 파이프라인(닉·SB 불필요)으로 대체됨 |
| **구 배치** | `batch_ace_pipeline.py` (위와 중복) | `batch_hud_ace_pipeline.py`가 대체 |

### C-2. ⚠ 삭제하면 안 되는 "레거시처럼 보이는" 살아있는 의존성

| 파일 | 살아있는 이유 |
|------|---------------|
| `train_scoreboard_clf.py` → `scoreboard_clf_best.pt` | **현행** CNN 경계검증기(`hud_boundary_verify.py`)가 이 모델을 로드함 |
| `train_win_clf.py` → `win_clf_best.pt` | `hud_score_wins.py`(R6, 현행)가 사용 |
| `game_roi.py` / `train_game_roi.py` → `game_roi_best.pt` | `extract_game_crop_bgr`(`detect_ace_hud.py:640`, 현행 스캔 루프 핵심) |
| `scoreboard_k_reader.py`의 `nick_match_score` | 진단 도구 `_nick_fail_diag.py`가 import (미추적 파일) |
| `extract_labeled_clips.py`의 `run_ffmpeg_extract` | **현행** 클립 추출(`detect_ace_hud.py:878`)이 import |

**제안 작업**: 삭제 대신 §B-5의 태깅+격리. 유일하게 안전한 완전 제거 후보는
`batch_ace_pipeline.py`(구 배치 진입점, 어떤 현행 코드도 참조 안 함) — 단 git 히스토리에 남으므로
급할 것 없음.

---

## D. 데이터 자산 리스크

### D-1. GT 원본 영상 일부 소실 — 재검증 불가 구간 존재

**현상**: HANDOFF(07-16)에 "GT 50영상(원본 있는 것 전부)"이라 기록 — 즉 **GT 61영상 중 ~11개는
원본이 이미 없다**. 이번 세션에서 `2026-03-21 00-40-56`, `2026-03-22 00-44-50` 소실을 직접 확인
(E:\OBS·D: 전체 검색). 해당 영상들의 GT 구간은 hud_timeline JSON 재생성이 영구히 불가능하며,
recall 분모에는 계속 포함된다.

**제안 작업**:
1. GT 61개 각각에 대해 원본 존재 여부를 스크립트로 전수 조사 → `gt_aces.json`(§B-1)에
   `source_available: bool` 필드로 기록.
2. `_compare_hud_gt.py` 리포트에 "재스캔 가능 GT" 서브셋 지표 추가 — 앞으로의 개선 실험은
   이 서브셋에서만 유효하게 비교 가능하다.

### D-2. 원본·검수 자산 백업 없음 (단일 사본)

**현상**: `E:\OBS` 115개 615GB, D: 루트 원본 162개, 사람 라벨이 붙은 검수 클립
(`ace_clips_candidates`의 오답 29건), `D:\hud_result`(D-영상 처리 결과 유일본 8건) — 전부 단일 사본.
D-1이 증명하듯 원본은 실제로 사라지고 있다.

**제안 작업**(코드 아님, 운영): 최소한 (a) 사람 라벨 자산(수백 MB)과 (b) GT 61영상 원본만이라도
별도 매체/클라우드에 복제. 615GB 전체가 부담이면 GT 서브셋(~50개)이 우선순위.

### D-3. 검수 라벨 체계가 음성-전용이라 "미검수"와 "정답"이 구분 불가

**현상**: `E:\clipai_result` 클립 441개 중 파일명 태그는 `오답` 33개뿐, `정답` 태그 0개.
태그 없음 = 정답인지 미검수인지 알 수 없음 — 이번 세션에서 사용자도 "뭐가 진짜고 가짜인지
모르겠다"고 직접 호소한 문제.

**제안 작업**: 파일명 인코딩을 버리고 검수 대장 하나로 이관 —
`E:\clipai_result\review_ledger.csv` (`stem, round, t_start, verdict{tp,fp,unreviewed}, note`).
기존 파일명의 `오답_*` 태그 29건을 파싱해 시드로 넣고, GT 대조(`_compare_hud_gt`)로 tp를 자동 채움.
이후 클립 파일명은 불변으로 유지(리네임이 해시 대조·중복 정리를 방해했던 사례가 이번 세션에 있음).

---

## E. 문서·저장소 위생

### E-1. 핸드오프 문서 비대화

**현상**: `HANDOFF.md` 34KB + `HUD_ACE_HANDOFF.md` 76KB + `SONNET_TASK.md` 72KB.
HANDOFF는 "역쌓기 로그"라 최신 결론과 뒤집힌 과거 결론(07-15 결론이 07-16에 반전된 사례)이
공존한다 — 새 세션이 낡은 절을 진실로 오독할 위험.

**제안 작업**: HANDOFF 상단에 "현재 유효한 사실만" 요약 블록(≤30줄)을 두고, 반전된 절에는
`[폐기됨 → X절 참고]` 머리표를 다는 경량 규칙 도입. 대수술(파일 분리)은 불필요.

### E-2. 미추적 실험 스크립트 32개 방치

**현상**: `files/_*.py` 32개가 커밋도 gitignore도 아닌 상태로 3주째 존재(git status 소음,
백업 없음). 이 중 `_nick_fail_diag.py`는 추적 파일이 의존하는 함수를 import하는 등 가치 편차가 큼.

**제안 작업**: 3분류 — (a) 재사용 가치 있는 진단 도구는 커밋, (b) 일회성 캘리브레이션 산출물
(`_calibrate_*` 9개, `_grid_*` 3개 등)은 `files/attic/`으로 이동 후 커밋(히스토리 보존) 또는
사용자 승인 하에 삭제, (c) 나머지는 개별 판단. **삭제는 반드시 사용자 확인 후** — 전부
git 밖이라 복구 불가.

### E-3. requirements.txt 불완전

**현상**: `torch, torchvision, opencv-python, numpy` 4개뿐. **현행** 파이프라인의
`hud_kda.py`와 `scouter_nick.py`가 `easyocr`를 import하지만 목록에 없음. 신규 환경 재현 시
런타임에서야 실패한다.

**제안 작업**: 실제 import 전수 조사로 requirements 갱신(easyocr 포함, 레거시 전용 의존성은
주석 분리). `scikit-learn`(`average_precision_score` 사용처 확인) 등도 점검.

---

## 우선순위 로드맵

| 순위 | 항목 | 근거 | 난이도 | 성능 영향 검증 필요 |
|------|------|------|--------|---------------------|
| **P0** | A-1 캐시를 배치에 연결 | 모든 후속 실험 비용을 시간→초로 | 중 | 아니오 (판독 불변) |
| **P0** | D-1 GT 원본 가용성 전수조사 | 이미 진행 중인 데이터 소실 | 하 | 아니오 |
| **P0** | E-3 requirements 보수 | 환경 재현 불가 상태 | 하 | 아니오 |
| **P1** | B-1 GT를 데이터 파일로 | 이중 관리 사고 재발 방지 | 하 | 아니오 (값 보존 검증) |
| **P1** | B-3 테스트 도입(회귀+골든) | 버그를 GT 재측정으로만 잡는 구조 탈피 | 중 | 아니오 |
| **P1** | B-4 병렬 락 | 문서만으로는 사고 재발 | 하 | 아니오 |
| **P1** | D-3 검수 대장 | 사용자가 직접 호소한 혼란 | 중 | 아니오 |
| **P2** | A-2 NVDEC 디코딩 | CPU 100% 해소, 단 판독 동일성 증명 필수 | 상 | **예 — 61영상 동일성** |
| **P2** | B-2 경로 통합, B-5 레거시 격리, E-1, E-2 | 위생 | 하~중 | 아니오 |

**실행 시 공통 규칙**: ① R10 평가 종료 확인 후 시작(`Get-Process python` + `_r10_eval` 갱신 중단 확인)
② 판독 경로 변경은 브랜치에서 `_tp_diff --compare-to r10_cleanbase` 무변화 증명 후 머지
③ 배치 실행은 단일 프로세스.
