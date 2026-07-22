# -*- coding: utf-8 -*-
"""CNN-8 후보 승격 (R16, 2026-07-23 Fable) — FABLE_HANDOFF.md 요청 #1 대응.

배경: CNN-8을 프레임 단위에서 직접 k=8로 주입했더니(구 _CNN_V2_EIGHT 방식),
"원래 완전 미판독이던 구간에 조밀한 새 증거가 갑자기 등장 → carry(5)→8 gap=3이
폭0 가짜 트리플로 정산"되는 FP 2건 발생(파일럿 실측, SONNET_TASK_DIGIT_CNN.md).

핸드오프의 후보안 (a) '시간적 동의'만으로는 부족함이 실측으로 증명돼 있다:
FP #1(05-26 R09)은 CNN-8이 450.0/450.5/450.75s 연속 3프레임 — 동의 요건을
통과한다. CNN이 6을 8로 '일관되게' 오분류하면 시간 축 동의는 무력하다.

R16 설계 — 2중 조건, 특히 (ii)가 안전성의 뼈대:
  (i)  시간적 동의: _PAIR_SEC 내 후보 2개 이상 (고립 단발 차단)
  (ii) 체인 인접 불변식: 직전 '지지된' 템플릿 K값이 정확히 7 또는 8일 때만
       승격. 승격된 8은 구조적으로 직전값+1(또는 유지)만 가능 —
       **가짜 트리플(+3 점프) 조작이 원리적으로 불가능**해진다.
       5→8, 6(오분류)→8 같은 점프는 여기서 전부 기각된다 (FP 2건 모두 차단).
       대가: 7이 아예 안 읽힌 8-크로싱(예: 6→[7,8 미판독]→9)은 못 구제 —
       그건 기존 Rule A/B(8-브리지)가 계속 담당. 손실 수용(정밀도 우선).

동작 방식: hud_kda가 (플래그 ON일 때) 단일 글리프 K자리에서 IoU 미판독 +
CNN p>=문턱인 프레임을 k=None, method='cnn8_cand'로만 표기(값 주입 안 함).
이 모듈의 promote_cnn8_reads()가 전체 타임라인을 보고 위 2중 조건을 통과한
후보만 k=8/method='template'로 승격. 미승격 후보는 miss로 남는다(기존과 동일).

승격 conf=0.85: 지지 필터(_SUPPORT_SINGLE_CONF=0.88) 아래 — 승격분도 정산
체인에서 단발로는 채택 불가, n>=2 필요(3중 방어).

자가 검증: python -u hud_cnn8_promote.py
"""

from __future__ import annotations

_PAIR_SEC = 1.5          # (i) 시간적 동의: 이 간격 내 다른 후보 필요
_LOOKBACK_SEC = 120.0    # (ii) 직전 지지값 탐색 창 (hud_round_settle._CARRY_MAX_GAP와 동일)
_SUPPORT_MIN = 2         # 지지: 같은 값 2회 이상 …또는
_SUPPORT_CONF = 0.88     # 단발 고신뢰 (hud_round_settle._SUPPORT_SINGLE_CONF와 동일)
_PROMOTE_CONF = 0.85     # 승격 read의 conf — 지지 필터 문턱(0.88) 아래 고정 (3중 방어)
_ALLOWED_PREV = (7, 8)   # 체인 인접 불변식: 직전 지지값이 이 중 하나일 때만 승격

CAND_METHOD = "cnn8_cand"


def _last_supported_value(reads: list, i: int) -> int | None:
    """reads[i] 이전 _LOOKBACK_SEC 내 템플릿 판독에서, 가장 최근의 '지지된' 값.

    최근 값 그룹부터 역방향으로: 같은 값의 (비연속 허용) 관측을 묶어
    n>=_SUPPORT_MIN 또는 max conf>=_SUPPORT_CONF면 그 값을 반환.
    미지지 그룹(고립 오독 의심)은 건너뛰고 다음 그룹을 본다.
    """
    t0 = reads[i].t
    groups: list[list] = []  # [[v, n, cmax], ...] 최근 순
    for j in range(i - 1, -1, -1):
        r = reads[j]
        if t0 - r.t > _LOOKBACK_SEC:
            break
        if r.method != "template" or r.k is None:
            continue
        if groups and groups[-1][0] == r.k:
            groups[-1][1] += 1
            groups[-1][2] = max(groups[-1][2], r.conf)
        else:
            groups.append([r.k, 1, r.conf])
    for v, n, cmax in groups:
        if n >= _SUPPORT_MIN or cmax >= _SUPPORT_CONF:
            return v
    return None


def promote_cnn8_reads(reads: list) -> int:
    """method=='cnn8_cand' 후보를 조건 통과 시 k=8 판독으로 승격 (제자리 수정).

    reads: KRead 리스트 (t 오름차순 가정 — collect_reads/캐시 로드 순서 그대로).
    반환: 승격 수. 후보가 없으면(구 캐시·플래그 OFF) 0 — 완전 no-op.
    """
    cand_idx = [i for i, r in enumerate(reads) if r.method == CAND_METHOD]
    if not cand_idx:
        return 0
    cand_times = [reads[i].t for i in cand_idx]
    n_promoted = 0
    for pos, i in enumerate(cand_idx):
        t = reads[i].t
        # (i) 시간적 동의 — 인접 후보 존재 (자기 제외)
        has_pair = (
            (pos > 0 and t - cand_times[pos - 1] <= _PAIR_SEC)
            or (pos + 1 < len(cand_times) and cand_times[pos + 1] - t <= _PAIR_SEC)
        )
        if not has_pair:
            continue
        # (ii) 체인 인접 불변식 — 직전 지지값이 7/8일 때만
        prev = _last_supported_value(reads, i)
        if prev not in _ALLOWED_PREV:
            continue
        r = reads[i]
        r.k = 8
        r.conf = _PROMOTE_CONF
        r.method = "template"
        n_promoted += 1
    return n_promoted


# ---------------------------------------------------------------------------
# 자가 검증 — 파일럿 FP 실측 지문 + 정탐 목표 픽스처. python -u hud_cnn8_promote.py
# ---------------------------------------------------------------------------

def _selftest() -> None:
    from dataclasses import dataclass

    @dataclass
    class _R:  # KRead 대역 (필요 필드만)
        t: float
        k: int | None
        conf: float
        method: str

    def T(t, k, c=0.7):
        return _R(t, k, c, "template")

    def M(t):
        return _R(t, None, 0.6, "template_miss")

    def C(t):
        return _R(t, None, 0.92, "cnn8_cand")

    # ① FP #1 실측 지문(05-26 R09 유형) — 5 안정 후 미스 구간에 후보 3연발.
    #    시간적 동의(i)는 통과하지만 직전 지지값 5 → (ii)에서 전원 기각.
    reads = [T(440 + i * 0.5, 5) for i in range(16)] + [M(448.5), M(449.0)] \
        + [C(450.0), C(450.5), C(450.75)] + [M(451.5)]
    assert promote_cnn8_reads(reads) == 0
    assert all(r.k is None for r in reads if r.method == CAND_METHOD)

    # ② 정탐 목표(7→8 크로싱, 02-21-23 79:51 유형) — 7 지지 후 후보 쌍 → 승격.
    reads = [T(10 + i, 7) for i in range(5)] + [M(15.5)] + [C(16.0), C(16.5), C(17.0)]
    assert promote_cnn8_reads(reads) == 3
    assert all(r.k == 8 and r.method == "template" and r.conf == _PROMOTE_CONF
               for r in reads[-3:])

    # ③ 고립 단발 — 7 지지라도 (i) 미충족 → 기각.
    reads = [T(10 + i, 7) for i in range(5)] + [C(20.0)]
    assert promote_cnn8_reads(reads) == 0

    # ④ 직전 그룹이 고립 오독(5 단발 저신뢰)이어도 그 앞의 지지된 7을 채택 → 승격.
    reads = [T(10 + i, 7) for i in range(5)] + [T(15.5, 5, 0.6)] + [C(16.5), C(17.0)]
    assert promote_cnn8_reads(reads) == 2

    # ⑤ 단발이라도 고신뢰(0.9) 7이면 지지 인정 → 승격.
    reads = [T(10.0, 7, 0.9)] + [C(11.0), C(11.5)]
    assert promote_cnn8_reads(reads) == 2

    # ⑥ 8 유지(이미 8 확립 후 재관측) — prev=8 허용 → 승격.
    reads = [T(10 + i, 8) for i in range(3)] + [M(13.5)] + [C(14.0), C(14.5)]
    assert promote_cnn8_reads(reads) == 2

    # ⑦ 룩백 밖의 7(200초 전)은 무효 → 기각.
    reads = [T(10 + i, 7) for i in range(5)] + [C(200.0), C(200.5)]
    assert promote_cnn8_reads(reads) == 0

    # ⑧ 후보 없음(구 캐시/플래그 OFF) → no-op.
    reads = [T(10, 5), M(11)]
    assert promote_cnn8_reads(reads) == 0

    print("hud_cnn8_promote selftest: 8/8 OK")


if __name__ == "__main__":
    _selftest()
