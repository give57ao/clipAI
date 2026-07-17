# -*- coding: utf-8 -*-
"""hud_boundary_verify fail-open 계약 회귀 테스트 (SONNET_TASKS.md T3-2).

계약(hud_boundary_verify.py:115-116, 147-155): 기각(False)은 프레임 3장이
전부 판독됐을 때만 허용. read 실패가 하나라도 섞이면(디스크 I/O 지연 등)
무조건 True(경계 유지) — 실영상/실모델 불필요, cv2.VideoCapture와
classify_frame을 몽키패치해 시뮬레이션한다.
"""

from __future__ import annotations

from pathlib import Path

import hud_boundary_verify as hbv
from detect_ace_hud import KRead

_REJECT_PROB = 0.1   # SCORE_THRESHOLD(0.6) 미만 — 증거만 온전하면 기각돼야 함


def _row_miss_reads(n: int = 20, start_t: float = 10.0, fps: float = 4.0) -> list[KRead]:
    """단일 row_miss run, 길이 n (18<=n<40 → LONG_RUN 자동유지 없이 CNN 검증 경로로 진입)."""
    return [KRead(t=start_t + i / fps, k=None, conf=0.0, method="row_miss") for i in range(n)]


class _FakeCap:
    """read_fn(call_index)->bool 로 프레임 판독 성공/실패를 호출 순서대로 제어."""

    def __init__(self, read_fn):
        self._read_fn = read_fn
        self._n = 0

    def isOpened(self):
        return True

    def get(self, _prop):
        return 30.0

    def set(self, _prop, _value):
        pass

    def read(self):
        ok = self._read_fn(self._n)
        self._n += 1
        return (True, "frame") if ok else (False, None)

    def release(self):
        pass


def _verify(monkeypatch, read_fn, classify_prob: float) -> bool:
    monkeypatch.setattr(hbv.cv2, "VideoCapture", lambda _path: _FakeCap(read_fn))
    monkeypatch.setattr(hbv, "classify_frame", lambda *_a, **_k: classify_prob)
    verdicts = hbv.verify_runs_live(
        Path("dummy.mp4"), _row_miss_reads(), model=None, transform=None, device=None,
    )
    assert len(verdicts) == 1
    _start, _last, is_boundary = verdicts[0]
    return is_boundary


def test_full_evidence_can_reject(monkeypatch):
    """대조군: 3장 전부 판독되고 점수가 낮으면 기각(False)이 실제로 나와야 함
    (그래야 아래 fail-open 테스트들이 '항상 True를 반환하는 깨진 함수'로 통과하는 걸 방지)."""
    is_boundary = _verify(monkeypatch, read_fn=lambda i: True, classify_prob=_REJECT_PROB)
    assert is_boundary is False


def test_all_reads_fail_keeps_boundary(monkeypatch):
    """3장 전부 read 실패 → n_ok=0 → 점수와 무관하게 무조건 유지(True)."""
    is_boundary = _verify(monkeypatch, read_fn=lambda i: False, classify_prob=_REJECT_PROB)
    assert is_boundary is True


def test_partial_reads_keep_boundary(monkeypatch):
    """3장 중 2장만 판독(1번째·3번째 프레임 성공, 2번째는 재시도까지 실패) → n_ok=2 → 유지(True)."""
    # 호출 순서: frac1 attempt0(성공,idx0) / frac2 attempt0·1(실패,idx1,idx2) / frac3 attempt0(성공,idx3)
    is_boundary = _verify(monkeypatch, read_fn=lambda i: i in (0, 3), classify_prob=_REJECT_PROB)
    assert is_boundary is True
