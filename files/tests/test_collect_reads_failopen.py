# -*- coding: utf-8 -*-
"""collect_reads I/O fail-open 재시도 회귀 (2026-07-17, R10 재평가 중 실측 발견).

배경: read/grab 실패를 곧장 EOF로 취급해 2-way 병렬 스캔 중 ffmpeg Stream timeout이
걸리면 영상 뒷부분이 경고 없이 통째로 사라지는 사고(`2026-05-31 21-57-14`, 정상
163라운드 → 66라운드에서 조용히 절단)가 실측됨. verify_runs_live의 fail-open
계약(hud_boundary_verify.py, 07-16)과 동일한 사고로 재시도 1회 추가.

실영상 없이 cv2.VideoCapture를 몽키패치해 시뮬레이션 — 실모델(OCR/템플릿매칭)도
get_hud_digit_matcher/extract_game_crop_bgr/read_kda_triple_from_game을 몽키패치해 우회.
"""

from __future__ import annotations

import detect_ace_hud as dah


class _FakeCap:
    """total_frames개 프레임을 순서대로 공급. fail_at에서 1회(transient) 또는
    영구(persistent) 실패를 주입해 read()/grab() 양쪽에 동일하게 적용."""

    def __init__(self, total_frames: int, fps: float = 4.0, fail_at: int | None = None,
                 persistent: bool = False):
        self.total_frames = total_frames
        self.fps = fps
        self.idx = 0
        self.fail_at = fail_at
        self.persistent = persistent
        self._transient_consumed = False

    def isOpened(self) -> bool:
        return True

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.total_frames
        return 0

    def _advance(self):
        if self.idx >= self.total_frames:
            return False, None
        should_fail = (
            self.fail_at is not None and self.idx == self.fail_at
            and (self.persistent or not self._transient_consumed)
        )
        if should_fail:
            if not self.persistent:
                self._transient_consumed = True
            return False, None
        self.idx += 1
        return True, "frame"

    def read(self):
        return self._advance()

    def grab(self) -> bool:
        ok, _ = self._advance()
        return ok

    def release(self):
        pass


def _patch_common(monkeypatch, fake_cap):
    monkeypatch.setattr(dah.cv2, "VideoCapture", lambda _path: fake_cap)
    monkeypatch.setattr(dah, "get_hud_digit_matcher", lambda: None)
    monkeypatch.setattr(dah, "extract_game_crop_bgr", lambda frame, dataset_root=None: (frame, None))
    monkeypatch.setattr(dah, "read_kda_triple_from_game", lambda game: (None, None, None, 0.0, "template_miss"))


def test_transient_read_failure_recovers_via_retry(monkeypatch):
    """scan_fps==cap fps(step=1)라 매 프레임이 샘플 프레임 — read() 재시도 경로 검증.
    5번째 프레임(idx=4)에서 1회 실패해도 재시도로 복구, 전체 20프레임 다 읽혀야 함."""
    cap = _FakeCap(total_frames=20, fps=4.0, fail_at=4, persistent=False)
    _patch_common(monkeypatch, cap)

    reads, duration, err = dah.collect_reads(dah.Path("dummy.mp4"), scan_fps=4.0)

    assert err is None
    assert len(reads) == 20
    assert duration == 5.0  # 20 frames / 4fps


def test_transient_grab_failure_recovers_via_retry(monkeypatch):
    """scan_fps < cap fps(step>1)라 grab()만 쓰는 스킵 프레임 구간에 실패 주입."""
    cap = _FakeCap(total_frames=40, fps=8.0, fail_at=5, persistent=False)  # step=2 → idx 5는 grab 대상
    _patch_common(monkeypatch, cap)

    reads, duration, err = dah.collect_reads(dah.Path("dummy.mp4"), scan_fps=4.0)

    assert err is None
    assert duration == 5.0  # 40 frames / 8fps
    assert len(reads) == 20  # step=2 → 40/2


def test_persistent_failure_reports_early_break_instead_of_silent_truncation(monkeypatch):
    """양쪽 재시도 다 실패하면(영구 I/O 장애) 조용히 끝내지 않고 early_break 경고를 반환."""
    cap = _FakeCap(total_frames=100, fps=4.0, fail_at=10, persistent=True)
    _patch_common(monkeypatch, cap)

    reads, duration, err = dah.collect_reads(dah.Path("dummy.mp4"), scan_fps=4.0)

    assert duration == 25.0  # 100 frames / 4fps (영상 원래 길이)
    assert len(reads) == 10  # fail_at(프레임 idx=10) 이전까지만 판독됨
    assert err == "early_break_2s_of_25s"  # 10프레임/4fps=2.5s 지점에서 끊김(반올림 2s)


def test_true_eof_within_tolerance_reports_no_warning(monkeypatch):
    """프레임카운트 추정과 실제 종료 지점이 5초 이내로 어긋나는 정상 EOF는 경고 없음."""
    cap = _FakeCap(total_frames=19, fps=4.0)  # 실제로는 19프레임에서 끝(정상 EOF, 재시도 무의미)
    _patch_common(monkeypatch, cap)

    reads, duration, err = dah.collect_reads(dah.Path("dummy.mp4"), scan_fps=4.0)

    assert duration == 4.75  # cap이 보고한 total_frames 기준 산출 duration
    assert len(reads) == 19
    assert err is None
