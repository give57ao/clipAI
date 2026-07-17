# -*- coding: utf-8 -*-
"""batch_hud_ace_pipeline 동시실행 락 회귀 테스트 (SONNET_TASKS.md T5).

수용 기준: 동시 3개 기동 시 3번째가 거부됨. 단일 실행은 영향 없음.
_pid_alive를 몽키패치해 실제 프로세스 기동 없이 "생존 PID N개" 상황을 시뮬레이션.
"""

from __future__ import annotations

import pytest

import batch_hud_ace_pipeline as bhp


def test_single_run_creates_and_releases_lock(tmp_path):
    with bhp.scan_lock(tmp_path):
        locks = list(tmp_path.glob("*.lock"))
        assert len(locks) == 1
    assert list(tmp_path.glob("*.lock")) == []


def test_third_concurrent_run_rejected(tmp_path, monkeypatch):
    """2개(_MAX_CONCURRENT_SCANS) 활성 락이 있는 상태에서 3번째 진입은 거부돼야 함."""
    monkeypatch.setattr(bhp, "_pid_alive", lambda pid: True)
    (tmp_path / "111.lock").write_text("111", encoding="utf-8")
    (tmp_path / "222.lock").write_text("222", encoding="utf-8")

    with pytest.raises(bhp.ScanLockError):
        with bhp.scan_lock(tmp_path):
            pass

    # 거부된 시도는 자기 락을 만들지 않았어야 함 — 기존 2개만 유지
    assert len(list(tmp_path.glob("*.lock"))) == 2


def test_two_concurrent_runs_both_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(bhp, "_pid_alive", lambda pid: True)
    (tmp_path / "111.lock").write_text("111", encoding="utf-8")

    with bhp.scan_lock(tmp_path):
        locks = list(tmp_path.glob("*.lock"))
        assert len(locks) == 2  # 기존 1개 + 이번 실행 1개


def test_stale_lock_is_cleaned_before_counting(tmp_path, monkeypatch):
    monkeypatch.setattr(bhp, "_pid_alive", lambda pid: False)
    (tmp_path / "999999.lock").write_text("999999", encoding="utf-8")

    with bhp.scan_lock(tmp_path):
        locks = list(tmp_path.glob("*.lock"))
        assert len(locks) == 1
        assert locks[0].stem != "999999"


def test_force_parallel_bypasses_lock_check(tmp_path, monkeypatch):
    monkeypatch.setattr(bhp, "_pid_alive", lambda pid: True)
    (tmp_path / "111.lock").write_text("111", encoding="utf-8")
    (tmp_path / "222.lock").write_text("222", encoding="utf-8")

    with bhp.scan_lock(tmp_path, force=True):
        pass  # 락 검사 자체가 생략되므로 예외 없음


def test_lock_released_on_exception(tmp_path):
    with pytest.raises(ValueError):
        with bhp.scan_lock(tmp_path):
            raise ValueError("boom")
    assert list(tmp_path.glob("*.lock")) == []


def test_load_stems_filter_strips_blank_lines(tmp_path):
    p = tmp_path / "stems.txt"
    p.write_text("2026-03-19 23-00-50\n\n2026-03-22 03-02-03\n  \n", encoding="utf-8")
    assert bhp._load_stems_filter(p) == {"2026-03-19 23-00-50", "2026-03-22 03-02-03"}
