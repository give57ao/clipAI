# -*- coding: utf-8 -*-
"""hud_cache_io 왕복 + hud_from_cache 하위호환 검증 (SONNET_TASKS.md T3-1)."""

from __future__ import annotations

import json

from detect_ace_hud import KRead
from hud_cache_io import load_scan_cache, save_scan_cache
from hud_from_cache import load_reads as load_reads_compat


def _sample_reads() -> list[KRead]:
    return [
        KRead(t=1.2345, k=5, conf=0.9123, method="template", d=1, a=2),
        KRead(t=2.5, k=None, conf=0.0, method="row_miss", d=None, a=None),
    ]


def _sample_verdicts() -> list[list]:
    return [[10.0, 12.5, True], [30.0, 33.0, False]]


def _sample_score_events() -> list[dict]:
    return [
        {"t_lo": 5.0, "t_hi": 6.2, "side": "R", "kind": "win", "old": 3, "new": 4},
        {"t_lo": 40.0, "t_hi": 41.0, "side": "B", "kind": "reset", "old": 10, "new": None},
    ]


def test_save_load_round_trip(tmp_path):
    reads = _sample_reads()
    verdicts = _sample_verdicts()
    events = _sample_score_events()

    out = save_scan_cache(
        "sample_stem", reads, duration=123.456, scan_fps=4.0, cache_dir=tmp_path,
        boundary_verdicts=verdicts, score_win_events=events,
    )
    assert out == tmp_path / "sample_stem.json"

    loaded = load_scan_cache(out)
    assert loaded["version"] == 2
    assert loaded["stem"] == "sample_stem"
    assert loaded["scan_fps"] == 4.0
    assert loaded["duration"] == 123.456
    assert loaded["boundary_verdicts"] == verdicts
    assert loaded["score_win_events"] == events

    assert len(loaded["reads"]) == len(reads)
    for orig, back in zip(reads, loaded["reads"]):
        assert back.t == round(orig.t, 3)
        assert back.k == orig.k
        assert back.conf == round(orig.conf, 3)
        assert back.method == orig.method
        assert back.d == orig.d
        assert back.a == orig.a


def test_hud_from_cache_load_reads_compat(tmp_path):
    """확장 캐시(hud_cache_io 산출물)를 hud_from_cache.load_reads로도 읽을 수 있어야 함."""
    reads = _sample_reads()
    out = save_scan_cache("compat_stem", reads, duration=50.0, scan_fps=4.0, cache_dir=tmp_path)

    back_reads, duration, scan_fps, stem = load_reads_compat(out)
    assert stem == "compat_stem"
    assert duration == 50.0
    assert scan_fps == 4.0
    assert len(back_reads) == len(reads)
    for orig, back in zip(reads, back_reads):
        assert back.t == round(orig.t, 3)
        assert back.k == orig.k
        assert back.method == orig.method
        assert back.d == orig.d
        assert back.a == orig.a


def test_v1_schema_backward_compat(tmp_path):
    """구 sig_cache(v1, verdicts/events 키 없음)도 읽히고 None으로 채워져야 함."""
    data = {
        "stem": "legacy",
        "scan_fps": 4.0,
        "duration": 10.0,
        "reads": [[1.0, 3, 0.9, "T", None, None]],
    }
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_scan_cache(p)
    assert loaded["version"] == 1
    assert loaded["boundary_verdicts"] is None
    assert loaded["score_win_events"] is None
    assert loaded["reads"][0].k == 3
    assert loaded["reads"][0].method == "template"
