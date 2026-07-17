# -*- coding: utf-8 -*-
"""_build_review_ledger 파싱/GT매칭 회귀 (SONNET_TASKS.md T6)."""

from __future__ import annotations

import json

import _build_review_ledger as brl


def test_parse_ace_clip_plain():
    row = brl._parse_ace_clip("2026-03-21 00-40-56_R36_38m02s_ace.mp4")
    assert row == {
        "stem": "2026-03-21 00-40-56", "round": 36, "t": 38 * 60 + 2,
        "verdict": "unreviewed", "note": "",
    }


def test_parse_ace_clip_hud_ace():
    row = brl._parse_ace_clip("2026-03-26 01-26-52_R35_26m24s_hud_ace.mp4")
    assert row["verdict"] == "unreviewed"
    assert row["round"] == 35
    assert row["t"] == 26 * 60 + 24


def test_parse_ace_clip_odap_no_underscore():
    row = brl._parse_ace_clip("2026-03-24 00-42-33_R39_33m46s_ace오답.mp4")
    assert row["verdict"] == "fp"
    assert row["note"] == ""


def test_parse_ace_clip_odap_with_note():
    row = brl._parse_ace_clip("2026-04-08 02-15-28_R34_28m09s_ace_오답_라운드넘어감.mp4")
    assert row["verdict"] == "fp"
    assert row["note"] == "라운드넘어감"


def test_parse_ace_clip_non_matching_returns_none():
    assert brl._parse_ace_clip("2026-01-08 02-33-22_하이라이트(1).mp4") is None
    assert brl._parse_ace_clip("02-21-23_miss_54m20-54m41.mp4") is None
    assert brl._parse_ace_clip("2026-04-08 03-23-57_R055_킬38m06s.mp4") is None


def test_matches_gt_within_tolerance():
    gt = {"stem1": [(100.0, 130.0)]}
    assert brl._matches_gt("stem1", 128.0, gt) is True    # 구간 안
    assert brl._matches_gt("stem1", 140.0, gt) is True    # 허용오차(15s) 안
    assert brl._matches_gt("stem1", 200.0, gt) is False   # 범위 밖
    assert brl._matches_gt("stem2", 110.0, gt) is False   # 다른 stem


def test_build_ledger_row_count_equals_scanned_files(tmp_path):
    d = tmp_path / "ace_clips" / "stemA"
    d.mkdir(parents=True)
    (d / "stemA_R01_1m00s_ace.mp4").write_bytes(b"")
    (d / "stemA_R02_2m00s_ace_오답_note.mp4").write_bytes(b"")
    (d / "stemA_하이라이트(1).mp4").write_bytes(b"")  # 미파싱

    gt_path = tmp_path / "gt_aces.json"
    gt_path.write_text(
        json.dumps({"stemA": {"spans": [[60.0, 62.0]], "source_available": True}}),
        encoding="utf-8",
    )

    rows = brl.build_ledger(tmp_path, gt_path)
    assert len(rows) == 3  # 스캔된 mp4 수와 항상 동일

    by_round = {r["round"]: r for r in rows if r["round"] is not None}
    assert by_round[1]["verdict"] == "tp"   # GT span과 일치 → 자동 tp
    assert by_round[2]["verdict"] == "fp"   # 오답 태그 유지(GT 일치와 무관하게 사람 판정 우선)

    unparsed = [r for r in rows if r["stem"] is None]
    assert len(unparsed) == 1
    assert "하이라이트" in unparsed[0]["note"]


def test_build_ledger_fp_conflicting_with_gt_is_flagged(tmp_path):
    d = tmp_path / "ace_clips_candidates" / "stemB"
    d.mkdir(parents=True)
    (d / "stemB_R05_1m00s_ace_오답_잘못봄.mp4").write_bytes(b"")

    gt_path = tmp_path / "gt_aces.json"
    gt_path.write_text(
        json.dumps({"stemB": {"spans": [[59.0, 61.0]], "source_available": True}}),
        encoding="utf-8",
    )

    rows = brl.build_ledger(tmp_path, gt_path)
    assert len(rows) == 1
    assert rows[0]["verdict"] == "fp"
    assert "GT_CONFLICT" in rows[0]["note"]
