# -*- coding: utf-8 -*-
"""batch_hud_ace_pipeline --files-list 옵션 회귀 테스트."""

from __future__ import annotations

from pathlib import Path

import batch_hud_ace_pipeline as bhp


def test_load_files_list_reads_absolute_paths(tmp_path):
    v1 = tmp_path / "a.mp4"
    v2 = tmp_path / "b.mp4"
    v1.write_bytes(b"")
    v2.write_bytes(b"")
    list_file = tmp_path / "files.txt"
    list_file.write_text(f"{v1}\n\n{v2}\n  \n", encoding="utf-8")

    result = bhp._load_files_list(list_file)

    assert result == [v1, v2]


def test_load_files_list_empty_file_returns_empty_list(tmp_path):
    list_file = tmp_path / "empty.txt"
    list_file.write_text("", encoding="utf-8")

    assert bhp._load_files_list(list_file) == []
