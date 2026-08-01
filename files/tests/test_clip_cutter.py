# -*- coding: utf-8 -*-
"""clip_cutter.py(수동 구간 자르기 exe) 단위 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest

import clip_cutter


def test_parse_timecode_minutes_seconds():
    assert clip_cutter.parse_timecode("13:45") == 825.0


def test_parse_timecode_hours_minutes_seconds():
    assert clip_cutter.parse_timecode("1:8:55") == 4135.0


def test_parse_timecode_invalid_format_raises():
    with pytest.raises(ValueError):
        clip_cutter.parse_timecode("abc")


def test_parse_timecode_too_many_parts_raises():
    with pytest.raises(ValueError):
        clip_cutter.parse_timecode("1:2:3:4")


def test_split_path_and_ranges_basic():
    line = r"E:\OBS\2026-06-30 22-24-03.mp4        13:45 - 14:20, 25:18 - 25:35"

    result = clip_cutter.split_path_and_ranges(line)

    assert result == (
        r"E:\OBS\2026-06-30 22-24-03.mp4",
        "13:45 - 14:20, 25:18 - 25:35",
    )


def test_split_path_and_ranges_no_mp4_returns_none():
    assert clip_cutter.split_path_and_ranges("이건 그냥 텍스트") is None


def test_parse_ranges_multiple_with_trailing_comma():
    result = clip_cutter.parse_ranges("13:45 - 14:20, 25:18 - 25:35, ")

    assert result == [
        ("13:45 - 14:20", 825.0, 860.0, None),
        ("25:18 - 25:35", 1518.0, 1535.0, None),
    ]


def test_parse_ranges_none_keyword_returns_empty():
    assert clip_cutter.parse_ranges("없음") == []


def test_parse_ranges_empty_string_returns_empty():
    assert clip_cutter.parse_ranges("") == []


def test_parse_ranges_invalid_token_marks_error():
    result = clip_cutter.parse_ranges("abc")

    assert result == [("abc", None, None, "형식 오류")]


def test_parse_ranges_end_before_start_marks_error():
    result = clip_cutter.parse_ranges("14:20 - 13:45")

    assert result == [("14:20 - 13:45", 860.0, 825.0, "시작>=종료")]


def test_format_start_label_minutes_seconds():
    assert clip_cutter.format_start_label(825.0) == "13m45s"


def test_format_start_label_with_hours():
    assert clip_cutter.format_start_label(4135.0) == "1h08m55s"


def test_build_output_path():
    result = clip_cutter.build_output_path(
        Path("E:/clipai_result/manual_clips"), "video", 825.0
    )

    assert result == Path("E:/clipai_result/manual_clips/video_13m45s.mp4")


def test_build_ffmpeg_command_applies_preroll_and_duration():
    cmd = clip_cutter.build_ffmpeg_command(
        "ffmpeg.exe", Path("in.mp4"), 100.0, 130.0, Path("out.mp4")
    )

    assert cmd[0] == "ffmpeg.exe"
    assert cmd[cmd.index("-ss") + 1] == "98.000"
    assert cmd[cmd.index("-i") + 1] == "in.mp4"
    assert cmd[cmd.index("-t") + 1] == "32.000"
    assert cmd[cmd.index("-c") + 1] == "copy"
    assert "-avoid_negative_ts" in cmd
    assert cmd[-1] == "out.mp4"


def test_build_ffmpeg_command_clamps_negative_start():
    cmd = clip_cutter.build_ffmpeg_command(
        "ffmpeg.exe", Path("in.mp4"), 1.0, 10.0, Path("out.mp4")
    )

    assert cmd[cmd.index("-ss") + 1] == "0.000"


def test_build_ffmpeg_command_duration_matches_requested_end_when_clamped():
    cmd = clip_cutter.build_ffmpeg_command(
        "ffmpeg.exe", Path("in.mp4"), 1.0, 10.0, Path("out.mp4")
    )

    assert cmd[cmd.index("-ss") + 1] == "0.000"
    assert cmd[cmd.index("-t") + 1] == "10.000"


def test_process_txt_file_skips_missing_source_file(tmp_path):
    txt = tmp_path / "req.txt"
    txt.write_text(r"E:\OBS\gone.mp4   13:45 - 14:20" + "\n", encoding="utf-8")

    success, skipped = clip_cutter.process_txt_file(txt, tmp_path, "ffmpeg.exe")

    assert success == 0
    assert skipped == 1


def test_process_txt_file_skips_none_marker(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    txt = tmp_path / "req.txt"
    txt.write_text(f"{video}   없음\n", encoding="utf-8")

    success, skipped = clip_cutter.process_txt_file(txt, tmp_path, "ffmpeg.exe")

    assert success == 0
    assert skipped == 0


def test_process_txt_file_runs_ffmpeg_for_valid_range(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    txt = tmp_path / "req.txt"
    txt.write_text(f"{video}   13:45 - 14:20\n", encoding="utf-8")

    captured = {}

    class FakeCompleted:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeCompleted()

    monkeypatch.setattr(clip_cutter.subprocess, "run", fake_run)

    success, skipped = clip_cutter.process_txt_file(txt, tmp_path, "ffmpeg.exe")

    assert success == 1
    assert skipped == 0
    assert captured["cmd"][0] == "ffmpeg.exe"
    assert str(video) in captured["cmd"]


def test_process_txt_file_skips_ffmpeg_failure(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    txt = tmp_path / "req.txt"
    txt.write_text(f"{video}   13:45 - 14:20\n", encoding="utf-8")

    class FakeCompleted:
        returncode = 1
        stderr = "boom"

    monkeypatch.setattr(clip_cutter.subprocess, "run", lambda cmd, **kwargs: FakeCompleted())

    success, skipped = clip_cutter.process_txt_file(txt, tmp_path, "ffmpeg.exe")

    assert success == 0
    assert skipped == 1


def test_main_no_args_prints_usage(monkeypatch, capsys):
    monkeypatch.setattr(clip_cutter.sys, "argv", ["clip_cutter.py"])
    monkeypatch.setattr(clip_cutter, "_wait_for_key", lambda: None)

    rc = clip_cutter.main()

    out = capsys.readouterr().out
    assert rc == 0
    assert "드래그하세요" in out


def test_main_skips_non_txt_files(tmp_path, monkeypatch, capsys):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    monkeypatch.setattr(clip_cutter.sys, "argv", ["clip_cutter.py", str(video)])
    monkeypatch.setattr(clip_cutter, "find_ffmpeg", lambda: "ffmpeg.exe")
    monkeypatch.setattr(clip_cutter, "_wait_for_key", lambda: None)

    rc = clip_cutter.main()

    out = capsys.readouterr().out
    assert rc == 0
    assert "txt 아님" in out
    assert "처리할 .txt 파일이 없습니다." in out


def test_main_shows_folder_prompt_only_when_success_gt_zero(tmp_path, monkeypatch, capsys):
    txt = tmp_path / "req.txt"
    txt.write_text("dummy\n", encoding="utf-8")
    monkeypatch.setattr(clip_cutter.sys, "argv", ["clip_cutter.py", str(txt)])
    monkeypatch.setattr(clip_cutter, "find_ffmpeg", lambda: "ffmpeg.exe")
    monkeypatch.setattr(clip_cutter, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(clip_cutter, "process_txt_file", lambda *a, **k: (0, 1))
    monkeypatch.setattr(clip_cutter, "_wait_for_key", lambda: None)

    rc = clip_cutter.main()

    out = capsys.readouterr().out
    assert rc == 0
    assert "결과 폴더" not in out
    assert "처리할 구간이 없습니다." in out


def test_main_folder_open_guarded_against_oserror(tmp_path, monkeypatch, capsys):
    txt = tmp_path / "req.txt"
    txt.write_text("dummy\n", encoding="utf-8")
    monkeypatch.setattr(clip_cutter.sys, "argv", ["clip_cutter.py", str(txt)])
    monkeypatch.setattr(clip_cutter, "find_ffmpeg", lambda: "ffmpeg.exe")
    monkeypatch.setattr(clip_cutter, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(clip_cutter, "process_txt_file", lambda *a, **k: (1, 0))
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    def boom(path):
        raise OSError("no such folder")

    monkeypatch.setattr(clip_cutter.os, "startfile", boom, raising=False)
    monkeypatch.setattr(clip_cutter, "_wait_for_key", lambda: None)

    rc = clip_cutter.main()

    out = capsys.readouterr().out
    assert rc == 0
    assert "폴더를 열 수 없습니다" in out
