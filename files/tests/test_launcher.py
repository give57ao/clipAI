# -*- coding: utf-8 -*-
"""launcher.py(드래그앤드롭 exe 실행기) 단위 테스트."""

from __future__ import annotations

import sys
from pathlib import Path

import launcher


def test_filter_video_paths_accepts_existing_mp4(tmp_path):
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"")

    valid, skipped = launcher.filter_video_paths([str(v)])

    assert valid == [v.resolve()]
    assert skipped == []


def test_filter_video_paths_skips_non_mp4(tmp_path):
    t = tmp_path / "notes.txt"
    t.write_bytes(b"")

    valid, skipped = launcher.filter_video_paths([str(t)])

    assert valid == []
    assert len(skipped) == 1
    assert "notes.txt" in skipped[0]
    assert "영상 파일 아님" in skipped[0]


def test_filter_video_paths_skips_missing_file(tmp_path):
    missing = tmp_path / "gone.mp4"

    valid, skipped = launcher.filter_video_paths([str(missing)])

    assert valid == []
    assert len(skipped) == 1
    assert "파일 없음" in skipped[0]


def test_filter_video_paths_mixed_inputs(tmp_path):
    good = tmp_path / "good.mp4"
    good.write_bytes(b"")
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"")

    valid, skipped = launcher.filter_video_paths([str(good), str(bad)])

    assert valid == [good.resolve()]
    assert len(skipped) == 1


def test_run_pipeline_invokes_venv_python_with_files_list(tmp_path, monkeypatch):
    v1 = tmp_path / "a.mp4"
    v1.write_bytes(b"")
    captured = {}

    class FakeCompleted:
        returncode = 0

    def fake_run(cmd, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        # subprocess.run에 넘겨진 --files-list 인자의 임시파일 내용을 검증
        list_path = Path(cmd[cmd.index("--files-list") + 1])
        captured["list_contents"] = list_path.read_text(encoding="utf-8")
        return FakeCompleted()

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    rc = launcher.run_pipeline([v1])

    assert rc == 0
    cmd = captured["cmd"]
    assert cmd[0] == str(launcher.PIPELINE_PYTHON)
    assert str(launcher.PIPELINE_SCRIPT) in cmd
    assert "--files-list" in cmd
    assert str(v1) in captured["list_contents"]
    # 임시 목록 파일은 실행 후 정리돼야 함
    list_path = Path(cmd[cmd.index("--files-list") + 1])
    assert not list_path.exists()


def test_main_skips_folder_prompt_and_returns_code_on_pipeline_failure(
    tmp_path, monkeypatch, capsys
):
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", ["launcher.py", str(v)])
    monkeypatch.setattr(launcher, "run_pipeline", lambda video_paths: 3)
    monkeypatch.setattr(launcher, "_wait_for_key", lambda: None)

    def fail_input(prompt=""):
        raise AssertionError("파이프라인 실패 시에는 결과 폴더 프롬프트를 띄우면 안 됨")

    monkeypatch.setattr("builtins.input", fail_input)

    rc = launcher.main()

    out = capsys.readouterr().out
    assert rc == 3
    assert "결과 폴더" not in out


def test_main_shows_folder_prompt_on_pipeline_success(tmp_path, monkeypatch, capsys):
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", ["launcher.py", str(v)])
    monkeypatch.setattr(launcher, "run_pipeline", lambda video_paths: 0)
    monkeypatch.setattr(launcher, "_wait_for_key", lambda: None)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")

    rc = launcher.main()

    out = capsys.readouterr().out
    assert rc == 0
    assert "결과 폴더" in out
