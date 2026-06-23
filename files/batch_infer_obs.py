# -*- coding: utf-8 -*-
"""1단계 학습 완료 후 E:\\OBS mp4 배치 추론 (동시 N개).

MAX_VIDEOS>0 이면 파일명 정렬 기준 상위 N개만 처리 (파일럿).
0 이면 OBS 폴더 전체(82개) 처리.
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

OBS_DIR = Path(r"E:\OBS")
RESULT_ROOT = Path(r"E:\clipai_result")
DATASET_ROOT = Path(r"E:\Highlights\ml_dataset")
INFER_SCRIPT = Path(r"C:\clipAI\files\infer_highlights.py")
PARALLEL = 3
MAX_VIDEOS = 3  # 파일럿: 3개만. 전체 실행 시 0
LOG_ROOT = RESULT_ROOT / "_logs"


def wait_for_binary_training() -> None:
    print("[batch] waiting for train_binary.py...", flush=True)
    while True:
        running = False
        try:
            out = subprocess.run(
                ["wmic", "process", "where", "name='python.exe'", "get", "commandline"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if "train_binary.py" in (out.stdout or ""):
                running = True
        except OSError:
            pass
        if not running:
            break
        time.sleep(30)
    time.sleep(5)
    print("[batch] train_binary.py finished", flush=True)


def run_one_infer(mp4_path: Path) -> tuple[str, int]:
    stem = mp4_path.stem
    out_dir = RESULT_ROOT / f"{stem}_하이라이트"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{stem}.log"

    cmd = [
        sys.executable,
        "-u",
        str(INFER_SCRIPT),
        str(mp4_path),
        "--dataset-root",
        str(DATASET_ROOT),
        "--output-dir",
        str(out_dir),
        "--binary-only",
        "--stride-sec",
        "8",
        "--binary-threshold",
        "0.55",
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    return mp4_path.name, result.returncode


def main() -> int:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    if not INFER_SCRIPT.exists():
        print(f"[batch] missing: {INFER_SCRIPT}")
        return 1

    wait_for_binary_training()

    all_mp4 = sorted(OBS_DIR.glob("*.mp4"))
    if not all_mp4:
        print(f"[batch] no mp4 in {OBS_DIR}")
        return 0

    mp4_files = all_mp4[:MAX_VIDEOS] if MAX_VIDEOS > 0 else all_mp4
    if MAX_VIDEOS > 0:
        print(
            f"[batch] pilot: {len(mp4_files)}/{len(all_mp4)} videos "
            f"(MAX_VIDEOS={MAX_VIDEOS}) parallel={PARALLEL}",
            flush=True,
        )
    else:
        print(f"[batch] full run: {len(mp4_files)} videos parallel={PARALLEL}", flush=True)
    failed: list[str] = []

    for batch_start in range(0, len(mp4_files), PARALLEL):
        batch = mp4_files[batch_start : batch_start + PARALLEL]
        group_num = batch_start // PARALLEL + 1
        group_total = (len(mp4_files) + PARALLEL - 1) // PARALLEL
        print(f"[batch] group {group_num}/{group_total}", flush=True)

        with ProcessPoolExecutor(max_workers=len(batch)) as executor:
            futures = {executor.submit(run_one_infer, path): path for path in batch}
            for future in as_completed(futures):
                name, code = future.result()
                if code != 0:
                    failed.append(name)
                    print(f"  FAIL: {name} (exit {code})", flush=True)
                else:
                    print(f"  OK: {name}", flush=True)

    if MAX_VIDEOS > 0:
        print(
            f"[batch] pilot complete ({len(mp4_files)} videos) -> {RESULT_ROOT}",
            flush=True,
        )
        print(
            "[batch] review results, then set MAX_VIDEOS=0 for full OBS run",
            flush=True,
        )
    else:
        print(f"[batch] done -> {RESULT_ROOT}", flush=True)
    if failed:
        print(f"[batch] failed ({len(failed)}): {', '.join(failed)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
