# -*- coding: utf-8 -*-
"""1c 단계: ace_rounds → ffmpeg 클립 자동 추출.

사용법:
    python extract_ace_clips.py <kill_timeline.json> [옵션]

예:
    python extract_ace_clips.py "E:\\clipai_result\\kill_timeline\\2026-03-21 00-40-56_v1.json"
        --output-dir "E:\\clipai_result\\ace_clips"
        --pre-sec 10 --post-sec 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


DEFAULT_OUTPUT_DIR = Path(r"E:\clipai_result\ace_clips")
# 킬은 스코어보드가 뜨기 "전" 라운드 플레이 중에 발생한다.
# 따라서 스코어보드 시작 시점에서 LEAD초 앞으로 거슬러 플레이 구간을 담는다.
DEFAULT_LEAD_SEC = 35.0   # 스코어보드 시작 전 최대 몇 초 (라운드 플레이/킬 구간)
DEFAULT_TAIL_SEC = 4.0    # 확인용 스코어보드 노출 (시작 후 몇 초)
MIN_CLIP_SEC = 12.0       # 최소 클립 길이
DEFAULT_MAX_LEAD_SEC = 75.0  # 라운드 전체 포함을 위해 lead 자동 확장 시 상한


def sec_to_mss(sec: float) -> str:
    total = int(sec)
    return f"{total // 60}:{total % 60:02d}"


def extract_clip(
    video_path: Path,
    clip_start: float,
    clip_end: float,
    out_path: Path,
) -> bool:
    clip_start = max(0.0, clip_start)
    duration = max(MIN_CLIP_SEC, clip_end - clip_start)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{clip_start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [오류] ffmpeg 실패: {result.stderr[-300:]}")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="ace_rounds → ffmpeg 클립 추출")
    parser.add_argument("timeline_json", help="kill_timeline JSON 파일")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--lead-sec", type=float, default=DEFAULT_LEAD_SEC,
                        help="스코어보드 시작 전 플레이(킬) 포함 최대 초 (기본 35)")
    parser.add_argument("--tail-sec", type=float, default=DEFAULT_TAIL_SEC,
                        help="확인용 스코어보드 노출 초 (기본 4)")
    parser.add_argument(
        "--max-lead-sec",
        type=float,
        default=DEFAULT_MAX_LEAD_SEC,
        help="라운드 전체 포함을 위해 lead 자동 확장 시 상한 (기본 75)",
    )
    parser.add_argument("--skip-errors", action="store_true",
                        help="k_read_error 라운드도 포함 (기본 제외)")
    parser.add_argument(
        "--include-candidates",
        action="store_true",
        help="ace_candidates(후보)도 함께 추출 (기본: ace_rounds만)",
    )
    args = parser.parse_args()

    json_path = Path(args.timeline_json)
    if not json_path.exists():
        print(f"[오류] 파일 없음: {json_path}")
        return 1

    data = json.loads(json_path.read_text(encoding="utf-8"))
    video_path = Path(data["video_path"])
    player_nick = data.get("player_nick", "unknown")
    rounds = data.get("rounds", [])
    ace_round_indices = set(data.get("ace_rounds", []))
    cand_round_indices = set(data.get("ace_candidates", [])) if args.include_candidates else set()
    k_error_indices = set(data.get("k_error_rounds", []))

    if not video_path.exists():
        print(f"[오류] 영상 없음: {video_path}")
        return 1

    # ace_rounds(+candidates) 중 k_read_error 제외 (기본)
    target = ace_round_indices | cand_round_indices
    valid_ace = target - (k_error_indices if not args.skip_errors else set())

    if not valid_ace:
        print(
            f"[1c] 추출할 라운드 없음 "
            f"(ace={sorted(ace_round_indices)}, cand={sorted(cand_round_indices)}, errors={sorted(k_error_indices)})"
        )
        return 0

    output_dir = Path(args.output_dir) / video_path.stem
    video_stem = video_path.stem

    print(f"[1c] {video_path.name}")
    print(
        f"     player={player_nick!r}  "
        f"ace={sorted(ace_round_indices)}  cand={sorted(cand_round_indices)}  "
        f"extract={sorted(valid_ace)}"
    )

    # round_index → 직전 라운드 스코어보드 종료 시각 (라운드 플레이 시작 경계)
    prev_end_by_round: dict[int, float] = {}
    last_end = 0.0
    for rnd in rounds:
        prev_end_by_round[rnd["round_index"]] = last_end
        last_end = rnd["scoreboard_end_sec"]

    ok_count = 0
    for rnd in rounds:
        ridx = rnd["round_index"]
        if ridx not in valid_ace:
            continue
        sb_start = rnd["scoreboard_start_sec"]
        # 클립 시작 = 스코어보드 시작 - lead, 단 직전 스코어보드 종료보다 앞서지 않게
        round_floor = prev_end_by_round.get(ridx, 0.0)
        # '앞의 2킬이 잘림' 방지:
        # - 기본 lead_sec(35)만 쓰면 라운드가 40~60초일 때 초반 킬이 빠질 수 있음
        # - 라운드 플레이 구간(직전 SB 종료~이번 SB 시작)을 가능한 한 포함
        play_gap = max(0.0, sb_start - round_floor)
        auto_lead = min(args.max_lead_sec, play_gap)
        lead = max(args.lead_sec, auto_lead)
        clip_start = max(round_floor, sb_start - lead)
        clip_end = sb_start + args.tail_sec  # 확인용 스코어보드 살짝 노출

        time_tag = sec_to_mss(sb_start).replace(":", "m") + "s"
        out_name = f"{video_stem}_R{ridx:02d}_{time_tag}_ace.mp4"
        out_path = output_dir / out_name

        print(f"  R{ridx:02d} 플레이 {sec_to_mss(clip_start)}~SB {sec_to_mss(sb_start)} "
              f"→ {out_path.name} ...", end=" ", flush=True)
        if extract_clip(video_path, clip_start, clip_end, out_path):
            print("OK")
            ok_count += 1
        else:
            print("FAIL")

    print(f"\n[1c] 완료 {ok_count}/{len(valid_ace)} 클립 → {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
