# -*- coding: utf-8 -*-
"""샘플별 닉 추출 실패 원인 분석."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from scouter_nick import read_scouter


def classify_fail(readout, min_conf: float) -> str:
    if readout.mode == "unknown":
        if not readout.rows:
            return "unknown_no_ocr"
        if not any("스카" in t or "scouter" in t.lower() for t, _ in readout.rows):
            return "unknown_no_header"
        return "unknown_header_ocr_garbled"
    if readout.mode == "scouter" and not readout.player_nick:
        return "scouter_dot_miss"
    if readout.player_nick and readout.player_conf < min_conf:
        return f"low_conf"
    if not readout.player_nick:
        return "empty_nick"
    return "ok"


def analyze_video(video_path: Path, max_samples: int, min_conf: float) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": "open_failed"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_total / fps if frame_total > 0 else 0.0

    t_start = min(60.0, duration * 0.05)
    t_end = max(t_start + 1.0, duration - 30.0)
    sample_times = np.linspace(t_start, t_end, num=max_samples)

    reasons: Counter = Counter()
    details: list[dict] = []

    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok or frame is None:
            reasons["frame_read_fail"] += 1
            details.append({"sec": round(float(t), 1), "reason": "frame_read_fail"})
            continue

        r = read_scouter(frame)
        reason = classify_fail(r, min_conf)
        reasons[reason] += 1
        details.append(
            {
                "sec": round(float(t), 1),
                "reason": reason,
                "mode": r.mode,
                "conf": round(r.player_conf, 3),
                "nick": r.player_nick,
                "rows": r.rows[:5],
            }
        )

    cap.release()
    ok_n = reasons.get("ok", 0)
    return {
        "name": video_path.name,
        "duration_sec": round(duration, 1),
        "samples": max_samples,
        "ok": ok_n,
        "fail": max_samples - ok_n,
        "reasons": dict(reasons),
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--min-conf", type=float, default=0.3)
    parser.add_argument(
        "--output",
        default=r"E:\Highlights\ml_dataset\manifests\ocr_nick_miss_analysis.txt",
    )
    args = parser.parse_args()

    lines = [
        "# 닉 추출 실패 원인 분석",
        f"# min_conf={args.min_conf}  samples={args.max_samples}",
        "",
    ]

    for v in args.videos:
        result = analyze_video(Path(v), args.max_samples, args.min_conf)
        lines.append(f"## {result['name']}  dur={result['duration_sec']}s")
        lines.append(f"성공: {result['ok']}/{result['samples']}  실패: {result['fail']}")
        fail_reasons = {k: v for k, v in result["reasons"].items() if k != "ok"}
        lines.append(f"실패 사유 집계: {fail_reasons}")
        lines.append("")
        for d in result["details"]:
            mark = "OK" if d["reason"] == "ok" else "MISS"
            nick = d.get("nick", "")
            lines.append(
                f"  [{mark}] {d['sec']:7.1f}s  {d['reason']:28s} "
                f"mode={d.get('mode','?'):10s} conf={d.get('conf',0)} nick={nick!r}"
            )
            if d["reason"] != "ok" and d.get("rows"):
                lines.append(f"         rows={d['rows']}")
        lines.append("")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
