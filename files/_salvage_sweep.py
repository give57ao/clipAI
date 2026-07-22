# -*- coding: utf-8 -*-
"""R13 파라미터 스윕 (T3, SONNET_TASK_UNDERREAD.md).

측정 루프를 그대로(서브프로세스로) 반복 실행 — 소스 상수 4개를 임시로 치환한 뒤
"python hud_streak_salvage.py && python hud_round_settle.py && python hud_from_cache.py
&& python _compare_hud_gt.py"를 돌려 recall/precision과 탐지/FP 집합을 비교한다.
끝나면 원본 소스로 반드시 복원 + timeline 재생성.

수용 기준: 베이스라인 대비 탐지 집합 상실 0, FP 집합 순증 0, 셀프테스트 통과.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
SRC = HERE / "hud_streak_salvage.py"
ORIG = SRC.read_text(encoding="utf-8")

PATTERNS = {
    "step": re.compile(r"^_STEP_MAX_SEC = [\d.]+", re.M),
    "win": re.compile(r"^_WIN_AFTER_SEC = [\d.]+", re.M),
    "conf": re.compile(r"^_EVID_MIN_CONF = [\d.]+", re.M),
    "stable": re.compile(r"^_STABLE_SINGLE_CONF = [\d.]+", re.M),
}
BASE = dict(step=50.0, win=25.0, conf=0.55, stable=0.88)


def patch(step, win, conf, stable) -> None:
    txt = ORIG
    txt = PATTERNS["step"].sub(f"_STEP_MAX_SEC = {step}", txt, count=1)
    txt = PATTERNS["win"].sub(f"_WIN_AFTER_SEC = {win}", txt, count=1)
    txt = PATTERNS["conf"].sub(f"_EVID_MIN_CONF = {conf}", txt, count=1)
    txt = PATTERNS["stable"].sub(f"_STABLE_SINGLE_CONF = {stable}", txt, count=1)
    SRC.write_text(txt, encoding="utf-8")


def run(cmd) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, p.stdout + p.stderr


def selftests_ok() -> bool:
    rc1, out1 = run([sys.executable, "-u", "hud_streak_salvage.py"])
    rc2, out2 = run([sys.executable, "-u", "hud_round_settle.py"])
    return rc1 == 0 and rc2 == 0 and "OK" in out1 and "OK" in out2


def parse_compare(out: str):
    det, fp = set(), set()
    stem = None
    stats = {}
    for l in out.splitlines():
        m = re.match(r"## (.+?):", l)
        if m:
            stem = m.group(1)
        m = re.match(r"\s+GT (\S+)-(\S+) → R(\d+)", l)
        if m:
            det.add((stem, m.group(1)))
        m = re.match(r"\s+FP R(\d+) (\S+)", l)
        if m:
            fp.add((stem, m.group(1), m.group(2)))
        m = re.match(r"GT (\d+)건 \| 탐지 (\d+) \(recall ([\d.]+)%\) \| 검출 (\d+)건 중 TP (\d+) \(precision ([\d.]+)%\)", l)
        if m and not stats:
            stats = {
                "gt": int(m.group(1)), "hit": int(m.group(2)), "recall": float(m.group(3)),
                "det": int(m.group(4)), "tp": int(m.group(5)), "precision": float(m.group(6)),
            }
    return det, fp, stats


def regen_and_compare():
    run([sys.executable, "-u", "hud_from_cache.py"])
    rc, out = run([sys.executable, "-u", "_compare_hud_gt.py"])
    return parse_compare(out)


def main() -> int:
    try:
        patch(**BASE)
        assert selftests_ok(), "베이스라인 셀프테스트 실패 — 스윕 중단"
        base_det, base_fp, base_stats = regen_and_compare()
        print(f"[baseline] {base_stats}")

        combos = []
        for step in (40.0, 50.0, 60.0, 75.0):
            for win in (20.0, 25.0, 30.0):
                for conf in (0.50, 0.55, 0.60):
                    for stable in (0.85, 0.88):
                        c = dict(step=step, win=win, conf=conf, stable=stable)
                        if c != BASE:
                            combos.append(c)

        results = []
        for c in combos:
            patch(**c)
            if not selftests_ok():
                results.append({**c, "verdict": "selftest_fail"})
                print(f"[REJECT] {c} -> 셀프테스트 실패")
                continue
            det, fp, stats = regen_and_compare()
            lost = base_det - det
            new_fp = fp - base_fp
            new_det_not_gt = det - base_det  # GT엔 있는데 베이스라인엔 없던 탐지 = 순수 개선분
            ok = not lost and not new_fp
            results.append({**c, "stats": stats, "lost": sorted(lost), "new_fp": sorted(new_fp),
                             "new_det": sorted(new_det_not_gt), "ok": ok})
            tag = "OK" if ok else "REJECT"
            print(f"[{tag}] {c} -> {stats} lost={sorted(lost)} new_fp={sorted(new_fp)} "
                  f"new_det={sorted(new_det_not_gt)}")

        accepted = [r for r in results if r.get("ok")]
        accepted.sort(key=lambda r: (-r["stats"].get("tp", 0)))
        print(f"\n===== 채택 가능 {len(accepted)}/{len(combos)} =====")
        for r in accepted[:10]:
            print(f"  step={r['step']} win={r['win']} conf={r['conf']} stable={r['stable']} "
                  f"-> {r['stats']} new_det={r['new_det']}")
    finally:
        SRC.write_text(ORIG, encoding="utf-8")
        run([sys.executable, "-u", "hud_from_cache.py"])  # 기본값으로 timeline 복원
        print("\n[복원] 원본 상수 + timeline 재생성 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
