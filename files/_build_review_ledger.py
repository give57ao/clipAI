# -*- coding: utf-8 -*-
"""ace_clips*/ mp4 전수 스캔 → 검수 대장(review_ledger.csv) 생성 (SONNET_TASKS.md T6,
IMPROVEMENT_REPORT.md §D-3).

**현상**(§D-3): `E:\\clipai_result` 클립 중 파일명 태그는 `오답` 33건뿐, `정답` 태그는 0건 —
태그 없음이 "정답"인지 "미검수"인지 파일명만으로는 구분 불가. 이 스크립트는 파일명
인코딩을 버리고 검수 대장(CSV) 하나로 이관한다.

(SONNET_TASKS.md 원문의 "기존 오답 29건"은 오기 — IMPROVEMENT_REPORT.md:196 원본 실측이
33건이고, 이 스크립트로 재확인해도 33건. 아래 acceptance는 33건 기준.)

읽기 전용 — 클립 파일 리네임·이동·삭제 없음(요구사항).

스캔 대상: `E:\\clipai_result\\ace_clips*\\**\\*.mp4`
(ace_clips/ace_clips_candidates — 레거시 ML 파이프라인, ace_clips_hud — 현행 HUD 파이프라인).

파일명 컨벤션은 도구마다 다르다:
    - ace-clip 후보(현행+레거시 공통 골격): `{stem}_R{round}_{M}m{S}s_(hud_ace|ace)[_오답[_설명]].mp4`
    - 그 외(하이라이트 수동추출, miss_피드백, 탐색/확인 진단 클립 등)는 round/시각 개념이 없어
      파싱 불가 — verdict=unreviewed로 대장에는 포함하되 note에 원인을 남겨 행 수 불변을 지킨다
      (필드 없이 조용히 건너뛰면 "행 수 = 스캔된 mp4 수" 수용 기준이 깨짐).

사용:
    python -u _build_review_ledger.py                # review_ledger.csv 생성
    python -u _build_review_ledger.py --dry-run       # 파일 미생성, 통계만 출력
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

CLIPS_ROOT = Path(r"E:\clipai_result")
CLIPS_GLOB = "ace_clips*"
GT_ACES_PATH = Path(__file__).parent / "gt_aces.json"
OUT_PATH = Path(r"E:\clipai_result\review_ledger.csv")

# 클립 파일명 시각(라운드 종료)과 GT span 경계 사이 허용 오차 — GT span은 킬 구간
# 자체를, 파일명은 라운드 종료(ace_clip_window 이후) 시각을 담아 초 단위로 어긋날 수 있음.
_GT_MATCH_TOLERANCE_SEC = 15.0

# 현행(detect_ace_hud.extract_ace_clips)·레거시 ace-clip 추출기가 공유하는 파일명 골격.
_ACE_CLIP_RE = re.compile(
    r"^(?P<stem>.+)_R(?P<round>\d+)_(?P<min>\d+)m(?P<sec>\d+)s_(?:hud_ace|ace)(?P<tag>.*)$"
)
_ODAP_RE = re.compile(r"오답_?(?P<note>.*)$")


def _parse_ace_clip(name: str) -> dict | None:
    """ace-clip 파일명(확장자 포함) 1개 파싱. 골격이 안 맞으면 None(호출측이 unparsed 처리)."""
    m = _ACE_CLIP_RE.match(name.rsplit(".", 1)[0] if name.lower().endswith(".mp4") else name)
    if not m:
        return None
    t = int(m.group("min")) * 60 + int(m.group("sec"))
    odap = _ODAP_RE.search(m.group("tag"))
    if odap:
        verdict = "fp"
        note = odap.group("note").strip("_")
    else:
        verdict = "unreviewed"
        note = ""
    return {
        "stem": m.group("stem"),
        "round": int(m.group("round")),
        "t": t,
        "verdict": verdict,
        "note": note,
    }


def _load_gt_spans(path: Path) -> dict[str, list[tuple[float, float]]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {stem: [tuple(span) for span in v["spans"]] for stem, v in data.items()}


def _matches_gt(stem: str, t: float, gt_spans: dict[str, list[tuple[float, float]]]) -> bool:
    return any(
        lo - _GT_MATCH_TOLERANCE_SEC <= t <= hi + _GT_MATCH_TOLERANCE_SEC
        for lo, hi in gt_spans.get(stem, [])
    )


def build_ledger(clips_root: Path = CLIPS_ROOT, gt_aces_path: Path = GT_ACES_PATH) -> list[dict]:
    """읽기 전용 스캔. 반환: stem/round/t/verdict/note/path 딕셔너리 리스트,
    행 수는 항상 스캔된 mp4 개수와 동일."""
    gt_spans = _load_gt_spans(gt_aces_path)
    rows: list[dict] = []
    for p in sorted(clips_root.glob(f"{CLIPS_GLOB}/**/*.mp4")):
        parsed = _parse_ace_clip(p.name)
        if parsed is None:
            rows.append({
                "stem": None, "round": None, "t": None,
                "verdict": "unreviewed",
                "note": f"unparsed_filename:{p.name}",
                "path": str(p),
            })
            continue
        row = dict(parsed)
        matched_gt = _matches_gt(row["stem"], row["t"], gt_spans)
        if row["verdict"] == "unreviewed" and matched_gt:
            row["verdict"] = "tp"  # T2 gt_aces.json 대조 자동 기입
        elif row["verdict"] == "fp" and matched_gt:
            # 사람이 오답으로 태깅한 클립이 GT 구간과도 겹침 — 자동 재분류하지 않고 플래그만
            row["note"] = (row["note"] + " " if row["note"] else "") + "[GT_CONFLICT]"
        row["path"] = str(p)
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="ace_clips* mp4 전수 스캔 → 검수 대장 생성 (읽기 전용)")
    ap.add_argument("--clips-root", default=str(CLIPS_ROOT))
    ap.add_argument("--gt-aces", default=str(GT_ACES_PATH))
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--dry-run", action="store_true", help="CSV 미생성, 통계만 출력")
    args = ap.parse_args()

    rows = build_ledger(Path(args.clips_root), Path(args.gt_aces))
    n_tp = sum(1 for r in rows if r["verdict"] == "tp")
    n_fp = sum(1 for r in rows if r["verdict"] == "fp")
    n_unreviewed = sum(1 for r in rows if r["verdict"] == "unreviewed")
    n_unparsed = sum(1 for r in rows if r["stem"] is None)
    n_conflict = sum(1 for r in rows if "GT_CONFLICT" in r["note"])
    print(
        f"[review-ledger] 스캔 {len(rows)}건  tp={n_tp} fp={n_fp} unreviewed={n_unreviewed}"
        f"  (파일명 미파싱 {n_unparsed}건, GT_CONFLICT {n_conflict}건)"
    )

    if args.dry_run:
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "round", "t", "verdict", "note", "path"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[review-ledger] -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
