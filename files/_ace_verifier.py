# -*- coding: utf-8 -*-
"""신호 조합 검증기 — ace 후보 신뢰도 점수 (R9, SONNET_TASK.md 참고).

★★★ 역할 한계 (절대 불변) ★★★
이 스크립트가 산출하는 점수는 "검수 우선순위 정렬 + 저신뢰 플래깅" 용도로만
쓰인다. ace 판정(hud_timeline JSON의 `ace: true/false`) 자체는 이 스크립트가
절대 바꾸지 않으며, 어떤 임계값으로도 자동 기각하지 않는다. 점수가 낮다고
해서 그 라운드를 파이프라인에서 빼거나 숨기지 말 것 — 사람이 검수 큐에서
우선순위로만 참고한다.

사용:
    python -u _ace_verifier.py --train    # 라벨 구축 → LOVO CV 비교 → 최종 학습 → 저장
    python -u _ace_verifier.py --score    # 전체 ace 후보 채점 → CSV + 검수 큐 출력
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from _compare_hud_gt import GT, TIMELINE_DIR, _overlaps, mss  # noqa: E402
from hud_score_wins import DEFAULT_SCORE_CACHE_DIR, load_score_timeline, score_events  # noqa: E402

MODEL_PATH = Path(r"E:\clipai_result\ace_verifier.pkl")
SCORES_CSV = Path(r"E:\clipai_result\ace_verifier_scores.csv")

FEATURE_NAMES = [
    "width",
    "width0",
    "n_start_kills",
    "first_kill_off",
    "ace_off_end",
    "dur",
    "k_density",
    "k_samples",
    "resets",
    "prev_kills",
    "next_kills",
    "read_quality",
    "round_pos",
    "kt_rel",
    "score_win_count",
]

_TOL = 15.0  # _compare_hud_gt와 동일 매칭 허용 오차(초)


_REQUIRED_ROUND_KEYS = {"kills", "kill_times", "resets", "reset_times"}


def _safe_load_json(path: Path) -> dict | None:
    """배치 재스캔이 동시에 쓰는 중일 수 있으므로 실패하면 조용히 None 반환.

    또한 구버전 스키마(예: k_base/max_k 필드만 있고 kills/kill_times/resets가
    없는 예전 파이프라인 산출물)도 스킵한다 — 아직 재스캔 전인 스텁 파일이므로
    이 라운드 필드 셋 명세(kills/kill_times/resets/reset_times 등)를 만족하지
    않는 파일은 신뢰할 수 없는 데이터로 취급한다.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        print(f"  [경고] {path.name} 읽기 실패(스킵): {e}", file=sys.stderr)
        return None
    for r in data.get("rounds", []):
        if not _REQUIRED_ROUND_KEYS.issubset(r.keys()):
            print(f"  [경고] {path.name} 구버전 스키마(스킵): 필드 누락 {sorted(_REQUIRED_ROUND_KEYS - r.keys())}", file=sys.stderr)
            return None
    return data


def _match_labels(data: dict, gts: list[tuple[float, float]], tol: float = _TOL) -> dict[int, int]:
    """_compare_hud_gt.main()과 동일한 그리디 매칭 → {round_index: 1(TP)/0(FP)}."""
    aces = [r for r in data.get("rounds", []) if r.get("ace")]
    det = []
    for r in aces:
        d1 = r.get("first_kill_sec") or r["start_sec"]
        d2 = r.get("ace_sec") or r["end_sec"]
        det.append((r["round_index"], d1, max(d1, d2)))

    used: set[int] = set()
    for (g1, g2) in gts:
        for (ri, d1, d2) in det:
            if ri in used:
                continue
            if _overlaps(g1, g2, d1, d2, tol):
                used.add(ri)
                break

    return {ri: (1 if ri in used else 0) for (ri, _d1, _d2) in det}


def _read_quality(data: dict) -> float:
    hits = data.get("k_template_hits", 0) or 0
    miss = data.get("k_template_miss", 0) or 0
    row_miss = data.get("k_row_miss", 0) or 0
    denom = hits + miss + row_miss
    return (hits / denom) if denom > 0 else -1.0


def _round_time(r: dict) -> float:
    """검수/CSV 보고용 대표 시각: ace_sec 우선, 없으면 first_kill_sec, 없으면 start_sec."""
    for key in ("ace_sec", "first_kill_sec"):
        v = r.get(key)
        if v is not None:
            return float(v)
    return float(r["start_sec"])


def extract_video_features(stem: str, data: dict, score_cache_dir: Path = DEFAULT_SCORE_CACHE_DIR) -> dict[int, dict]:
    """영상 하나의 hud_timeline JSON → {round_index: {feature_name: value}} (ace 라운드만)."""
    rounds = data.get("rounds", [])
    total_rounds = len(rounds)
    read_q = _read_quality(data)

    timeline = None
    try:
        timeline = load_score_timeline(stem, cache_dir=score_cache_dir)
    except Exception as e:  # noqa: BLE001 — score_cache 결측/손상은 -1 처리, 크래시 금지
        print(f"  [경고] score_cache 로드 실패({stem}): {e}", file=sys.stderr)
        timeline = None
    events = None
    if timeline is not None:
        try:
            events = score_events(timeline)
        except Exception as e:  # noqa: BLE001
            print(f"  [경고] score_events 계산 실패({stem}): {e}", file=sys.stderr)
            events = None

    out: dict[int, dict] = {}
    for idx, r in enumerate(rounds):
        if not r.get("ace"):
            continue
        ri = r["round_index"]
        start = float(r["start_sec"])
        end = float(r["end_sec"])
        kill_times = [float(kt) for kt in (r.get("kill_times") or [])]
        dur = end - start

        width = (max(kill_times) - min(kill_times)) if kill_times else 0.0
        width0 = 1 if width == 0.0 else 0
        n_start_kills = sum(1 for kt in kill_times if kt == start)
        first_kill_off = (float(r["first_kill_sec"]) - start) if r.get("first_kill_sec") is not None else -1.0
        ace_off_end = (end - float(r["ace_sec"])) if r.get("ace_sec") is not None else -1.0
        k_samples = r.get("k_samples", 0) or 0
        k_density = k_samples / max(dur, 1.0)
        resets = r.get("resets", 0) or 0
        prev_kills = rounds[idx - 1]["kills"] if idx - 1 >= 0 else -1
        next_kills = rounds[idx + 1]["kills"] if idx + 1 < len(rounds) else -1
        round_pos = (ri / total_rounds) if total_rounds else 0.0
        kt_rel = (
            sum((kt - start) / max(dur, 1.0) for kt in kill_times) / len(kill_times)
            if kill_times
            else -1.0
        )

        if events is not None:
            score_win_count = sum(1 for e in events if e.kind == "win" and start <= e.t_hi <= end)
        else:
            score_win_count = -1

        out[ri] = {
            "width": width,
            "width0": width0,
            "n_start_kills": n_start_kills,
            "first_kill_off": first_kill_off,
            "ace_off_end": ace_off_end,
            "dur": dur,
            "k_density": k_density,
            "k_samples": k_samples,
            "resets": resets,
            "prev_kills": prev_kills,
            "next_kills": next_kills,
            "read_quality": read_q,
            "round_pos": round_pos,
            "kt_rel": kt_rel,
            "score_win_count": score_win_count,
            "_time": _round_time(r),
        }
    return out


def _vec(feat: dict) -> list[float]:
    return [float(feat[name]) for name in FEATURE_NAMES]


# ---------------------------------------------------------------------------
# 라벨 구축
# ---------------------------------------------------------------------------


def build_dataset(tol: float = _TOL):
    """GT 딕셔너리 영상만 순회 → (X, y, groups, keys) 라벨 있는 데이터셋."""
    X: list[list[float]] = []
    y: list[int] = []
    groups: list[str] = []
    keys: list[tuple[str, int]] = []
    times: list[float] = []

    n_tp = n_fp = 0
    for stem, gts in GT.items():
        jp = TIMELINE_DIR / f"{stem}.json"
        if not jp.exists():
            print(f"  [정보] {stem}: hud_timeline JSON 없음 — 스킵")
            continue
        data = _safe_load_json(jp)
        if data is None:
            continue
        labels = _match_labels(data, gts, tol=tol)
        if not labels:
            continue
        feats = extract_video_features(stem, data)
        for ri, label in labels.items():
            if ri not in feats:
                continue
            X.append(_vec(feats[ri]))
            y.append(label)
            groups.append(stem)
            keys.append((stem, ri))
            times.append(feats[ri]["_time"])
            if label == 1:
                n_tp += 1
            else:
                n_fp += 1

    print(f"라벨 데이터셋: TP {n_tp}건, FP {n_fp}건 (영상 {len(set(groups))}개)")
    return np.array(X, dtype=float), np.array(y, dtype=int), np.array(groups), keys, np.array(times)


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------


def make_logreg() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        ]
    )


def make_hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=2,
        max_iter=50,
        min_samples_leaf=5,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=42,
    )


MODELS = {
    "logreg": make_logreg,
    "hgb": make_hgb,
}


def lovo_cv(model_factory, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Leave-one-video-out CV. out-of-fold 확률을 모아 전체 ROC-AUC/PR-AUC 계산."""
    logo = LeaveOneGroupOut()
    oof = np.full(len(y), np.nan)
    skipped = []
    for tr_idx, te_idx in logo.split(X, y, groups):
        y_tr = y[tr_idx]
        if len(np.unique(y_tr)) < 2:
            # 이 영상을 빼면 학습셋이 단일 클래스가 됨 — 그 영상은 CV에서 제외
            skipped.append(groups[te_idx[0]])
            continue
        m = model_factory()
        m.fit(X[tr_idx], y_tr)
        proba = m.predict_proba(X[te_idx])[:, 1]
        oof[te_idx] = proba

    mask = ~np.isnan(oof)
    if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
        return float("nan"), float("nan"), oof, skipped
    auc = roc_auc_score(y[mask], oof[mask])
    ap = average_precision_score(y[mask], oof[mask])
    return auc, ap, oof, skipped


def fp_concentration(y: np.ndarray, scores: np.ndarray, ks=(10, 20)) -> dict[int, tuple[int, int]]:
    """점수 오름차순 정렬 시 상위 K개 중 FP 개수. 반환: {K: (fp_in_topk, total_fp)}."""
    mask = ~np.isnan(scores)
    yv = y[mask]
    sv = scores[mask]
    order = np.argsort(sv)  # 오름차순(낮은 점수=저신뢰가 먼저)
    y_sorted = yv[order]
    total_fp = int((yv == 0).sum())
    out = {}
    for k in ks:
        k_eff = min(k, len(y_sorted))
        fp_in_k = int((y_sorted[:k_eff] == 0).sum())
        out[k] = (fp_in_k, total_fp)
    return out


def feature_importance_report(model_name: str, model, X: np.ndarray, y: np.ndarray) -> list[tuple[str, float]]:
    if model_name == "logreg":
        coefs = model.named_steps["clf"].coef_[0]
        pairs = list(zip(FEATURE_NAMES, np.abs(coefs)))
    else:
        r = permutation_importance(model, X, y, n_repeats=20, random_state=42, scoring="roc_auc")
        pairs = list(zip(FEATURE_NAMES, r.importances_mean))
    pairs.sort(key=lambda t: t[1], reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# --train
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    print("=== 라벨 데이터셋 구축 ===")
    X, y, groups, keys, times = build_dataset(tol=args.tol)
    if len(y) == 0:
        print("라벨 데이터가 없습니다. GT/hud_timeline 경로를 확인하세요.")
        return 1
    n_videos = len(set(groups))
    print(f"총 {len(y)}행, 영상 {n_videos}개, 특징 {len(FEATURE_NAMES)}개\n")

    print("=== Leave-One-Video-Out CV (모델 비교) ===")
    results = {}
    for name, factory in MODELS.items():
        auc, ap, oof, skipped = lovo_cv(factory, X, y, groups)
        results[name] = (auc, ap, oof)
        skip_note = f" (CV 제외 {len(skipped)}개 영상: 단일클래스)" if skipped else ""
        print(f"  {name:8s}: ROC-AUC={auc:.4f}  PR-AUC={ap:.4f}{skip_note}")

    winner = max(results, key=lambda n: (results[n][0] if not np.isnan(results[n][0]) else -1))
    win_auc, win_ap, win_oof = results[winner]
    print(f"\n우승 모델: {winner} (LOVO ROC-AUC={win_auc:.4f}, PR-AUC={win_ap:.4f})")
    if win_auc < 0.75:
        print("  [정직 보고] ROC-AUC < 0.75 — 정렬용으로도 약함. 신뢰 구간 낮음으로 취급할 것.")

    print("\n=== 운영 지표: 점수 오름차순 상위 K FP 몰림 (LOVO out-of-fold 기준) ===")
    conc = fp_concentration(y, win_oof, ks=(10, 20))
    for k, (fp_in_k, total_fp) in conc.items():
        print(f"  상위 K={k}: FP {fp_in_k}건 / 전체 FP {total_fp}건")

    print("\n=== 최종 학습 (전체 라벨 데이터) ===")
    final_model = MODELS[winner]()
    final_model.fit(X, y)

    print(f"\n=== 피처 중요도 ({winner}) 상위 5 ===")
    imp = feature_importance_report(winner, final_model, X, y)
    for name, val in imp[:5]:
        print(f"  {name:16s} {val:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "model": final_model,
                "model_type": winner,
                "feature_names": FEATURE_NAMES,
                "lovo_roc_auc": win_auc,
                "lovo_pr_auc": win_ap,
            },
            f,
        )
    print(f"\n저장 완료: {MODEL_PATH}")
    return 0


# ---------------------------------------------------------------------------
# --score
# ---------------------------------------------------------------------------


def cmd_score(args: argparse.Namespace) -> int:
    if not MODEL_PATH.exists():
        print(f"모델 파일이 없습니다: {MODEL_PATH} — 먼저 --train 을 실행하세요.")
        return 1
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    # GT 영상들의 라벨(TP/FP) — 검수 큐에서 라벨 유무 구분용
    label_by_key: dict[tuple[str, int], int] = {}
    for stem, gts in GT.items():
        jp = TIMELINE_DIR / f"{stem}.json"
        if not jp.exists():
            continue
        data = _safe_load_json(jp)
        if data is None:
            continue
        for ri, label in _match_labels(data, gts, tol=args.tol).items():
            label_by_key[(stem, ri)] = label

    rows = []
    n_ok = n_skip = 0
    for jp in sorted(TIMELINE_DIR.glob("*.json")):
        stem = jp.stem
        data = _safe_load_json(jp)
        if data is None:
            n_skip += 1
            continue
        feats = extract_video_features(stem, data)
        if not feats:
            n_ok += 1
            continue
        for ri, feat in feats.items():
            x = np.array([_vec(feat)], dtype=float)
            score = float(model.predict_proba(x)[0, 1])
            label = label_by_key.get((stem, ri))
            rows.append(
                {
                    "stem": stem,
                    "round": ri,
                    "mss": mss(feat["_time"]),
                    "label": ("" if label is None else label),
                    "score": score,
                }
            )
        n_ok += 1
    print(f"영상 {n_ok}개 처리, {n_skip}개 스킵(읽기 실패)")
    print(f"ace 후보 총 {len(rows)}건 채점 완료\n")

    SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SCORES_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "round", "mss", "label", "score"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"CSV 저장: {SCORES_CSV}")

    unlabeled = [r for r in rows if r["label"] == ""]
    unlabeled.sort(key=lambda r: r["score"])
    print(f"\n=== 검수 큐 (라벨 없음, {len(unlabeled)}건, 점수 오름차순 — 저신뢰부터) 상위 10 ===")
    print("  ※ 이 순서는 검수 우선순위일 뿐 — ace 판정 자체를 바꾸거나 자동 기각하지 않는다.")
    for r in unlabeled[:10]:
        print(f"  {r['stem']}  R{r['round']:02d}  {r['mss']:>7s}  score={r['score']:.4f}")

    return 0


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--tol", type=float, default=_TOL)
    args = ap.parse_args()

    if not args.train and not args.score:
        print("사용: python -u _ace_verifier.py --train | --score")
        return 1

    rc = 0
    if args.train:
        rc = cmd_train(args) or rc
    if args.score:
        rc = cmd_score(args) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
