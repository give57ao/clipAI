# -*- coding: utf-8 -*-
"""닉네임 fuzzy 정규화·매칭·투표 클러스터링 (scouter / scoreboard 공용)."""

from __future__ import annotations

import re
from collections import Counter

from scouter_nick import levenshtein

_NULL_VARIANTS = {
    "null", "nul", "nuli", "nu11", "nu1l", "nill", "mull", "nu", "jyu", "nuii",
}


def normalize_nick(text: str) -> str:
    return text.strip().lower().replace(" ", "")


def strip_special(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", text.lower())


def nick_core(text: str, min_len: int = 3) -> str:
    """특수문자·'...' 잘림 제거 후 닉 핵심부 (한글/영문 연속 최장 구간).

    예: '◇ㅁ깜띸겅쥬..' → '깜띸겅쥬' (OCR이 앞 기호를 붙여도 핵심 4글자 추출)
    """
    stripped = strip_special(text)
    if not stripped:
        return ""
    runs = re.findall(r"[0-9a-z가-힣]+", stripped)
    if not runs:
        return stripped if len(stripped) >= 2 else ""
    best = max(runs, key=len)
    if len(best) >= min_len:
        return best
    # 짧은 닉(null 등)은 전체 반환
    return stripped if len(stripped) >= 2 else best


def cores_match(a: str, b: str, min_core: int = 3) -> bool:
    """핵심부 일치 — 한쪽이 다른 쪽에 포함되거나 fuzzy 일치."""
    if not a or not b:
        return False
    if nick_match_text(a, b):
        return True
    ca, cb = nick_core(a), nick_core(b)
    if not ca or not cb:
        return False
    if ca == cb:
        return True
    shorter, longer = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
    if len(shorter) >= 4 and shorter in longer:
        return True
    if len(shorter) >= 2 and nick_match(shorter, longer):
        return True
    return False


def is_null_variant(norm: str) -> bool:
    return norm in _NULL_VARIANTS


def is_null_variant_text(text: str) -> bool:
    norm = re.sub(r"[^0-9a-z]", "", normalize_nick(text))
    if not norm:
        return False
    if norm in _NULL_VARIANTS:
        return True
    return len(norm) <= 5 and levenshtein(norm, "null") <= 1


def nick_match(norm_a: str, norm_b: str) -> bool:
    if not norm_a or not norm_b:
        return False
    if is_null_variant(norm_a) and is_null_variant(norm_b):
        return True
    stripped_a, stripped_b = strip_special(norm_a), strip_special(norm_b)
    if stripped_a and stripped_b:
        if stripped_a == stripped_b:
            return True
        if len(stripped_a) >= 4 and len(stripped_b) >= 4:
            if stripped_a in stripped_b or stripped_b in stripped_a:
                return True
    # 핵심부(4글자+) 포함 매칭 — 특수문자·잘림 OCR 변형 (3글자는 오병합 위험)
    ca, cb = nick_core(norm_a), nick_core(norm_b)
    if ca and cb:
        if ca == cb:
            return True
        shorter, longer = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
        if len(shorter) >= 4 and shorter in longer:
            return True
    min_len = min(len(norm_a), len(norm_b))
    # 긴 닉일수록 허용 오류 수 확대 (OCR 변형 허용)
    threshold = 1 if min_len <= 3 else (3 if min_len >= 6 else 2)
    return levenshtein(norm_a, norm_b) <= threshold


def nick_match_text(a: str, b: str) -> bool:
    return nick_match(normalize_nick(a), normalize_nick(b))


def canonicalize_nick(text: str) -> str:
    if is_null_variant_text(text):
        return "null"
    return text.strip()


def cluster_votes(votes: list[dict]) -> list[dict]:
    aggregated: dict[str, dict] = {}
    for vote in votes:
        bucket = aggregated.setdefault(
            vote["text"],
            {"text": vote["text"], "weight": 0.0, "conf": 0.0, "samples": 0},
        )
        bucket["weight"] += vote["weight"]
        bucket["conf"] = max(bucket["conf"], vote["conf"])
        bucket["samples"] += 1

    candidates = sorted(aggregated.values(), key=lambda d: -d["weight"])
    clusters: list[dict] = []
    for cand in candidates:
        norm = normalize_nick(cand["text"])
        core = nick_core(cand["text"])
        placed = False
        for cluster in clusters:
            if nick_match(norm, cluster["norm"]) or (
                core and cluster.get("core") and cores_match(core, cluster["core"])
            ):
                cluster["members"].append(cand)
                cluster["weight"] += cand["weight"]
                cluster["samples"] += cand["samples"]
                cluster["best_conf"] = max(cluster["best_conf"], cand["conf"])
                placed = True
                break
        if not placed:
            clusters.append(
                {
                    "norm": norm,
                    "core": core or norm,
                    "canonical": cand["text"],
                    "members": [cand],
                    "weight": cand["weight"],
                    "samples": cand["samples"],
                    "best_conf": cand["conf"],
                }
            )

    for cluster in clusters:
        if any(is_null_variant(normalize_nick(m["text"])) for m in cluster["members"]):
            cluster["canonical"] = "null"
            cluster["core"] = "null"
        elif cluster.get("core") and len(cluster["core"]) >= 4:
            # 확정 닉은 핵심부로 통일 (4글자 이상만 — 짧은 조각 오병합 방지)
            cluster["canonical"] = cluster["core"]

    clusters.sort(key=lambda c: -c["weight"])
    return clusters


def clusters_to_summary(clusters: list[dict], limit: int = 8) -> list[dict]:
    return [
        {
            "canonical": c["canonical"],
            "weight": round(c["weight"], 2),
            "samples": c["samples"],
            "best_conf": round(c["best_conf"], 3),
            "variants": dict(Counter(m["text"] for m in c["members"])),
        }
        for c in clusters[:limit]
    ]
