# -*- coding: utf-8 -*-
"""확장 신호 캐시 I/O — reads + 경계 verdicts + score 이벤트를 한 파일에 저장.

배경 (IMPROVEMENT_REPORT.md §A-1): 배치 스캔이 collect_reads 결과를 버려서
판정 실험마다 영상 전체를 재디코딩해 왔다. 이 모듈은 스캔 산출물 전체를
JSON 하나로 영속화해 "영상 없이 재실험"을 성립시킨다.

포맷 계약:
    hud_sig_cache.build_cache 스키마(stem/scan_fps/duration/reads)의 상위집합.
    hud_from_cache.load_reads는 아는 키만 읽으므로 이 파일을 그대로 소비 가능
    (하위호환). 추가 키:
        boundary_verdicts: verify_runs_live 반환 리스트 그대로
                           (timeline_from_reads가 직접 소비하는 형식)
        score_win_events:  scan_hud_aces가 만드는 asdict 리스트
        version:           2 (확장 스키마 표식)

배선 지점 (평가 종료 후 — SONNET_TASKS.md A-1 참고):
    detect_ace_hud.scan_hud_aces가 reads/boundary_verdicts/score_win_events를
    모두 손에 쥔 시점(timeline_from_reads 호출 직전)에서 save_scan_cache 1회 호출.
"""

from __future__ import annotations

import json
from pathlib import Path

from detect_ace_hud import KRead
from hud_sig_cache import _METHOD_CODE, METHOD_DECODE

SCHEMA_VERSION = 2


def _jsonable(obj):
    """numpy 스칼라 등 비-JSON 타입을 재귀적으로 순수 파이썬 타입으로 강등."""
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return round(obj, 4)
    if hasattr(obj, "item"):  # numpy 스칼라
        return _jsonable(obj.item())
    return str(obj)


def save_scan_cache(
    stem: str,
    reads: list[KRead],
    duration: float,
    scan_fps: float,
    cache_dir: Path,
    *,
    boundary_verdicts: list | None = None,
    score_win_events: list | None = None,
) -> Path:
    """스캔 산출물 전체를 <cache_dir>/<stem>.json 으로 저장. 반환: 저장 경로.

    reads 직렬화는 hud_sig_cache.build_cache와 바이트 수준 동일 규칙
    (_METHOD_CODE 1글자 축약, t/conf 반올림) — 로더 공유를 위한 계약.
    """
    data = {
        "version": SCHEMA_VERSION,
        "stem": stem,
        "scan_fps": scan_fps,
        "duration": duration,
        "reads": [
            [round(r.t, 3), r.k, round(r.conf, 3), _METHOD_CODE.get(r.method, "?"),
             r.d, r.a]
            for r in reads
        ],
        "boundary_verdicts": _jsonable(boundary_verdicts) if boundary_verdicts is not None else None,
        "score_win_events": _jsonable(score_win_events) if score_win_events is not None else None,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{stem}.json"
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return out


def load_scan_cache(cache_path: Path) -> dict:
    """확장 캐시 로드. 반환 dict 키:
    reads(list[KRead]) / duration / scan_fps / stem /
    boundary_verdicts(list|None) / score_win_events(list|None) / version(int)

    v1(구 sig_cache) 파일도 읽는다 — verdicts/events는 None으로 채움.
    """
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    reads = [
        KRead(t=row[0], k=row[1], conf=row[2],
              method=METHOD_DECODE.get(row[3], row[3]),
              d=row[4] if len(row) > 4 else None,
              a=row[5] if len(row) > 5 else None)
        for row in data["reads"]
    ]
    return {
        "version": data.get("version", 1),
        "stem": data["stem"],
        "scan_fps": data.get("scan_fps", 4.0),
        "duration": data["duration"],
        "reads": reads,
        "boundary_verdicts": data.get("boundary_verdicts"),
        "score_win_events": data.get("score_win_events"),
    }
