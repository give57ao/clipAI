# -*- coding: utf-8 -*-
"""상단 라이브 HUD 팀 승수 판독 — "005" 가운데 박스를 풀프레임 앵커로 삼아
game_roi ML crop의 프레임별 지터를 완전히 우회한다 (R6, SONNET_TASK.md 참고).

배경: game_roi가 프레임마다 몇 픽셀씩 다르게 예측 → 그 crop 위에 비율 좌표를
얹으면 좌표가 흔들림. UI 자체는 절대 안 움직이므로, 매치 내내 불변인 "005"
텍스트를 풀프레임에서 직접 템플릿 매칭해 영상당 1회 앵커를 잡으면 이 문제가
구조적으로 사라진다 (Fable 실측: 8영상 전부 위치 표준편차 0.0px).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np

from hud_digit_match import DEFAULT_TEMPLATE_DIR, get_hud_digit_matcher, match_glyph, normalize_glyph, white_mask

_ANCHOR_PATH = DEFAULT_TEMPLATE_DIR / "score_anchor_005.png"
_ANCHOR_SCALES = (0.70, 0.78, 0.86, 0.94, 1.0, 1.06, 1.14, 1.22)
_ANCHOR_BAND_Y = (0.0, 0.16)   # 풀프레임 비율 — 앵커 탐색 밴드
_ANCHOR_BAND_X = (0.20, 0.80)
_ANCHOR_MIN_CONF = 0.75
_ANCHOR_MIN_HITS = 5

# §1-3 측정된 기하 상수 (scale 1.0, 앵커 좌상단 (ax,ay) 기준 오프셋)
_NUM_Y_OFF = (7, 45)          # 숫자 밴드 y 오프셋 (높이 38)
_RED_X_OFF = (-142, 1)
_MID_X_OFF = (8, 151)
_BLUE_X_OFF = (158, 301)

_WHITE_THRESH = 155


@dataclass(frozen=True)
class ScoreAnchor:
    ax: int
    ay: int
    scale: float
    conf: float


def _load_anchor_template() -> np.ndarray | None:
    if not _ANCHOR_PATH.exists():
        return None
    img = cv2.imread(str(_ANCHOR_PATH), cv2.IMREAD_GRAYSCALE)
    return img


def _match_anchor_in_frame(gray_full: np.ndarray, tmpl: np.ndarray) -> tuple[float, int, int, float] | None:
    """풀프레임에서 앵커 다중 스케일 매칭. 반환: (conf, x, y, scale) 또는 None."""
    h, w = gray_full.shape[:2]
    y1, y2 = int(h * _ANCHOR_BAND_Y[0]), int(h * _ANCHOR_BAND_Y[1])
    x1, x2 = int(w * _ANCHOR_BAND_X[0]), int(w * _ANCHOR_BAND_X[1])
    band = gray_full[y1:y2, x1:x2]
    if band.size == 0:
        return None
    best: tuple[float, int, int, float] | None = None
    for s in _ANCHOR_SCALES:
        tw, th = int(tmpl.shape[1] * s), int(tmpl.shape[0] * s)
        if tw >= band.shape[1] or th >= band.shape[0] or tw < 4 or th < 4:
            continue
        t2 = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(band, t2, cv2.TM_CCOEFF_NORMED)
        _minv, maxv, _minl, maxl = cv2.minMaxLoc(res)
        if best is None or maxv > best[0]:
            best = (float(maxv), maxl[0] + x1, maxl[1] + y1, s)
    return best


def find_score_anchor(video_path: str | Path, n_samples: int = 9) -> ScoreAnchor | None:
    """영상 전체에 고르게 분포한 n_samples 프레임에서 "005" 앵커 위치를 탐지.

    conf>=_ANCHOR_MIN_CONF인 프레임이 _ANCHOR_MIN_HITS개 이상이고 그 프레임들의
    위치(x,y,scale) 최빈값이 일치해야 채택. 실패하면 None(이 영상은 기능 비활성 —
    회귀 없이 안전하게 꺼짐).
    """
    tmpl = _load_anchor_template()
    if tmpl is None:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps if fps > 0 else 0.0
    if duration <= 0:
        cap.release()
        return None

    hits: list[tuple[float, int, int, float]] = []
    for frac in np.linspace(0.06, 0.94, n_samples):
        t = duration * float(frac)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        r = _match_anchor_in_frame(gray, tmpl)
        if r is not None and r[0] >= _ANCHOR_MIN_CONF:
            hits.append(r)
    cap.release()

    if len(hits) < _ANCHOR_MIN_HITS:
        return None

    # 위치 최빈값 (반올림 좌표로 그룹화 — 정합 프레임은 지터 없이 정확히 일치해야 함)
    from collections import Counter

    keyed = [((round(x), round(y), round(s, 2)), (conf, x, y, s)) for (conf, x, y, s) in hits]
    counts = Counter(k for k, _v in keyed)
    mode_key, mode_n = counts.most_common(1)[0]
    if mode_n < _ANCHOR_MIN_HITS:
        return None
    mode_vals = [v for k, v in keyed if k == mode_key]
    best_conf = max(v[0] for v in mode_vals)
    _c, ax, ay, scale = mode_vals[0]
    return ScoreAnchor(ax=ax, ay=ay, scale=scale, conf=best_conf)


def _digit_band_crop(frame_bgr: np.ndarray, anchor: ScoreAnchor, x_off: tuple[int, int]) -> np.ndarray:
    s = anchor.scale
    y1 = anchor.ay + int(_NUM_Y_OFF[0] * s)
    y2 = anchor.ay + int(_NUM_Y_OFF[1] * s)
    x1 = anchor.ax + int(x_off[0] * s)
    x2 = anchor.ax + int(x_off[1] * s)
    h, w = frame_bgr.shape[:2]
    y1, y2 = max(0, y1), min(h, y2)
    x1, x2 = max(0, x1), min(w, x2)
    if y2 <= y1 or x2 <= x1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return frame_bgr[y1:y2, x1:x2]


def _read_wins_box(box_bgr: np.ndarray, white_templates: dict) -> int | None:
    """3자리 제로패딩 승수 하나 판독. CC 정확히 3개 아니면 None(함정1 방지).
    앞 두 자리가 0이 아니면 None(함정2 — 승수는 한 자리 값)."""
    if box_bgr is None or box_bgr.size <= 1:
        return None
    mask = white_mask(box_bgr, thresh=_WHITE_THRESH)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mh = mask.shape[0]
    cells = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h < mh * 0.35 or w * h < 200:
            continue
        cells.append((x, mask[y : y + h, x : x + w]))
    if len(cells) != 3:
        return None  # 함정 1: CC 3개 아니면 안전 미스
    cells.sort(key=lambda t: t[0])
    digits: list[int] = []
    for _x, cell in cells:
        glyph = normalize_glyph(cell)
        d, _score = match_glyph(glyph, white_templates, min_score=0.45)
        if d is None:
            return None
        digits.append(d)
    if digits[0] != 0 or digits[1] != 0:
        return None  # 함정 2: 승수는 한 자리 (0~9), 앞 두 자리는 항상 0
    return digits[2]


def read_wins(frame_bgr: np.ndarray, anchor: ScoreAnchor) -> tuple[int | None, int | None]:
    """앵커 기준으로 (red_wins, blue_wins) 판독. 실패 슬롯은 None."""
    matcher = get_hud_digit_matcher()
    red_box = _digit_band_crop(frame_bgr, anchor, _RED_X_OFF)
    blue_box = _digit_band_crop(frame_bgr, anchor, _BLUE_X_OFF)
    red = _read_wins_box(red_box, matcher.white_templates)
    blue = _read_wins_box(blue_box, matcher.white_templates)
    return red, blue


# ---------------------------------------------------------------------------
# Task 2 — 타임라인 스캐너 + 이벤트 추출 + 캐시 (SONNET_TASK.md R6 §4)
# ---------------------------------------------------------------------------

import json  # noqa: E402

DEFAULT_SCORE_CACHE_DIR = Path(r"E:\clipai_result\score_cache")


@dataclass(frozen=True)
class ScoreEvent:
    t_lo: float    # 이전(old)값 마지막 안정 관측 시각
    t_hi: float    # 새(new)값 첫 안정 관측 시각 (지지 확정 시점)
    side: str      # 'R' 또는 'B'
    kind: str      # 'win'(승수 +) 또는 'reset'(하프타임/매치종료로 하락)
    old: int | None
    new: int


def scan_score_timeline(
    video_path: str | Path,
    scan_fps: float = 2.0,
    out_dir: Path | None = DEFAULT_SCORE_CACHE_DIR,
) -> dict | None:
    """영상 전체를 스캔해 (t, red, blue) 원시 판독을 캐시 JSON으로 저장.

    앵커 탐지 실패 시 None(이 영상은 R6 기능 비활성 — 하위 호환, 회귀 없음).
    """
    video_path = Path(video_path)
    anchor = find_score_anchor(video_path)
    if anchor is None:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / scan_fps))) if scan_fps > 0 else 1

    reads: list[list] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_idx % step == 0:
            t = frame_idx / fps
            r, b = read_wins(frame, anchor)
            reads.append([round(t, 3), r, b])
        frame_idx += 1
    cap.release()

    data = {
        "stem": video_path.stem,
        "video_path": str(video_path),
        "anchor": {"ax": anchor.ax, "ay": anchor.ay, "scale": anchor.scale, "conf": anchor.conf},
        "scan_fps": scan_fps,
        "reads": reads,
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{video_path.stem}.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
    return data


def load_score_timeline(stem: str, cache_dir: Path = DEFAULT_SCORE_CACHE_DIR) -> dict | None:
    p = Path(cache_dir) / f"{stem}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


_SUPPORT_CONFIRM = 2  # 지지 필터: 새 값이 연속(창 내) 2회 이상 관측돼야 확정


def _side_events(reads: list[list], side_idx: int) -> list[ScoreEvent]:
    """한쪽(R 또는 B) 승수의 확정 전이만 추출 — 순차 상태머신.

    ★ 하프타임 처리(§4 (a)(b)(d)): 이 함수는 confirmed(직전 확정값) → 새 확정값의
    '국소' 전이만 비교하고 매치 전체에 걸친 전역 단조성은 절대 가정하지 않는다 —
    그래서 하프 경계에서 별도 분기 처리가 필요 없다: 리셋(new<old)이 오면 그 자체로
    kind='reset' 이벤트가 되고, confirmed가 그 낮은 값으로 갱신되므로 이후의
    "0→1" 같은 후반전 첫 전이도 자동으로 kind='win'(new>old)으로 정확히 분류된다.
    리셋 이벤트 자체도 반환 목록에 포함되므로 (b) 라운드 경계 신호로 그대로 쓸 수 있다.
    """
    events: list[ScoreEvent] = []
    confirmed: int | None = None
    confirmed_t: float | None = None
    pending: int | None = None
    pending_first_t: float | None = None
    pending_count = 0

    for t, r, b in reads:
        v = r if side_idx == 0 else b
        if v is None:
            continue
        if v == confirmed:
            confirmed_t = t
            pending = None
            pending_count = 0
            continue
        if v == pending:
            pending_count += 1
        else:
            pending = v
            pending_first_t = t
            pending_count = 1
        if pending_count >= _SUPPORT_CONFIRM:
            if confirmed is not None:
                kind = "win" if v > confirmed else "reset"
                events.append(
                    ScoreEvent(
                        t_lo=confirmed_t if confirmed_t is not None else pending_first_t,
                        t_hi=pending_first_t,
                        side="R" if side_idx == 0 else "B",
                        kind=kind,
                        old=confirmed,
                        new=v,
                    )
                )
            confirmed = v
            confirmed_t = t
            pending = None
            pending_count = 0
    return events


def score_events(timeline: dict) -> list[ScoreEvent]:
    """캐시된 타임라인 → (R,B) 양쪽 확정 전이 이벤트, 시각순 정렬."""
    reads = timeline["reads"]
    ev = _side_events(reads, 0) + _side_events(reads, 1)
    ev.sort(key=lambda e: e.t_hi)
    return ev


# ---------------------------------------------------------------------------
# selftest — §1-2 검증 영상에서 앵커 재현 확인 (E:\OBS 필요, 없으면 skip)
# ---------------------------------------------------------------------------

_SELFTEST_VIDEOS = [
    r"E:\OBS\2026-03-29 01-01-04.mp4",
    r"E:\OBS\2026-03-29 03-12-36.mp4",
    r"E:\OBS\2026-04-08 02-15-28.mp4",
    r"E:\OBS\2026-04-28 00-07-24.mp4",
    r"E:\OBS\2026-04-10 00-08-17.mp4",
    r"E:\OBS\2026-04-09 02-05-29.mp4",
    r"E:\OBS\2026-04-27 22-50-54.mp4",
    r"E:\OBS\2026-04-29 01-14-15.mp4",
]


def _selftest_events() -> None:
    """합성 픽스처 — 하프타임 리셋·리셋직후 win이 올바르게 분류되는지 (§4 (a)(b)(d))."""
    reads = [
        [0.0, 0, 5], [1.0, 0, 5],
        [10.0, 1, 5], [11.0, 1, 5],
        [20.0, 2, 5], [21.0, 2, 5],
        [30.0, 0, 5], [31.0, 0, 5],   # 하프타임: R만 리셋 (B는 유지되는 합성 케이스)
        [40.0, 1, 5], [41.0, 1, 5],
    ]
    timeline = {"reads": reads}
    events = [e for e in score_events(timeline) if e.side == "R"]
    kinds = [(e.old, e.new, e.kind) for e in events]
    assert kinds == [
        (0, 1, "win"),
        (1, 2, "win"),
        (2, 0, "reset"),
        (0, 1, "win"),  # 리셋 직후 0->1도 win으로 정확히 분류 (§4-d)
    ], f"score_events 분류 오류: {kinds}"
    print("hud_score_wins selftest(events): OK")


def _selftest() -> None:
    n_ok = 0
    n_skip = 0
    for vp in _SELFTEST_VIDEOS:
        if not Path(vp).exists():
            n_skip += 1
            continue
        anchor = find_score_anchor(vp)
        assert anchor is not None, f"앵커 탐지 실패: {vp}"
        assert anchor.ax == 879 and anchor.ay == 0 and abs(anchor.scale - 1.0) < 1e-6, (
            f"앵커 좌표 불일치 {vp}: {anchor}"
        )
        n_ok += 1
    print(f"hud_score_wins selftest(anchor): {n_ok} OK, {n_skip} skipped (영상 없음)")
    assert n_ok > 0, "검증 영상이 하나도 없어 selftest 무의미"
    _selftest_events()


if __name__ == "__main__":
    _selftest()
