# -*- coding: utf-8
"""HUD K/D/A 기반 올킬 탐지 — 단일 패스 + ΔK + 템플릿."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import cv2

from game_frame import extract_game_crop_bgr
from hud_digit_match import get_hud_digit_matcher
from hud_kda import read_kda_triple_from_game

DEFAULT_DATASET_ROOT = Path(r"E:\Highlights\ml_dataset")
DEFAULT_OUTPUT_DIR = Path(r"E:\clipai_result\ace_clips_hud")
DEFAULT_JSON_DIR = Path(r"E:\clipai_result\hud_timeline")
ACE_KILLS = 3
_REQ_CONFIRM = 2           # K 전이 확정에 필요한 연속 동일 판독 — 3은 라운드 막판 킬 유실(실측)
_REQ_REBASE = 5            # 하향/큰 점프(하프타임·리조인) 리베이스 확정 — 진짜 리셋은 경기당 1~2회
# ★ 라운드 경계 = 연속 row_miss (KDA HUD 사라짐 = 풀 스코어보드/전환 화면).
#   도메인 정의 "라운드 = 풀 스코어보드 사이 구간"과 일치. 전멸 아이콘(analyze_hud_icons)보다
#   훨씬 깨끗한 신호 — 캐시 스윕에서 recall 33%→79%, FP 대폭 감소 (2026-07-06 Opus).
_BOUNDARY_ROWMISS = 18     # 경계 인정 최소 연속 row_miss 프레임(4fps≈4.5s). 실측: 18~20 안정,
                           #   <16이면 플레이 중 순간 가림에 과분할, ≥24면 스코어보드 놓쳐 병합
_MAX_ACE_SPAN_SEC = 90.0   # 올킬 3킬은 한 라운드(≤90s) 내 (그 이상은 경계 놓친 병합 의심)

# 클립 구간 — end-35 방식 대신 라운드 시작·첫 킬 기준
_CLIP_PRE_ROUND_SEC = 6.0   # 라운드 시작 직전
_CLIP_PRE_KILL_SEC = 14.0   # 첫 킬 이전 (킬 장면이 클립 앞쪽에 오도록)
_CLIP_POST_END_SEC = 1.5    # 라운드 종료 직후 (스코어 잠깐만)
_CLIP_MAX_SEC = 55.0


@dataclass
class RoundTrack:
    round_index: int
    start_sec: float
    end_sec: float = 0.0
    kills: int = 0
    kill_times: list[float] = field(default_factory=list)
    resets: int = 0
    reset_times: list[float] = field(default_factory=list)
    k_samples: int = 0
    ace: bool = False
    end_reason: str = "hud_elim"
    first_kill_sec: float | None = None
    ace_sec: float | None = None


@dataclass
class KillEvent:
    t: float          # 새 K값이 처음 관측된 시각
    from_k: int
    to_k: int


@dataclass
class KRead:
    """프레임 1개의 원시 K 판독 — 신호 캐시의 단위 (hud_sig_cache.py).

    method: 'template'(성공) | 'template_miss' | 'row_miss' | 'triple_incomplete'
    """
    t: float
    k: int | None
    conf: float
    method: str


@dataclass
class HudAceTimeline:
    video_path: str
    scan_fps: float
    ace_kills: int
    rounds: list[RoundTrack] = field(default_factory=list)
    ace_rounds: list[int] = field(default_factory=list)
    hud_end_count: int = 0
    k_template_hits: int = 0
    k_template_miss: int = 0
    k_row_miss: int = 0
    kill_events: list[KillEvent] = field(default_factory=list)
    reset_events: list[list[float]] = field(default_factory=list)  # [t, old, new]
    warnings: list[str] = field(default_factory=list)


class _KTracker:
    """누적 K 확정값 상태머신 — 연속 동일 판독으로만 전이 확정.

    - +1..+3 전이: _REQ_CONFIRM 회 연속 → 킬 이벤트 (증가량 = 킬 수)
    - 하향 또는 +4 이상: _REQ_REBASE 회 연속 → 리셋(하프타임/리조인/오독) 리베이스
    - None 판독은 pending 카운트를 깨지 않음 (판독 공백 허용)
    """

    def __init__(self) -> None:
        self.confirmed: int | None = None
        self.pend_val: int | None = None
        self.pend_n = 0
        self.pend_t0 = 0.0
        self.kills: list[KillEvent] = []
        self.resets: list[tuple[float, int, int]] = []

    def update(self, t: float, k: int | None, conf: float = 1.0) -> None:
        # conf: 판독 IoU 신뢰도 — 현행 v1은 미사용, 증거창(v2) 확정 로직용 통로
        if k is None:
            return
        if k == self.confirmed:
            self.pend_val, self.pend_n = None, 0
            return
        if k == self.pend_val:
            self.pend_n += 1
        else:
            self.pend_val, self.pend_n, self.pend_t0 = k, 1, t

        if self.confirmed is None:
            if self.pend_n >= _REQ_CONFIRM:
                self.confirmed = k
                self.pend_val, self.pend_n = None, 0
            return

        delta = k - self.confirmed
        if 1 <= delta <= ACE_KILLS:
            if self.pend_n >= _REQ_CONFIRM:
                self.kills.append(KillEvent(self.pend_t0, self.confirmed, k))
                self.confirmed = k
                self.pend_val, self.pend_n = None, 0
        else:
            if self.pend_n >= _REQ_REBASE:
                self.resets.append((self.pend_t0, self.confirmed, k))
                self.confirmed = k
                self.pend_val, self.pend_n = None, 0


def _build_rounds(
    boundaries: list[float],
    tracker: _KTracker,
    duration: float,
) -> list[RoundTrack]:
    """경계(연속 row_miss 중앙) + 리셋 시각으로 라운드 세그먼트 생성.

    리셋(하프타임·리조인)도 라운드 경계 — 서로 다른 매치/반의 킬이 섞이지 않도록.
    """
    reset_ts = [t for (t, _o, _n) in tracker.resets]
    seps = sorted(set(boundaries) | set(reset_ts))
    rounds: list[RoundTrack] = []
    prev = 0.0
    idx = 0
    for sp in seps + [duration]:
        if sp - prev > 0.5:
            rounds.append(RoundTrack(round_index=idx, start_sec=prev, end_sec=sp))
            idx += 1
        prev = sp
    return rounds


def _assign_events(
    rounds: list[RoundTrack],
    tracker: _KTracker,
) -> None:
    """킬/리셋 이벤트를 라운드(세그먼트)에 귀속하고 ace 판정.

    ace = 세그먼트 내 킬 합 정확히 3 + 리셋 없음 + 3킬 스팬 ≤ _MAX_ACE_SPAN_SEC.
    경계가 row_miss(스코어보드)라 세그먼트가 곧 도메인상 라운드와 일치.
    """
    if not rounds:
        return

    def round_for(t: float) -> int | None:
        for i, r in enumerate(rounds):
            if r.start_sec <= t < r.end_sec:
                return i
        return len(rounds) - 1 if t >= rounds[-1].end_sec else None

    for ev in tracker.kills:
        idx = round_for(ev.t)
        if idx is None:
            continue
        r = rounds[idx]
        r.kills += ev.to_k - ev.from_k
        r.k_samples += 1
        r.kill_times.append(ev.t)
        if r.first_kill_sec is None or ev.t < r.first_kill_sec:
            r.first_kill_sec = ev.t
        if r.ace_sec is None and r.kills >= ACE_KILLS:
            r.ace_sec = ev.t

    # 리셋은 세그먼트 경계(_build_rounds에서 분리) — 경계에 걸친 리셋을 내부로
    # 오귀속하면 그 세그먼트의 정상 올킬을 잘못 기각(02-21-23 64:44 실측).
    # 엄격 내부(start < t < end)만 카운트: 정상적으론 발생 안 하는 방어용.
    for (t, _old, _new) in tracker.resets:
        for r in rounds:
            if r.start_sec < t < r.end_sec:
                r.resets += 1
                r.reset_times.append(t)
                break

    for r in rounds:
        span = (
            (r.ace_sec - r.first_kill_sec)
            if (r.ace_sec is not None and r.first_kill_sec is not None)
            else 0.0
        )
        r.ace = (
            r.kills == ACE_KILLS
            and r.resets == 0
            and span <= _MAX_ACE_SPAN_SEC
        )


def scan_hud_aces(
    video_path: Path,
    *,
    scan_fps: float = 4.0,
    ace_kills: int = ACE_KILLS,
    dataset_root: Path | None = None,
) -> HudAceTimeline:
    reads, duration, err = collect_reads(
        Path(video_path), scan_fps=scan_fps, dataset_root=dataset_root
    )
    timeline = timeline_from_reads(
        reads,
        duration=duration,
        video_path=Path(video_path),
        scan_fps=scan_fps,
        ace_kills=ace_kills,
    )
    if err:
        timeline.warnings.append(err)
    return timeline


def collect_reads(
    video_path: Path,
    *,
    scan_fps: float = 4.0,
    dataset_root: Path | None = None,
) -> tuple[list[KRead], float, str | None]:
    """영상 1패스 디코드 → 프레임별 원시 K 판독 리스트.

    이 결과(신호 캐시)만 있으면 판정 로직은 영상 재판독 없이 무한 재실험 가능
    — hud_sig_cache.py 로 저장, hud_from_cache.py / timeline_from_reads 로 소비.
    """
    get_hud_digit_matcher()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0, "video_open_failed"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_total / fps if frame_total > 0 else 0.0
    step = max(1, int(round(fps / scan_fps))) if scan_fps > 0 else int(fps)

    reads: list[KRead] = []
    frame_idx = 0
    while True:
        if frame_idx % step == 0:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            t = frame_idx / fps
            game, _ = extract_game_crop_bgr(frame, dataset_root=dataset_root)
            # 오독 방어: K/D/A 세 슬롯 모두 파싱된 프레임만 채택 —
            # 행이 배너·페이드로 오염되면 보통 셋 다 깨짐
            k, d, a, conf, method = read_kda_triple_from_game(game)
            if k is not None and (d is None or a is None):
                k = None
                method = "triple_incomplete"
            reads.append(KRead(t=t, k=k, conf=float(conf), method=method))
        else:
            if not cap.grab():
                break
            frame_idx += 1
            if frame_idx / fps >= duration:
                break
            continue

        frame_idx += 1
        if frame_idx / fps >= duration:
            break

    cap.release()
    return reads, duration, None


def timeline_from_reads(
    reads: list[KRead],
    *,
    duration: float,
    video_path: Path,
    scan_fps: float = 4.0,
    ace_kills: int = ACE_KILLS,
) -> HudAceTimeline:
    """원시 판독 리스트 → 라운드 분할·킬 이벤트·ace 판정 (영상 접근 없음).

    판정 로직의 단일 진입점: 실스캔(scan_hud_aces)과 캐시 재계산(hud_from_cache)이
    모두 이 함수를 쓰므로 로직 수정 시 두 경로가 자동 일치.
    """
    timeline = HudAceTimeline(
        video_path=str(video_path),
        scan_fps=scan_fps,
        ace_kills=ace_kills,
    )
    tracker = _KTracker()
    boundaries: list[float] = []   # 라운드 경계 시각 (연속 row_miss run 중앙)
    rowmiss_run = 0
    run_start = 0.0
    run_last = 0.0

    for r in reads:
        if r.k is not None:
            timeline.k_template_hits += 1
        elif r.method in ("template_miss", "triple_incomplete"):
            timeline.k_template_miss += 1
        elif r.method == "row_miss":
            timeline.k_row_miss += 1
        tracker.update(r.t, r.k, r.conf)

        # 경계 검출: KDA 행이 사라진(row_miss) 연속 구간 = 스코어보드/전환
        if r.method == "row_miss":
            if rowmiss_run == 0:
                run_start = r.t
            rowmiss_run += 1
            run_last = r.t
        else:
            if rowmiss_run >= _BOUNDARY_ROWMISS:
                boundaries.append((run_start + run_last) / 2)
            rowmiss_run = 0
    if rowmiss_run >= _BOUNDARY_ROWMISS:
        boundaries.append((run_start + run_last) / 2)

    rounds = _build_rounds(boundaries, tracker, duration)
    timeline.hud_end_count = len(boundaries)
    _assign_events(rounds, tracker)
    timeline.rounds = rounds
    timeline.ace_rounds = [r.round_index for r in rounds if r.ace]
    timeline.kill_events = tracker.kills
    timeline.reset_events = [[t, float(o), float(n)] for (t, o, n) in tracker.resets]

    if timeline.k_template_hits == 0:
        timeline.warnings.append("k_never_read")

    return timeline


def ace_clip_window(r: RoundTrack) -> tuple[float, float]:
    """올킬 클립 [start, end] — 라운드 시작·첫 킬 기준, 종료 후 tail 최소."""
    start = r.start_sec - _CLIP_PRE_ROUND_SEC
    if r.first_kill_sec is not None:
        start = min(start, r.first_kill_sec - _CLIP_PRE_KILL_SEC)
    start = max(0.0, start)

    end = r.end_sec + _CLIP_POST_END_SEC
    if r.ace_sec is not None:
        end = min(end, max(r.ace_sec + 12.0, r.end_sec + 0.5))

    if end - start > _CLIP_MAX_SEC:
        start = end - _CLIP_MAX_SEC
    if end - start < 12.0:
        start = max(0.0, end - 12.0)
    return start, end


def extract_ace_clips(
    video_path: Path,
    timeline: HudAceTimeline,
    output_dir: Path,
) -> list[Path]:
    from extract_labeled_clips import run_ffmpeg_extract

    out_dir = output_dir / Path(timeline.video_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for r in timeline.rounds:
        if not r.ace:
            continue
        clip_start, clip_end = ace_clip_window(r)
        label = sec_to_mss(r.end_sec).replace(":", "m")
        out_path = out_dir / f"{Path(timeline.video_path).stem}_R{r.round_index:02d}_{label}s_hud_ace.mp4"
        ok = run_ffmpeg_extract(video_path, clip_start, clip_end, out_path)
        tag = "OK" if ok else "FAIL"
        print(
            f"  R{r.round_index} {sec_to_mss(clip_start)}-{sec_to_mss(clip_end)} "
            f"(round {sec_to_mss(r.start_sec)}-{sec_to_mss(r.end_sec)}) -> {out_path.name} {tag}",
            flush=True,
        )
        if ok:
            written.append(out_path)
    return written


def sec_to_mss(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}:{s:02d}"


def format_report(timeline: HudAceTimeline) -> str:
    lines = [
        f"## {Path(timeline.video_path).name}",
        f"scan_fps={timeline.scan_fps}  rounds={len(timeline.rounds)}  "
        f"ace={timeline.ace_rounds}  hud_ends={timeline.hud_end_count}",
        f"k_read: hit={timeline.k_template_hits} template_miss={timeline.k_template_miss} "
        f"row_miss={timeline.k_row_miss}  kills={len(timeline.kill_events)} "
        f"resets={len(timeline.reset_events)}",
    ]
    if timeline.warnings:
        lines.append(f"warnings: {timeline.warnings}")
    for r in timeline.rounds:
        tag = " **ACE**" if r.ace else ""
        kt = ",".join(sec_to_mss(t) for t in r.kill_times)
        lines.append(
            f"  R{r.round_index:02d} {sec_to_mss(r.start_sec)}-{sec_to_mss(r.end_sec)} "
            f"kills={r.kills}[{kt}] resets={r.resets} n={r.k_samples}{tag}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HUD K/D/A 올킬 스캔 (닉 불필요)")
    p.add_argument("video_path")
    p.add_argument("--scan-fps", type=float, default=4.0)
    p.add_argument("--ace-kills", type=int, default=ACE_KILLS)
    p.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    p.add_argument("--json-out", default=None)
    p.add_argument("--extract", action="store_true")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"[hud-ace] 영상 없음: {video_path}")
        return 1

    print(f"[hud-ace] 스캔: {video_path.name} (fps={args.scan_fps})", flush=True)
    timeline = scan_hud_aces(
        video_path,
        scan_fps=args.scan_fps,
        ace_kills=args.ace_kills,
        dataset_root=Path(args.dataset_root),
    )
    print(format_report(timeline), flush=True)

    json_out = Path(args.json_out) if args.json_out else (
        DEFAULT_JSON_DIR / f"{video_path.stem}.json"
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(asdict(timeline), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[hud-ace] saved -> {json_out}", flush=True)

    if args.extract and timeline.ace_rounds:
        extract_ace_clips(video_path, timeline, Path(args.output_dir))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
