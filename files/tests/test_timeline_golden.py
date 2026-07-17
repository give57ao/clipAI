# -*- coding: utf-8 -*-
"""timeline_from_reads 스냅샷 회귀 (SONNET_TASKS.md T3-3).

files/tests/fixtures/sig_cache_v2/ 에 복사해 둔 실캐시 2개(E:\\clipai_result\\sig_cache_v2
원본, 2026-07-17 스냅)를 재생 없이 로드해 라운드/ace 판정이 고정값과 같은지 확인한다.
E: 드라이브(원본 캐시 디렉터리) 유무와 무관하게 통과해야 하므로 fixtures/ 사본만 사용.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from detect_ace_hud import timeline_from_reads
from hud_from_cache import load_boundary_verdicts, load_reads

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sig_cache_v2"

# stem -> (rounds 수, ace_rounds, hud_end_count, k_template_hits)
GOLDEN = {
    "2026-03-19 23-00-50": (17, [16], 16, 973),
    "2026-03-22 03-02-03": (27, [14], 26, 1427),
}


@pytest.mark.parametrize("stem", sorted(GOLDEN))
def test_timeline_snapshot(stem):
    cache_path = FIXTURE_DIR / f"{stem}.json"
    if not cache_path.exists():
        pytest.skip(f"fixture missing: {cache_path}")

    reads, duration, scan_fps, loaded_stem = load_reads(cache_path)
    assert loaded_stem == stem
    boundary_verdicts = load_boundary_verdicts(stem, FIXTURE_DIR)

    tl = timeline_from_reads(
        reads,
        duration=duration,
        video_path=Path(rf"E:\OBS\{stem}.mp4"),
        scan_fps=scan_fps,
        boundary_verdicts=boundary_verdicts,
    )

    n_rounds, ace_rounds, hud_end_count, k_template_hits = GOLDEN[stem]
    assert len(tl.rounds) == n_rounds
    assert tl.ace_rounds == ace_rounds
    assert tl.hud_end_count == hud_end_count
    assert tl.k_template_hits == k_template_hits
