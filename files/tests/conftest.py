# -*- coding: utf-8 -*-
"""files/ 스크립트들은 상호 bare import(`from detect_ace_hud import ...`)를 쓰므로
테스트에서도 files/를 sys.path에 넣어야 동일하게 임포트된다."""

from __future__ import annotations

import sys
from pathlib import Path

FILES_DIR = Path(__file__).resolve().parent.parent
if str(FILES_DIR) not in sys.path:
    sys.path.insert(0, str(FILES_DIR))
