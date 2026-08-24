#!/usr/bin/env python3
"""运行公开生产仓内容门禁。"""

from __future__ import annotations

import json
from pathlib import Path

from xuannv_embedding.utils.repository_policy import validate_repository

if __name__ == "__main__":
    print(json.dumps(validate_repository(Path.cwd()), ensure_ascii=False, sort_keys=True))
