"""Credential detection — S-2/S-3 のクレデンシャル走査。

released AgentCore SDK には `framework.security` が無いため（旧スタブ環境が
提供していた）、本テンプレ内に self-contained で実装する（F.Liu GovernmentProcedureQAAgent の inline
`_CREDENTIAL_PATTERNS` 方式に倣う）。マスク（PII redaction）は src/masking.py が担い、
本モジュールは「検出したらブロック」する credential 専用。
"""

from __future__ import annotations

import re

_CREDENTIAL_PATTERNS = [
    r"sk-[a-zA-Z0-9]{20,}",
    r"eyJ[a-zA-Z0-9._-]{10,}",
    r"AKIA[A-Z0-9]{16}",
    r"Bearer\s+[a-zA-Z0-9._-]{20,}",
    r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",
    r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*\S+",
]


def detect_credentials(text: str) -> list[str]:
    """text に含まれる credential パターンを返す（空なら検出なし）。"""
    found: list[str] = []
    for pat in _CREDENTIAL_PATTERNS:
        if re.search(pat, str(text)):
            found.append(pat)
    return found
