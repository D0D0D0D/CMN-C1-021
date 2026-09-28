"""PII マスク機構 — S-3 出力ゲートの mask_patterns を実適用する。

docs/02_design.md §8。credential（sk-/JWT 等）は framework.security の credential scan が
**ブロック**するのに対し、本モジュールはドメイン PII を **マスク（redact）して通過**させる。

`MASK_REGISTRY` は config の mask_patterns ラベル → 正規表現。未登録ラベルは拡張ポイント
（派生テンプレが登録）として skip する。標準 OptimizationSolverAgent は PII を網羅列挙しない
方針のため、高信頼なパターンのみ同梱する。
"""

from __future__ import annotations

import re
from typing import Iterable

# ラベル → コンパイル済み正規表現。マッチ部分は [REDACTED:<label>] に置換される。
MASK_REGISTRY: dict[str, re.Pattern] = {
    # マイナンバー（12 桁。区切りあり/なし）
    "mynumber": re.compile(r"\b\d{4}-?\d{4}-?\d{4}\b"),
    # 口座番号（誤検出を避けるためラベル文脈付き）
    "account_number": re.compile(
        r"(?i)(?:口座番号|account[\s_-]?(?:number|no)\.?)\s*[:：]?\s*\d{6,}"
    ),
}

_REDACTION = "[REDACTED:{label}]"


def apply_masks(text: str, labels: Iterable[str]) -> tuple[str, list[str]]:
    """text に対し、labels で有効化された既知パターンをマスクする。

    Returns: (マスク後テキスト, 実際に適用されたラベル一覧)。
    未登録ラベルは拡張ポイントとして無視する（例外を投げない）。
    """
    masked = str(text)
    applied: list[str] = []
    for label in labels or []:
        pattern = MASK_REGISTRY.get(label)
        if pattern is None:
            continue  # 拡張ポイント（派生が登録）
        new_masked, n = pattern.subn(_REDACTION.format(label=label), masked)
        if n > 0:
            masked = new_masked
            applied.append(label)
    return masked, applied
