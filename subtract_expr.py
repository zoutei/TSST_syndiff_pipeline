"""
Parse ``subtract`` stage linear combinations of workspace labels, e.g.
``hp_d + hp_b - bkg_temp_smooth``.
"""

from __future__ import annotations

import re


_LABEL = r"[a-zA-Z_][a-zA-Z0-9_]*"


def parse_subtract_expression(expr: str) -> list[tuple[int, str]]:
    """
    Parse a sum of signed workspace labels.

    Examples::

        hp_d + hp_b - bkg_temp_smooth  ->  (+1, hp_d), (+1, hp_b), (-1, bkg_temp_smooth)
        -hp_d + hp_b              ->  (-1, hp_d), (+1, hp_b)

    Labels match ``[a-zA-Z_][a-zA-Z0-9_]*`` (YAML keys / workspace names).
    """
    s = (expr or "").strip()
    if not s:
        raise ValueError("subtract expression is empty")
    if s[0] not in "+-":
        s = "+" + s
    matches = re.findall(rf"([+-])\s*({_LABEL})", s)
    if not matches:
        raise ValueError(f"subtract expression has no terms: {expr!r}")
    # Reject junk after last term (e.g. "a + b )")
    consumed = 0
    for m in re.finditer(rf"[+-]\s*{_LABEL}", s):
        consumed = m.end()
    tail = s[consumed:].strip()
    if tail:
        raise ValueError(f"subtract expression has trailing garbage: {expr!r}")
    out: list[tuple[int, str]] = []
    for sign_ch, lab in matches:
        out.append((1 if sign_ch == "+" else -1, lab))
    return out


def labels_in_subtract_expression(expr: str) -> list[str]:
    """Workspace labels referenced by an expression (order preserved, duplicates kept)."""
    return [lab for _, lab in parse_subtract_expression(expr)]
