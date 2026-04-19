#!/usr/bin/env python3
"""Parity harness for IQ-564 default-stride no-partial bridge wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_compute_scaled_qk_row_checked_default_stride import (
    attention_q16_compute_scaled_qk_row_checked_default_stride,
    explicit_default_stride_nopartial_bridge,
    q16_from_text,
)


def test_bridge_multilingual_case() -> None:
    head_dim = 16
    token_count = 3
    score_scale_q16 = 16384

    q_row = q16_from_text("bridge-世界🙂", head_dim)
    token_texts = ["alpha", "βήτα", "гамма"]

    k_rows = [0] * (token_count * head_dim)
    for token_index, text in enumerate(token_texts):
        row = q16_from_text(text, head_dim)
        base = token_index * head_dim
        for lane in range(head_dim):
            k_rows[base + lane] = row[lane]

    out_capacity = (token_count - 1) * token_count + 1
    out_a = [77] * out_capacity
    out_b = out_a.copy()

    err_a = attention_q16_compute_scaled_qk_row_checked_default_stride(
        q_row,
        len(q_row),
        k_rows,
        len(k_rows),
        token_count,
        head_dim,
        score_scale_q16,
        out_a,
        out_capacity,
    )
    err_b = explicit_default_stride_nopartial_bridge(
        q_row,
        len(q_row),
        k_rows,
        len(k_rows),
        token_count,
        head_dim,
        score_scale_q16,
        out_b,
        out_capacity,
    )

    assert err_a == err_b == 0
    assert out_a == out_b


if __name__ == "__main__":
    test_bridge_multilingual_case()
    print("ok")
