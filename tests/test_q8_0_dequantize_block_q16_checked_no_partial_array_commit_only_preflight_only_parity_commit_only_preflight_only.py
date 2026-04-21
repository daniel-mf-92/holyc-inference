#!/usr/bin/env python3
"""Parity harness for IQ-954 companion wrapper."""

from pathlib import Path


def test_source_contains_iq954_signature() -> None:
    source = Path("src/quant/q8_0.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "status = Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "if (!Q8_0TryMulI64NonNeg(snapshot_block_count - 1," in body
    assert "if (!Q8_0TryMulI64NonNeg(recomputed_required_src_blocks," in body
    assert "if (staged_block_count != snapshot_block_count ||" in body
    assert "*out_required_dst_bytes = staged_required_dst_bytes;" in body


def run() -> None:
    test_source_contains_iq954_signature()
    print("iq954_source_contract=ok")


if __name__ == "__main__":
    run()
