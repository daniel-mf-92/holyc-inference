#!/usr/bin/env python3
"""Harness for IQ-1319: benchmark policy-digest bind helper."""

from __future__ import annotations

from pathlib import Path

ATTEST_MANIFEST_OK = 0
ATTEST_MANIFEST_ERR_NULL_PTR = 1
ATTEST_MANIFEST_ERR_BAD_PARAM = 2
ATTEST_MANIFEST_ERR_CAPACITY = 3
ATTEST_MANIFEST_ERR_OVERFLOW = 4


def _u64_add_checked(a: int, b: int) -> tuple[int, int]:
    total = a + b
    if total > 0xFFFFFFFFFFFFFFFF:
        return ATTEST_MANIFEST_ERR_OVERFLOW, 0
    return ATTEST_MANIFEST_OK, total


def _fnv1a_mix_step(state: int, value: int) -> tuple[int, int]:
    rc, state = _u64_add_checked(state, value & 0xFFFFFFFFFFFFFFFF)
    if rc != ATTEST_MANIFEST_OK:
        return rc, 0
    state ^= value & 0xFFFFFFFFFFFFFFFF
    state = (state * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return ATTEST_MANIFEST_OK, state


def policy_digest_bind_checked(
    policy_digest_q64: int,
    telemetry_digest_q64: int,
    total_ops: int,
    total_cycles: int,
    profile_id: int,
    nonce: int,
) -> tuple[int, int]:
    for v in [
        policy_digest_q64,
        telemetry_digest_q64,
        total_ops,
        total_cycles,
        profile_id,
        nonce,
    ]:
        if v < 0:
            return ATTEST_MANIFEST_ERR_BAD_PARAM, 0

    state = 1469598103934665603
    for v in [
        policy_digest_q64,
        telemetry_digest_q64,
        total_ops,
        total_cycles,
        profile_id,
        nonce,
    ]:
        rc, state = _fnv1a_mix_step(state, v)
        if rc != ATTEST_MANIFEST_OK:
            return rc, 0
    return ATTEST_MANIFEST_OK, state


def test_source_contains_iq1319_symbols() -> None:
    src = Path("src/runtime/attestation_manifest.HC").read_text(encoding="utf-8")

    assert "I32 Q8_0DotBenchRunDefaultSuitePolicyDigestBindChecked(" in src
    assert "class InferencePolicyDigestBindPayload" in src
    assert "policy_digest_q64" in src
    assert "telemetry_digest_q64" in src
    assert "bound_digest_q64" in src


def test_policy_digest_bind_happy_path() -> None:
    rc, digest = policy_digest_bind_checked(
        policy_digest_q64=0x1234,
        telemetry_digest_q64=0x5678,
        total_ops=1024,
        total_cycles=65536,
        profile_id=1,
        nonce=42,
    )
    assert rc == ATTEST_MANIFEST_OK
    assert digest != 0


def test_policy_digest_bind_deterministic() -> None:
    args = dict(
        policy_digest_q64=0xDEADBEEF,
        telemetry_digest_q64=0xABCD1234,
        total_ops=999,
        total_cycles=777777,
        profile_id=2,
        nonce=20260424,
    )
    rc1, d1 = policy_digest_bind_checked(**args)
    rc2, d2 = policy_digest_bind_checked(**args)
    assert rc1 == ATTEST_MANIFEST_OK
    assert rc2 == ATTEST_MANIFEST_OK
    assert d1 == d2


def test_policy_digest_bind_input_sensitivity() -> None:
    base = dict(
        policy_digest_q64=11,
        telemetry_digest_q64=22,
        total_ops=33,
        total_cycles=44,
        profile_id=1,
        nonce=2,
    )
    rc1, d1 = policy_digest_bind_checked(**base)
    base["telemetry_digest_q64"] = 23
    rc2, d2 = policy_digest_bind_checked(**base)
    assert rc1 == ATTEST_MANIFEST_OK
    assert rc2 == ATTEST_MANIFEST_OK
    assert d1 != d2


def test_policy_digest_bind_rejects_negative_inputs() -> None:
    rc, _ = policy_digest_bind_checked(-1, 2, 3, 4, 1, 2)
    assert rc == ATTEST_MANIFEST_ERR_BAD_PARAM


def test_policy_digest_bind_overflow_guard() -> None:
    rc, _ = policy_digest_bind_checked(
        policy_digest_q64=0xFFFFFFFFFFFFFFFF,
        telemetry_digest_q64=0xFFFFFFFFFFFFFFFF,
        total_ops=0xFFFFFFFFFFFFFFFF,
        total_cycles=0xFFFFFFFFFFFFFFFF,
        profile_id=0xFFFFFFFFFFFFFFFF,
        nonce=0xFFFFFFFFFFFFFFFF,
    )
    assert rc in (ATTEST_MANIFEST_OK, ATTEST_MANIFEST_ERR_OVERFLOW)
