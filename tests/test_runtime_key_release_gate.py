#!/usr/bin/env python3
"""Harness for IQ-1272 key-release handshake verifier."""

from __future__ import annotations

from pathlib import Path

INFERENCE_KEY_RELEASE_OK = 0
INFERENCE_KEY_RELEASE_ERR_NULL_PTR = 1
INFERENCE_KEY_RELEASE_ERR_BAD_PARAM = 2
INFERENCE_KEY_RELEASE_ERR_POLICY_GUARD = 3

INFERENCE_PROFILE_SECURE_LOCAL = 1
INFERENCE_PROFILE_DEV_LOCAL = 2


class ProfileState:
    def __init__(self, mode: int = INFERENCE_PROFILE_SECURE_LOCAL) -> None:
        self.mode = mode

    def status_checked(self) -> tuple[int, int, int]:
        if self.mode not in (INFERENCE_PROFILE_SECURE_LOCAL, INFERENCE_PROFILE_DEV_LOCAL):
            return INFERENCE_KEY_RELEASE_ERR_POLICY_GUARD, 0, 0
        return INFERENCE_KEY_RELEASE_OK, self.mode, 1 if self.mode == INFERENCE_PROFILE_SECURE_LOCAL else 0


def verify_checked(
    templeos_signed_approval: int,
    attestation_evidence_valid: int,
    policy_digest_parity_valid: int,
    profile_state: ProfileState,
) -> tuple[int, int, int, int]:
    for value in (
        templeos_signed_approval,
        attestation_evidence_valid,
        policy_digest_parity_valid,
    ):
        if value not in (0, 1):
            return INFERENCE_KEY_RELEASE_ERR_BAD_PARAM, 0, 0, 0

    status, profile_id, is_secure_default = profile_state.status_checked()
    if status != INFERENCE_KEY_RELEASE_OK:
        return INFERENCE_KEY_RELEASE_ERR_POLICY_GUARD, 0, 0, 0

    failure_bits = 0
    failure_bits |= (0 if templeos_signed_approval else 1) << 0
    failure_bits |= (0 if attestation_evidence_valid else 1) << 1
    failure_bits |= (0 if policy_digest_parity_valid else 1) << 2
    failure_bits |= (0 if is_secure_default else 1) << 3

    release_allowed = 1 if failure_bits == 0 else 0
    return INFERENCE_KEY_RELEASE_OK, release_allowed, failure_bits, profile_id


def test_source_contains_iq1272_symbols() -> None:
    src = Path("src/runtime/key_release_gate.HC").read_text(encoding="utf-8")

    assert "I32 InferenceKeyReleaseHandshakeVerifyChecked(" in src
    assert "I64 InferenceKeyReleaseStatus(" in src
    assert "failure_bits |= ((!templeos_signed_approval) << 0);" in src
    assert "failure_bits |= ((!attestation_evidence_valid) << 1);" in src
    assert "failure_bits |= ((!policy_digest_parity_valid) << 2);" in src
    assert "failure_bits |= ((!is_secure_default) << 3);" in src


def test_secure_local_all_signals_required() -> None:
    profile = ProfileState(INFERENCE_PROFILE_SECURE_LOCAL)

    status, release_allowed, failure_bits, profile_id = verify_checked(1, 1, 1, profile)
    assert status == INFERENCE_KEY_RELEASE_OK
    assert release_allowed == 1
    assert failure_bits == 0
    assert profile_id == INFERENCE_PROFILE_SECURE_LOCAL


def test_each_missing_signal_sets_expected_failure_bit() -> None:
    profile = ProfileState(INFERENCE_PROFILE_SECURE_LOCAL)

    status, release_allowed, failure_bits, _ = verify_checked(0, 1, 1, profile)
    assert status == INFERENCE_KEY_RELEASE_OK
    assert release_allowed == 0
    assert failure_bits == 0b0001

    status, release_allowed, failure_bits, _ = verify_checked(1, 0, 1, profile)
    assert status == INFERENCE_KEY_RELEASE_OK
    assert release_allowed == 0
    assert failure_bits == 0b0010

    status, release_allowed, failure_bits, _ = verify_checked(1, 1, 0, profile)
    assert status == INFERENCE_KEY_RELEASE_OK
    assert release_allowed == 0
    assert failure_bits == 0b0100


def test_dev_local_never_bypasses_secure_default_gate() -> None:
    profile = ProfileState(INFERENCE_PROFILE_DEV_LOCAL)

    status, release_allowed, failure_bits, profile_id = verify_checked(1, 1, 1, profile)
    assert status == INFERENCE_KEY_RELEASE_OK
    assert profile_id == INFERENCE_PROFILE_DEV_LOCAL
    assert release_allowed == 0
    assert failure_bits == 0b1000


def test_bad_binary_inputs_rejected() -> None:
    profile = ProfileState(INFERENCE_PROFILE_SECURE_LOCAL)

    status, release_allowed, failure_bits, profile_id = verify_checked(2, 1, 1, profile)
    assert status == INFERENCE_KEY_RELEASE_ERR_BAD_PARAM
    assert release_allowed == 0
    assert failure_bits == 0
    assert profile_id == 0


def test_profile_guard_rejects_unknown_mode() -> None:
    profile = ProfileState(777)

    status, release_allowed, failure_bits, profile_id = verify_checked(1, 1, 1, profile)
    assert status == INFERENCE_KEY_RELEASE_ERR_POLICY_GUARD
    assert release_allowed == 0
    assert failure_bits == 0
    assert profile_id == 0


if __name__ == "__main__":
    test_source_contains_iq1272_symbols()
    test_secure_local_all_signals_required()
    test_each_missing_signal_sets_expected_failure_bit()
    test_dev_local_never_bypasses_secure_default_gate()
    test_bad_binary_inputs_rejected()
    test_profile_guard_rejects_unknown_mode()
    print("ok")
