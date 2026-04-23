#!/usr/bin/env python3
"""Harness for IQ-1264 policy digest emitter (`InferencePolicyDigest;`)."""

from __future__ import annotations

from pathlib import Path

INFERENCE_POLICY_OK = 0
INFERENCE_POLICY_ERR_NULL_PTR = 1
INFERENCE_POLICY_ERR_BAD_PARAM = 2
INFERENCE_POLICY_ERR_PROFILE_GUARD = 3

INFERENCE_POLICY_PROFILE_SECURE_LOCAL = 1
INFERENCE_POLICY_PROFILE_DEV_LOCAL = 2

INFERENCE_POLICY_DIGEST_VERSION = 1
INFERENCE_POLICY_FNV_OFFSET_BASIS = 1469598103934665603
INFERENCE_POLICY_FNV_PRIME = 1099511628211
MASK_U64 = (1 << 64) - 1


class RuntimePolicyDigestState:
    def __init__(self) -> None:
        self.profile_id = INFERENCE_POLICY_PROFILE_SECURE_LOCAL
        self.iommu_enabled = 1
        self.bot_dma_log_enabled = 1
        self.bot_mmio_log_enabled = 1
        self.bot_dispatch_log_enabled = 1
        self.quarantine_gate_enabled = 1
        self.hash_manifest_gate_enabled = 1

    @staticmethod
    def _binary(value: int) -> bool:
        return value in (0, 1)

    @staticmethod
    def _u64(value: int) -> int:
        return value & MASK_U64

    @classmethod
    def _mix_u64(cls, state: int, value: int) -> int:
        state = cls._u64(state ^ cls._u64(value))
        state = cls._u64(state * INFERENCE_POLICY_FNV_PRIME)
        return state

    @classmethod
    def _finalize_u64(cls, state: int) -> int:
        state ^= state >> 31
        state = cls._u64(state * 1140071481932319845)
        state ^= state >> 27
        state = cls._u64(state * 7046029254386353131)
        state ^= state >> 33
        return cls._u64(state)

    def set_profile(self, profile_id: int) -> int:
        if profile_id not in (INFERENCE_POLICY_PROFILE_SECURE_LOCAL, INFERENCE_POLICY_PROFILE_DEV_LOCAL):
            return INFERENCE_POLICY_ERR_PROFILE_GUARD
        self.profile_id = profile_id
        return INFERENCE_POLICY_OK

    def set_runtime_guards_checked(
        self,
        iommu_enabled: int,
        bot_dma_log_enabled: int,
        bot_mmio_log_enabled: int,
        bot_dispatch_log_enabled: int,
        quarantine_gate_enabled: int,
        hash_manifest_gate_enabled: int,
    ) -> int:
        if not all(
            self._binary(v)
            for v in (
                iommu_enabled,
                bot_dma_log_enabled,
                bot_mmio_log_enabled,
                bot_dispatch_log_enabled,
                quarantine_gate_enabled,
                hash_manifest_gate_enabled,
            )
        ):
            return INFERENCE_POLICY_ERR_BAD_PARAM

        self.iommu_enabled = iommu_enabled
        self.bot_dma_log_enabled = bot_dma_log_enabled
        self.bot_mmio_log_enabled = bot_mmio_log_enabled
        self.bot_dispatch_log_enabled = bot_dispatch_log_enabled
        self.quarantine_gate_enabled = quarantine_gate_enabled
        self.hash_manifest_gate_enabled = hash_manifest_gate_enabled
        return INFERENCE_POLICY_OK

    def digest_checked(self) -> tuple[int, int, int, int]:
        if self.profile_id not in (INFERENCE_POLICY_PROFILE_SECURE_LOCAL, INFERENCE_POLICY_PROFILE_DEV_LOCAL):
            return INFERENCE_POLICY_ERR_PROFILE_GUARD, 0, 0, 0

        is_secure_default = 1 if self.profile_id == INFERENCE_POLICY_PROFILE_SECURE_LOCAL else 0
        if not all(
            self._binary(v)
            for v in (
                self.iommu_enabled,
                self.bot_dma_log_enabled,
                self.bot_mmio_log_enabled,
                self.bot_dispatch_log_enabled,
                self.quarantine_gate_enabled,
                self.hash_manifest_gate_enabled,
                is_secure_default,
            )
        ):
            return INFERENCE_POLICY_ERR_BAD_PARAM, 0, 0, 0

        policy_bits = 0
        policy_bits |= self.iommu_enabled << 0
        policy_bits |= self.bot_dma_log_enabled << 1
        policy_bits |= self.bot_mmio_log_enabled << 2
        policy_bits |= self.bot_dispatch_log_enabled << 3
        policy_bits |= self.quarantine_gate_enabled << 4
        policy_bits |= self.hash_manifest_gate_enabled << 5
        policy_bits |= is_secure_default << 6
        policy_bits |= (1 if self.profile_id == INFERENCE_POLICY_PROFILE_SECURE_LOCAL else 0) << 7

        state = INFERENCE_POLICY_FNV_OFFSET_BASIS
        state = self._mix_u64(state, 0x4951504F4C494359)
        state = self._mix_u64(state, INFERENCE_POLICY_DIGEST_VERSION)
        state = self._mix_u64(state, self.profile_id)
        state = self._mix_u64(state, is_secure_default)
        state = self._mix_u64(state, self.iommu_enabled)
        state = self._mix_u64(state, self.bot_dma_log_enabled)
        state = self._mix_u64(state, self.bot_mmio_log_enabled)
        state = self._mix_u64(state, self.bot_dispatch_log_enabled)
        state = self._mix_u64(state, self.quarantine_gate_enabled)
        state = self._mix_u64(state, self.hash_manifest_gate_enabled)
        state = self._mix_u64(state, INFERENCE_POLICY_PROFILE_SECURE_LOCAL)
        state = self._mix_u64(state, INFERENCE_POLICY_PROFILE_DEV_LOCAL)
        state = self._mix_u64(state, policy_bits)
        digest = self._finalize_u64(state)

        return INFERENCE_POLICY_OK, digest, policy_bits, self.profile_id

    def digest_cli(self) -> int:
        status, digest, _policy_bits, _profile_id = self.digest_checked()
        return digest if status == INFERENCE_POLICY_OK else -status


def test_source_contains_iq1264_symbols() -> None:
    src = Path("src/runtime/policy_digest.HC").read_text(encoding="utf-8")

    assert "I32 InferencePolicyDigestChecked(" in src
    assert "I32 InferencePolicyRuntimeGuardsSetChecked(" in src
    assert "I64 InferencePolicyDigest()" in src
    assert "INFERENCE_POLICY_FNV_OFFSET_BASIS" in src
    assert "INFERENCE_POLICY_PROFILE_SECURE_LOCAL" in src
    assert "InferenceProfileStatusChecked" in src


def test_secure_default_digest_emits_expected_policy_bits() -> None:
    state = RuntimePolicyDigestState()

    status, digest, policy_bits, profile_id = state.digest_checked()

    assert status == INFERENCE_POLICY_OK
    assert digest != 0
    assert profile_id == INFERENCE_POLICY_PROFILE_SECURE_LOCAL
    assert policy_bits == 0b11111111


def test_dev_local_profile_changes_digest_and_bits() -> None:
    state = RuntimePolicyDigestState()
    assert state.set_profile(INFERENCE_POLICY_PROFILE_DEV_LOCAL) == INFERENCE_POLICY_OK

    status, digest_dev, bits_dev, profile_id_dev = state.digest_checked()

    assert status == INFERENCE_POLICY_OK
    assert profile_id_dev == INFERENCE_POLICY_PROFILE_DEV_LOCAL
    assert bits_dev == 0b00111111

    state2 = RuntimePolicyDigestState()
    status2, digest_secure, _bits_secure, _profile_id_secure = state2.digest_checked()
    assert status2 == INFERENCE_POLICY_OK
    assert digest_dev != digest_secure


def test_guard_toggle_produces_deterministic_digest_delta() -> None:
    state = RuntimePolicyDigestState()
    status, baseline_digest, baseline_bits, _ = state.digest_checked()
    assert status == INFERENCE_POLICY_OK
    assert baseline_bits == 0xFF

    assert state.set_runtime_guards_checked(1, 1, 0, 1, 1, 1) == INFERENCE_POLICY_OK
    status2, digest2, bits2, _ = state.digest_checked()

    assert status2 == INFERENCE_POLICY_OK
    assert bits2 == 0b11111011
    assert digest2 != baseline_digest

    # Repeat call must be deterministic.
    status3, digest3, bits3, _ = state.digest_checked()
    assert status3 == INFERENCE_POLICY_OK
    assert bits3 == bits2
    assert digest3 == digest2


def test_rejects_non_binary_runtime_guard_inputs() -> None:
    state = RuntimePolicyDigestState()

    assert state.set_runtime_guards_checked(2, 1, 1, 1, 1, 1) == INFERENCE_POLICY_ERR_BAD_PARAM
    assert state.set_runtime_guards_checked(1, 1, 1, 1, 1, -1) == INFERENCE_POLICY_ERR_BAD_PARAM


def test_invalid_profile_is_rejected() -> None:
    state = RuntimePolicyDigestState()

    assert state.set_profile(99) == INFERENCE_POLICY_ERR_PROFILE_GUARD
    state.profile_id = 77

    status, digest, bits, profile = state.digest_checked()
    assert status == INFERENCE_POLICY_ERR_PROFILE_GUARD
    assert digest == 0
    assert bits == 0
    assert profile == 0


def test_cli_helper_returns_digest_or_negative_status() -> None:
    state = RuntimePolicyDigestState()

    value = state.digest_cli()
    assert value > 0

    state.profile_id = -1
    assert state.digest_cli() == -INFERENCE_POLICY_ERR_PROFILE_GUARD


if __name__ == "__main__":
    test_source_contains_iq1264_symbols()
    test_secure_default_digest_emits_expected_policy_bits()
    test_dev_local_profile_changes_digest_and_bits()
    test_guard_toggle_produces_deterministic_digest_delta()
    test_rejects_non_binary_runtime_guard_inputs()
    test_invalid_profile_is_rejected()
    test_cli_helper_returns_digest_or_negative_status()
    print("ok")
