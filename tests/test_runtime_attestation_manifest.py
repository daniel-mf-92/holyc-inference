#!/usr/bin/env python3
"""Harness for IQ-1271 attestation manifest emitter."""

from __future__ import annotations

from pathlib import Path

ATTEST_MANIFEST_OK = 0
ATTEST_MANIFEST_ERR_NULL_PTR = 1
ATTEST_MANIFEST_ERR_BAD_PARAM = 2
ATTEST_MANIFEST_ERR_CAPACITY = 3


class Manifest:
    def __init__(self) -> None:
        self.session_id = ""
        self.profile_name = ""
        self.policy_digest_hex = ""
        self.profile_id = 0
        self.attestation_nonce = 0
        self.trusted_model_count = 0
        self.quarantine_block_count = 0
        self.gpu_dispatch_allowed = 0
        self.iommu_active = 0
        self.bot_gpu_hooks_active = 0
        self.lines: list[str] = []


def _safe_char(ch: str) -> bool:
    return ch.isalnum() or ch in "-_ .:/".replace(" ", "")


def _copy_sanitized(value: str, max_chars: int) -> tuple[int, str]:
    if not value or len(value) > max_chars:
        return ATTEST_MANIFEST_ERR_CAPACITY if len(value) > max_chars else ATTEST_MANIFEST_ERR_BAD_PARAM, ""
    if any(not _safe_char(ch) for ch in value):
        return ATTEST_MANIFEST_ERR_BAD_PARAM, ""
    return ATTEST_MANIFEST_OK, value


def init_checked(
    manifest: Manifest,
    session_id: str,
    profile_id: int,
    profile_name: str,
    policy_digest_hex: str,
    nonce: int,
) -> int:
    if profile_id <= 0 or nonce < 0:
        return ATTEST_MANIFEST_ERR_BAD_PARAM

    rc, session = _copy_sanitized(session_id, 64)
    if rc != ATTEST_MANIFEST_OK:
        return rc
    rc, pname = _copy_sanitized(profile_name, 31)
    if rc != ATTEST_MANIFEST_OK:
        return rc
    rc, digest = _copy_sanitized(policy_digest_hex, 64)
    if rc != ATTEST_MANIFEST_OK:
        return rc

    manifest.session_id = session
    manifest.profile_id = profile_id
    manifest.profile_name = pname
    manifest.policy_digest_hex = digest
    manifest.attestation_nonce = nonce
    return ATTEST_MANIFEST_OK


def emit_checked(manifest: Manifest) -> int:
    manifest.lines = [
        f"session_id={manifest.session_id}",
        f"profile_name={manifest.profile_name}",
        f"policy_digest={manifest.policy_digest_hex}",
        f"profile_id={manifest.profile_id}",
        f"nonce={manifest.attestation_nonce}",
        f"trusted_models={manifest.trusted_model_count}",
        f"quarantine_blocks={manifest.quarantine_block_count}",
        f"gpu_dispatch_allowed={manifest.gpu_dispatch_allowed}",
        f"iommu_active={manifest.iommu_active}",
        f"bot_gpu_hooks_active={manifest.bot_gpu_hooks_active}",
    ]
    return ATTEST_MANIFEST_OK


def test_source_contains_iq1271_symbols() -> None:
    src = Path("src/runtime/attestation_manifest.HC").read_text(encoding="utf-8")

    assert "class InferenceAttestationManifest" in src
    assert "I32 InferenceAttestationManifestInitChecked(" in src
    assert "I32 InferenceAttestationManifestSetTrustCountsChecked(" in src
    assert "I32 InferenceAttestationManifestSetGPUStateChecked(" in src
    assert "I32 InferenceAttestationManifestEmitChecked(" in src
    assert "gpu_dispatch_allowed" in src


def test_manifest_emit_secure_local_payload() -> None:
    m = Manifest()
    digest = "a" * 64
    assert init_checked(m, "sess-20260423-1", 1, "secure-local", digest, 42) == ATTEST_MANIFEST_OK

    m.trusted_model_count = 3
    m.quarantine_block_count = 0
    m.gpu_dispatch_allowed = 1
    m.iommu_active = 1
    m.bot_gpu_hooks_active = 1

    assert emit_checked(m) == ATTEST_MANIFEST_OK
    assert m.lines[0] == "session_id=sess-20260423-1"
    assert m.lines[1] == "profile_name=secure-local"
    assert m.lines[2] == f"policy_digest={digest}"
    assert "trusted_models=3" in m.lines
    assert "gpu_dispatch_allowed=1" in m.lines
    assert "iommu_active=1" in m.lines
    assert "bot_gpu_hooks_active=1" in m.lines


def test_manifest_rejects_unsafe_text() -> None:
    m = Manifest()
    bad_digest = "abc$" + ("0" * 60)
    assert init_checked(m, "sess", 1, "secure-local", bad_digest, 1) == ATTEST_MANIFEST_ERR_BAD_PARAM


def test_manifest_rejects_bad_numeric_fields() -> None:
    m = Manifest()
    digest = "b" * 64
    assert init_checked(m, "sess", 0, "secure-local", digest, 1) == ATTEST_MANIFEST_ERR_BAD_PARAM
    assert init_checked(m, "sess", 1, "secure-local", digest, -1) == ATTEST_MANIFEST_ERR_BAD_PARAM
