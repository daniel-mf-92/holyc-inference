#!/usr/bin/env python3
"""Harness for IQ-1253 model quarantine import->verify->promote workflow."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

QUARANTINE_OK = 0
QUARANTINE_ERR_NULL_PTR = 1
QUARANTINE_ERR_BAD_PARAM = 2
QUARANTINE_ERR_OVERFLOW = 3
QUARANTINE_ERR_MALFORMED = 4
QUARANTINE_ERR_BAD_STATE = 5
QUARANTINE_ERR_PROFILE_GUARD = 6
QUARANTINE_ERR_ENTRY_NOT_FOUND = 7
QUARANTINE_ERR_SIZE_MISMATCH = 8
QUARANTINE_ERR_HASH_MISMATCH = 9

QUARANTINE_STAGE_EMPTY = 0
QUARANTINE_STAGE_IMPORTED = 1
QUARANTINE_STAGE_VERIFIED = 2
QUARANTINE_STAGE_PROMOTED = 3

QUARANTINE_MAX_PATH_CHARS = 240
QUARANTINE_SHA256_HEX_CHARS = 64

INFERENCE_PROFILE_SECURE_LOCAL = 1
INFERENCE_PROFILE_DEV_LOCAL = 2


@dataclass
class ModelQuarantineState:
    stage: int = QUARANTINE_STAGE_EMPTY
    import_rel_path: str = ""
    promoted_rel_path: str = ""
    imported_model_nbytes: int = 0
    verified_manifest_entry: int = -1
    verified_hash_hex: str = ""
    verified_profile_id: int = 0


class ProfileState:
    def __init__(self) -> None:
        self.profile_id = INFERENCE_PROFILE_SECURE_LOCAL

    def set_dev_local_checked(self) -> int:
        self.profile_id = INFERENCE_PROFILE_DEV_LOCAL
        return 0

    def set_secure_local_checked(self) -> int:
        self.profile_id = INFERENCE_PROFILE_SECURE_LOCAL
        return 0

    def status_checked(self) -> tuple[int, int, int]:
        is_secure = 1 if self.profile_id == INFERENCE_PROFILE_SECURE_LOCAL else 0
        return 0, self.profile_id, is_secure


def _path_safe(path: str) -> bool:
    if not path or len(path) > QUARANTINE_MAX_PATH_CHARS:
        return False
    if path.startswith("/") or path.startswith("\\"):
        return False
    if "\\" in path or ":" in path or "\n" in path:
        return False
    parts = path.split("/")
    return not any(part == ".." for part in parts)


def _find_manifest_entry(
    manifest: bytes,
    target_path: str,
    target_size: int,
) -> tuple[int, str, int]:
    entry_index = 0
    for raw in manifest.decode("ascii").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        chunks = line.split()
        if len(chunks) != 3:
            return QUARANTINE_ERR_MALFORMED, "", -1

        hash_hex, size_tok, rel_path = chunks
        if len(hash_hex) != QUARANTINE_SHA256_HEX_CHARS:
            return QUARANTINE_ERR_MALFORMED, "", -1
        if any(ch not in "0123456789abcdefABCDEF" for ch in hash_hex):
            return QUARANTINE_ERR_MALFORMED, "", -1

        if not size_tok.isdigit():
            return QUARANTINE_ERR_MALFORMED, "", -1
        size_val = int(size_tok)

        if rel_path == target_path:
            if size_val != target_size:
                return QUARANTINE_ERR_SIZE_MISMATCH, "", -1
            return QUARANTINE_OK, hash_hex.lower(), entry_index

        entry_index += 1

    return QUARANTINE_ERR_ENTRY_NOT_FOUND, "", -1


def model_quarantine_import_checked(
    state: ModelQuarantineState,
    candidate_rel_path: str,
    model_bytes: bytes,
) -> int:
    if not _path_safe(candidate_rel_path):
        return QUARANTINE_ERR_BAD_PARAM
    if len(model_bytes) <= 0:
        return QUARANTINE_ERR_BAD_PARAM

    state.stage = QUARANTINE_STAGE_IMPORTED
    state.import_rel_path = candidate_rel_path
    state.promoted_rel_path = ""
    state.imported_model_nbytes = len(model_bytes)
    state.verified_manifest_entry = -1
    state.verified_hash_hex = ""
    state.verified_profile_id = 0
    return QUARANTINE_OK


def model_quarantine_verify_checked(
    state: ModelQuarantineState,
    profile: ProfileState,
    manifest: bytes,
    model_bytes: bytes,
) -> int:
    if state.stage < QUARANTINE_STAGE_IMPORTED:
        return QUARANTINE_ERR_BAD_STATE
    if len(model_bytes) != state.imported_model_nbytes:
        return QUARANTINE_ERR_SIZE_MISMATCH

    err, expected_hash, entry_index = _find_manifest_entry(
        manifest,
        state.import_rel_path,
        len(model_bytes),
    )
    if err != QUARANTINE_OK:
        return err

    computed_hash = hashlib.sha256(model_bytes).hexdigest()
    if computed_hash != expected_hash:
        return QUARANTINE_ERR_HASH_MISMATCH

    status, profile_id, _ = profile.status_checked()
    if status != 0:
        return QUARANTINE_ERR_PROFILE_GUARD

    state.stage = QUARANTINE_STAGE_VERIFIED
    state.verified_manifest_entry = entry_index
    state.verified_hash_hex = computed_hash
    state.verified_profile_id = profile_id
    return QUARANTINE_OK


def model_quarantine_promote_checked(
    state: ModelQuarantineState,
    profile: ProfileState,
    trusted_rel_path: str,
) -> int:
    if state.stage < QUARANTINE_STAGE_VERIFIED:
        return QUARANTINE_ERR_BAD_STATE
    if not _path_safe(trusted_rel_path):
        return QUARANTINE_ERR_BAD_PARAM

    status, _, is_secure = profile.status_checked()
    if status != 0 or is_secure != 1:
        return QUARANTINE_ERR_PROFILE_GUARD

    state.stage = QUARANTINE_STAGE_PROMOTED
    state.promoted_rel_path = trusted_rel_path
    return QUARANTINE_OK


def model_quarantine_import_verify_promote_checked(
    state: ModelQuarantineState,
    profile: ProfileState,
    candidate_rel_path: str,
    trusted_rel_path: str,
    manifest: bytes,
    model_bytes: bytes,
) -> int:
    err = model_quarantine_import_checked(state, candidate_rel_path, model_bytes)
    if err != QUARANTINE_OK:
        return err
    err = model_quarantine_verify_checked(state, profile, manifest, model_bytes)
    if err != QUARANTINE_OK:
        return err
    return model_quarantine_promote_checked(state, profile, trusted_rel_path)


def test_source_contains_iq1253_symbols() -> None:
    src = open("src/model/quarantine.HC", "r", encoding="utf-8").read()
    assert "I32 ModelQuarantineImportChecked(" in src
    assert "I32 ModelQuarantineVerifyChecked(" in src
    assert "I32 ModelQuarantinePromoteChecked(" in src
    assert "I32 ModelQuarantineImportVerifyPromoteChecked(" in src
    assert "if (!is_secure_default)" in src


def test_import_verify_promote_success_secure_local() -> None:
    profile = ProfileState()
    state = ModelQuarantineState()
    model = b"holy-model-bytes"
    digest = hashlib.sha256(model).hexdigest()
    manifest = f"{digest} {len(model)} models/q4.gguf\n".encode("ascii")

    err = model_quarantine_import_verify_promote_checked(
        state,
        profile,
        "models/q4.gguf",
        "trusted/models/q4.gguf",
        manifest,
        model,
    )
    assert err == QUARANTINE_OK
    assert state.stage == QUARANTINE_STAGE_PROMOTED
    assert state.import_rel_path == "models/q4.gguf"
    assert state.promoted_rel_path == "trusted/models/q4.gguf"


def test_promote_blocked_in_dev_local_after_verify() -> None:
    profile = ProfileState()
    state = ModelQuarantineState()
    model = b"guarded"
    digest = hashlib.sha256(model).hexdigest()
    manifest = f"{digest} {len(model)} models/a.gguf\n".encode("ascii")

    assert model_quarantine_import_checked(state, "models/a.gguf", model) == QUARANTINE_OK
    assert model_quarantine_verify_checked(state, profile, manifest, model) == QUARANTINE_OK

    assert profile.set_dev_local_checked() == 0
    err = model_quarantine_promote_checked(state, profile, "trusted/models/a.gguf")
    assert err == QUARANTINE_ERR_PROFILE_GUARD
    assert state.stage == QUARANTINE_STAGE_VERIFIED
    assert state.promoted_rel_path == ""


def test_verify_hash_mismatch_keeps_import_stage() -> None:
    profile = ProfileState()
    state = ModelQuarantineState()
    model = b"abc"
    wrong_digest = hashlib.sha256(b"xyz").hexdigest()
    manifest = f"{wrong_digest} {len(model)} models/b.gguf\n".encode("ascii")

    assert model_quarantine_import_checked(state, "models/b.gguf", model) == QUARANTINE_OK
    err = model_quarantine_verify_checked(state, profile, manifest, model)
    assert err == QUARANTINE_ERR_HASH_MISMATCH
    assert state.stage == QUARANTINE_STAGE_IMPORTED
    assert state.verified_hash_hex == ""


def test_import_rejects_unsafe_paths() -> None:
    state = ModelQuarantineState()
    model = b"abc"

    assert model_quarantine_import_checked(state, "../escape.gguf", model) == QUARANTINE_ERR_BAD_PARAM
    assert model_quarantine_import_checked(state, "/abs/path.gguf", model) == QUARANTINE_ERR_BAD_PARAM
    assert state.stage == QUARANTINE_STAGE_EMPTY


if __name__ == "__main__":
    test_source_contains_iq1253_symbols()
    test_import_verify_promote_success_secure_local()
    test_promote_blocked_in_dev_local_after_verify()
    test_verify_hash_mismatch_keeps_import_stage()
    test_import_rejects_unsafe_paths()
    print("ok")
