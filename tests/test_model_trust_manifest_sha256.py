#!/usr/bin/env python3
"""Contract checks for trusted model manifest parser + SHA256 verifier."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

TRUST_MANIFEST_OK = 0
TRUST_MANIFEST_ERR_NULL_PTR = 1
TRUST_MANIFEST_ERR_BAD_PARAM = 2
TRUST_MANIFEST_ERR_OVERFLOW = 3
TRUST_MANIFEST_ERR_MALFORMED = 4
TRUST_MANIFEST_ERR_CAPACITY = 5
TRUST_MANIFEST_ERR_ENTRY_NOT_FOUND = 6
TRUST_MANIFEST_ERR_SIZE_MISMATCH = 7
TRUST_MANIFEST_ERR_HASH_MISMATCH = 8

TRUST_MANIFEST_SHA256_HEX_CHARS = 64
TRUST_MANIFEST_MAX_PATH_CHARS = 240
TRUST_U64_MAX_VALUE = (1 << 64) - 1


@dataclass(frozen=True)
class TrustManifestEntry:
    sha256_hex: str
    size_bytes: int
    rel_path: str


def _is_ws(ch: str) -> bool:
    return ch in (" ", "\t", "\r")


def _is_hex(ch: str) -> bool:
    return ch.isdigit() or ("a" <= ch <= "f") or ("A" <= ch <= "F")


def _validate_rel_path(path: str) -> int:
    if not path:
        return TRUST_MANIFEST_ERR_MALFORMED
    if len(path) > TRUST_MANIFEST_MAX_PATH_CHARS:
        return TRUST_MANIFEST_ERR_CAPACITY
    if path[0] in "/\\":
        return TRUST_MANIFEST_ERR_MALFORMED
    if "\\" in path or ":" in path or "\x00" in path:
        return TRUST_MANIFEST_ERR_MALFORMED

    segs = path.split("/")
    if any(seg in ("", ".", "..") for seg in segs):
        return TRUST_MANIFEST_ERR_MALFORMED
    return TRUST_MANIFEST_OK


def _parse_u64_decimal(tok: str) -> tuple[int, int]:
    if not tok:
        return TRUST_MANIFEST_ERR_MALFORMED, 0
    value = 0
    for ch in tok:
        if not ch.isdigit():
            return TRUST_MANIFEST_ERR_MALFORMED, 0
        if value > TRUST_U64_MAX_VALUE // 10:
            return TRUST_MANIFEST_ERR_OVERFLOW, 0
        value = value * 10 + (ord(ch) - ord("0"))
        if value > TRUST_U64_MAX_VALUE:
            return TRUST_MANIFEST_ERR_OVERFLOW, 0
    return TRUST_MANIFEST_OK, value


def _parse_line(line: str) -> tuple[int, TrustManifestEntry | None]:
    i = 0
    while i < len(line) and _is_ws(line[i]):
        i += 1

    if i >= len(line) or line[i] == "#":
        return TRUST_MANIFEST_OK, None

    hash_start = i
    while i < len(line) and not _is_ws(line[i]):
        i += 1
    hash_tok = line[hash_start:i]

    while i < len(line) and _is_ws(line[i]):
        i += 1
    size_start = i
    while i < len(line) and not _is_ws(line[i]):
        i += 1
    size_tok = line[size_start:i]

    while i < len(line) and _is_ws(line[i]):
        i += 1
    path_tok = line[i:].rstrip(" \t\r")

    if len(hash_tok) != TRUST_MANIFEST_SHA256_HEX_CHARS:
        return TRUST_MANIFEST_ERR_MALFORMED, None
    if any(not _is_hex(ch) for ch in hash_tok):
        return TRUST_MANIFEST_ERR_MALFORMED, None

    err, size_value = _parse_u64_decimal(size_tok)
    if err != TRUST_MANIFEST_OK:
        return err, None

    err = _validate_rel_path(path_tok)
    if err != TRUST_MANIFEST_OK:
        return err, None

    return (
        TRUST_MANIFEST_OK,
        TrustManifestEntry(hash_tok.lower(), size_value, path_tok),
    )


def trust_manifest_parse_checked_no_partial(
    manifest_bytes: bytes | None,
    entry_capacity: int,
) -> tuple[int, list[TrustManifestEntry]]:
    if manifest_bytes is None:
        return TRUST_MANIFEST_ERR_NULL_PTR, []
    if entry_capacity < 0:
        return TRUST_MANIFEST_ERR_BAD_PARAM, []

    text = manifest_bytes.decode("ascii")
    staged: list[TrustManifestEntry] = []

    for line in text.split("\n"):
        err, entry = _parse_line(line)
        if err != TRUST_MANIFEST_OK:
            return err, []
        if entry is None:
            continue
        staged.append(entry)

    if len(staged) > entry_capacity:
        return TRUST_MANIFEST_ERR_CAPACITY, []

    return TRUST_MANIFEST_OK, staged


def trust_manifest_verify_path_checked_no_partial(
    manifest_bytes: bytes | None,
    entry_capacity: int,
    target_rel_path: str | None,
    model_bytes: bytes | None,
) -> tuple[int, int]:
    if manifest_bytes is None or target_rel_path is None or model_bytes is None:
        return TRUST_MANIFEST_ERR_NULL_PTR, -1

    err, entries = trust_manifest_parse_checked_no_partial(manifest_bytes, entry_capacity)
    if err != TRUST_MANIFEST_OK:
        return err, -1

    target_err = _validate_rel_path(target_rel_path)
    if target_err != TRUST_MANIFEST_OK:
        return target_err, -1

    for idx, entry in enumerate(entries):
        if entry.rel_path != target_rel_path:
            continue

        if entry.size_bytes != len(model_bytes):
            return TRUST_MANIFEST_ERR_SIZE_MISMATCH, -1

        digest = hashlib.sha256(model_bytes).hexdigest()
        if digest != entry.sha256_hex:
            return TRUST_MANIFEST_ERR_HASH_MISMATCH, -1

        return TRUST_MANIFEST_OK, idx

    return TRUST_MANIFEST_ERR_ENTRY_NOT_FOUND, -1


def test_parse_accepts_comment_blank_and_valid_lines() -> None:
    model = b"abc"
    digest = hashlib.sha256(model).hexdigest()
    manifest = (
        b"# trust manifest\n"
        + f"{digest} 3 models/a.gguf\n".encode("ascii")
        + b"   \n"
        + f"{digest.upper()} 3 models/b.gguf\n".encode("ascii")
    )

    err, entries = trust_manifest_parse_checked_no_partial(manifest, 8)
    assert err == TRUST_MANIFEST_OK
    assert len(entries) == 2
    assert entries[0].sha256_hex == digest
    assert entries[1].sha256_hex == digest
    assert entries[1].rel_path == "models/b.gguf"


def test_parse_rejects_path_traversal_and_preserves_no_partial_contract() -> None:
    model = b"abc"
    digest = hashlib.sha256(model).hexdigest()
    good_line = f"{digest} 3 models/a.gguf\n".encode("ascii")
    bad_line = f"{digest} 3 ../escape.gguf\n".encode("ascii")

    err, entries = trust_manifest_parse_checked_no_partial(good_line + bad_line, 8)
    assert err == TRUST_MANIFEST_ERR_MALFORMED
    assert entries == []


def test_parse_rejects_u64_size_overflow() -> None:
    digest = hashlib.sha256(b"x").hexdigest()
    # 2^64, one over max u64.
    manifest = f"{digest} 18446744073709551616 models/ovf.gguf\n".encode("ascii")

    err, _ = trust_manifest_parse_checked_no_partial(manifest, 4)
    assert err == TRUST_MANIFEST_ERR_OVERFLOW


def test_verify_path_success() -> None:
    model = b"holy-c-inference"
    digest = hashlib.sha256(model).hexdigest()
    manifest = f"{digest} {len(model)} models/tiny.gguf\n".encode("ascii")

    err, idx = trust_manifest_verify_path_checked_no_partial(
        manifest,
        4,
        "models/tiny.gguf",
        model,
    )
    assert err == TRUST_MANIFEST_OK
    assert idx == 0


def test_verify_path_size_mismatch() -> None:
    model = b"data"
    digest = hashlib.sha256(model).hexdigest()
    manifest = f"{digest} 5 models/tiny.gguf\n".encode("ascii")

    err, _ = trust_manifest_verify_path_checked_no_partial(
        manifest,
        4,
        "models/tiny.gguf",
        model,
    )
    assert err == TRUST_MANIFEST_ERR_SIZE_MISMATCH


def test_verify_path_hash_mismatch() -> None:
    model = b"data"
    wrong_digest = hashlib.sha256(b"other").hexdigest()
    manifest = f"{wrong_digest} {len(model)} models/tiny.gguf\n".encode("ascii")

    err, _ = trust_manifest_verify_path_checked_no_partial(
        manifest,
        4,
        "models/tiny.gguf",
        model,
    )
    assert err == TRUST_MANIFEST_ERR_HASH_MISMATCH


def test_verify_path_not_found() -> None:
    model = b"data"
    digest = hashlib.sha256(model).hexdigest()
    manifest = f"{digest} {len(model)} models/tiny.gguf\n".encode("ascii")

    err, _ = trust_manifest_verify_path_checked_no_partial(
        manifest,
        4,
        "models/missing.gguf",
        model,
    )
    assert err == TRUST_MANIFEST_ERR_ENTRY_NOT_FOUND


def test_capacity_enforced() -> None:
    model = b"z"
    digest = hashlib.sha256(model).hexdigest()
    manifest = (
        f"{digest} 1 models/a.gguf\n"
        + f"{digest} 1 models/b.gguf\n"
    ).encode("ascii")

    err, _ = trust_manifest_parse_checked_no_partial(manifest, 1)
    assert err == TRUST_MANIFEST_ERR_CAPACITY


if __name__ == "__main__":
    test_parse_accepts_comment_blank_and_valid_lines()
    test_parse_rejects_path_traversal_and_preserves_no_partial_contract()
    test_parse_rejects_u64_size_overflow()
    test_verify_path_success()
    test_verify_path_size_mismatch()
    test_verify_path_hash_mismatch()
    test_verify_path_not_found()
    test_capacity_enforced()
    print("ok")
