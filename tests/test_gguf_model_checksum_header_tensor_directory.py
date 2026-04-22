#!/usr/bin/env python3
"""Harness for GGUFModelChecksumHeaderAndTensorDirectory (IQ-1164)."""

from __future__ import annotations

from pathlib import Path

GGUF_MODEL_VALIDATE_OK = 0
GGUF_MODEL_VALIDATE_ERR_NULL_PTR = 1
GGUF_MODEL_VALIDATE_ERR_BAD_PARAM = 2
GGUF_MODEL_VALIDATE_ERR_OVERFLOW = 3

U64_MAX = (1 << 64) - 1


def _u64_add(a: int, b: int) -> int | None:
    s = a + b
    if s > U64_MAX:
        return None
    return s


def gguf_model_checksum_header_tensor_directory_reference(
    *,
    header_bytes: bytes | None,
    tensor_dir_bytes: bytes | None,
    seed: int,
    out_checksum_ref: list[int] | None,
    out_alias_header: bool = False,
    out_alias_tensor: bool = False,
) -> int:
    if header_bytes is None or tensor_dir_bytes is None or out_checksum_ref is None:
        return GGUF_MODEL_VALIDATE_ERR_NULL_PTR

    if out_alias_header and len(header_bytes) > 0:
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM
    if out_alias_tensor and len(tensor_dir_bytes) > 0:
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM

    state = (seed ^ 0xCBF29CE484222325) & U64_MAX
    abs_pos = 0

    for b in header_bytes:
        state ^= (b ^ ((abs_pos << 8) & U64_MAX))
        state = (state * 0x100000001B3) & U64_MAX
        abs_pos += 1

    for b in tensor_dir_bytes:
        state ^= (b ^ ((abs_pos << 8) & U64_MAX))
        state = (state * 0x100000001B3) & U64_MAX
        abs_pos += 1

    len_mix = len(header_bytes)
    shifted = (len(tensor_dir_bytes) << 1) & U64_MAX
    len_mix = _u64_add(len_mix, shifted)
    if len_mix is None:
        return GGUF_MODEL_VALIDATE_ERR_OVERFLOW

    state ^= len_mix

    state ^= state >> 33
    state = (state * 0xFF51AFD7ED558CCD) & U64_MAX
    state ^= state >> 33
    state = (state * 0xC4CEB9FE1A85EC53) & U64_MAX
    state ^= state >> 33

    out_checksum_ref[0] = state
    return GGUF_MODEL_VALIDATE_OK


def test_source_contains_iq1164_signature_and_contract() -> None:
    source = Path("src/gguf/validator.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFModelChecksumHeaderAndTensorDirectory("
    assert source.count(sig) == 1

    body = source.split(sig, 1)[1]
    assert "state *= 0x100000001B3" in body
    assert "state ^= state >> 33" in body
    assert "*out_checksum = state" in body


def test_success_known_vector_and_sensitivity() -> None:
    header = bytes([0x47, 0x47, 0x55, 0x46, 0x03, 0x00, 0x00, 0x00])
    tensor_dir = bytes([
        0x06, 0x74, 0x6F, 0x6B, 0x5F, 0x65, 0x6D, 0x62,
        0x02, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00,
    ])
    seed = 0x123456789ABCDEF0

    out = [0]
    rc = gguf_model_checksum_header_tensor_directory_reference(
        header_bytes=header,
        tensor_dir_bytes=tensor_dir,
        seed=seed,
        out_checksum_ref=out,
    )
    assert rc == GGUF_MODEL_VALIDATE_OK
    checksum_a = out[0]

    # deterministic re-run
    out2 = [0]
    rc = gguf_model_checksum_header_tensor_directory_reference(
        header_bytes=header,
        tensor_dir_bytes=tensor_dir,
        seed=seed,
        out_checksum_ref=out2,
    )
    assert rc == GGUF_MODEL_VALIDATE_OK
    assert out2[0] == checksum_a

    # one-byte mutation must perturb checksum
    mutated = bytearray(tensor_dir)
    mutated[3] ^= 0x01
    out3 = [0]
    rc = gguf_model_checksum_header_tensor_directory_reference(
        header_bytes=header,
        tensor_dir_bytes=bytes(mutated),
        seed=seed,
        out_checksum_ref=out3,
    )
    assert rc == GGUF_MODEL_VALIDATE_OK
    assert out3[0] != checksum_a


def test_null_and_no_partial_behaviors() -> None:
    header = b"\x01\x02\x03"
    tensor_dir = b"\x10\x20"

    out = [777]
    rc = gguf_model_checksum_header_tensor_directory_reference(
        header_bytes=None,
        tensor_dir_bytes=tensor_dir,
        seed=0,
        out_checksum_ref=out,
    )
    assert rc == GGUF_MODEL_VALIDATE_ERR_NULL_PTR
    assert out == [777]

    rc = gguf_model_checksum_header_tensor_directory_reference(
        header_bytes=header,
        tensor_dir_bytes=tensor_dir,
        seed=0,
        out_checksum_ref=None,
    )
    assert rc == GGUF_MODEL_VALIDATE_ERR_NULL_PTR


def test_output_alias_rejection_simulated() -> None:
    # Simulates the HolyC alias guard by forcing alias flags.
    out = [555]
    rc = gguf_model_checksum_header_tensor_directory_reference(
        header_bytes=b"\xAA\xBB",
        tensor_dir_bytes=b"\x11\x22",
        seed=0,
        out_checksum_ref=out,
        out_alias_header=True,
    )
    assert rc == GGUF_MODEL_VALIDATE_ERR_BAD_PARAM
    assert out == [555]

    rc = gguf_model_checksum_header_tensor_directory_reference(
        header_bytes=b"\xAA\xBB",
        tensor_dir_bytes=b"\x11\x22",
        seed=0,
        out_checksum_ref=out,
        out_alias_tensor=True,
    )
    assert rc == GGUF_MODEL_VALIDATE_ERR_BAD_PARAM
    assert out == [555]


def run() -> None:
    test_source_contains_iq1164_signature_and_contract()
    test_success_known_vector_and_sensitivity()
    test_null_and_no_partial_behaviors()
    test_output_alias_rejection_simulated()
    print("gguf_model_checksum_header_tensor_directory=ok")


if __name__ == "__main__":
    run()
