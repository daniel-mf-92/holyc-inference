#!/usr/bin/env python3
"""Parity/spec harness for TokenizerBPESpecialTokensResolveCheckedNoPartial (IQ-1166)."""

from __future__ import annotations

from pathlib import Path

TOKENIZER_BPE_OK = 0
TOKENIZER_BPE_ERR_NULL_PTR = 101
TOKENIZER_BPE_ERR_BAD_PARAM = 102
TOKENIZER_BPE_ERR_OVERFLOW = 103
TOKENIZER_BPE_ERR_NOT_FOUND = 104
TOKENIZER_BPE_ERR_TYPE_MISMATCH = 105
TOKENIZER_BPE_ERR_RANGE = 106

TOKENIZER_GGUF_META_PARSE_OK = 0
TOKENIZER_GGUF_META_PARSE_ERR_NULL_PTR = 1
TOKENIZER_GGUF_META_PARSE_ERR_NOT_FOUND = 9
TOKENIZER_GGUF_META_PARSE_ERR_TYPE_MISMATCH = 10

I32_MAX = 0x7FFFFFFF


class MetaI64TypeMismatch(Exception):
    pass


def gguf_meta_get_i64_by_key(metadata: dict[str, object] | None, key: str) -> tuple[int, int]:
    if metadata is None:
        return TOKENIZER_GGUF_META_PARSE_ERR_NULL_PTR, 0
    if key not in metadata:
        return TOKENIZER_GGUF_META_PARSE_ERR_NOT_FOUND, 0

    value = metadata[key]
    if not isinstance(value, int):
        return TOKENIZER_GGUF_META_PARSE_ERR_TYPE_MISMATCH, 0

    return TOKENIZER_GGUF_META_PARSE_OK, value


def tokenizer_bpe_special_token_resolve_one_checked(
    metadata: dict[str, object] | None,
    key: str,
    fallback_token_id: int,
    allow_missing_without_fallback: bool,
    out_token_id: list[int] | None,
    out_has_token: list[bool] | None,
) -> int:
    if metadata is None or out_token_id is None or out_has_token is None:
        return TOKENIZER_BPE_ERR_NULL_PTR

    status, value = gguf_meta_get_i64_by_key(metadata, key)

    if status == TOKENIZER_GGUF_META_PARSE_OK:
        if value < 0:
            return TOKENIZER_BPE_ERR_RANGE
        if value > I32_MAX:
            return TOKENIZER_BPE_ERR_OVERFLOW
        out_token_id[0] = value
        out_has_token[0] = True
        return TOKENIZER_BPE_OK

    if status == TOKENIZER_GGUF_META_PARSE_ERR_NOT_FOUND:
        if fallback_token_id >= 0:
            out_token_id[0] = fallback_token_id
            out_has_token[0] = True
            return TOKENIZER_BPE_OK

        if allow_missing_without_fallback:
            out_token_id[0] = -1
            out_has_token[0] = False
            return TOKENIZER_BPE_OK

        return TOKENIZER_BPE_ERR_NOT_FOUND

    if status == TOKENIZER_GGUF_META_PARSE_ERR_TYPE_MISMATCH:
        return TOKENIZER_BPE_ERR_TYPE_MISMATCH

    if status == TOKENIZER_GGUF_META_PARSE_ERR_NULL_PTR:
        return TOKENIZER_BPE_ERR_NULL_PTR

    return TOKENIZER_BPE_ERR_BAD_PARAM


def tokenizer_bpe_special_tokens_resolve_checked_nopartial(
    metadata: dict[str, object] | None,
    fallback_bos_token_id: int,
    fallback_eos_token_id: int,
    fallback_pad_token_id: int,
    allow_missing_without_fallback: bool,
    out_bos_token_id: list[int] | None,
    out_eos_token_id: list[int] | None,
    out_pad_token_id: list[int] | None,
    out_has_bos: list[bool] | None,
    out_has_eos: list[bool] | None,
    out_has_pad: list[bool] | None,
) -> int:
    if (
        metadata is None
        or out_bos_token_id is None
        or out_eos_token_id is None
        or out_pad_token_id is None
        or out_has_bos is None
        or out_has_eos is None
        or out_has_pad is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if fallback_bos_token_id < -1 or fallback_eos_token_id < -1 or fallback_pad_token_id < -1:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    staged_bos = [out_bos_token_id[0]]
    staged_eos = [out_eos_token_id[0]]
    staged_pad = [out_pad_token_id[0]]
    staged_has_bos = [out_has_bos[0]]
    staged_has_eos = [out_has_eos[0]]
    staged_has_pad = [out_has_pad[0]]

    err = tokenizer_bpe_special_token_resolve_one_checked(
        metadata,
        "tokenizer.ggml.bos_token_id",
        fallback_bos_token_id,
        allow_missing_without_fallback,
        staged_bos,
        staged_has_bos,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    err = tokenizer_bpe_special_token_resolve_one_checked(
        metadata,
        "tokenizer.ggml.eos_token_id",
        fallback_eos_token_id,
        allow_missing_without_fallback,
        staged_eos,
        staged_has_eos,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    err = tokenizer_bpe_special_token_resolve_one_checked(
        metadata,
        "tokenizer.ggml.padding_token_id",
        fallback_pad_token_id,
        allow_missing_without_fallback,
        staged_pad,
        staged_has_pad,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    out_bos_token_id[0] = staged_bos[0]
    out_eos_token_id[0] = staged_eos[0]
    out_pad_token_id[0] = staged_pad[0]
    out_has_bos[0] = staged_has_bos[0]
    out_has_eos[0] = staged_has_eos[0]
    out_has_pad[0] = staged_has_pad[0]
    return TOKENIZER_BPE_OK


def run_case(
    metadata: dict[str, object] | None,
    fallback_triplet: tuple[int, int, int],
    allow_missing_without_fallback: bool,
    expected_err: int,
    expected_triplet: tuple[int, int, int],
    expected_has: tuple[bool, bool, bool],
    unchanged_on_error: bool,
) -> None:
    bos = [111]
    eos = [222]
    pad = [333]
    has_bos = [False]
    has_eos = [False]
    has_pad = [False]

    err = tokenizer_bpe_special_tokens_resolve_checked_nopartial(
        metadata,
        fallback_triplet[0],
        fallback_triplet[1],
        fallback_triplet[2],
        allow_missing_without_fallback,
        bos,
        eos,
        pad,
        has_bos,
        has_eos,
        has_pad,
    )

    assert err == expected_err

    if expected_err == TOKENIZER_BPE_OK:
        assert (bos[0], eos[0], pad[0]) == expected_triplet
        assert (has_bos[0], has_eos[0], has_pad[0]) == expected_has
    elif unchanged_on_error:
        assert (bos[0], eos[0], pad[0]) == (111, 222, 333)
        assert (has_bos[0], has_eos[0], has_pad[0]) == (False, False, False)


def test_source_contains_resolver_and_metadata_keys() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")

    assert "I32 TokenizerBPESpecialTokensResolveCheckedNoPartial(" in source
    assert "I32 TokenizerBPESpecialTokenResolveOneChecked(" in source
    assert '"tokenizer.ggml.bos_token_id"' in source
    assert '"tokenizer.ggml.eos_token_id"' in source
    assert '"tokenizer.ggml.padding_token_id"' in source


def test_metadata_values_override_fallbacks() -> None:
    metadata = {
        "tokenizer.ggml.bos_token_id": 1,
        "tokenizer.ggml.eos_token_id": 2,
        "tokenizer.ggml.padding_token_id": 3,
    }
    run_case(
        metadata=metadata,
        fallback_triplet=(11, 22, 33),
        allow_missing_without_fallback=False,
        expected_err=TOKENIZER_BPE_OK,
        expected_triplet=(1, 2, 3),
        expected_has=(True, True, True),
        unchanged_on_error=False,
    )


def test_missing_key_uses_fallback_id() -> None:
    metadata = {
        "tokenizer.ggml.bos_token_id": 7,
        "tokenizer.ggml.padding_token_id": 9,
    }
    run_case(
        metadata=metadata,
        fallback_triplet=(-1, 55, -1),
        allow_missing_without_fallback=False,
        expected_err=TOKENIZER_BPE_OK,
        expected_triplet=(7, 55, 9),
        expected_has=(True, True, True),
        unchanged_on_error=False,
    )


def test_missing_key_without_fallback_can_be_explicitly_allowed() -> None:
    metadata = {
        "tokenizer.ggml.bos_token_id": 101,
        "tokenizer.ggml.eos_token_id": 102,
    }
    run_case(
        metadata=metadata,
        fallback_triplet=(-1, -1, -1),
        allow_missing_without_fallback=True,
        expected_err=TOKENIZER_BPE_OK,
        expected_triplet=(101, 102, -1),
        expected_has=(True, True, False),
        unchanged_on_error=False,
    )


def test_missing_required_key_fails_closed_without_partial_commit() -> None:
    metadata = {
        "tokenizer.ggml.bos_token_id": 7,
        "tokenizer.ggml.eos_token_id": 8,
    }
    run_case(
        metadata=metadata,
        fallback_triplet=(-1, -1, -1),
        allow_missing_without_fallback=False,
        expected_err=TOKENIZER_BPE_ERR_NOT_FOUND,
        expected_triplet=(0, 0, 0),
        expected_has=(False, False, False),
        unchanged_on_error=True,
    )


def test_type_mismatch_fails_closed_without_partial_commit() -> None:
    metadata = {
        "tokenizer.ggml.bos_token_id": 7,
        "tokenizer.ggml.eos_token_id": "8",
        "tokenizer.ggml.padding_token_id": 9,
    }
    run_case(
        metadata=metadata,
        fallback_triplet=(-1, -1, -1),
        allow_missing_without_fallback=True,
        expected_err=TOKENIZER_BPE_ERR_TYPE_MISMATCH,
        expected_triplet=(0, 0, 0),
        expected_has=(False, False, False),
        unchanged_on_error=True,
    )


def test_negative_metadata_id_rejected_as_range_error() -> None:
    metadata = {
        "tokenizer.ggml.bos_token_id": -7,
        "tokenizer.ggml.eos_token_id": 8,
        "tokenizer.ggml.padding_token_id": 9,
    }
    run_case(
        metadata=metadata,
        fallback_triplet=(-1, -1, -1),
        allow_missing_without_fallback=True,
        expected_err=TOKENIZER_BPE_ERR_RANGE,
        expected_triplet=(0, 0, 0),
        expected_has=(False, False, False),
        unchanged_on_error=True,
    )


def test_large_metadata_id_rejected_as_overflow() -> None:
    metadata = {
        "tokenizer.ggml.bos_token_id": I32_MAX + 1,
        "tokenizer.ggml.eos_token_id": 8,
        "tokenizer.ggml.padding_token_id": 9,
    }
    run_case(
        metadata=metadata,
        fallback_triplet=(-1, -1, -1),
        allow_missing_without_fallback=True,
        expected_err=TOKENIZER_BPE_ERR_OVERFLOW,
        expected_triplet=(0, 0, 0),
        expected_has=(False, False, False),
        unchanged_on_error=True,
    )


def test_invalid_fallback_domain_rejected() -> None:
    metadata = {
        "tokenizer.ggml.bos_token_id": 1,
        "tokenizer.ggml.eos_token_id": 2,
        "tokenizer.ggml.padding_token_id": 3,
    }
    run_case(
        metadata=metadata,
        fallback_triplet=(-2, -1, -1),
        allow_missing_without_fallback=True,
        expected_err=TOKENIZER_BPE_ERR_BAD_PARAM,
        expected_triplet=(0, 0, 0),
        expected_has=(False, False, False),
        unchanged_on_error=True,
    )


if __name__ == "__main__":
    test_source_contains_resolver_and_metadata_keys()
    test_metadata_values_override_fallbacks()
    test_missing_key_uses_fallback_id()
    test_missing_key_without_fallback_can_be_explicitly_allowed()
    test_missing_required_key_fails_closed_without_partial_commit()
    test_type_mismatch_fails_closed_without_partial_commit()
    test_negative_metadata_id_rejected_as_range_error()
    test_large_metadata_id_rejected_as_overflow()
    test_invalid_fallback_domain_rejected()
    print("ok")
