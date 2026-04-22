from __future__ import annotations

from pathlib import Path
import random

GGUF_TDBASE_OK = 0
GGUF_TDBASE_ERR_NULL_PTR = 1
GGUF_TDBASE_ERR_OVERFLOW = 3
GGUF_TDBASE_ERR_OUT_OF_BOUNDS = 7

U64_MAX = (1 << 64) - 1


def tensor_lookup_by_abs_offset(
    abs_offset: int,
    tensor_abs_starts: list[int] | None,
    tensor_abs_ends: list[int] | None,
    sorted_tensor_indices: list[int] | None,
) -> tuple[int, int, int, int]:
    if (
        tensor_abs_starts is None
        or tensor_abs_ends is None
        or sorted_tensor_indices is None
    ):
        return GGUF_TDBASE_ERR_NULL_PTR, 0, 0, 0

    for sorted_pos, tensor_index in enumerate(sorted_tensor_indices):
        if tensor_index < 0 or tensor_index >= len(tensor_abs_starts):
            return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, 0, 0, 0

        start = tensor_abs_starts[sorted_pos]
        end = tensor_abs_ends[sorted_pos]
        if end < start:
            return GGUF_TDBASE_ERR_OVERFLOW, 0, 0, 0

        if abs_offset >= start and abs_offset < end:
            return GGUF_TDBASE_OK, tensor_index, start, end

    return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, 0, 0, 0


def tensor_lookup_span_by_index_and_abs_range(
    tensor_index: int,
    span_abs_start: int,
    span_abs_end: int,
    tensor_abs_starts: list[int] | None,
    tensor_abs_ends: list[int] | None,
    sorted_position_by_tensor: list[int] | None,
) -> tuple[int, int, int, int, int]:
    if (
        tensor_abs_starts is None
        or tensor_abs_ends is None
        or sorted_position_by_tensor is None
    ):
        return GGUF_TDBASE_ERR_NULL_PTR, 0, 0, 0, 0

    if span_abs_end < span_abs_start:
        return GGUF_TDBASE_ERR_OVERFLOW, 0, 0, 0, 0

    if tensor_index < 0 or tensor_index >= len(sorted_position_by_tensor):
        return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, 0, 0, 0, 0

    sorted_pos = sorted_position_by_tensor[tensor_index]
    if sorted_pos < 0 or sorted_pos >= len(tensor_abs_starts):
        return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, 0, 0, 0, 0

    tensor_abs_start = tensor_abs_starts[sorted_pos]
    tensor_abs_end = tensor_abs_ends[sorted_pos]

    if tensor_abs_end < tensor_abs_start:
        return GGUF_TDBASE_ERR_OVERFLOW, 0, 0, 0, 0

    if span_abs_start < tensor_abs_start or span_abs_end > tensor_abs_end:
        return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, 0, 0, 0, 0

    return (
        GGUF_TDBASE_OK,
        tensor_abs_start,
        tensor_abs_end,
        span_abs_start - tensor_abs_start,
        span_abs_end - tensor_abs_start,
    )


def tensor_payload_read_window_checked(
    owner_tensor_index: int,
    payload_abs_start: int,
    payload_nbytes: int,
    tensor_abs_starts: list[int] | None,
    tensor_abs_ends: list[int] | None,
    sorted_tensor_indices: list[int] | None,
    sorted_position_by_tensor: list[int] | None,
    out_abs_start: list[int] | None,
    out_abs_end: list[int] | None,
    out_span_start_in_tensor: list[int] | None,
    out_span_end_in_tensor: list[int] | None,
) -> int:
    if (
        tensor_abs_starts is None
        or tensor_abs_ends is None
        or sorted_tensor_indices is None
        or sorted_position_by_tensor is None
        or out_abs_start is None
        or out_abs_end is None
        or out_span_start_in_tensor is None
        or out_span_end_in_tensor is None
    ):
        return GGUF_TDBASE_ERR_NULL_PTR

    out_abs_start[0] = 0
    out_abs_end[0] = 0
    out_span_start_in_tensor[0] = 0
    out_span_end_in_tensor[0] = 0

    if payload_abs_start < 0 or payload_nbytes < 0:
        return GGUF_TDBASE_ERR_OVERFLOW

    if payload_abs_start > U64_MAX - payload_nbytes:
        return GGUF_TDBASE_ERR_OVERFLOW
    payload_abs_end = payload_abs_start + payload_nbytes

    if payload_nbytes != 0:
        err, owner, _, _ = tensor_lookup_by_abs_offset(
            payload_abs_start,
            tensor_abs_starts,
            tensor_abs_ends,
            sorted_tensor_indices,
        )
        if err != GGUF_TDBASE_OK:
            return err
        if owner != owner_tensor_index:
            return GGUF_TDBASE_ERR_OUT_OF_BOUNDS

    err, _, _, span_start, span_end = tensor_lookup_span_by_index_and_abs_range(
        owner_tensor_index,
        payload_abs_start,
        payload_abs_end,
        tensor_abs_starts,
        tensor_abs_ends,
        sorted_position_by_tensor,
    )
    if err != GGUF_TDBASE_OK:
        return err

    out_abs_start[0] = payload_abs_start
    out_abs_end[0] = payload_abs_end
    out_span_start_in_tensor[0] = span_start
    out_span_end_in_tensor[0] = span_end
    return GGUF_TDBASE_OK


def test_source_contains_iq1139_function_and_owner_verification_logic() -> None:
    source = Path("src/gguf/tensor_data_base.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorPayloadReadWindowChecked(U64 owner_tensor_index,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I64 GGUFTensorInfoLoadPayloadWindowByRelSpan(", 1)[0]

    assert "payload_abs_start > 0xFFFFFFFFFFFFFFFF - payload_nbytes" in body
    assert "if (payload_nbytes)" in body
    assert "GGUFTensorLookupByAbsOffset(payload_abs_start," in body
    assert "if (resolved_owner_tensor_index != owner_tensor_index)" in body
    assert "GGUFTensorLookupSpanByIndexAndAbsRange(owner_tensor_index," in body


def test_happy_path_non_empty_payload_matches_owner_and_span() -> None:
    starts = [0x1000, 0x1080, 0x1100]
    ends = [0x1020, 0x10B0, 0x1180]
    sorted_idx = [0, 1, 2]
    sorted_pos = [0, 1, 2]

    out_abs_start = [999]
    out_abs_end = [999]
    out_span_start = [999]
    out_span_end = [999]

    err = tensor_payload_read_window_checked(
        owner_tensor_index=1,
        payload_abs_start=0x1090,
        payload_nbytes=0x10,
        tensor_abs_starts=starts,
        tensor_abs_ends=ends,
        sorted_tensor_indices=sorted_idx,
        sorted_position_by_tensor=sorted_pos,
        out_abs_start=out_abs_start,
        out_abs_end=out_abs_end,
        out_span_start_in_tensor=out_span_start,
        out_span_end_in_tensor=out_span_end,
    )

    assert err == GGUF_TDBASE_OK
    assert out_abs_start[0] == 0x1090
    assert out_abs_end[0] == 0x10A0
    assert out_span_start[0] == 0x10
    assert out_span_end[0] == 0x20


def test_owner_mismatch_is_rejected_before_publish() -> None:
    starts = [0x2000, 0x2100]
    ends = [0x2040, 0x2180]
    sorted_idx = [0, 1]
    sorted_pos = [0, 1]

    out_abs_start = [777]
    out_abs_end = [777]
    out_span_start = [777]
    out_span_end = [777]

    err = tensor_payload_read_window_checked(
        owner_tensor_index=1,
        payload_abs_start=0x2010,
        payload_nbytes=8,
        tensor_abs_starts=starts,
        tensor_abs_ends=ends,
        sorted_tensor_indices=sorted_idx,
        sorted_position_by_tensor=sorted_pos,
        out_abs_start=out_abs_start,
        out_abs_end=out_abs_end,
        out_span_start_in_tensor=out_span_start,
        out_span_end_in_tensor=out_span_end,
    )

    assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
    assert out_abs_start == [0]
    assert out_abs_end == [0]
    assert out_span_start == [0]
    assert out_span_end == [0]


def test_zero_length_payload_skips_owner_lookup_but_requires_owner_span_containment() -> None:
    starts = [0x3000, 0x3040]
    ends = [0x3040, 0x3080]
    sorted_idx = [0, 1]
    sorted_pos = [0, 1]

    out_abs_start = [1]
    out_abs_end = [1]
    out_span_start = [1]
    out_span_end = [1]

    err = tensor_payload_read_window_checked(
        owner_tensor_index=1,
        payload_abs_start=0x3060,
        payload_nbytes=0,
        tensor_abs_starts=starts,
        tensor_abs_ends=ends,
        sorted_tensor_indices=sorted_idx,
        sorted_position_by_tensor=sorted_pos,
        out_abs_start=out_abs_start,
        out_abs_end=out_abs_end,
        out_span_start_in_tensor=out_span_start,
        out_span_end_in_tensor=out_span_end,
    )

    assert err == GGUF_TDBASE_OK
    assert out_abs_start[0] == 0x3060
    assert out_abs_end[0] == 0x3060
    assert out_span_start[0] == 0x20
    assert out_span_end[0] == 0x20


def test_overflow_and_cross_tensor_windows_fail() -> None:
    starts = [0x4000, 0x4080]
    ends = [0x4040, 0x40E0]
    sorted_idx = [0, 1]
    sorted_pos = [0, 1]

    out_abs_start = [5]
    out_abs_end = [5]
    out_span_start = [5]
    out_span_end = [5]

    err = tensor_payload_read_window_checked(
        owner_tensor_index=0,
        payload_abs_start=U64_MAX - 3,
        payload_nbytes=8,
        tensor_abs_starts=starts,
        tensor_abs_ends=ends,
        sorted_tensor_indices=sorted_idx,
        sorted_position_by_tensor=sorted_pos,
        out_abs_start=out_abs_start,
        out_abs_end=out_abs_end,
        out_span_start_in_tensor=out_span_start,
        out_span_end_in_tensor=out_span_end,
    )
    assert err == GGUF_TDBASE_ERR_OVERFLOW

    err = tensor_payload_read_window_checked(
        owner_tensor_index=0,
        payload_abs_start=0x4038,
        payload_nbytes=0x10,
        tensor_abs_starts=starts,
        tensor_abs_ends=ends,
        sorted_tensor_indices=sorted_idx,
        sorted_position_by_tensor=sorted_pos,
        out_abs_start=out_abs_start,
        out_abs_end=out_abs_end,
        out_span_start_in_tensor=out_span_start,
        out_span_end_in_tensor=out_span_end,
    )
    assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
    assert out_abs_start == [0]
    assert out_abs_end == [0]
    assert out_span_start == [0]
    assert out_span_end == [0]


def test_null_ptr_contract_and_no_partial_publish() -> None:
    starts = [0x5000]
    ends = [0x5080]
    sorted_idx = [0]
    sorted_pos = [0]

    out_abs_start = [111]
    out_abs_end = [222]
    out_span_start = [333]
    out_span_end = [444]

    err = tensor_payload_read_window_checked(
        owner_tensor_index=0,
        payload_abs_start=0x5000,
        payload_nbytes=0x10,
        tensor_abs_starts=None,
        tensor_abs_ends=ends,
        sorted_tensor_indices=sorted_idx,
        sorted_position_by_tensor=sorted_pos,
        out_abs_start=out_abs_start,
        out_abs_end=out_abs_end,
        out_span_start_in_tensor=out_span_start,
        out_span_end_in_tensor=out_span_end,
    )
    assert err == GGUF_TDBASE_ERR_NULL_PTR
    assert out_abs_start == [111]
    assert out_abs_end == [222]
    assert out_span_start == [333]
    assert out_span_end == [444]


def test_randomized_owner_and_window_validation() -> None:
    rng = random.Random(1139)

    starts = [0x6000, 0x6100, 0x6200]
    ends = [0x6080, 0x6180, 0x6280]
    sorted_idx = [0, 1, 2]
    sorted_pos = [0, 1, 2]

    for _ in range(800):
        owner = rng.randrange(0, 3)
        base = starts[owner]
        limit = ends[owner]

        span_len = rng.randrange(0, 0x50)
        start = rng.randrange(base, limit + 0x20)

        out_abs_start = [0xAA]
        out_abs_end = [0xBB]
        out_span_start = [0xCC]
        out_span_end = [0xDD]

        err = tensor_payload_read_window_checked(
            owner_tensor_index=owner,
            payload_abs_start=start,
            payload_nbytes=span_len,
            tensor_abs_starts=starts,
            tensor_abs_ends=ends,
            sorted_tensor_indices=sorted_idx,
            sorted_position_by_tensor=sorted_pos,
            out_abs_start=out_abs_start,
            out_abs_end=out_abs_end,
            out_span_start_in_tensor=out_span_start,
            out_span_end_in_tensor=out_span_end,
        )

        expected_end = start + span_len
        in_owner = start >= base and expected_end <= limit
        start_in_any = any(start >= s and start < e for s, e in zip(starts, ends))

        if span_len == 0:
            if in_owner:
                assert err == GGUF_TDBASE_OK
                assert out_abs_start[0] == start
                assert out_abs_end[0] == start
                assert out_span_start[0] == start - base
                assert out_span_end[0] == start - base
            else:
                assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
                assert out_abs_start == [0]
                assert out_abs_end == [0]
                assert out_span_start == [0]
                assert out_span_end == [0]
            continue

        if not start_in_any:
            assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
            assert out_abs_start == [0]
            assert out_abs_end == [0]
            assert out_span_start == [0]
            assert out_span_end == [0]
            continue

        actual_owner = 0
        for idx, (s, e) in enumerate(zip(starts, ends)):
            if start >= s and start < e:
                actual_owner = idx
                break

        if actual_owner != owner or not in_owner:
            assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
            assert out_abs_start == [0]
            assert out_abs_end == [0]
            assert out_span_start == [0]
            assert out_span_end == [0]
        else:
            assert err == GGUF_TDBASE_OK
            assert out_abs_start[0] == start
            assert out_abs_end[0] == expected_end
            assert out_span_start[0] == start - base
            assert out_span_end[0] == expected_end - base
