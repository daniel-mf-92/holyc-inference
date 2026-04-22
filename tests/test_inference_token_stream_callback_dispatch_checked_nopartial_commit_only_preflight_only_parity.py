#!/usr/bin/env python3
"""Parity harness for IQ-1173 callback-dispatch parity gate."""

from __future__ import annotations

import random
from pathlib import Path

SAMPLING_Q16_OK = 0
SAMPLING_Q16_ERR_NULL_PTR = 1
SAMPLING_Q16_ERR_BAD_PARAM = 2

INFERENCE_TOKEN_STREAM_CALLBACK_ABI_VERSION = 1
INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS = 5
INFERENCE_TOKEN_STREAM_CALLBACK_FLAG_NONE = 0


def token_stream_callback_dispatch_checked_nopartial_model(
    *,
    token_callback,
    abi_tuple_buffer: list[int] | None,
    abi_tuple_buffer_capacity: int,
    token_index: int,
    token_id: int,
    token_prob_q16: int,
    callback_flags: int,
    user_ctx: int,
    out_dispatch_count: list[int] | None,
    out_callback_status: list[int] | None,
) -> int:
    if token_callback is None or abi_tuple_buffer is None:
        return SAMPLING_Q16_ERR_NULL_PTR
    if out_dispatch_count is None or out_callback_status is None:
        return SAMPLING_Q16_ERR_NULL_PTR
    if out_dispatch_count is out_callback_status:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if abi_tuple_buffer_capacity < INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if token_index < 0 or token_id < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if callback_flags != INFERENCE_TOKEN_STREAM_CALLBACK_FLAG_NONE:
        return SAMPLING_Q16_ERR_BAD_PARAM

    staged = [
        INFERENCE_TOKEN_STREAM_CALLBACK_ABI_VERSION,
        callback_flags,
        token_index,
        token_id,
        token_prob_q16,
    ]

    callback_status = token_callback(staged, INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS, user_ctx)
    if callback_status != SAMPLING_Q16_OK:
        return SAMPLING_Q16_ERR_BAD_PARAM

    canonical = [
        INFERENCE_TOKEN_STREAM_CALLBACK_ABI_VERSION,
        INFERENCE_TOKEN_STREAM_CALLBACK_FLAG_NONE,
        token_index,
        token_id,
        token_prob_q16,
    ]
    if staged != canonical:
        return SAMPLING_Q16_ERR_BAD_PARAM

    for lane in range(INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS):
        abi_tuple_buffer[lane] = staged[lane]

    out_dispatch_count[0] = 1
    out_callback_status[0] = SAMPLING_Q16_OK
    return SAMPLING_Q16_OK


def token_stream_callback_dispatch_checked_nopartial_commit_only_preflight_only_model(
    *,
    token_callback,
    abi_tuple_buffer: list[int] | None,
    abi_tuple_buffer_capacity: int,
    token_index: int,
    token_id: int,
    token_prob_q16: int,
    callback_flags: int,
    user_ctx: int,
    out_dispatch_count: list[int] | None,
    out_callback_status: list[int] | None,
) -> int:
    if token_callback is None or abi_tuple_buffer is None:
        return SAMPLING_Q16_ERR_NULL_PTR
    if out_dispatch_count is None or out_callback_status is None:
        return SAMPLING_Q16_ERR_NULL_PTR
    if out_dispatch_count is out_callback_status:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if abi_tuple_buffer_capacity < INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if token_index < 0 or token_id < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if callback_flags != INFERENCE_TOKEN_STREAM_CALLBACK_FLAG_NONE:
        return SAMPLING_Q16_ERR_BAD_PARAM

    staged_tuple = [0] * INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS
    staged_dispatch = [0]
    staged_callback_status = [0]

    status = token_stream_callback_dispatch_checked_nopartial_model(
        token_callback=token_callback,
        abi_tuple_buffer=staged_tuple,
        abi_tuple_buffer_capacity=INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS,
        token_index=token_index,
        token_id=token_id,
        token_prob_q16=token_prob_q16,
        callback_flags=callback_flags,
        user_ctx=user_ctx,
        out_dispatch_count=staged_dispatch,
        out_callback_status=staged_callback_status,
    )
    if status != SAMPLING_Q16_OK:
        return status

    canonical_tuple = [
        INFERENCE_TOKEN_STREAM_CALLBACK_ABI_VERSION,
        INFERENCE_TOKEN_STREAM_CALLBACK_FLAG_NONE,
        token_index,
        token_id,
        token_prob_q16,
    ]

    if staged_tuple != canonical_tuple:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if staged_dispatch[0] != 1:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if staged_callback_status[0] != SAMPLING_Q16_OK:
        return SAMPLING_Q16_ERR_BAD_PARAM

    for lane in range(INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS):
        abi_tuple_buffer[lane] = staged_tuple[lane]

    out_dispatch_count[0] = staged_dispatch[0]
    out_callback_status[0] = staged_callback_status[0]
    return SAMPLING_Q16_OK


def token_stream_callback_dispatch_checked_nopartial_commit_only_preflight_only_parity_model(
    *,
    token_callback,
    abi_tuple_buffer: list[int] | None,
    abi_tuple_buffer_capacity: int,
    token_index: int,
    token_id: int,
    token_prob_q16: int,
    callback_flags: int,
    user_ctx: int,
    out_dispatch_count: list[int] | None,
    out_callback_status: list[int] | None,
    preflight_model=token_stream_callback_dispatch_checked_nopartial_commit_only_preflight_only_model,
    canonical_model=token_stream_callback_dispatch_checked_nopartial_model,
) -> int:
    if token_callback is None or abi_tuple_buffer is None:
        return SAMPLING_Q16_ERR_NULL_PTR
    if out_dispatch_count is None or out_callback_status is None:
        return SAMPLING_Q16_ERR_NULL_PTR
    if out_dispatch_count is out_callback_status:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if abi_tuple_buffer_capacity < INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if token_index < 0 or token_id < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if callback_flags != INFERENCE_TOKEN_STREAM_CALLBACK_FLAG_NONE:
        return SAMPLING_Q16_ERR_BAD_PARAM

    snap_tuple = list(abi_tuple_buffer)
    snap_dispatch = out_dispatch_count[0]
    snap_status = out_callback_status[0]

    staged_tuple = [0] * INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS
    staged_dispatch = [0]
    staged_callback_status = [0]

    canonical_tuple = [0] * INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS
    canonical_dispatch = [0]
    canonical_callback_status = [0]

    status = preflight_model(
        token_callback=token_callback,
        abi_tuple_buffer=staged_tuple,
        abi_tuple_buffer_capacity=INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS,
        token_index=token_index,
        token_id=token_id,
        token_prob_q16=token_prob_q16,
        callback_flags=callback_flags,
        user_ctx=user_ctx,
        out_dispatch_count=staged_dispatch,
        out_callback_status=staged_callback_status,
    )
    if status != SAMPLING_Q16_OK:
        assert abi_tuple_buffer == snap_tuple
        assert out_dispatch_count[0] == snap_dispatch
        assert out_callback_status[0] == snap_status
        return status

    status = canonical_model(
        token_callback=token_callback,
        abi_tuple_buffer=canonical_tuple,
        abi_tuple_buffer_capacity=INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS,
        token_index=token_index,
        token_id=token_id,
        token_prob_q16=token_prob_q16,
        callback_flags=callback_flags,
        user_ctx=user_ctx,
        out_dispatch_count=canonical_dispatch,
        out_callback_status=canonical_callback_status,
    )
    if status != SAMPLING_Q16_OK:
        assert abi_tuple_buffer == snap_tuple
        assert out_dispatch_count[0] == snap_dispatch
        assert out_callback_status[0] == snap_status
        return status

    if staged_tuple != canonical_tuple:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if staged_dispatch[0] != canonical_dispatch[0]:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if staged_callback_status[0] != canonical_callback_status[0]:
        return SAMPLING_Q16_ERR_BAD_PARAM

    for lane in range(INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS):
        abi_tuple_buffer[lane] = staged_tuple[lane]

    out_dispatch_count[0] = staged_dispatch[0]
    out_callback_status[0] = staged_callback_status[0]
    return SAMPLING_Q16_OK


def _extract_function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 1
    index = brace + 1
    while depth:
        ch = source[index]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        index += 1
    return source[brace + 1 : index - 1]


def test_source_contains_parity_contract() -> None:
    source = Path("src/model/inference.HC").read_text(encoding="utf-8")
    signature = "I32 InferenceTokenStreamCallbackDispatchCheckedNoPartialCommitOnlyPreflightOnlyParity("
    assert signature in source

    body = _extract_function_body(source, signature)
    assert "status = InferenceTokenStreamCallbackDispatchCheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "status = InferenceTokenStreamCallbackDispatchCheckedNoPartial(" in body
    assert "if (snapshot_token_callback != token_callback ||" in body
    assert "if (staged_dispatch_count != canonical_dispatch_count)" in body
    assert "if (staged_callback_status != canonical_callback_status)" in body
    assert "*out_dispatch_count = staged_dispatch_count;" in body
    assert "*out_callback_status = staged_callback_status;" in body


def test_fail_closed_vectors_keep_outputs_unchanged() -> None:
    tuple_buf = [41, 42, 43, 44, 45]
    dispatch_out = [71]
    callback_out = [72]

    def ok_callback(_tuple: list[int], _cells: int, _ctx: int) -> int:
        return SAMPLING_Q16_OK

    vectors = [
        dict(token_callback=None),
        dict(abi_tuple_buffer=None),
        dict(out_dispatch_count=None),
        dict(out_callback_status=None),
        dict(abi_tuple_buffer_capacity=4),
        dict(token_index=-1),
        dict(token_id=-2),
        dict(callback_flags=1),
    ]

    for override in vectors:
        local_tuple = list(tuple_buf)
        local_dispatch = list(dispatch_out)
        local_status = list(callback_out)

        kwargs = dict(
            token_callback=ok_callback,
            abi_tuple_buffer=local_tuple,
            abi_tuple_buffer_capacity=5,
            token_index=9,
            token_id=12,
            token_prob_q16=33,
            callback_flags=0,
            user_ctx=777,
            out_dispatch_count=local_dispatch,
            out_callback_status=local_status,
        )
        kwargs.update(override)

        status = token_stream_callback_dispatch_checked_nopartial_commit_only_preflight_only_parity_model(**kwargs)
        assert status in {SAMPLING_Q16_ERR_NULL_PTR, SAMPLING_Q16_ERR_BAD_PARAM}
        assert local_tuple == tuple_buf
        assert local_dispatch == dispatch_out
        assert local_status == callback_out


def test_rejects_tuple_or_diagnostics_parity_mismatch_without_publish() -> None:
    tuple_buf = [3, 3, 3, 3, 3]
    dispatch_out = [8]
    callback_out = [9]

    def ok_callback(_tuple: list[int], _cells: int, _ctx: int) -> int:
        return SAMPLING_Q16_OK

    def bad_preflight(**kwargs) -> int:
        status = token_stream_callback_dispatch_checked_nopartial_commit_only_preflight_only_model(**kwargs)
        if status == SAMPLING_Q16_OK:
            kwargs["out_dispatch_count"][0] += 1
        return status

    status = token_stream_callback_dispatch_checked_nopartial_commit_only_preflight_only_parity_model(
        token_callback=ok_callback,
        abi_tuple_buffer=tuple_buf,
        abi_tuple_buffer_capacity=5,
        token_index=44,
        token_id=55,
        token_prob_q16=66,
        callback_flags=0,
        user_ctx=11,
        out_dispatch_count=dispatch_out,
        out_callback_status=callback_out,
        preflight_model=bad_preflight,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert tuple_buf == [3, 3, 3, 3, 3]
    assert dispatch_out == [8]
    assert callback_out == [9]


def test_success_publishes_exact_parity_tuple() -> None:
    tuple_buf = [-1, -1, -1, -1, -1]
    dispatch_out = [-1]
    callback_out = [-1]

    def strict_callback(staged: list[int], cells: int, user_ctx: int) -> int:
        assert cells == INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS
        assert user_ctx == 31337
        assert staged[0] == INFERENCE_TOKEN_STREAM_CALLBACK_ABI_VERSION
        assert staged[1] == INFERENCE_TOKEN_STREAM_CALLBACK_FLAG_NONE
        return SAMPLING_Q16_OK

    status = token_stream_callback_dispatch_checked_nopartial_commit_only_preflight_only_parity_model(
        token_callback=strict_callback,
        abi_tuple_buffer=tuple_buf,
        abi_tuple_buffer_capacity=5,
        token_index=101,
        token_id=202,
        token_prob_q16=303,
        callback_flags=0,
        user_ctx=31337,
        out_dispatch_count=dispatch_out,
        out_callback_status=callback_out,
    )
    assert status == SAMPLING_Q16_OK
    assert tuple_buf == [1, 0, 101, 202, 303]
    assert dispatch_out == [1]
    assert callback_out == [0]


def test_randomized_success_vectors_publish_exact_parity_outputs() -> None:
    rng = random.Random(1173)

    def callback(staged: list[int], cells: int, _ctx: int) -> int:
        if cells != INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS:
            return 77
        return SAMPLING_Q16_OK

    for _ in range(300):
        token_index = rng.randint(0, 200_000)
        token_id = rng.randint(0, 200_000)
        token_prob_q16 = rng.randint(-(1 << 21), 1 << 21)
        user_ctx = rng.randint(-(1 << 31), 1 << 31)

        tuple_buf = [rng.randint(-2000, 2000) for _ in range(5)]
        dispatch_out = [rng.randint(-50, 50)]
        callback_out = [rng.randint(-50, 50)]

        status = token_stream_callback_dispatch_checked_nopartial_commit_only_preflight_only_parity_model(
            token_callback=callback,
            abi_tuple_buffer=tuple_buf,
            abi_tuple_buffer_capacity=5,
            token_index=token_index,
            token_id=token_id,
            token_prob_q16=token_prob_q16,
            callback_flags=0,
            user_ctx=user_ctx,
            out_dispatch_count=dispatch_out,
            out_callback_status=callback_out,
        )
        assert status == SAMPLING_Q16_OK
        assert tuple_buf == [1, 0, token_index, token_id, token_prob_q16]
        assert dispatch_out == [1]
        assert callback_out == [0]


if __name__ == "__main__":
    test_source_contains_parity_contract()
    test_fail_closed_vectors_keep_outputs_unchanged()
    test_rejects_tuple_or_diagnostics_parity_mismatch_without_publish()
    test_success_publishes_exact_parity_tuple()
    test_randomized_success_vectors_publish_exact_parity_outputs()
    print("ok")
