#!/usr/bin/env python3
"""Parity harness for IQ-1170 token stream callback dispatcher."""

from __future__ import annotations

from pathlib import Path
import random

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

    snap_tuple = list(abi_tuple_buffer)
    snap_dispatch = out_dispatch_count[0]
    snap_callback_status = out_callback_status[0]

    staged = [
        INFERENCE_TOKEN_STREAM_CALLBACK_ABI_VERSION,
        callback_flags,
        token_index,
        token_id,
        token_prob_q16,
    ]

    callback_status = token_callback(staged, INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS, user_ctx)
    if callback_status != SAMPLING_Q16_OK:
        assert abi_tuple_buffer == snap_tuple
        assert out_dispatch_count[0] == snap_dispatch
        assert out_callback_status[0] == snap_callback_status
        return SAMPLING_Q16_ERR_BAD_PARAM

    if staged[0] != INFERENCE_TOKEN_STREAM_CALLBACK_ABI_VERSION:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if staged[1] != INFERENCE_TOKEN_STREAM_CALLBACK_FLAG_NONE:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if staged[2] != token_index:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if staged[3] != token_id:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if staged[4] != token_prob_q16:
        return SAMPLING_Q16_ERR_BAD_PARAM

    for lane in range(INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS):
        abi_tuple_buffer[lane] = staged[lane]

    out_dispatch_count[0] = 1
    out_callback_status[0] = callback_status
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


def test_source_contains_callback_dispatch_contract() -> None:
    source = Path("src/model/inference.HC").read_text(encoding="utf-8")
    signature = "I32 InferenceTokenStreamCallbackDispatchCheckedNoPartial("
    assert signature in source

    body = _extract_function_body(source, signature)
    assert "INFERENCE_TOKEN_STREAM_CALLBACK_ABI_VERSION" in source
    assert "INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS" in source
    assert "if (!token_callback || !abi_tuple_buffer ||" in body
    assert "if (callback_flags != INFERENCE_TOKEN_STREAM_CALLBACK_FLAG_NONE)" in body
    assert "callback_status = token_callback(staged_tuple," in body
    assert "if (staged_tuple[0] != INFERENCE_TOKEN_STREAM_CALLBACK_ABI_VERSION)" in body
    assert "*out_dispatch_count = staged_dispatch_count;" in body
    assert "*out_callback_status = staged_callback_status;" in body


def test_invalid_inputs_fail_closed_without_publish() -> None:
    tuple_buf = [99, 98, 97, 96, 95]
    dispatch_count = [77]
    callback_status_out = [66]

    def ok_callback(_tuple: list[int], _cells: int, _ctx: int) -> int:
        return SAMPLING_Q16_OK

    vectors = [
        dict(token_callback=None),
        dict(abi_tuple_buffer=None),
        dict(abi_tuple_buffer_capacity=4),
        dict(token_index=-1),
        dict(token_id=-1),
        dict(callback_flags=1),
    ]

    for override in vectors:
        local_tuple = list(tuple_buf)
        local_dispatch = list(dispatch_count)
        local_status = list(callback_status_out)

        kwargs = dict(
            token_callback=ok_callback,
            abi_tuple_buffer=local_tuple,
            abi_tuple_buffer_capacity=5,
            token_index=2,
            token_id=123,
            token_prob_q16=456,
            callback_flags=0,
            user_ctx=17,
            out_dispatch_count=local_dispatch,
            out_callback_status=local_status,
        )
        kwargs.update(override)

        status = token_stream_callback_dispatch_checked_nopartial_model(**kwargs)
        assert status in {SAMPLING_Q16_ERR_NULL_PTR, SAMPLING_Q16_ERR_BAD_PARAM}
        assert local_tuple == tuple_buf
        assert local_dispatch == dispatch_count
        assert local_status == callback_status_out


def test_callback_failure_and_tuple_mutation_are_rejected() -> None:
    tuple_buf = [1, 2, 3, 4, 5]
    dispatch_count = [10]
    callback_status_out = [20]

    def fail_callback(_tuple: list[int], _cells: int, _ctx: int) -> int:
        return 91

    status = token_stream_callback_dispatch_checked_nopartial_model(
        token_callback=fail_callback,
        abi_tuple_buffer=tuple_buf,
        abi_tuple_buffer_capacity=5,
        token_index=11,
        token_id=22,
        token_prob_q16=33,
        callback_flags=0,
        user_ctx=44,
        out_dispatch_count=dispatch_count,
        out_callback_status=callback_status_out,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert tuple_buf == [1, 2, 3, 4, 5]
    assert dispatch_count == [10]
    assert callback_status_out == [20]

    def tamper_callback(staged: list[int], _cells: int, _ctx: int) -> int:
        staged[0] = 999
        return SAMPLING_Q16_OK

    status = token_stream_callback_dispatch_checked_nopartial_model(
        token_callback=tamper_callback,
        abi_tuple_buffer=tuple_buf,
        abi_tuple_buffer_capacity=5,
        token_index=11,
        token_id=22,
        token_prob_q16=33,
        callback_flags=0,
        user_ctx=44,
        out_dispatch_count=dispatch_count,
        out_callback_status=callback_status_out,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert tuple_buf == [1, 2, 3, 4, 5]
    assert dispatch_count == [10]
    assert callback_status_out == [20]


def test_success_dispatch_publishes_abi_tuple_atomically() -> None:
    tuple_buf = [0, 0, 0, 0, 0]
    dispatch_count = [-1]
    callback_status_out = [-1]

    def strict_callback(staged: list[int], cells: int, user_ctx: int) -> int:
        assert cells == INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS
        assert user_ctx == 919
        assert staged[0] == INFERENCE_TOKEN_STREAM_CALLBACK_ABI_VERSION
        assert staged[1] == 0
        return SAMPLING_Q16_OK

    status = token_stream_callback_dispatch_checked_nopartial_model(
        token_callback=strict_callback,
        abi_tuple_buffer=tuple_buf,
        abi_tuple_buffer_capacity=5,
        token_index=7,
        token_id=32001,
        token_prob_q16=12345,
        callback_flags=0,
        user_ctx=919,
        out_dispatch_count=dispatch_count,
        out_callback_status=callback_status_out,
    )
    assert status == SAMPLING_Q16_OK
    assert tuple_buf == [1, 0, 7, 32001, 12345]
    assert dispatch_count == [1]
    assert callback_status_out == [0]


def test_randomized_success_vectors_keep_callback_tuple_exact() -> None:
    rng = random.Random(1170)

    def callback(staged: list[int], cells: int, _ctx: int) -> int:
        if cells != 5:
            return 55
        return SAMPLING_Q16_OK

    for _ in range(200):
        token_index = rng.randint(0, 100_000)
        token_id = rng.randint(0, 128_000)
        token_prob_q16 = rng.randint(-(1 << 20), 1 << 20)
        user_ctx = rng.randint(-(1 << 31), 1 << 31)

        tuple_buf = [rng.randint(-99, 99) for _ in range(5)]
        dispatch_count = [rng.randint(-100, 100)]
        callback_status_out = [rng.randint(-100, 100)]

        status = token_stream_callback_dispatch_checked_nopartial_model(
            token_callback=callback,
            abi_tuple_buffer=tuple_buf,
            abi_tuple_buffer_capacity=5,
            token_index=token_index,
            token_id=token_id,
            token_prob_q16=token_prob_q16,
            callback_flags=0,
            user_ctx=user_ctx,
            out_dispatch_count=dispatch_count,
            out_callback_status=callback_status_out,
        )
        assert status == SAMPLING_Q16_OK
        assert tuple_buf == [1, 0, token_index, token_id, token_prob_q16]
        assert dispatch_count == [1]
        assert callback_status_out == [0]


if __name__ == "__main__":
    test_source_contains_callback_dispatch_contract()
    test_invalid_inputs_fail_closed_without_publish()
    test_callback_failure_and_tuple_mutation_are_rejected()
    test_success_dispatch_publishes_abi_tuple_atomically()
    test_randomized_success_vectors_keep_callback_tuple_exact()
    print("ok")
