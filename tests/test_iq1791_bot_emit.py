#!/usr/bin/env python3
"""IQ-1791 harness for Book-of-Truth token event emission gate."""

from __future__ import annotations

from pathlib import Path

SAMPLING_Q16_OK = 0
SAMPLING_Q16_ERR_NULL_PTR = 1
SAMPLING_Q16_ERR_BAD_PARAM = 2

INFERENCE_BOT_EVENT_TUPLE_CELLS = 6
INFERENCE_BOT_PROFILE_SECURE = 1
INFERENCE_BOT_PROFILE_DEV = 2
INFERENCE_BOT_STATUS_BLOCKED = 0
INFERENCE_BOT_STATUS_EMITTED = 1
INFERENCE_BOT_DIGEST_OFFSET = 1469598103934665603
INFERENCE_BOT_DIGEST_PRIME = 1099511628211
I64_MASK = (1 << 64) - 1


def _i64_wrap(value: int) -> int:
    value &= I64_MASK
    if value >= (1 << 63):
        value -= 1 << 64
    return value


def _digest(event: list[int], status: int, count: int) -> int:
    acc = INFERENCE_BOT_DIGEST_OFFSET
    for lane in event:
        acc = _i64_wrap(acc ^ lane)
        acc = _i64_wrap(acc * INFERENCE_BOT_DIGEST_PRIME)
    acc = _i64_wrap(acc ^ status)
    acc = _i64_wrap(acc * INFERENCE_BOT_DIGEST_PRIME)
    acc = _i64_wrap(acc ^ count)
    acc = _i64_wrap(acc * INFERENCE_BOT_DIGEST_PRIME)
    return acc


def bot_emit_model(
    *,
    event_buffer: list[int] | None,
    event_buffer_capacity: int,
    session_id: int,
    step_index: int,
    token_id: int,
    logit_q16: int,
    policy_digest_q64: int,
    expected_policy_digest_q64: int,
    profile_mode: int,
    out_event_status: list[int] | None,
    out_event_count: list[int] | None,
    out_event_digest_q64: list[int] | None,
) -> int:
    if event_buffer is None or out_event_status is None or out_event_count is None or out_event_digest_q64 is None:
        return SAMPLING_Q16_ERR_NULL_PTR

    if out_event_status is out_event_count or out_event_status is out_event_digest_q64 or out_event_count is out_event_digest_q64:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if event_buffer_capacity < INFERENCE_BOT_EVENT_TUPLE_CELLS:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if session_id < 0 or step_index < 0 or token_id < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if profile_mode not in (INFERENCE_BOT_PROFILE_SECURE, INFERENCE_BOT_PROFILE_DEV):
        return SAMPLING_Q16_ERR_BAD_PARAM

    snap_buffer = list(event_buffer)
    snap_status = out_event_status[0]
    snap_count = out_event_count[0]
    snap_digest = out_event_digest_q64[0]

    staged_event = [session_id, step_index, token_id, logit_q16, policy_digest_q64, profile_mode]
    if profile_mode == INFERENCE_BOT_PROFILE_SECURE and policy_digest_q64 == expected_policy_digest_q64:
        staged_status = INFERENCE_BOT_STATUS_EMITTED
        staged_count = 1
    else:
        staged_status = INFERENCE_BOT_STATUS_BLOCKED
        staged_count = 0

    if staged_status not in (INFERENCE_BOT_STATUS_BLOCKED, INFERENCE_BOT_STATUS_EMITTED):
        assert event_buffer == snap_buffer
        assert out_event_status[0] == snap_status
        assert out_event_count[0] == snap_count
        assert out_event_digest_q64[0] == snap_digest
        return SAMPLING_Q16_ERR_BAD_PARAM

    staged_digest = _digest(staged_event, staged_status, staged_count)

    if staged_status == INFERENCE_BOT_STATUS_EMITTED:
        for lane in range(INFERENCE_BOT_EVENT_TUPLE_CELLS):
            event_buffer[lane] = staged_event[lane]

    out_event_status[0] = staged_status
    out_event_count[0] = staged_count
    out_event_digest_q64[0] = staged_digest
    return SAMPLING_Q16_OK


def bot_emit_commit_only_model(
    *,
    event_buffer: list[int] | None,
    event_buffer_capacity: int,
    session_id: int,
    step_index: int,
    token_id: int,
    logit_q16: int,
    policy_digest_q64: int,
    expected_policy_digest_q64: int,
    profile_mode: int,
    out_event_status: list[int] | None,
    out_event_count: list[int] | None,
    out_event_digest_q64: list[int] | None,
) -> int:
    if event_buffer is None or out_event_status is None or out_event_count is None or out_event_digest_q64 is None:
        return SAMPLING_Q16_ERR_NULL_PTR

    if out_event_status is out_event_count or out_event_status is out_event_digest_q64 or out_event_count is out_event_digest_q64:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if event_buffer_capacity < INFERENCE_BOT_EVENT_TUPLE_CELLS:
        return SAMPLING_Q16_ERR_BAD_PARAM

    primary_event = [0] * INFERENCE_BOT_EVENT_TUPLE_CELLS
    primary_status = [0]
    primary_count = [0]
    primary_digest = [0]
    replay_event = [0] * INFERENCE_BOT_EVENT_TUPLE_CELLS
    replay_status = [0]
    replay_count = [0]
    replay_digest = [0]

    status_primary = bot_emit_model(
        event_buffer=primary_event,
        event_buffer_capacity=INFERENCE_BOT_EVENT_TUPLE_CELLS,
        session_id=session_id,
        step_index=step_index,
        token_id=token_id,
        logit_q16=logit_q16,
        policy_digest_q64=policy_digest_q64,
        expected_policy_digest_q64=expected_policy_digest_q64,
        profile_mode=profile_mode,
        out_event_status=primary_status,
        out_event_count=primary_count,
        out_event_digest_q64=primary_digest,
    )
    if status_primary != SAMPLING_Q16_OK:
        return status_primary

    status_replay = bot_emit_model(
        event_buffer=replay_event,
        event_buffer_capacity=INFERENCE_BOT_EVENT_TUPLE_CELLS,
        session_id=session_id,
        step_index=step_index,
        token_id=token_id,
        logit_q16=logit_q16,
        policy_digest_q64=policy_digest_q64,
        expected_policy_digest_q64=expected_policy_digest_q64,
        profile_mode=profile_mode,
        out_event_status=replay_status,
        out_event_count=replay_count,
        out_event_digest_q64=replay_digest,
    )
    if status_replay != SAMPLING_Q16_OK:
        return status_replay

    if (
        primary_status[0] != replay_status[0]
        or primary_count[0] != replay_count[0]
        or primary_digest[0] != replay_digest[0]
        or primary_event != replay_event
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if primary_status[0] == INFERENCE_BOT_STATUS_EMITTED:
        for lane in range(INFERENCE_BOT_EVENT_TUPLE_CELLS):
            event_buffer[lane] = primary_event[lane]

    out_event_status[0] = primary_status[0]
    out_event_count[0] = primary_count[0]
    out_event_digest_q64[0] = primary_digest[0]
    return SAMPLING_Q16_OK


def _extract_body(source: str, signature: str) -> str:
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


def test_source_contains_iq_1791_contract() -> None:
    source = Path("src/model/inference.HC").read_text(encoding="utf-8")
    signature = "I32 InferenceBookOfTruthTokenEventEmitChecked("
    assert signature in source
    body = _extract_body(source, signature)
    assert "staged_event[0] = session_id;" in body
    assert "if (profile_mode == INFERENCE_BOT_PROFILE_SECURE &&" in body
    assert "staged_event_status = INFERENCE_BOT_STATUS_BLOCKED;" in body
    assert "if (snapshot_event_buffer != event_buffer ||" in body
    assert "*out_event_digest_q64 = staged_event_digest_q64;" in body


def test_source_contains_iq_1792_contract() -> None:
    source = Path("src/model/inference.HC").read_text(encoding="utf-8")
    assert "#define InferenceBookOfTruthTokenEventEmitCheckedCommitOnly BotTokenEmitCommitOnly" in source
    signature = "I32 BotTokenEmitCommitOnly("
    assert signature in source
    body = _extract_body(source, signature)
    assert "status_primary = InferenceBookOfTruthTokenEventEmitChecked(" in body
    assert "status_replay = InferenceBookOfTruthTokenEventEmitChecked(" in body
    assert "if (staged_event_status_primary != staged_event_status_replay ||" in body
    assert "*out_event_digest_q64 = staged_event_digest_primary;" in body


def test_secure_local_success_emits_event() -> None:
    event_buffer = [999] * INFERENCE_BOT_EVENT_TUPLE_CELLS
    out_status = [77]
    out_count = [66]
    out_digest = [55]

    status = bot_emit_model(
        event_buffer=event_buffer,
        event_buffer_capacity=INFERENCE_BOT_EVENT_TUPLE_CELLS,
        session_id=41,
        step_index=2,
        token_id=15496,
        logit_q16=123456,
        policy_digest_q64=0x1234,
        expected_policy_digest_q64=0x1234,
        profile_mode=INFERENCE_BOT_PROFILE_SECURE,
        out_event_status=out_status,
        out_event_count=out_count,
        out_event_digest_q64=out_digest,
    )

    assert status == SAMPLING_Q16_OK
    assert event_buffer == [41, 2, 15496, 123456, 0x1234, INFERENCE_BOT_PROFILE_SECURE]
    assert out_status == [INFERENCE_BOT_STATUS_EMITTED]
    assert out_count == [1]
    assert out_digest[0] == _digest(event_buffer, INFERENCE_BOT_STATUS_EMITTED, 1)


def test_digest_mismatch_blocks_and_preserves_event_buffer() -> None:
    event_buffer = [8, 7, 6, 5, 4, 3]
    out_status = [9]
    out_count = [9]
    out_digest = [9]

    status = bot_emit_model(
        event_buffer=event_buffer,
        event_buffer_capacity=INFERENCE_BOT_EVENT_TUPLE_CELLS,
        session_id=7,
        step_index=8,
        token_id=9,
        logit_q16=10,
        policy_digest_q64=123,
        expected_policy_digest_q64=999,
        profile_mode=INFERENCE_BOT_PROFILE_SECURE,
        out_event_status=out_status,
        out_event_count=out_count,
        out_event_digest_q64=out_digest,
    )

    assert status == SAMPLING_Q16_OK
    assert event_buffer == [8, 7, 6, 5, 4, 3]
    assert out_status == [INFERENCE_BOT_STATUS_BLOCKED]
    assert out_count == [0]


def test_commit_only_secure_local_success_emits_event() -> None:
    event_buffer = [111] * INFERENCE_BOT_EVENT_TUPLE_CELLS
    out_status = [0]
    out_count = [0]
    out_digest = [0]

    status = bot_emit_commit_only_model(
        event_buffer=event_buffer,
        event_buffer_capacity=INFERENCE_BOT_EVENT_TUPLE_CELLS,
        session_id=9,
        step_index=10,
        token_id=11,
        logit_q16=12,
        policy_digest_q64=0x55AA,
        expected_policy_digest_q64=0x55AA,
        profile_mode=INFERENCE_BOT_PROFILE_SECURE,
        out_event_status=out_status,
        out_event_count=out_count,
        out_event_digest_q64=out_digest,
    )

    assert status == SAMPLING_Q16_OK
    assert event_buffer == [9, 10, 11, 12, 0x55AA, INFERENCE_BOT_PROFILE_SECURE]
    assert out_status == [INFERENCE_BOT_STATUS_EMITTED]
    assert out_count == [1]
    assert out_digest[0] == _digest(event_buffer, INFERENCE_BOT_STATUS_EMITTED, 1)


def test_commit_only_digest_mismatch_blocks_and_preserves_event_buffer() -> None:
    event_buffer = [4, 5, 6, 7, 8, 9]
    out_status = [3]
    out_count = [2]
    out_digest = [1]

    status = bot_emit_commit_only_model(
        event_buffer=event_buffer,
        event_buffer_capacity=INFERENCE_BOT_EVENT_TUPLE_CELLS,
        session_id=1,
        step_index=2,
        token_id=3,
        logit_q16=4,
        policy_digest_q64=999,
        expected_policy_digest_q64=111,
        profile_mode=INFERENCE_BOT_PROFILE_SECURE,
        out_event_status=out_status,
        out_event_count=out_count,
        out_event_digest_q64=out_digest,
    )

    assert status == SAMPLING_Q16_OK
    assert event_buffer == [4, 5, 6, 7, 8, 9]
    assert out_status == [INFERENCE_BOT_STATUS_BLOCKED]
    assert out_count == [0]


def test_capacity_underflow_is_fail_closed() -> None:
    event_buffer = [1, 2, 3, 4, 5, 6]
    out_status = [1]
    out_count = [2]
    out_digest = [3]

    status = bot_emit_model(
        event_buffer=event_buffer,
        event_buffer_capacity=INFERENCE_BOT_EVENT_TUPLE_CELLS - 1,
        session_id=1,
        step_index=1,
        token_id=1,
        logit_q16=1,
        policy_digest_q64=1,
        expected_policy_digest_q64=1,
        profile_mode=INFERENCE_BOT_PROFILE_SECURE,
        out_event_status=out_status,
        out_event_count=out_count,
        out_event_digest_q64=out_digest,
    )

    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert event_buffer == [1, 2, 3, 4, 5, 6]
    assert out_status == [1]
    assert out_count == [2]
    assert out_digest == [3]


def test_deterministic_replay_digest() -> None:
    args = dict(
        event_buffer_capacity=INFERENCE_BOT_EVENT_TUPLE_CELLS,
        session_id=123,
        step_index=9,
        token_id=2048,
        logit_q16=-321,
        policy_digest_q64=0xABCD,
        expected_policy_digest_q64=0xABCD,
        profile_mode=INFERENCE_BOT_PROFILE_SECURE,
    )

    buffer_a = [0] * INFERENCE_BOT_EVENT_TUPLE_CELLS
    buffer_b = [0] * INFERENCE_BOT_EVENT_TUPLE_CELLS
    status_a = [0]
    status_b = [0]
    count_a = [0]
    count_b = [0]
    digest_a = [0]
    digest_b = [0]

    result_a = bot_emit_model(
        event_buffer=buffer_a,
        out_event_status=status_a,
        out_event_count=count_a,
        out_event_digest_q64=digest_a,
        **args,
    )
    result_b = bot_emit_model(
        event_buffer=buffer_b,
        out_event_status=status_b,
        out_event_count=count_b,
        out_event_digest_q64=digest_b,
        **args,
    )

    assert result_a == SAMPLING_Q16_OK
    assert result_b == SAMPLING_Q16_OK
    assert buffer_a == buffer_b
    assert status_a == status_b
    assert count_a == count_b
    assert digest_a == digest_b


if __name__ == "__main__":
    test_source_contains_iq_1791_contract()
    test_source_contains_iq_1792_contract()
    test_secure_local_success_emits_event()
    test_digest_mismatch_blocks_and_preserves_event_buffer()
    test_commit_only_secure_local_success_emits_event()
    test_commit_only_digest_mismatch_blocks_and_preserves_event_buffer()
    test_capacity_underflow_is_fail_closed()
    test_deterministic_replay_digest()
    print("ok")
