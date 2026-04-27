#!/usr/bin/env python3
"""IQ-1796 harness for BotTokenEmitParityCommitPreflight."""

from __future__ import annotations

from pathlib import Path

SAMPLING_Q16_OK = 0
SAMPLING_Q16_ERR_BAD_PARAM = 2

INFERENCE_BOT_EVENT_TUPLE_CELLS = 6
INFERENCE_BOT_PROFILE_SECURE = 1
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


def _emit_base(
    *,
    session_id: int,
    step_index: int,
    token_id: int,
    logit_q16: int,
    policy_digest_q64: int,
    expected_policy_digest_q64: int,
) -> tuple[int, int, int, list[int]]:
    event = [
        session_id,
        step_index,
        token_id,
        logit_q16,
        policy_digest_q64,
        INFERENCE_BOT_PROFILE_SECURE,
    ]
    if policy_digest_q64 == expected_policy_digest_q64:
        status = INFERENCE_BOT_STATUS_EMITTED
        count = 1
    else:
        status = INFERENCE_BOT_STATUS_BLOCKED
        count = 0
    return status, count, _digest(event, status, count), event


def bot_emit_parity_commit_model(
    *,
    event_buffer: list[int],
    session_id: int,
    step_index: int,
    token_id: int,
    logit_q16: int,
    policy_digest_q64: int,
    expected_policy_digest_q64: int,
) -> tuple[int, int, int, int, list[int]]:
    status_a, count_a, digest_a, event_a = _emit_base(
        session_id=session_id,
        step_index=step_index,
        token_id=token_id,
        logit_q16=logit_q16,
        policy_digest_q64=policy_digest_q64,
        expected_policy_digest_q64=expected_policy_digest_q64,
    )
    status_b, count_b, digest_b, event_b = _emit_base(
        session_id=session_id,
        step_index=step_index,
        token_id=token_id,
        logit_q16=logit_q16,
        policy_digest_q64=policy_digest_q64,
        expected_policy_digest_q64=expected_policy_digest_q64,
    )
    if (status_a, count_a, digest_a, event_a) != (status_b, count_b, digest_b, event_b):
        return SAMPLING_Q16_ERR_BAD_PARAM, 0, 0, 0, []
    if status_a == INFERENCE_BOT_STATUS_EMITTED:
        for lane in range(INFERENCE_BOT_EVENT_TUPLE_CELLS):
            event_buffer[lane] = event_a[lane]
    return SAMPLING_Q16_OK, status_a, count_a, digest_a, event_a


def bot_emit_parity_model(
    *,
    event_buffer: list[int],
    session_id: int,
    step_index: int,
    token_id: int,
    logit_q16: int,
    policy_digest_q64: int,
    expected_policy_digest_q64: int,
) -> tuple[int, int, int, int, list[int]]:
    return bot_emit_parity_commit_model(
        event_buffer=event_buffer,
        session_id=session_id,
        step_index=step_index,
        token_id=token_id,
        logit_q16=logit_q16,
        policy_digest_q64=policy_digest_q64,
        expected_policy_digest_q64=expected_policy_digest_q64,
    )


def bot_emit_parity_commit_pre_model(
    *,
    event_buffer: list[int],
    out_status: list[int],
    out_count: list[int],
    out_digest: list[int],
    session_id: int,
    step_index: int,
    token_id: int,
    logit_q16: int,
    policy_digest_q64: int,
    expected_policy_digest_q64: int,
) -> int:
    snap_event = list(event_buffer)
    snap_status = out_status[0]
    snap_count = out_count[0]
    snap_digest = out_digest[0]

    staged_commit = [0] * INFERENCE_BOT_EVENT_TUPLE_CELLS
    staged_parity = [0] * INFERENCE_BOT_EVENT_TUPLE_CELLS

    rc_commit, status_commit, count_commit, digest_commit, _ = bot_emit_parity_commit_model(
        event_buffer=staged_commit,
        session_id=session_id,
        step_index=step_index,
        token_id=token_id,
        logit_q16=logit_q16,
        policy_digest_q64=policy_digest_q64,
        expected_policy_digest_q64=expected_policy_digest_q64,
    )
    if rc_commit != SAMPLING_Q16_OK:
        return rc_commit

    rc_parity, status_parity, count_parity, digest_parity, _ = bot_emit_parity_model(
        event_buffer=staged_parity,
        session_id=session_id,
        step_index=step_index,
        token_id=token_id,
        logit_q16=logit_q16,
        policy_digest_q64=policy_digest_q64,
        expected_policy_digest_q64=expected_policy_digest_q64,
    )
    if rc_parity != SAMPLING_Q16_OK:
        return rc_parity

    if (status_commit, count_commit, digest_commit) != (status_parity, count_parity, digest_parity):
        return SAMPLING_Q16_ERR_BAD_PARAM
    if staged_commit != staged_parity:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if event_buffer != snap_event:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if (out_status[0], out_count[0], out_digest[0]) != (snap_status, snap_count, snap_digest):
        return SAMPLING_Q16_ERR_BAD_PARAM
    return SAMPLING_Q16_OK


def _extract_body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 1
    idx = brace + 1
    while depth:
        ch = source[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        idx += 1
    return source[brace + 1 : idx - 1]


def test_source_has_iq_1796_gate() -> None:
    source = Path("src/model/inference.HC").read_text(encoding="utf-8")
    signature = "I32 BotTokenEmitParityCommitPreflight("
    assert signature in source
    body = _extract_body(source, signature)
    assert "status_commit = BotTokenEmitParityCommit(" in body
    assert "status_parity = BotTokenEmitParity(" in body
    assert "if (staged_commit_status != staged_parity_status ||" in body
    assert "if (*out_event_status != snapshot_event_status ||" in body
    assert "if (event_buffer[lane] != snapshot_event[lane])" in body


def test_secure_match_keeps_outputs_untouched() -> None:
    event_buffer = [7, 6, 5, 4, 3, 2]
    out_status = [91]
    out_count = [92]
    out_digest = [93]
    rc = bot_emit_parity_commit_pre_model(
        event_buffer=event_buffer,
        out_status=out_status,
        out_count=out_count,
        out_digest=out_digest,
        session_id=8,
        step_index=9,
        token_id=10,
        logit_q16=11,
        policy_digest_q64=0x55AA,
        expected_policy_digest_q64=0x55AA,
    )
    assert rc == SAMPLING_Q16_OK
    assert event_buffer == [7, 6, 5, 4, 3, 2]
    assert out_status == [91]
    assert out_count == [92]
    assert out_digest == [93]


def test_secure_mismatch_keeps_outputs_untouched() -> None:
    event_buffer = [1, 1, 2, 3, 5, 8]
    out_status = [41]
    out_count = [42]
    out_digest = [43]
    rc = bot_emit_parity_commit_pre_model(
        event_buffer=event_buffer,
        out_status=out_status,
        out_count=out_count,
        out_digest=out_digest,
        session_id=15,
        step_index=16,
        token_id=17,
        logit_q16=18,
        policy_digest_q64=0xA0,
        expected_policy_digest_q64=0xB0,
    )
    assert rc == SAMPLING_Q16_OK
    assert event_buffer == [1, 1, 2, 3, 5, 8]
    assert out_status == [41]
    assert out_count == [42]
    assert out_digest == [43]


if __name__ == "__main__":
    test_source_has_iq_1796_gate()
    test_secure_match_keeps_outputs_untouched()
    test_secure_mismatch_keeps_outputs_untouched()
    print("ok")
