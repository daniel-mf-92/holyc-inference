#!/usr/bin/env python3
"""IQ-1795 harness for BotTokenEmitParityCommit."""

from __future__ import annotations

from pathlib import Path

SAMPLING_Q16_OK = 0
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


def bot_emit_base(
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
    status_a, count_a, digest_a, event_a = bot_emit_base(
        session_id=session_id,
        step_index=step_index,
        token_id=token_id,
        logit_q16=logit_q16,
        policy_digest_q64=policy_digest_q64,
        expected_policy_digest_q64=expected_policy_digest_q64,
    )
    status_b, count_b, digest_b, event_b = bot_emit_base(
        session_id=session_id,
        step_index=step_index,
        token_id=token_id,
        logit_q16=logit_q16,
        policy_digest_q64=policy_digest_q64,
        expected_policy_digest_q64=expected_policy_digest_q64,
    )

    assert status_a == status_b
    assert count_a == count_b
    assert digest_a == digest_b
    assert event_a == event_b

    if status_a == INFERENCE_BOT_STATUS_EMITTED:
        for lane in range(INFERENCE_BOT_EVENT_TUPLE_CELLS):
            event_buffer[lane] = event_a[lane]

    return SAMPLING_Q16_OK, status_a, count_a, digest_a, event_a


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


def test_source_has_iq_1795_gate() -> None:
    source = Path("src/model/inference.HC").read_text(encoding="utf-8")
    signature = "I32 BotTokenEmitParityCommit("
    assert signature in source
    body = _extract_body(source, signature)
    assert "status_parity = BotTokenEmitParity(" in body
    assert "status_preflight = BotTokenEmitPreflightOnly(" in body
    assert "status_primary = BotTokenEmitCommitOnly(" in body
    assert "status_replay = BotTokenEmitCommitOnly(" in body
    assert "if (staged_primary_status != staged_replay_status ||" in body


def test_secure_match_emits() -> None:
    event_buffer = [99] * INFERENCE_BOT_EVENT_TUPLE_CELLS
    rc, status, count, digest, event = bot_emit_parity_commit_model(
        event_buffer=event_buffer,
        session_id=11,
        step_index=22,
        token_id=33,
        logit_q16=44,
        policy_digest_q64=0xABCD,
        expected_policy_digest_q64=0xABCD,
    )

    assert rc == SAMPLING_Q16_OK
    assert status == INFERENCE_BOT_STATUS_EMITTED
    assert count == 1
    assert event_buffer == event
    assert digest == _digest(event, status, count)


def test_secure_mismatch_blocks() -> None:
    event_buffer = [5, 4, 3, 2, 1, 0]
    rc, status, count, digest, event = bot_emit_parity_commit_model(
        event_buffer=event_buffer,
        session_id=1,
        step_index=2,
        token_id=3,
        logit_q16=4,
        policy_digest_q64=0x1111,
        expected_policy_digest_q64=0x2222,
    )

    assert rc == SAMPLING_Q16_OK
    assert status == INFERENCE_BOT_STATUS_BLOCKED
    assert count == 0
    assert event_buffer == [5, 4, 3, 2, 1, 0]
    assert digest == _digest(event, status, count)


if __name__ == "__main__":
    test_source_has_iq_1795_gate()
    test_secure_match_emits()
    test_secure_mismatch_blocks()
    print("ok")
