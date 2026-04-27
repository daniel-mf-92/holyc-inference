#!/usr/bin/env python3
"""IQ-1798 harness for BotTokenEmitDiagReplayCommit."""

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


def _emit_event(
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


def _model_iq1798(
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
    staged_diag_status = 0x8181
    staged_diag_count = 0x8282
    staged_diag_digest = 0x8383
    if (staged_diag_status, staged_diag_count, staged_diag_digest) != (0x8181, 0x8282, 0x8383):
        return SAMPLING_Q16_ERR_BAD_PARAM

    staged_pre_status = 0x8484
    staged_pre_count = 0x8585
    staged_pre_digest = 0x8686
    if (staged_pre_status, staged_pre_count, staged_pre_digest) != (0x8484, 0x8585, 0x8686):
        return SAMPLING_Q16_ERR_BAD_PARAM

    status_a, count_a, digest_a, event_a = _emit_event(
        session_id=session_id,
        step_index=step_index,
        token_id=token_id,
        logit_q16=logit_q16,
        policy_digest_q64=policy_digest_q64,
        expected_policy_digest_q64=expected_policy_digest_q64,
    )
    status_b, count_b, digest_b, event_b = _emit_event(
        session_id=session_id,
        step_index=step_index,
        token_id=token_id,
        logit_q16=logit_q16,
        policy_digest_q64=policy_digest_q64,
        expected_policy_digest_q64=expected_policy_digest_q64,
    )
    if status_a not in (INFERENCE_BOT_STATUS_BLOCKED, INFERENCE_BOT_STATUS_EMITTED):
        return SAMPLING_Q16_ERR_BAD_PARAM
    if status_b not in (INFERENCE_BOT_STATUS_BLOCKED, INFERENCE_BOT_STATUS_EMITTED):
        return SAMPLING_Q16_ERR_BAD_PARAM
    if (status_a, count_a, digest_a, event_a) != (status_b, count_b, digest_b, event_b):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if status_a == INFERENCE_BOT_STATUS_EMITTED:
        for lane in range(INFERENCE_BOT_EVENT_TUPLE_CELLS):
            event_buffer[lane] = event_a[lane]
    out_status[0] = status_a
    out_count[0] = count_a
    out_digest[0] = digest_a
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


def test_source_has_iq_1798_gate() -> None:
    source = Path("src/model/inference.HC").read_text(encoding="utf-8")
    signature = "I32 BotTokenEmitDiagReplayCommit("
    assert signature in source
    body = _extract_body(source, signature)
    assert "status_diag = BotTokenEmitDiagReplay(" in body
    assert "status_preflight = BotTokenEmitParityCommitPreflight(" in body
    assert "status_primary = BotTokenEmitParityCommit(" in body
    assert "status_replay = BotTokenEmitParityCommit(" in body
    assert "if (staged_diag_status != 0x8181 ||" in body
    assert "if (staged_preflight_status != 0x8484 ||" in body
    assert "if (staged_primary_status != staged_replay_status ||" in body


def test_secure_match_commits_tuple_and_event() -> None:
    event_buffer = [99] * INFERENCE_BOT_EVENT_TUPLE_CELLS
    out_status = [91]
    out_count = [92]
    out_digest = [93]
    rc = _model_iq1798(
        event_buffer=event_buffer,
        out_status=out_status,
        out_count=out_count,
        out_digest=out_digest,
        session_id=71,
        step_index=72,
        token_id=73,
        logit_q16=74,
        policy_digest_q64=0x1234,
        expected_policy_digest_q64=0x1234,
    )
    assert rc == SAMPLING_Q16_OK
    assert event_buffer == [71, 72, 73, 74, 0x1234, INFERENCE_BOT_PROFILE_SECURE]
    assert out_status == [INFERENCE_BOT_STATUS_EMITTED]
    assert out_count == [1]
    assert out_digest == [_digest(event_buffer, INFERENCE_BOT_STATUS_EMITTED, 1)]


def test_secure_mismatch_commits_blocked_tuple_only() -> None:
    event_buffer = [7, 8, 9, 10, 11, 12]
    snap_event = list(event_buffer)
    out_status = [44]
    out_count = [45]
    out_digest = [46]
    rc = _model_iq1798(
        event_buffer=event_buffer,
        out_status=out_status,
        out_count=out_count,
        out_digest=out_digest,
        session_id=81,
        step_index=82,
        token_id=83,
        logit_q16=84,
        policy_digest_q64=0x2221,
        expected_policy_digest_q64=0x2222,
    )
    assert rc == SAMPLING_Q16_OK
    assert event_buffer == snap_event
    assert out_status == [INFERENCE_BOT_STATUS_BLOCKED]
    assert out_count == [0]
    blocked_event = [81, 82, 83, 84, 0x2221, INFERENCE_BOT_PROFILE_SECURE]
    assert out_digest == [_digest(blocked_event, INFERENCE_BOT_STATUS_BLOCKED, 0)]


if __name__ == "__main__":
    test_source_has_iq_1798_gate()
    test_secure_match_commits_tuple_and_event()
    test_secure_mismatch_commits_blocked_tuple_only()
    print("ok")
