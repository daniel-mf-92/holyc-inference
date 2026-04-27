#!/usr/bin/env python3
"""IQ-1799 harness for BotTokenEmitDiagReplayCommitPreflight."""

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


def _emit_parity_commit(
    *,
    event_buffer: list[int],
    session_id: int,
    step_index: int,
    token_id: int,
    logit_q16: int,
    policy_digest_q64: int,
    expected_policy_digest_q64: int,
) -> tuple[int, int, int, int]:
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
    if status_a not in (INFERENCE_BOT_STATUS_BLOCKED, INFERENCE_BOT_STATUS_EMITTED):
        return SAMPLING_Q16_ERR_BAD_PARAM, 0, 0, 0
    if status_b not in (INFERENCE_BOT_STATUS_BLOCKED, INFERENCE_BOT_STATUS_EMITTED):
        return SAMPLING_Q16_ERR_BAD_PARAM, 0, 0, 0
    if (status_a, count_a, digest_a, event_a) != (status_b, count_b, digest_b, event_b):
        return SAMPLING_Q16_ERR_BAD_PARAM, 0, 0, 0
    if status_a == INFERENCE_BOT_STATUS_EMITTED:
        for lane in range(INFERENCE_BOT_EVENT_TUPLE_CELLS):
            event_buffer[lane] = event_a[lane]
    return SAMPLING_Q16_OK, status_a, count_a, digest_a


def _emit_diag_replay(
    *,
    event_buffer: list[int],
    session_id: int,
    step_index: int,
    token_id: int,
    logit_q16: int,
    policy_digest_q64: int,
    expected_policy_digest_q64: int,
) -> tuple[int, int, int, int]:
    staged_preflight_status = 0x7171
    staged_preflight_count = 0x7272
    staged_preflight_digest = 0x7373
    if (staged_preflight_status, staged_preflight_count, staged_preflight_digest) != (
        0x7171,
        0x7272,
        0x7373,
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM, 0, 0, 0

    staged_commit = [0] * INFERENCE_BOT_EVENT_TUPLE_CELLS
    rc_commit, status_commit, count_commit, digest_commit = _emit_parity_commit(
        event_buffer=staged_commit,
        session_id=session_id,
        step_index=step_index,
        token_id=token_id,
        logit_q16=logit_q16,
        policy_digest_q64=policy_digest_q64,
        expected_policy_digest_q64=expected_policy_digest_q64,
    )
    if rc_commit != SAMPLING_Q16_OK:
        return rc_commit, 0, 0, 0

    staged_replay = [0] * INFERENCE_BOT_EVENT_TUPLE_CELLS
    rc_replay, status_replay, count_replay, digest_replay = _emit_parity_commit(
        event_buffer=staged_replay,
        session_id=session_id,
        step_index=step_index,
        token_id=token_id,
        logit_q16=logit_q16,
        policy_digest_q64=policy_digest_q64,
        expected_policy_digest_q64=expected_policy_digest_q64,
    )
    if rc_replay != SAMPLING_Q16_OK:
        return rc_replay, 0, 0, 0

    if (status_commit, count_commit, digest_commit) != (status_replay, count_replay, digest_replay):
        return SAMPLING_Q16_ERR_BAD_PARAM, 0, 0, 0
    if staged_commit != staged_replay:
        return SAMPLING_Q16_ERR_BAD_PARAM, 0, 0, 0
    return SAMPLING_Q16_OK, status_commit, count_commit, digest_commit


def _emit_diag_replay_commit(
    *,
    event_buffer: list[int],
    session_id: int,
    step_index: int,
    token_id: int,
    logit_q16: int,
    policy_digest_q64: int,
    expected_policy_digest_q64: int,
) -> tuple[int, int, int, int]:
    staged_diag_status = 0x8181
    staged_diag_count = 0x8282
    staged_diag_digest = 0x8383
    if (staged_diag_status, staged_diag_count, staged_diag_digest) != (0x8181, 0x8282, 0x8383):
        return SAMPLING_Q16_ERR_BAD_PARAM, 0, 0, 0

    staged_pre_status = 0x8484
    staged_pre_count = 0x8585
    staged_pre_digest = 0x8686
    if (staged_pre_status, staged_pre_count, staged_pre_digest) != (0x8484, 0x8585, 0x8686):
        return SAMPLING_Q16_ERR_BAD_PARAM, 0, 0, 0

    staged_primary = [0] * INFERENCE_BOT_EVENT_TUPLE_CELLS
    rc_primary, status_primary, count_primary, digest_primary = _emit_parity_commit(
        event_buffer=staged_primary,
        session_id=session_id,
        step_index=step_index,
        token_id=token_id,
        logit_q16=logit_q16,
        policy_digest_q64=policy_digest_q64,
        expected_policy_digest_q64=expected_policy_digest_q64,
    )
    if rc_primary != SAMPLING_Q16_OK:
        return rc_primary, 0, 0, 0

    staged_replay = [0] * INFERENCE_BOT_EVENT_TUPLE_CELLS
    rc_replay, status_replay, count_replay, digest_replay = _emit_parity_commit(
        event_buffer=staged_replay,
        session_id=session_id,
        step_index=step_index,
        token_id=token_id,
        logit_q16=logit_q16,
        policy_digest_q64=policy_digest_q64,
        expected_policy_digest_q64=expected_policy_digest_q64,
    )
    if rc_replay != SAMPLING_Q16_OK:
        return rc_replay, 0, 0, 0

    if (status_primary, count_primary, digest_primary) != (status_replay, count_replay, digest_replay):
        return SAMPLING_Q16_ERR_BAD_PARAM, 0, 0, 0
    if staged_primary != staged_replay:
        return SAMPLING_Q16_ERR_BAD_PARAM, 0, 0, 0
    if status_primary == INFERENCE_BOT_STATUS_EMITTED:
        for lane in range(INFERENCE_BOT_EVENT_TUPLE_CELLS):
            event_buffer[lane] = staged_primary[lane]
    return SAMPLING_Q16_OK, status_primary, count_primary, digest_primary


def _model_iq1799(
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
    rc_commit, status_commit, count_commit, digest_commit = _emit_diag_replay_commit(
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

    staged_canonical = [0] * INFERENCE_BOT_EVENT_TUPLE_CELLS
    rc_canonical, status_canonical, count_canonical, digest_canonical = _emit_diag_replay(
        event_buffer=staged_canonical,
        session_id=session_id,
        step_index=step_index,
        token_id=token_id,
        logit_q16=logit_q16,
        policy_digest_q64=policy_digest_q64,
        expected_policy_digest_q64=expected_policy_digest_q64,
    )
    if rc_canonical != SAMPLING_Q16_OK:
        return rc_canonical

    if status_commit not in (INFERENCE_BOT_STATUS_BLOCKED, INFERENCE_BOT_STATUS_EMITTED):
        return SAMPLING_Q16_ERR_BAD_PARAM
    if status_canonical not in (INFERENCE_BOT_STATUS_BLOCKED, INFERENCE_BOT_STATUS_EMITTED):
        return SAMPLING_Q16_ERR_BAD_PARAM
    if (status_commit, count_commit, digest_commit) != (
        status_canonical,
        count_canonical,
        digest_canonical,
    ):
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


def test_source_has_iq_1799_gate() -> None:
    source = Path("src/model/inference.HC").read_text(encoding="utf-8")
    signature = "I32 BotTokenEmitDiagReplayCommitPreflight("
    assert signature in source
    body = _extract_body(source, signature)
    assert "status_commit = BotTokenEmitDiagReplayCommit(" in body
    assert "status_canonical = BotTokenEmitDiagReplay(" in body
    assert "if (staged_commit_status != staged_canonical_status ||" in body
    assert "if (*out_event_status != snapshot_event_status ||" in body
    assert "if (event_buffer[lane] != snapshot_event[lane])" in body


def test_secure_match_keeps_outputs_untouched() -> None:
    event_buffer = [5, 4, 3, 2, 1, 0]
    out_status = [31]
    out_count = [32]
    out_digest = [33]
    rc = _model_iq1799(
        event_buffer=event_buffer,
        out_status=out_status,
        out_count=out_count,
        out_digest=out_digest,
        session_id=101,
        step_index=102,
        token_id=103,
        logit_q16=104,
        policy_digest_q64=0x4455,
        expected_policy_digest_q64=0x4455,
    )
    assert rc == SAMPLING_Q16_OK
    assert event_buffer == [5, 4, 3, 2, 1, 0]
    assert out_status == [31]
    assert out_count == [32]
    assert out_digest == [33]


def test_secure_mismatch_keeps_outputs_untouched() -> None:
    event_buffer = [13, 21, 34, 55, 89, 144]
    out_status = [41]
    out_count = [42]
    out_digest = [43]
    rc = _model_iq1799(
        event_buffer=event_buffer,
        out_status=out_status,
        out_count=out_count,
        out_digest=out_digest,
        session_id=201,
        step_index=202,
        token_id=203,
        logit_q16=204,
        policy_digest_q64=0x8898,
        expected_policy_digest_q64=0x8899,
    )
    assert rc == SAMPLING_Q16_OK
    assert event_buffer == [13, 21, 34, 55, 89, 144]
    assert out_status == [41]
    assert out_count == [42]
    assert out_digest == [43]


if __name__ == "__main__":
    test_source_has_iq_1799_gate()
    test_secure_match_keeps_outputs_untouched()
    test_secure_mismatch_keeps_outputs_untouched()
    print("ok")
