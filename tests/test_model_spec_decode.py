#!/usr/bin/env python3
"""Validation harness for src/model/spec_decode.HC."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HC = ROOT / "src/model/spec_decode.HC"


def _read() -> str:
    return HC.read_text(encoding="utf-8")


def _spec_decode_coordinator_step_py(token_ids, token_capacity, used_tokens, draft_capacity,
                                     draft_window, accepted_target_count, vocab_size, seed):
    if draft_window <= 0 or token_capacity <= 0 or used_tokens < 0:
        raise ValueError("bad params")
    if used_tokens > token_capacity:
        raise ValueError("bad params")
    slack = token_capacity - used_tokens
    if slack <= 0:
        raise ValueError("bad params")
    max_window = min(draft_window, slack)
    if max_window > draft_capacity:
        raise ValueError("bad params")

    draft_tokens = []
    for idx in range(max_window):
        raw = (used_tokens + idx + 1) * 1103515245
        raw = raw + seed + 12345
        if raw < 0:
            raw = -raw
        draft_tokens.append(raw % vocab_size)

    if accepted_target_count < 0 or accepted_target_count > len(draft_tokens):
        raise ValueError("bad params")

    accepted_count = accepted_target_count
    reject_index = -1 if accepted_count == len(draft_tokens) else accepted_count

    if used_tokens + accepted_count > token_capacity:
        raise ValueError("bad params")

    out_tokens = list(token_ids)
    for idx in range(accepted_count):
        out_tokens[used_tokens + idx] = draft_tokens[idx]

    return {
        "draft_tokens": draft_tokens,
        "drafted_count": len(draft_tokens),
        "accepted_count": accepted_count,
        "reject_index": reject_index,
        "committed_count": accepted_count,
        "used_tokens": used_tokens + accepted_count,
        "token_ids": out_tokens,
    }


def test_required_functions_exist():
    text = _read()
    for name in [
        "SpecDecodeValidateWindowChecked",
        "SpecDecodeDraftTokensChecked",
        "SpecDecodeVerifyDraftChecked",
        "SpecDecodeCommitAcceptedChecked",
        "SpecDecodeCoordinatorStepChecked",
    ]:
        assert f"I32 {name}(" in text


def test_has_integer_only_lcg_and_no_float_types():
    text = _read()
    assert "1103515245" in text
    assert "% vocab_size" in text
    assert not re.search(r"\b(F32|F64|double|float)\b", text)


def test_python_model_partial_accept_commit():
    token_ids = [10, 11, 12, 0, 0, 0, 0, 0]
    out = _spec_decode_coordinator_step_py(
        token_ids=token_ids,
        token_capacity=8,
        used_tokens=3,
        draft_capacity=4,
        draft_window=4,
        accepted_target_count=2,
        vocab_size=32000,
        seed=7,
    )
    assert out["drafted_count"] == 4
    assert out["accepted_count"] == 2
    assert out["reject_index"] == 2
    assert out["committed_count"] == 2
    assert out["used_tokens"] == 5
    assert out["token_ids"][:3] == [10, 11, 12]
    assert out["token_ids"][3] == out["draft_tokens"][0]
    assert out["token_ids"][4] == out["draft_tokens"][1]


def test_python_model_full_accept_commit():
    token_ids = [1, 2, 3, 4, 0, 0, 0]
    out = _spec_decode_coordinator_step_py(
        token_ids=token_ids,
        token_capacity=7,
        used_tokens=4,
        draft_capacity=3,
        draft_window=3,
        accepted_target_count=3,
        vocab_size=8192,
        seed=9,
    )
    assert out["drafted_count"] == 3
    assert out["accepted_count"] == 3
    assert out["reject_index"] == -1
    assert out["committed_count"] == 3
    assert out["used_tokens"] == 7


def test_python_model_window_clamps_to_slack():
    token_ids = [5, 6, 0, 0, 0]
    out = _spec_decode_coordinator_step_py(
        token_ids=token_ids,
        token_capacity=5,
        used_tokens=2,
        draft_capacity=5,
        draft_window=5,
        accepted_target_count=1,
        vocab_size=1024,
        seed=3,
    )
    assert out["drafted_count"] == 3
    assert out["accepted_count"] == 1
    assert out["reject_index"] == 1


def test_python_model_bad_accept_target_rejected():
    token_ids = [1, 2, 3, 0]
    try:
        _spec_decode_coordinator_step_py(
            token_ids=token_ids,
            token_capacity=4,
            used_tokens=3,
            draft_capacity=1,
            draft_window=1,
            accepted_target_count=2,
            vocab_size=2048,
            seed=1,
        )
        assert False, "expected ValueError"
    except ValueError:
        pass
