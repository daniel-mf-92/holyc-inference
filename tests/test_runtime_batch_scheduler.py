#!/usr/bin/env python3
"""Harness for IQ-1266 continuous batching scheduler semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BATCH_SCHED_OK = 0
BATCH_SCHED_ERR_NULL_PTR = 1
BATCH_SCHED_ERR_BAD_PARAM = 2
BATCH_SCHED_ERR_OVERFLOW = 3
BATCH_SCHED_ERR_POLICY_GUARD = 4

PROFILE_SECURE_LOCAL = 1
PROFILE_DEV_LOCAL = 2


@dataclass
class Request:
    request_id: int
    arrival_seq: int
    remaining_prefill_tokens: int
    remaining_decode_tokens: int
    state_flags: int


class ProfileState:
    def __init__(self) -> None:
        self.profile_id = PROFILE_SECURE_LOCAL


class BatchSchedulerState:
    def __init__(self, profile: ProfileState) -> None:
        self.profile = profile
        self.require_policy_digest_match = 1
        self.require_attestation_ok = 1

    def set_security_guards_checked(self, require_policy_digest_match: int, require_attestation_ok: int) -> int:
        if require_policy_digest_match not in (0, 1) or require_attestation_ok not in (0, 1):
            return BATCH_SCHED_ERR_BAD_PARAM
        self.require_policy_digest_match = require_policy_digest_match
        self.require_attestation_ok = require_attestation_ok
        return BATCH_SCHED_OK

    def policy_gate_checked(self, policy_digest_match: int, attestation_ok: int) -> int:
        if policy_digest_match not in (0, 1) or attestation_ok not in (0, 1):
            return BATCH_SCHED_ERR_BAD_PARAM
        if self.profile.profile_id not in (PROFILE_SECURE_LOCAL, PROFILE_DEV_LOCAL):
            return BATCH_SCHED_ERR_POLICY_GUARD

        require_policy = self.require_policy_digest_match
        require_attestation = self.require_attestation_ok
        if self.profile.profile_id == PROFILE_SECURE_LOCAL:
            require_policy = 1
            require_attestation = 1

        if (require_policy and not policy_digest_match) or (require_attestation and not attestation_ok):
            return BATCH_SCHED_ERR_POLICY_GUARD
        return BATCH_SCHED_OK

    def plan_step_checked(
        self,
        requests: list[Request],
        max_active_requests: int,
        token_budget: int,
        policy_digest_match: int,
        attestation_ok: int,
    ) -> tuple[int, list[int], list[int], int, int]:
        if max_active_requests < 0 or token_budget < 0:
            return (BATCH_SCHED_ERR_BAD_PARAM, [], [], 0, 0)

        policy_status = self.policy_gate_checked(policy_digest_match, attestation_ok)
        if policy_status != BATCH_SCHED_OK:
            return (policy_status, [], [], 0, 0)

        if max_active_requests == 0 or token_budget == 0:
            return (BATCH_SCHED_OK, [], [], 0, 0)

        active_indices: list[int] = []
        for idx, req in enumerate(requests):
            if len(active_indices) >= max_active_requests:
                break
            if not (req.state_flags & 1):
                continue
            if req.remaining_prefill_tokens < 0 or req.remaining_decode_tokens < 0:
                continue
            if req.remaining_prefill_tokens + req.remaining_decode_tokens <= 0:
                continue
            active_indices.append(idx)

        if not active_indices:
            return (BATCH_SCHED_OK, [], [], 0, 0)

        allocs = [0 for _ in active_indices]
        prefill_left = [requests[idx].remaining_prefill_tokens for idx in active_indices]
        decode_left = [requests[idx].remaining_decode_tokens for idx in active_indices]

        budget_left = token_budget
        planned_prefill = 0
        planned_decode = 0

        while budget_left > 0:
            progressed = False
            for lane in range(len(active_indices)):
                if budget_left == 0:
                    break
                if prefill_left[lane] <= 0:
                    continue
                prefill_left[lane] -= 1
                allocs[lane] += 1
                budget_left -= 1
                planned_prefill += 1
                progressed = True
            if not progressed:
                break

        while budget_left > 0:
            progressed = False
            for lane in range(len(active_indices)):
                if budget_left == 0:
                    break
                if decode_left[lane] <= 0:
                    continue
                decode_left[lane] -= 1
                allocs[lane] += 1
                budget_left -= 1
                planned_decode += 1
                progressed = True
            if not progressed:
                break

        return (BATCH_SCHED_OK, active_indices, allocs, planned_prefill, planned_decode)

    def advance_requests_checked(self, requests: list[Request], indices: list[int], allocs: list[int]) -> int:
        if len(indices) != len(allocs):
            return BATCH_SCHED_ERR_BAD_PARAM

        for req_idx, alloc in zip(indices, allocs):
            if req_idx < 0 or req_idx >= len(requests) or alloc < 0:
                return BATCH_SCHED_ERR_BAD_PARAM
            req = requests[req_idx]
            if req.remaining_prefill_tokens < 0 or req.remaining_decode_tokens < 0:
                return BATCH_SCHED_ERR_BAD_PARAM
            if alloc > (req.remaining_prefill_tokens + req.remaining_decode_tokens):
                return BATCH_SCHED_ERR_BAD_PARAM

        for req_idx, alloc in zip(indices, allocs):
            req = requests[req_idx]
            consume_prefill = min(alloc, req.remaining_prefill_tokens)
            req.remaining_prefill_tokens -= consume_prefill
            req.remaining_decode_tokens -= alloc - consume_prefill

        return BATCH_SCHED_OK


def _build_requests() -> list[Request]:
    return [
        Request(11, 100, 2, 2, 1),
        Request(12, 101, 1, 3, 1),
        Request(13, 102, 0, 2, 1),
    ]


def test_source_contains_iq1266_symbols() -> None:
    src = Path("src/runtime/batch_scheduler.HC").read_text(encoding="utf-8")
    assert "I32 BatchSchedulerPlanStepChecked(" in src
    assert "I32 BatchSchedulerAdvanceRequestsChecked(" in src
    assert "I32 BatchSchedulerPolicyGateChecked(" in src
    assert "secure-local is fail-closed" in src
    assert "round-robin prefill" in src


def test_secure_local_requires_policy_and_attestation() -> None:
    profile = ProfileState()
    scheduler = BatchSchedulerState(profile)

    status, _, _, _, _ = scheduler.plan_step_checked(_build_requests(), 2, 4, policy_digest_match=0, attestation_ok=1)
    assert status == BATCH_SCHED_ERR_POLICY_GUARD

    status, _, _, _, _ = scheduler.plan_step_checked(_build_requests(), 2, 4, policy_digest_match=1, attestation_ok=0)
    assert status == BATCH_SCHED_ERR_POLICY_GUARD

    status, _, _, _, _ = scheduler.plan_step_checked(_build_requests(), 2, 4, policy_digest_match=1, attestation_ok=1)
    assert status == BATCH_SCHED_OK


def test_dev_local_can_opt_out_policy_flags_explicitly() -> None:
    profile = ProfileState()
    profile.profile_id = PROFILE_DEV_LOCAL
    scheduler = BatchSchedulerState(profile)
    assert scheduler.set_security_guards_checked(0, 0) == BATCH_SCHED_OK

    status, indices, allocs, prefill, decode = scheduler.plan_step_checked(
        _build_requests(),
        max_active_requests=2,
        token_budget=3,
        policy_digest_match=0,
        attestation_ok=0,
    )
    assert status == BATCH_SCHED_OK
    assert indices == [0, 1]
    assert allocs == [2, 1]
    assert prefill == 3
    assert decode == 0


def test_round_robin_prefill_then_decode_allocation() -> None:
    profile = ProfileState()
    scheduler = BatchSchedulerState(profile)

    status, indices, allocs, prefill, decode = scheduler.plan_step_checked(
        _build_requests(),
        max_active_requests=2,
        token_budget=5,
        policy_digest_match=1,
        attestation_ok=1,
    )

    assert status == BATCH_SCHED_OK
    assert indices == [0, 1]
    assert allocs == [3, 2]
    assert prefill == 3
    assert decode == 2


def test_advance_requests_consumes_prefill_then_decode() -> None:
    profile = ProfileState()
    scheduler = BatchSchedulerState(profile)
    requests = _build_requests()

    status, indices, allocs, _, _ = scheduler.plan_step_checked(
        requests,
        max_active_requests=2,
        token_budget=5,
        policy_digest_match=1,
        attestation_ok=1,
    )
    assert status == BATCH_SCHED_OK

    assert scheduler.advance_requests_checked(requests, indices, allocs) == BATCH_SCHED_OK

    assert requests[0].remaining_prefill_tokens == 0
    assert requests[0].remaining_decode_tokens == 1
    assert requests[1].remaining_prefill_tokens == 0
    assert requests[1].remaining_decode_tokens == 2
    assert requests[2].remaining_prefill_tokens == 0
    assert requests[2].remaining_decode_tokens == 2


def test_advance_rejects_oversubscription_without_mutation() -> None:
    profile = ProfileState()
    scheduler = BatchSchedulerState(profile)
    requests = _build_requests()

    before = [(r.remaining_prefill_tokens, r.remaining_decode_tokens) for r in requests]
    err = scheduler.advance_requests_checked(requests, [0], [99])
    assert err == BATCH_SCHED_ERR_BAD_PARAM

    after = [(r.remaining_prefill_tokens, r.remaining_decode_tokens) for r in requests]
    assert after == before


if __name__ == "__main__":
    test_source_contains_iq1266_symbols()
    test_secure_local_requires_policy_and_attestation()
    test_dev_local_can_opt_out_policy_flags_explicitly()
    test_round_robin_prefill_then_decode_allocation()
    test_advance_requests_consumes_prefill_then_decode()
    test_advance_rejects_oversubscription_without_mutation()
    print("ok")
