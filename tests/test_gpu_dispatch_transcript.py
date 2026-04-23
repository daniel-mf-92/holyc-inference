#!/usr/bin/env python3
"""Harness for IQ-1263 deterministic GPU dispatch transcript recorder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

GPU_DISPATCH_TX_OK = 0
GPU_DISPATCH_TX_ERR_NULL_PTR = 1
GPU_DISPATCH_TX_ERR_BAD_PARAM = 2
GPU_DISPATCH_TX_ERR_BAD_STATE = 3
GPU_DISPATCH_TX_ERR_CAPACITY = 4
GPU_DISPATCH_TX_ERR_OVERFLOW = 5
GPU_DISPATCH_TX_ERR_ALIGNMENT = 6
GPU_DISPATCH_TX_ERR_SEQUENCE_OVERFLOW = 7
GPU_DISPATCH_TX_ERR_NOT_FOUND = 8

GPU_DISPATCH_TX_ALIGN_BYTES = 16
GPU_DISPATCH_TX_I64_MAX = 0x7FFFFFFFFFFFFFFF
I64_MASK = 0xFFFFFFFFFFFFFFFF


def _to_i64(value: int) -> int:
    value &= I64_MASK
    if value >= (1 << 63):
        value -= 1 << 64
    return value


def _is_aligned_pow2(value: int, alignment: int) -> bool:
    if alignment <= 0:
        return False
    return (value & (alignment - 1)) == 0


def _add_checked(lhs: int, rhs: int) -> int | None:
    if lhs < 0 or rhs < 0:
        return None
    if lhs > GPU_DISPATCH_TX_I64_MAX - rhs:
        return None
    return lhs + rhs


def _mix64(seed: int, value: int) -> int:
    x = _to_i64(seed ^ value)
    x = _to_i64(x ^ (x >> 30))
    x = _to_i64(x * 0xBF58476D1CE4E5B9)
    x = _to_i64(x ^ (x >> 27))
    x = _to_i64(x * 0x94D049BB133111EB)
    x = _to_i64(x ^ (x >> 31))
    if x == 0:
        x = 0x9E3779B97F4A7C15
    return _to_i64(x)


def _entry_hash(
    session_nonce: int,
    seq_id: int,
    queue_id: int,
    queue_depth_before: int,
    queue_depth_after: int,
    descriptor_addr: int,
    descriptor_bytes: int,
    descriptor_type: int,
    descriptor_op: int,
    cycle_start: int,
    cycle_end: int,
    status_code: int,
) -> int:
    h = _mix64(session_nonce, seq_id)
    h = _mix64(h, queue_id)
    h = _mix64(h, queue_depth_before)
    h = _mix64(h, queue_depth_after)
    h = _mix64(h, descriptor_addr)
    h = _mix64(h, descriptor_bytes)
    h = _mix64(h, descriptor_type)
    h = _mix64(h, descriptor_op)
    h = _mix64(h, cycle_start)
    h = _mix64(h, cycle_end)
    h = _mix64(h, status_code)
    return h


@dataclass
class Entry:
    seq_id: int = 0
    queue_id: int = 0
    queue_depth_before: int = 0
    queue_depth_after: int = 0
    descriptor_addr: int = 0
    descriptor_bytes: int = 0
    descriptor_type: int = 0
    descriptor_op: int = 0
    cycle_start: int = 0
    cycle_end: int = 0
    cycle_delta: int = 0
    status_code: int = 0
    entry_hash: int = 0
    chain_hash: int = 0


@dataclass
class Transcript:
    entries: list[Entry]
    capacity: int
    count: int = 0
    head: int = 0
    next_seq_id: int = 1
    session_nonce: int = 0
    chain_hash: int = 0
    total_descriptor_bytes: int = 0
    total_cycle_delta: int = 0


def tx_init(capacity: int, session_nonce: int) -> tuple[int, Transcript | None]:
    if capacity <= 0:
        return GPU_DISPATCH_TX_ERR_CAPACITY, None
    t = Transcript(entries=[Entry() for _ in range(capacity)], capacity=capacity)
    t.session_nonce = session_nonce
    t.chain_hash = _mix64(session_nonce, 0xD15A7C7001)
    return GPU_DISPATCH_TX_OK, t


def _state_is_valid(t: Transcript | None) -> bool:
    return (
        t is not None
        and t.entries is not None
        and t.capacity > 0
        and 0 <= t.count <= t.capacity
        and 0 <= t.head < t.capacity
        and t.next_seq_id > 0
        and t.total_descriptor_bytes >= 0
        and t.total_cycle_delta >= 0
    )


def tx_record(
    t: Transcript | None,
    queue_id: int,
    queue_depth_before: int,
    queue_depth_after: int,
    descriptor_addr: int,
    descriptor_bytes: int,
    descriptor_type: int,
    descriptor_op: int,
    cycle_start: int,
    cycle_end: int,
    status_code: int,
) -> tuple[int, int, int, int]:
    if t is None:
        return GPU_DISPATCH_TX_ERR_NULL_PTR, 0, 0, 0
    if not _state_is_valid(t):
        return GPU_DISPATCH_TX_ERR_BAD_STATE, 0, 0, 0

    if (
        queue_id < 0
        or queue_depth_before < 0
        or queue_depth_after < 0
        or descriptor_addr < 0
        or descriptor_bytes <= 0
        or descriptor_type <= 0
        or descriptor_op <= 0
        or cycle_start < 0
        or cycle_end < 0
        or status_code < 0
    ):
        return GPU_DISPATCH_TX_ERR_BAD_PARAM, 0, 0, 0

    if cycle_end < cycle_start:
        return GPU_DISPATCH_TX_ERR_BAD_PARAM, 0, 0, 0

    if not _is_aligned_pow2(descriptor_addr, GPU_DISPATCH_TX_ALIGN_BYTES):
        return GPU_DISPATCH_TX_ERR_ALIGNMENT, 0, 0, 0
    if not _is_aligned_pow2(descriptor_bytes, GPU_DISPATCH_TX_ALIGN_BYTES):
        return GPU_DISPATCH_TX_ERR_ALIGNMENT, 0, 0, 0

    if t.next_seq_id == GPU_DISPATCH_TX_I64_MAX:
        return GPU_DISPATCH_TX_ERR_SEQUENCE_OVERFLOW, 0, 0, 0

    cycle_delta = cycle_end - cycle_start

    total_descriptor_bytes = _add_checked(t.total_descriptor_bytes, descriptor_bytes)
    if total_descriptor_bytes is None:
        return GPU_DISPATCH_TX_ERR_OVERFLOW, 0, 0, 0

    total_cycle_delta = _add_checked(t.total_cycle_delta, cycle_delta)
    if total_cycle_delta is None:
        return GPU_DISPATCH_TX_ERR_OVERFLOW, 0, 0, 0

    seq_id = t.next_seq_id
    entry_hash = _entry_hash(
        t.session_nonce,
        seq_id,
        queue_id,
        queue_depth_before,
        queue_depth_after,
        descriptor_addr,
        descriptor_bytes,
        descriptor_type,
        descriptor_op,
        cycle_start,
        cycle_end,
        status_code,
    )
    chain_hash = _mix64(t.chain_hash, _to_i64(entry_hash ^ seq_id))

    e = t.entries[t.head]
    e.seq_id = seq_id
    e.queue_id = queue_id
    e.queue_depth_before = queue_depth_before
    e.queue_depth_after = queue_depth_after
    e.descriptor_addr = descriptor_addr
    e.descriptor_bytes = descriptor_bytes
    e.descriptor_type = descriptor_type
    e.descriptor_op = descriptor_op
    e.cycle_start = cycle_start
    e.cycle_end = cycle_end
    e.cycle_delta = cycle_delta
    e.status_code = status_code
    e.entry_hash = entry_hash
    e.chain_hash = chain_hash

    t.next_seq_id += 1
    t.chain_hash = chain_hash
    t.total_descriptor_bytes = total_descriptor_bytes
    t.total_cycle_delta = total_cycle_delta
    t.head = (t.head + 1) % t.capacity
    if t.count < t.capacity:
        t.count += 1

    return GPU_DISPATCH_TX_OK, seq_id, entry_hash, chain_hash


def tx_get_by_seq(t: Transcript | None, seq_id: int) -> tuple[int, Entry | None]:
    if t is None:
        return GPU_DISPATCH_TX_ERR_NULL_PTR, None
    if not _state_is_valid(t):
        return GPU_DISPATCH_TX_ERR_BAD_STATE, None
    if seq_id <= 0:
        return GPU_DISPATCH_TX_ERR_BAD_PARAM, None

    for entry in t.entries:
        if entry.seq_id == seq_id:
            return GPU_DISPATCH_TX_OK, entry

    return GPU_DISPATCH_TX_ERR_NOT_FOUND, None


def secure_dispatch_perf_overhead_plan() -> dict[str, int]:
    """Secure-local overhead plan (WS9-29/WS9-21 alignment)."""
    return {
        "warmup_iters": 128,
        "measure_iters": 2048,
        "secure_hooks_required": 1,
        "profiles_compared": 2,
        "min_dispatches_per_run": 256,
    }


def test_source_contains_iq1263_symbols() -> None:
    src = Path("src/gpu/dispatch_transcript.HC").read_text(encoding="utf-8")

    assert "class GPUDispatchTranscriptEntry" in src
    assert "class GPUDispatchTranscript" in src
    assert "I32 GPUDispatchTranscriptInit(" in src
    assert "I32 GPUDispatchTranscriptRecordChecked(" in src
    assert "I32 GPUDispatchTranscriptSnapshotChecked(" in src
    assert "I32 GPUDispatchTranscriptGetBySeqChecked(" in src
    assert "GPU_DISPATCH_TX_ERR_ALIGNMENT" in src


def test_monotonic_sequence_chain_and_snapshot_totals() -> None:
    status, t = tx_init(4, session_nonce=0x3456)
    assert status == GPU_DISPATCH_TX_OK
    assert t is not None

    s1, seq1, h1, c1 = tx_record(t, 3, 0, 1, 0x1000, 64, 1, 2, 100, 140, 0)
    assert s1 == GPU_DISPATCH_TX_OK
    assert seq1 == 1

    s2, seq2, h2, c2 = tx_record(t, 3, 1, 2, 0x1040, 64, 1, 2, 141, 191, 0)
    assert s2 == GPU_DISPATCH_TX_OK
    assert seq2 == 2

    assert h1 != h2
    assert c1 != c2
    assert t.count == 2
    assert t.head == 2
    assert t.total_descriptor_bytes == 128
    assert t.total_cycle_delta == (40 + 50)


def test_alignment_and_overflow_guards_fail_closed() -> None:
    status, t = tx_init(2, session_nonce=77)
    assert status == GPU_DISPATCH_TX_OK
    assert t is not None

    status, _, _, _ = tx_record(t, 0, 0, 0, 0x1008, 64, 1, 1, 10, 20, 0)
    assert status == GPU_DISPATCH_TX_ERR_ALIGNMENT

    status, _, _, _ = tx_record(t, 0, 0, 0, 0x1010, 18, 1, 1, 10, 20, 0)
    assert status == GPU_DISPATCH_TX_ERR_ALIGNMENT

    t.total_descriptor_bytes = GPU_DISPATCH_TX_I64_MAX - 8
    status, _, _, _ = tx_record(t, 0, 0, 0, 0x1020, 16, 1, 1, 10, 20, 0)
    assert status == GPU_DISPATCH_TX_ERR_OVERFLOW


def test_get_by_seq_and_tamper_evidence_chain_mismatch() -> None:
    status, t = tx_init(4, session_nonce=0x8BADF00D)
    assert status == GPU_DISPATCH_TX_OK
    assert t is not None

    s1, seq1, _, c1 = tx_record(t, 1, 0, 1, 0x2000, 64, 2, 2, 500, 540, 0)
    s2, seq2, _, c2 = tx_record(t, 1, 1, 2, 0x2040, 64, 2, 2, 541, 599, 0)
    assert s1 == GPU_DISPATCH_TX_OK
    assert s2 == GPU_DISPATCH_TX_OK

    status, e1 = tx_get_by_seq(t, seq1)
    assert status == GPU_DISPATCH_TX_OK
    assert e1 is not None

    status, e2 = tx_get_by_seq(t, seq2)
    assert status == GPU_DISPATCH_TX_OK
    assert e2 is not None

    recomputed_c2 = _mix64(c1, _to_i64(e2.entry_hash ^ e2.seq_id))
    assert recomputed_c2 == c2

    # Hardening check: any tamper in descriptor bytes changes derived chain.
    tampered_hash = _entry_hash(
        t.session_nonce,
        e2.seq_id,
        e2.queue_id,
        e2.queue_depth_before,
        e2.queue_depth_after,
        e2.descriptor_addr,
        e2.descriptor_bytes + 16,
        e2.descriptor_type,
        e2.descriptor_op,
        e2.cycle_start,
        e2.cycle_end,
        e2.status_code,
    )
    tampered_c2 = _mix64(c1, _to_i64(tampered_hash ^ e2.seq_id))
    assert tampered_c2 != c2


def test_secure_dispatch_perf_overhead_plan_shape() -> None:
    plan = secure_dispatch_perf_overhead_plan()
    assert plan["warmup_iters"] >= 64
    assert plan["measure_iters"] >= 1024
    assert plan["secure_hooks_required"] == 1
    assert plan["profiles_compared"] == 2
    assert plan["min_dispatches_per_run"] >= 128


if __name__ == "__main__":
    test_source_contains_iq1263_symbols()
    test_monotonic_sequence_chain_and_snapshot_totals()
    test_alignment_and_overflow_guards_fail_closed()
    test_get_by_seq_and_tamper_evidence_chain_mismatch()
    test_secure_dispatch_perf_overhead_plan_shape()
    print("ok")
