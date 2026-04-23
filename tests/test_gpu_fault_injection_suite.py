#!/usr/bin/env python3
"""Fault-injection suite for IQ-1257 (timeout/bad-map/malformed-descriptor)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

GPU_POLICY_OK = 0
GPU_POLICY_ERR_IOMMU_GUARD = 4
GPU_POLICY_ERR_BOOK_GUARD = 5

GPU_CMD_VERIFY_OK = 0
GPU_CMD_VERIFY_ERR_BAD_TYPE = 3
GPU_CMD_VERIFY_ERR_BAD_OP = 4
GPU_CMD_VERIFY_ERR_BAD_RANGE = 5
GPU_CMD_VERIFY_ERR_BAD_ALIGNMENT = 9

GPU_CMD_VERIFY_REASON_ALLOW = 0
GPU_CMD_VERIFY_REASON_BAD_TYPE = 3
GPU_CMD_VERIFY_REASON_BAD_OP = 4
GPU_CMD_VERIFY_REASON_BAD_BOUNDS = 5
GPU_CMD_VERIFY_REASON_BAD_ALIGNMENT = 9

GPU_CMD_DESC_TYPE_QMATMUL = 1
GPU_CMD_DESC_TYPE_ATTN = 2
GPU_CMD_DESC_TYPE_ELEMENT = 3

GPU_CMD_OP_COPY_IN = 1
GPU_CMD_OP_COMPUTE = 2
GPU_CMD_OP_COPY_OUT = 3

GPU_CMD_VERIFY_DMA_ALIGNMENT_BYTES = 16

BOT_GPU_BRIDGE_OK = 0
BOT_GPU_BRIDGE_ERR_EVENT_OP = 5

BOT_GPU_EVENT_DISPATCH = 3
BOT_GPU_DISPATCH_SUBMIT = 1
BOT_GPU_DISPATCH_TIMEOUT = 3

FAULT_SUITE_OK = 0
FAULT_SUITE_ERR_POLICY = 1
FAULT_SUITE_ERR_VERIFY = 2
FAULT_SUITE_ERR_TIMEOUT = 3


@dataclass
class GPUCommandDescriptor:
    desc_type: int
    desc_op: int
    src_offset_bytes: int
    dst_offset_bytes: int
    byte_count: int
    flags: int


@dataclass
class Event:
    seq_id: int = 0
    event_type: int = 0
    event_op: int = 0
    arg0: int = 0
    arg1: int = 0
    arg2: int = 0
    arg3: int = 0


@dataclass
class Bridge:
    capacity: int
    events: list[Event] = field(default_factory=list)
    count: int = 0
    head: int = 0
    next_seq_id: int = 1

    def __post_init__(self) -> None:
        if not self.events:
            self.events = [Event() for _ in range(self.capacity)]


def gpu_policy_allow_dispatch_checked(
    iommu_enabled: int,
    bot_dma_log_enabled: int,
    bot_mmio_log_enabled: int,
    bot_dispatch_log_enabled: int,
) -> int:
    if iommu_enabled != 1:
        return GPU_POLICY_ERR_IOMMU_GUARD
    if bot_dma_log_enabled != 1 or bot_mmio_log_enabled != 1 or bot_dispatch_log_enabled != 1:
        return GPU_POLICY_ERR_BOOK_GUARD
    return GPU_POLICY_OK


def _is_aligned_pow2(value: int, alignment: int) -> bool:
    if alignment <= 0:
        return False
    return (value & (alignment - 1)) == 0


def verify_descriptor_checked(
    desc: GPUCommandDescriptor,
    region_nbytes: int,
    max_cmd_nbytes: int,
) -> tuple[int, int]:
    if desc.desc_type not in (GPU_CMD_DESC_TYPE_QMATMUL, GPU_CMD_DESC_TYPE_ATTN, GPU_CMD_DESC_TYPE_ELEMENT):
        return GPU_CMD_VERIFY_ERR_BAD_TYPE, GPU_CMD_VERIFY_REASON_BAD_TYPE

    if desc.desc_op not in (GPU_CMD_OP_COPY_IN, GPU_CMD_OP_COMPUTE, GPU_CMD_OP_COPY_OUT):
        return GPU_CMD_VERIFY_ERR_BAD_OP, GPU_CMD_VERIFY_REASON_BAD_OP

    if desc.desc_type == GPU_CMD_DESC_TYPE_ELEMENT and desc.desc_op != GPU_CMD_OP_COMPUTE:
        return GPU_CMD_VERIFY_ERR_BAD_OP, GPU_CMD_VERIFY_REASON_BAD_OP

    if desc.src_offset_bytes < 0 or desc.dst_offset_bytes < 0 or desc.byte_count <= 0:
        return GPU_CMD_VERIFY_ERR_BAD_RANGE, GPU_CMD_VERIFY_REASON_BAD_BOUNDS

    if desc.byte_count > max_cmd_nbytes:
        return GPU_CMD_VERIFY_ERR_BAD_RANGE, GPU_CMD_VERIFY_REASON_BAD_BOUNDS

    if not _is_aligned_pow2(desc.src_offset_bytes, GPU_CMD_VERIFY_DMA_ALIGNMENT_BYTES):
        return GPU_CMD_VERIFY_ERR_BAD_ALIGNMENT, GPU_CMD_VERIFY_REASON_BAD_ALIGNMENT
    if not _is_aligned_pow2(desc.dst_offset_bytes, GPU_CMD_VERIFY_DMA_ALIGNMENT_BYTES):
        return GPU_CMD_VERIFY_ERR_BAD_ALIGNMENT, GPU_CMD_VERIFY_REASON_BAD_ALIGNMENT
    if not _is_aligned_pow2(desc.byte_count, GPU_CMD_VERIFY_DMA_ALIGNMENT_BYTES):
        return GPU_CMD_VERIFY_ERR_BAD_ALIGNMENT, GPU_CMD_VERIFY_REASON_BAD_ALIGNMENT

    src_end = desc.src_offset_bytes + desc.byte_count
    dst_end = desc.dst_offset_bytes + desc.byte_count
    if src_end > region_nbytes or dst_end > region_nbytes:
        return GPU_CMD_VERIFY_ERR_BAD_RANGE, GPU_CMD_VERIFY_REASON_BAD_BOUNDS

    return GPU_CMD_VERIFY_OK, GPU_CMD_VERIFY_REASON_ALLOW


def _bridge_append_dispatch(bridge: Bridge, dispatch_op: int, descriptor_addr: int, descriptor_bytes: int) -> tuple[int, int]:
    if dispatch_op not in (BOT_GPU_DISPATCH_SUBMIT, BOT_GPU_DISPATCH_TIMEOUT):
        return BOT_GPU_BRIDGE_ERR_EVENT_OP, 0

    slot = bridge.events[bridge.head]
    slot.seq_id = bridge.next_seq_id
    slot.event_type = BOT_GPU_EVENT_DISPATCH
    slot.event_op = dispatch_op
    slot.arg1 = descriptor_addr
    slot.arg2 = descriptor_bytes

    seq = bridge.next_seq_id
    bridge.next_seq_id += 1
    bridge.head += 1
    if bridge.head >= bridge.capacity:
        bridge.head = 0
    if bridge.count < bridge.capacity:
        bridge.count += 1

    return BOT_GPU_BRIDGE_OK, seq


def run_fault_vector(name: str) -> dict[str, int]:
    policy_status = gpu_policy_allow_dispatch_checked(1, 1, 1, 1)
    if policy_status != GPU_POLICY_OK:
        return {"status": FAULT_SUITE_ERR_POLICY, "policy": policy_status, "verify": -1, "reason": -1, "events": 0}

    bridge = Bridge(capacity=4)

    if name == "malformed-descriptor":
        desc = GPUCommandDescriptor(99, GPU_CMD_OP_COMPUTE, 0, 0, 64, 0)
    elif name == "bad-map":
        desc = GPUCommandDescriptor(GPU_CMD_DESC_TYPE_QMATMUL, GPU_CMD_OP_COPY_IN, 4080, 0, 64, 0)
    elif name == "timeout":
        desc = GPUCommandDescriptor(GPU_CMD_DESC_TYPE_ATTN, GPU_CMD_OP_COMPUTE, 0, 0, 64, 0)
    else:
        raise ValueError(name)

    verify_status, reason = verify_descriptor_checked(desc, region_nbytes=4096, max_cmd_nbytes=256)
    if verify_status != GPU_CMD_VERIFY_OK:
        return {
            "status": FAULT_SUITE_ERR_VERIFY,
            "policy": policy_status,
            "verify": verify_status,
            "reason": reason,
            "events": bridge.count,
        }

    status, _ = _bridge_append_dispatch(bridge, BOT_GPU_DISPATCH_SUBMIT, 0x9000, 64)
    assert status == BOT_GPU_BRIDGE_OK

    if name == "timeout":
        status, _ = _bridge_append_dispatch(bridge, BOT_GPU_DISPATCH_TIMEOUT, 0x9000, 64)
        assert status == BOT_GPU_BRIDGE_OK
        return {
            "status": FAULT_SUITE_ERR_TIMEOUT,
            "policy": policy_status,
            "verify": verify_status,
            "reason": reason,
            "events": bridge.count,
        }

    return {
        "status": FAULT_SUITE_OK,
        "policy": policy_status,
        "verify": verify_status,
        "reason": reason,
        "events": bridge.count,
    }


def secure_perf_overhead_plan() -> dict[str, int]:
    """Return secure-local perf plan shape for WS9 reporting.

    Plan model:
    - Measure verifier + Book-of-Truth hooks enabled in secure-local.
    - Compare against dev-local with same descriptors and token counts.
    - Export integer-only microsecond/counter deltas.
    """

    return {
        "warmup_iters": 64,
        "measure_iters": 1024,
        "secure_hooks_required": 1,
        "compare_profiles": 2,
        "min_descriptor_count": 128,
    }


def test_source_contains_iq1257_symbols() -> None:
    verify_src = Path("src/gpu/command_verify.HC").read_text(encoding="utf-8")
    bridge_src = Path("src/gpu/book_of_truth_bridge.HC").read_text(encoding="utf-8")
    policy_src = Path("src/gpu/policy.HC").read_text(encoding="utf-8")

    assert "I32 GPUCommandVerifyDescriptorChecked(" in verify_src
    assert "GPU_CMD_VERIFY_ERR_BAD_ALIGNMENT" in verify_src
    assert "I32 BOTGPUBridgeRecordDispatch(" in bridge_src
    assert "BOT_GPU_DISPATCH_TIMEOUT" in bridge_src
    assert "I32 GPUPolicyAllowDispatchChecked(" in policy_src
    assert "GPU_POLICY_ERR_IOMMU_GUARD" in policy_src


def test_fault_vector_malformed_descriptor_rejected() -> None:
    out = run_fault_vector("malformed-descriptor")
    assert out["status"] == FAULT_SUITE_ERR_VERIFY
    assert out["verify"] == GPU_CMD_VERIFY_ERR_BAD_TYPE
    assert out["reason"] == GPU_CMD_VERIFY_REASON_BAD_TYPE
    assert out["events"] == 0


def test_fault_vector_bad_map_rejected_before_dispatch() -> None:
    out = run_fault_vector("bad-map")
    assert out["status"] == FAULT_SUITE_ERR_VERIFY
    assert out["verify"] == GPU_CMD_VERIFY_ERR_BAD_RANGE
    assert out["reason"] == GPU_CMD_VERIFY_REASON_BAD_BOUNDS
    assert out["events"] == 0


def test_fault_vector_timeout_logged_as_fault() -> None:
    out = run_fault_vector("timeout")
    assert out["status"] == FAULT_SUITE_ERR_TIMEOUT
    assert out["verify"] == GPU_CMD_VERIFY_OK
    assert out["reason"] == GPU_CMD_VERIFY_REASON_ALLOW
    assert out["events"] == 2


def test_hardening_check_requires_iommu_and_all_book_hooks() -> None:
    assert gpu_policy_allow_dispatch_checked(0, 1, 1, 1) == GPU_POLICY_ERR_IOMMU_GUARD
    assert gpu_policy_allow_dispatch_checked(1, 1, 0, 1) == GPU_POLICY_ERR_BOOK_GUARD
    assert gpu_policy_allow_dispatch_checked(1, 1, 1, 1) == GPU_POLICY_OK


def test_perf_overhead_measurement_plan_shape() -> None:
    plan = secure_perf_overhead_plan()
    assert plan["secure_hooks_required"] == 1
    assert plan["compare_profiles"] == 2
    assert plan["warmup_iters"] > 0
    assert plan["measure_iters"] > plan["warmup_iters"]
    assert plan["min_descriptor_count"] >= 128


if __name__ == "__main__":
    test_source_contains_iq1257_symbols()
    test_fault_vector_malformed_descriptor_rejected()
    test_fault_vector_bad_map_rejected_before_dispatch()
    test_fault_vector_timeout_logged_as_fault()
    test_hardening_check_requires_iommu_and_all_book_hooks()
    test_perf_overhead_measurement_plan_shape()
    print("ok")
