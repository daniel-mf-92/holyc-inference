#!/usr/bin/env python3
"""Harness for IQ-1255 Book-of-Truth GPU event bridge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BOT_GPU_BRIDGE_OK = 0
BOT_GPU_BRIDGE_ERR_NULL_PTR = 1
BOT_GPU_BRIDGE_ERR_BAD_PARAM = 2
BOT_GPU_BRIDGE_ERR_CAPACITY = 3
BOT_GPU_BRIDGE_ERR_EVENT_TYPE = 4
BOT_GPU_BRIDGE_ERR_EVENT_OP = 5
BOT_GPU_BRIDGE_ERR_SEQUENCE_OVERFLOW = 6

BOT_GPU_EVENT_DMA = 1
BOT_GPU_EVENT_MMIO = 2
BOT_GPU_EVENT_DISPATCH = 3

BOT_GPU_DMA_MAP = 1
BOT_GPU_DMA_UPDATE = 2
BOT_GPU_DMA_UNMAP = 3

BOT_GPU_MMIO_WRITE = 1

BOT_GPU_DISPATCH_SUBMIT = 1
BOT_GPU_DISPATCH_COMPLETE = 2
BOT_GPU_DISPATCH_TIMEOUT = 3


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
    events: list[Event]
    capacity: int
    count: int = 0
    head: int = 0
    next_seq_id: int = 1


def _event_type_is_valid(event_type: int) -> bool:
    return event_type in (BOT_GPU_EVENT_DMA, BOT_GPU_EVENT_MMIO, BOT_GPU_EVENT_DISPATCH)


def _event_op_is_valid(event_type: int, event_op: int) -> bool:
    if event_type == BOT_GPU_EVENT_DMA:
        return event_op in (BOT_GPU_DMA_MAP, BOT_GPU_DMA_UPDATE, BOT_GPU_DMA_UNMAP)
    if event_type == BOT_GPU_EVENT_MMIO:
        return event_op == BOT_GPU_MMIO_WRITE
    if event_type == BOT_GPU_EVENT_DISPATCH:
        return event_op in (
            BOT_GPU_DISPATCH_SUBMIT,
            BOT_GPU_DISPATCH_COMPLETE,
            BOT_GPU_DISPATCH_TIMEOUT,
        )
    return False


def append_checked(
    bridge: Bridge,
    event_type: int,
    event_op: int,
    arg0: int,
    arg1: int,
    arg2: int,
    arg3: int,
) -> tuple[int, int]:
    if bridge is None:
        return BOT_GPU_BRIDGE_ERR_NULL_PTR, 0

    if bridge.capacity <= 0:
        return BOT_GPU_BRIDGE_ERR_CAPACITY, 0

    if not (0 <= bridge.count <= bridge.capacity and 0 <= bridge.head < bridge.capacity):
        return BOT_GPU_BRIDGE_ERR_BAD_PARAM, 0

    if not _event_type_is_valid(event_type):
        return BOT_GPU_BRIDGE_ERR_EVENT_TYPE, 0

    if not _event_op_is_valid(event_type, event_op):
        return BOT_GPU_BRIDGE_ERR_EVENT_OP, 0

    if bridge.next_seq_id <= 0:
        return BOT_GPU_BRIDGE_ERR_SEQUENCE_OVERFLOW, 0

    slot = bridge.events[bridge.head]
    slot.seq_id = bridge.next_seq_id
    slot.event_type = event_type
    slot.event_op = event_op
    slot.arg0 = arg0
    slot.arg1 = arg1
    slot.arg2 = arg2
    slot.arg3 = arg3

    out_seq_id = bridge.next_seq_id
    bridge.next_seq_id += 1
    bridge.head += 1
    if bridge.head >= bridge.capacity:
        bridge.head = 0

    if bridge.count < bridge.capacity:
        bridge.count += 1

    return BOT_GPU_BRIDGE_OK, out_seq_id


def record_dma(
    bridge: Bridge,
    dma_op: int,
    lease_id: int,
    phys_addr: int,
    nbytes: int,
    iommu_domain: int,
) -> tuple[int, int]:
    return append_checked(
        bridge,
        BOT_GPU_EVENT_DMA,
        dma_op,
        lease_id,
        phys_addr,
        nbytes,
        iommu_domain,
    )


def record_mmio_write(
    bridge: Bridge,
    bar_index: int,
    reg_offset: int,
    value: int,
    width_bytes: int,
) -> tuple[int, int]:
    return append_checked(
        bridge,
        BOT_GPU_EVENT_MMIO,
        BOT_GPU_MMIO_WRITE,
        bar_index,
        reg_offset,
        value,
        width_bytes,
    )


def record_dispatch(
    bridge: Bridge,
    dispatch_op: int,
    queue_id: int,
    descriptor_addr: int,
    descriptor_bytes: int,
    fence_id: int,
) -> tuple[int, int]:
    return append_checked(
        bridge,
        BOT_GPU_EVENT_DISPATCH,
        dispatch_op,
        queue_id,
        descriptor_addr,
        descriptor_bytes,
        fence_id,
    )


def test_source_contains_iq1255_symbols() -> None:
    src = Path("src/gpu/book_of_truth_bridge.HC").read_text(encoding="utf-8")

    assert "I32 BOTGPUBridgeInit(" in src
    assert "I32 BOTGPUBridgeAppendChecked(" in src
    assert "I32 BOTGPUBridgeRecordDMA(" in src
    assert "I32 BOTGPUBridgeRecordMMIOWrite(" in src
    assert "I32 BOTGPUBridgeRecordDispatch(" in src
    assert "BOT_GPU_EVENT_DMA" in src
    assert "BOT_GPU_EVENT_MMIO" in src
    assert "BOT_GPU_EVENT_DISPATCH" in src


def test_dma_mmio_dispatch_events_append_with_monotonic_sequence() -> None:
    bridge = Bridge(events=[Event() for _ in range(4)], capacity=4)

    status, seq1 = record_dma(bridge, BOT_GPU_DMA_MAP, 11, 0x1000, 4096, 7)
    assert status == BOT_GPU_BRIDGE_OK
    assert seq1 == 1

    status, seq2 = record_mmio_write(bridge, 2, 0x88, 0x1234, 4)
    assert status == BOT_GPU_BRIDGE_OK
    assert seq2 == 2

    status, seq3 = record_dispatch(bridge, BOT_GPU_DISPATCH_SUBMIT, 5, 0x9000, 128, 42)
    assert status == BOT_GPU_BRIDGE_OK
    assert seq3 == 3

    assert bridge.count == 3
    assert bridge.head == 3
    assert bridge.events[0].event_type == BOT_GPU_EVENT_DMA
    assert bridge.events[1].event_type == BOT_GPU_EVENT_MMIO
    assert bridge.events[2].event_type == BOT_GPU_EVENT_DISPATCH


def test_ring_overwrite_keeps_capacity_and_wraps_head() -> None:
    bridge = Bridge(events=[Event() for _ in range(2)], capacity=2)

    status, _ = record_dma(bridge, BOT_GPU_DMA_MAP, 1, 0x1000, 64, 1)
    assert status == BOT_GPU_BRIDGE_OK
    status, _ = record_dma(bridge, BOT_GPU_DMA_UPDATE, 1, 0x2000, 64, 1)
    assert status == BOT_GPU_BRIDGE_OK
    status, seq3 = record_dispatch(bridge, BOT_GPU_DISPATCH_COMPLETE, 0, 0, 0, 99)
    assert status == BOT_GPU_BRIDGE_OK

    assert seq3 == 3
    assert bridge.count == 2
    assert bridge.head == 1
    assert bridge.events[0].seq_id == 3
    assert bridge.events[0].event_type == BOT_GPU_EVENT_DISPATCH


def test_invalid_event_type_and_op_are_rejected() -> None:
    bridge = Bridge(events=[Event() for _ in range(3)], capacity=3)

    status, seq = append_checked(bridge, 99, 1, 0, 0, 0, 0)
    assert status == BOT_GPU_BRIDGE_ERR_EVENT_TYPE
    assert seq == 0

    status, seq = record_dispatch(bridge, 77, 0, 0, 0, 0)
    assert status == BOT_GPU_BRIDGE_ERR_EVENT_OP
    assert seq == 0


def test_bad_state_and_sequence_overflow_are_rejected() -> None:
    bridge = Bridge(events=[Event() for _ in range(2)], capacity=2)

    bridge.count = 3
    status, seq = record_mmio_write(bridge, 0, 0, 0, 4)
    assert status == BOT_GPU_BRIDGE_ERR_BAD_PARAM
    assert seq == 0

    bridge.count = 0
    bridge.head = 0
    bridge.next_seq_id = 0
    status, seq = record_dma(bridge, BOT_GPU_DMA_UNMAP, 7, 0x3000, 32, 1)
    assert status == BOT_GPU_BRIDGE_ERR_SEQUENCE_OVERFLOW
    assert seq == 0


if __name__ == "__main__":
    test_source_contains_iq1255_symbols()
    test_dma_mmio_dispatch_events_append_with_monotonic_sequence()
    test_ring_overwrite_keeps_capacity_and_wraps_head()
    test_invalid_event_type_and_op_are_rejected()
    test_bad_state_and_sequence_overflow_are_rejected()
    print("ok")

