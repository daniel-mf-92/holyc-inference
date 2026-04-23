#!/usr/bin/env python3
"""Parity harness for GPU command-stream preflight verifier (IQ-1256)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import time

GPU_CMD_VERIFY_OK = 0
GPU_CMD_VERIFY_ERR_NULL_PTR = 1
GPU_CMD_VERIFY_ERR_BAD_PARAM = 2
GPU_CMD_VERIFY_ERR_BAD_TYPE = 3
GPU_CMD_VERIFY_ERR_BAD_OP = 4
GPU_CMD_VERIFY_ERR_BAD_RANGE = 5
GPU_CMD_VERIFY_ERR_BAD_FLAGS = 6
GPU_CMD_VERIFY_ERR_OVERFLOW = 7
GPU_CMD_VERIFY_ERR_BAD_COMBO = 8
GPU_CMD_VERIFY_ERR_BAD_ALIGNMENT = 9
GPU_CMD_VERIFY_ERR_STREAM_BUDGET = 10

GPU_CMD_VERIFY_REASON_ALLOW = 0
GPU_CMD_VERIFY_REASON_NULL_DESC = 1
GPU_CMD_VERIFY_REASON_BAD_GEOMETRY = 2
GPU_CMD_VERIFY_REASON_BAD_TYPE = 3
GPU_CMD_VERIFY_REASON_BAD_OP = 4
GPU_CMD_VERIFY_REASON_BAD_BOUNDS = 5
GPU_CMD_VERIFY_REASON_BAD_FLAGS = 6
GPU_CMD_VERIFY_REASON_OVERFLOW = 7
GPU_CMD_VERIFY_REASON_BAD_COMBO = 8
GPU_CMD_VERIFY_REASON_BAD_ALIGNMENT = 9
GPU_CMD_VERIFY_REASON_STREAM_BUDGET = 10

GPU_CMD_DESC_TYPE_QMATMUL = 1
GPU_CMD_DESC_TYPE_ATTN = 2
GPU_CMD_DESC_TYPE_ELEMENT = 3

GPU_CMD_OP_COPY_IN = 1
GPU_CMD_OP_COMPUTE = 2
GPU_CMD_OP_COPY_OUT = 3

GPU_CMD_VERIFY_DMA_ALIGNMENT_BYTES = 16

I64_MAX = (1 << 63) - 1


@dataclass
class GPUCommandDescriptor:
    desc_type: int
    desc_op: int
    src_offset_bytes: int
    dst_offset_bytes: int
    byte_count: int
    flags: int


def desc_type_valid(desc_type: int) -> bool:
    return desc_type in {
        GPU_CMD_DESC_TYPE_QMATMUL,
        GPU_CMD_DESC_TYPE_ATTN,
        GPU_CMD_DESC_TYPE_ELEMENT,
    }


def desc_op_valid(desc_op: int) -> bool:
    return desc_op in {GPU_CMD_OP_COPY_IN, GPU_CMD_OP_COMPUTE, GPU_CMD_OP_COPY_OUT}


def desc_type_op_pair_valid(desc_type: int, desc_op: int) -> bool:
    if desc_type == GPU_CMD_DESC_TYPE_ELEMENT:
        return desc_op == GPU_CMD_OP_COMPUTE
    return desc_op in {GPU_CMD_OP_COPY_IN, GPU_CMD_OP_COMPUTE, GPU_CMD_OP_COPY_OUT}


def aligned_pow2(value: int, alignment: int) -> bool:
    if alignment <= 0:
        return False
    return (value & (alignment - 1)) == 0


def add_non_negative_i64_checked(a: int, b: int) -> int | None:
    if a < 0 or b < 0:
        return None
    if a > I64_MAX - b:
        return None
    return a + b


def verify_descriptor_checked(
    desc: GPUCommandDescriptor | None,
    region_nbytes: int,
    max_cmd_nbytes: int,
    allowed_flags_mask: int,
    out_reason_code: list[int] | None,
) -> int:
    if out_reason_code is None:
        return GPU_CMD_VERIFY_ERR_NULL_PTR

    out_reason_code[0] = GPU_CMD_VERIFY_REASON_BAD_GEOMETRY

    if desc is None:
        out_reason_code[0] = GPU_CMD_VERIFY_REASON_NULL_DESC
        return GPU_CMD_VERIFY_ERR_NULL_PTR

    if (
        region_nbytes <= 0
        or max_cmd_nbytes <= 0
        or max_cmd_nbytes > region_nbytes
        or allowed_flags_mask < 0
    ):
        return GPU_CMD_VERIFY_ERR_BAD_PARAM

    if not desc_type_valid(desc.desc_type):
        out_reason_code[0] = GPU_CMD_VERIFY_REASON_BAD_TYPE
        return GPU_CMD_VERIFY_ERR_BAD_TYPE

    if not desc_op_valid(desc.desc_op):
        out_reason_code[0] = GPU_CMD_VERIFY_REASON_BAD_OP
        return GPU_CMD_VERIFY_ERR_BAD_OP

    if not desc_type_op_pair_valid(desc.desc_type, desc.desc_op):
        out_reason_code[0] = GPU_CMD_VERIFY_REASON_BAD_COMBO
        return GPU_CMD_VERIFY_ERR_BAD_COMBO

    if desc.flags < 0 or (desc.flags & (~allowed_flags_mask)) != 0:
        out_reason_code[0] = GPU_CMD_VERIFY_REASON_BAD_FLAGS
        return GPU_CMD_VERIFY_ERR_BAD_FLAGS

    if (
        desc.src_offset_bytes < 0
        or desc.dst_offset_bytes < 0
        or desc.byte_count <= 0
        or desc.byte_count > max_cmd_nbytes
    ):
        out_reason_code[0] = GPU_CMD_VERIFY_REASON_BAD_BOUNDS
        return GPU_CMD_VERIFY_ERR_BAD_RANGE

    if (
        not aligned_pow2(desc.src_offset_bytes, GPU_CMD_VERIFY_DMA_ALIGNMENT_BYTES)
        or not aligned_pow2(desc.dst_offset_bytes, GPU_CMD_VERIFY_DMA_ALIGNMENT_BYTES)
        or not aligned_pow2(desc.byte_count, GPU_CMD_VERIFY_DMA_ALIGNMENT_BYTES)
    ):
        out_reason_code[0] = GPU_CMD_VERIFY_REASON_BAD_ALIGNMENT
        return GPU_CMD_VERIFY_ERR_BAD_ALIGNMENT

    src_end = add_non_negative_i64_checked(desc.src_offset_bytes, desc.byte_count)
    dst_end = add_non_negative_i64_checked(desc.dst_offset_bytes, desc.byte_count)
    if src_end is None or dst_end is None:
        out_reason_code[0] = GPU_CMD_VERIFY_REASON_OVERFLOW
        return GPU_CMD_VERIFY_ERR_OVERFLOW

    if src_end > region_nbytes or dst_end > region_nbytes:
        out_reason_code[0] = GPU_CMD_VERIFY_REASON_BAD_BOUNDS
        return GPU_CMD_VERIFY_ERR_BAD_RANGE

    out_reason_code[0] = GPU_CMD_VERIFY_REASON_ALLOW
    return GPU_CMD_VERIFY_OK


def verify_stream_checked(
    descs: list[GPUCommandDescriptor] | None,
    desc_count: int,
    region_nbytes: int,
    max_cmd_nbytes: int,
    allowed_flags_mask: int,
    out_bad_index: list[int] | None,
    out_reason_code: list[int] | None,
) -> int:
    if descs is None or out_bad_index is None or out_reason_code is None:
        return GPU_CMD_VERIFY_ERR_NULL_PTR

    out_bad_index[0] = -1
    out_reason_code[0] = GPU_CMD_VERIFY_REASON_BAD_GEOMETRY

    if desc_count <= 0:
        return GPU_CMD_VERIFY_ERR_BAD_PARAM

    stream_total_bytes = 0

    for i in range(desc_count):
        reason = [GPU_CMD_VERIFY_REASON_BAD_GEOMETRY]
        status = verify_descriptor_checked(
            descs[i], region_nbytes, max_cmd_nbytes, allowed_flags_mask, reason
        )
        if status != GPU_CMD_VERIFY_OK:
            out_bad_index[0] = i
            out_reason_code[0] = reason[0]
            return status

        maybe_total = add_non_negative_i64_checked(stream_total_bytes, descs[i].byte_count)
        if maybe_total is None:
            out_bad_index[0] = i
            out_reason_code[0] = GPU_CMD_VERIFY_REASON_OVERFLOW
            return GPU_CMD_VERIFY_ERR_OVERFLOW
        stream_total_bytes = maybe_total

        if stream_total_bytes > region_nbytes:
            out_bad_index[0] = i
            out_reason_code[0] = GPU_CMD_VERIFY_REASON_STREAM_BUDGET
            return GPU_CMD_VERIFY_ERR_STREAM_BUDGET

    out_reason_code[0] = GPU_CMD_VERIFY_REASON_ALLOW
    return GPU_CMD_VERIFY_OK


def test_source_contains_iq1256_functions_and_contracts() -> None:
    source = Path("src/gpu/command_verify.HC").read_text(encoding="utf-8")

    sig_desc = "I32 GPUCommandVerifyDescriptorChecked(GPUCommandDescriptor *desc,"
    assert sig_desc in source
    desc_body = source.split(sig_desc, 1)[1].split(
        "I32 GPUCommandVerifyStreamChecked(", 1
    )[0]
    assert "if (!GPUCommandDescTypeIsValid(desc->desc_type))" in desc_body
    assert "if (!GPUCommandDescOpIsValid(desc->desc_op))" in desc_body
    assert "if (desc->flags < 0 || (desc->flags & (~allowed_flags_mask)) != 0)" in desc_body
    assert "if (!GPUCommandDescTypeOpPairIsValid(desc->desc_type," in desc_body
    assert "GPU_CMD_VERIFY_DMA_ALIGNMENT_BYTES" in desc_body
    assert "if (src_end > region_nbytes || dst_end > region_nbytes)" in desc_body

    sig_stream = "I32 GPUCommandVerifyStreamChecked(GPUCommandDescriptor *descs,"
    assert sig_stream in source
    stream_body = source.split(sig_stream, 1)[1]
    assert "*out_bad_index = -1;" in stream_body
    assert "status = GPUCommandVerifyDescriptorChecked(&descs[i]," in stream_body
    assert "if (stream_total_bytes > region_nbytes)" in stream_body


def test_descriptor_and_stream_error_paths() -> None:
    reason = [999]
    status = verify_descriptor_checked(None, 4096, 256, 0x7, reason)
    assert status == GPU_CMD_VERIFY_ERR_NULL_PTR
    assert reason == [GPU_CMD_VERIFY_REASON_NULL_DESC]

    reason = [999]
    bad = GPUCommandDescriptor(99, GPU_CMD_OP_COPY_IN, 0, 0, 64, 0)
    status = verify_descriptor_checked(bad, 4096, 256, 0x7, reason)
    assert status == GPU_CMD_VERIFY_ERR_BAD_TYPE
    assert reason == [GPU_CMD_VERIFY_REASON_BAD_TYPE]

    reason = [999]
    bad = GPUCommandDescriptor(GPU_CMD_DESC_TYPE_QMATMUL, 42, 0, 0, 64, 0)
    status = verify_descriptor_checked(bad, 4096, 256, 0x7, reason)
    assert status == GPU_CMD_VERIFY_ERR_BAD_OP
    assert reason == [GPU_CMD_VERIFY_REASON_BAD_OP]

    reason = [999]
    bad = GPUCommandDescriptor(GPU_CMD_DESC_TYPE_QMATMUL, GPU_CMD_OP_COMPUTE, 0, 0, 64, 0x80)
    status = verify_descriptor_checked(bad, 4096, 256, 0x7, reason)
    assert status == GPU_CMD_VERIFY_ERR_BAD_FLAGS
    assert reason == [GPU_CMD_VERIFY_REASON_BAD_FLAGS]

    reason = [999]
    bad = GPUCommandDescriptor(GPU_CMD_DESC_TYPE_QMATMUL, GPU_CMD_OP_COPY_IN, 4000, 0, 128, 0)
    status = verify_descriptor_checked(bad, 4096, 256, 0x7, reason)
    assert status == GPU_CMD_VERIFY_ERR_BAD_RANGE
    assert reason == [GPU_CMD_VERIFY_REASON_BAD_BOUNDS]

    reason = [999]
    bad = GPUCommandDescriptor(GPU_CMD_DESC_TYPE_ELEMENT, GPU_CMD_OP_COPY_OUT, 0, 0, 64, 0)
    status = verify_descriptor_checked(bad, 4096, 256, 0x7, reason)
    assert status == GPU_CMD_VERIFY_ERR_BAD_COMBO
    assert reason == [GPU_CMD_VERIFY_REASON_BAD_COMBO]

    reason = [999]
    bad = GPUCommandDescriptor(GPU_CMD_DESC_TYPE_QMATMUL, GPU_CMD_OP_COPY_IN, 8, 0, 64, 0)
    status = verify_descriptor_checked(bad, 4096, 256, 0x7, reason)
    assert status == GPU_CMD_VERIFY_ERR_BAD_ALIGNMENT
    assert reason == [GPU_CMD_VERIFY_REASON_BAD_ALIGNMENT]

    reason = [999]
    bad = GPUCommandDescriptor(
        GPU_CMD_DESC_TYPE_QMATMUL, GPU_CMD_OP_COPY_IN, I64_MAX - 15, 0, 32, 0
    )
    status = verify_descriptor_checked(bad, I64_MAX, I64_MAX, 0x7, reason)
    assert status == GPU_CMD_VERIFY_ERR_OVERFLOW
    assert reason == [GPU_CMD_VERIFY_REASON_OVERFLOW]

    bad_index = [777]
    reason = [888]
    stream = [
        GPUCommandDescriptor(GPU_CMD_DESC_TYPE_QMATMUL, GPU_CMD_OP_COPY_IN, 0, 0, 64, 0),
        GPUCommandDescriptor(GPU_CMD_DESC_TYPE_ATTN, GPU_CMD_OP_COMPUTE, 128, 128, 64, 0),
        GPUCommandDescriptor(GPU_CMD_DESC_TYPE_ELEMENT, 77, 256, 256, 64, 0),
    ]
    status = verify_stream_checked(stream, len(stream), 4096, 256, 0x7, bad_index, reason)
    assert status == GPU_CMD_VERIFY_ERR_BAD_OP
    assert bad_index == [2]
    assert reason == [GPU_CMD_VERIFY_REASON_BAD_OP]

    bad_index = [777]
    reason = [888]
    stream = [
        GPUCommandDescriptor(GPU_CMD_DESC_TYPE_QMATMUL, GPU_CMD_OP_COPY_IN, 0, 0, 2048, 0),
        GPUCommandDescriptor(GPU_CMD_DESC_TYPE_QMATMUL, GPU_CMD_OP_COPY_OUT, 2048, 2048, 2048, 0),
        GPUCommandDescriptor(GPU_CMD_DESC_TYPE_ATTN, GPU_CMD_OP_COMPUTE, 0, 0, 16, 0),
    ]
    status = verify_stream_checked(stream, len(stream), 4096, 4096, 0x7, bad_index, reason)
    assert status == GPU_CMD_VERIFY_ERR_STREAM_BUDGET
    assert bad_index == [2]
    assert reason == [GPU_CMD_VERIFY_REASON_STREAM_BUDGET]


def test_success_randomized_parity() -> None:
    rng = random.Random(20260423_1256)
    types = [GPU_CMD_DESC_TYPE_QMATMUL, GPU_CMD_DESC_TYPE_ATTN, GPU_CMD_DESC_TYPE_ELEMENT]
    ops = [GPU_CMD_OP_COPY_IN, GPU_CMD_OP_COMPUTE, GPU_CMD_OP_COPY_OUT]

    for _ in range(500):
        region_nbytes = rng.randint(65536, 1 << 20)
        max_cmd_nbytes = rng.randint(64, min(region_nbytes, 4096))
        allowed_flags = 0xF

        desc_count = rng.randint(1, 8)
        stream: list[GPUCommandDescriptor] = []

        for _ in range(desc_count):
            byte_count = rng.randint(1, max_cmd_nbytes)
            byte_count -= byte_count % GPU_CMD_VERIFY_DMA_ALIGNMENT_BYTES
            if byte_count == 0:
                byte_count = GPU_CMD_VERIFY_DMA_ALIGNMENT_BYTES
            limit = region_nbytes - byte_count
            src = rng.randint(0, max(0, limit))
            src -= src % GPU_CMD_VERIFY_DMA_ALIGNMENT_BYTES
            dst = rng.randint(0, max(0, limit))
            dst -= dst % GPU_CMD_VERIFY_DMA_ALIGNMENT_BYTES

            desc_type = rng.choice(types)
            if desc_type == GPU_CMD_DESC_TYPE_ELEMENT:
                desc_op = GPU_CMD_OP_COMPUTE
            else:
                desc_op = rng.choice(ops)

            stream.append(
                GPUCommandDescriptor(
                    desc_type=desc_type,
                    desc_op=desc_op,
                    src_offset_bytes=src,
                    dst_offset_bytes=dst,
                    byte_count=byte_count,
                    flags=rng.randint(0, allowed_flags),
                )
            )

        bad_index = [-1]
        reason = [GPU_CMD_VERIFY_REASON_BAD_GEOMETRY]
        status = verify_stream_checked(
            stream,
            len(stream),
            region_nbytes,
            max_cmd_nbytes,
            allowed_flags,
            bad_index,
            reason,
        )

        assert status == GPU_CMD_VERIFY_OK
        assert bad_index == [-1]
        assert reason == [GPU_CMD_VERIFY_REASON_ALLOW]


def test_perf_overhead_measurement_plan_secure_profile() -> None:
    """Perf-overhead plan: baseline verifier cost per descriptor in secure-local."""

    desc = GPUCommandDescriptor(
        GPU_CMD_DESC_TYPE_QMATMUL,
        GPU_CMD_OP_COMPUTE,
        128,
        128,
        256,
        0x3,
    )

    loops = 50000
    reason = [0]

    t0 = time.perf_counter()
    for _ in range(loops):
        status = verify_descriptor_checked(desc, 1 << 20, 4096, 0xF, reason)
        if status != GPU_CMD_VERIFY_OK:
            raise AssertionError("Verifier unexpectedly failed in benchmark loop")
    dt = time.perf_counter() - t0

    # Keep this as a measurement plan guard: ensure benchmark emitted sane values.
    assert dt > 0.0
    per_call_ns = (dt / loops) * 1e9
    assert per_call_ns > 0.0
