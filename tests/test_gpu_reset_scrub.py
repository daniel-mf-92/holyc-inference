#!/usr/bin/env python3
"""Parity harness for GPU reset+scrub sequencing helper (IQ-1262)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

GPU_RESET_SCRUB_OK = 0
GPU_RESET_SCRUB_ERR_NULL_PTR = 1
GPU_RESET_SCRUB_ERR_BAD_PARAM = 2
GPU_RESET_SCRUB_ERR_BAD_STATE = 3
GPU_RESET_SCRUB_ERR_OVERFLOW = 4
GPU_RESET_SCRUB_ERR_POLICY_GUARD = 5
GPU_RESET_SCRUB_ERR_SEQUENCE_GUARD = 6

GPU_RESET_SCRUB_PROFILE_SECURE_LOCAL = 1
GPU_RESET_SCRUB_PROFILE_DEV_LOCAL = 2

I64_MAX = (1 << 63) - 1


@dataclass
class Context:
    profile_id: int
    session_active: int
    iommu_enabled: int
    bot_dma_log_enabled: int
    bot_mmio_log_enabled: int
    bot_dispatch_log_enabled: int
    total_resets: int = 0
    total_scrub_passes: int = 0
    total_scrub_blocks: int = 0
    total_scrub_bytes: int = 0
    sequence_id: int = 0


def _is_binary(value: int) -> bool:
    return value in (0, 1)


def _add_checked(lhs: int, rhs: int) -> int | None:
    if lhs < 0 or rhs < 0:
        return None
    if lhs > I64_MAX - rhs:
        return None
    return lhs + rhs


def _mul_checked(lhs: int, rhs: int) -> int | None:
    if lhs < 0 or rhs < 0:
        return None
    if lhs == 0 or rhs == 0:
        return 0
    if lhs > I64_MAX // rhs:
        return None
    return lhs * rhs


def _prereqs_active(ctx: Context) -> bool:
    return (
        _is_binary(ctx.iommu_enabled)
        and _is_binary(ctx.bot_dma_log_enabled)
        and _is_binary(ctx.bot_mmio_log_enabled)
        and _is_binary(ctx.bot_dispatch_log_enabled)
        and ctx.iommu_enabled == 1
        and ctx.bot_dma_log_enabled == 1
        and ctx.bot_mmio_log_enabled == 1
        and ctx.bot_dispatch_log_enabled == 1
    )


def context_init(
    profile_id: int,
    iommu_enabled: int,
    bot_dma_log_enabled: int,
    bot_mmio_log_enabled: int,
    bot_dispatch_log_enabled: int,
) -> tuple[int, Context | None]:
    if not all(
        _is_binary(flag)
        for flag in (
            iommu_enabled,
            bot_dma_log_enabled,
            bot_mmio_log_enabled,
            bot_dispatch_log_enabled,
        )
    ):
        return GPU_RESET_SCRUB_ERR_BAD_PARAM, None

    if profile_id not in (
        GPU_RESET_SCRUB_PROFILE_SECURE_LOCAL,
        GPU_RESET_SCRUB_PROFILE_DEV_LOCAL,
    ):
        return GPU_RESET_SCRUB_ERR_BAD_PARAM, None

    return (
        GPU_RESET_SCRUB_OK,
        Context(
            profile_id=profile_id,
            session_active=0,
            iommu_enabled=iommu_enabled,
            bot_dma_log_enabled=bot_dma_log_enabled,
            bot_mmio_log_enabled=bot_mmio_log_enabled,
            bot_dispatch_log_enabled=bot_dispatch_log_enabled,
        ),
    )


def plan_checked(
    partition_nbytes: int,
    scrub_chunk_nbytes: int,
    scrub_pass_count: int,
) -> tuple[int, int, int, int]:
    if partition_nbytes <= 0 or scrub_chunk_nbytes <= 0 or scrub_pass_count <= 0:
        return GPU_RESET_SCRUB_ERR_BAD_PARAM, 0, 0, 0

    ceil_numerator = _add_checked(partition_nbytes, scrub_chunk_nbytes - 1)
    if ceil_numerator is None:
        return GPU_RESET_SCRUB_ERR_OVERFLOW, 0, 0, 0

    blocks_per_pass = ceil_numerator // scrub_chunk_nbytes
    if blocks_per_pass <= 0:
        return GPU_RESET_SCRUB_ERR_BAD_STATE, 0, 0, 0

    total_blocks = _mul_checked(blocks_per_pass, scrub_pass_count)
    if total_blocks is None:
        return GPU_RESET_SCRUB_ERR_OVERFLOW, 0, 0, 0

    total_scrubbed = _mul_checked(partition_nbytes, scrub_pass_count)
    if total_scrubbed is None:
        return GPU_RESET_SCRUB_ERR_OVERFLOW, 0, 0, 0

    return GPU_RESET_SCRUB_OK, blocks_per_pass, total_blocks, total_scrubbed


def run_pre(
    ctx: Context,
    partition_nbytes: int,
    scrub_chunk_nbytes: int,
    scrub_pass_count: int,
) -> tuple[int, int, int, int, int]:
    if ctx.session_active:
        return GPU_RESET_SCRUB_ERR_SEQUENCE_GUARD, 0, 0, 0, 0

    if not _prereqs_active(ctx):
        return GPU_RESET_SCRUB_ERR_POLICY_GUARD, 0, 0, 0, 0

    status, _, total_blocks, total_scrubbed = plan_checked(
        partition_nbytes, scrub_chunk_nbytes, scrub_pass_count
    )
    if status != GPU_RESET_SCRUB_OK:
        return status, 0, 0, 0, 0

    step_count = _add_checked(1, total_blocks)
    if step_count is None:
        return GPU_RESET_SCRUB_ERR_OVERFLOW, 0, 0, 0, 0

    next_resets = _add_checked(ctx.total_resets, 1)
    next_passes = _add_checked(ctx.total_scrub_passes, scrub_pass_count)
    next_blocks = _add_checked(ctx.total_scrub_blocks, total_blocks)
    next_bytes = _add_checked(ctx.total_scrub_bytes, total_scrubbed)
    next_seq = _add_checked(ctx.sequence_id, step_count)
    if None in (next_resets, next_passes, next_blocks, next_bytes, next_seq):
        return GPU_RESET_SCRUB_ERR_OVERFLOW, 0, 0, 0, 0

    ctx.total_resets = int(next_resets)
    ctx.total_scrub_passes = int(next_passes)
    ctx.total_scrub_blocks = int(next_blocks)
    ctx.total_scrub_bytes = int(next_bytes)
    ctx.sequence_id = int(next_seq)
    ctx.session_active = 1

    return GPU_RESET_SCRUB_OK, int(step_count), int(total_blocks), int(total_scrubbed), int(next_seq)


def run_post(
    ctx: Context,
    partition_nbytes: int,
    scrub_chunk_nbytes: int,
    scrub_pass_count: int,
) -> tuple[int, int, int, int, int]:
    if not ctx.session_active:
        return GPU_RESET_SCRUB_ERR_SEQUENCE_GUARD, 0, 0, 0, 0

    if not _prereqs_active(ctx):
        return GPU_RESET_SCRUB_ERR_POLICY_GUARD, 0, 0, 0, 0

    status, _, total_blocks, total_scrubbed = plan_checked(
        partition_nbytes, scrub_chunk_nbytes, scrub_pass_count
    )
    if status != GPU_RESET_SCRUB_OK:
        return status, 0, 0, 0, 0

    step_count = _add_checked(total_blocks, 1)
    if step_count is None:
        return GPU_RESET_SCRUB_ERR_OVERFLOW, 0, 0, 0, 0

    next_resets = _add_checked(ctx.total_resets, 1)
    next_passes = _add_checked(ctx.total_scrub_passes, scrub_pass_count)
    next_blocks = _add_checked(ctx.total_scrub_blocks, total_blocks)
    next_bytes = _add_checked(ctx.total_scrub_bytes, total_scrubbed)
    next_seq = _add_checked(ctx.sequence_id, step_count)
    if None in (next_resets, next_passes, next_blocks, next_bytes, next_seq):
        return GPU_RESET_SCRUB_ERR_OVERFLOW, 0, 0, 0, 0

    ctx.total_resets = int(next_resets)
    ctx.total_scrub_passes = int(next_passes)
    ctx.total_scrub_blocks = int(next_blocks)
    ctx.total_scrub_bytes = int(next_bytes)
    ctx.sequence_id = int(next_seq)
    ctx.session_active = 0

    return GPU_RESET_SCRUB_OK, int(step_count), int(total_blocks), int(total_scrubbed), int(next_seq)


def test_source_contains_iq1262_symbols() -> None:
    src = Path("src/gpu/reset_scrub.HC").read_text(encoding="utf-8")

    assert "class GPUResetScrubContext" in src
    assert "I32 GPUResetScrubContextInit(" in src
    assert "I32 GPUResetScrubPlanChecked(" in src
    assert "I32 GPUResetScrubRunPreTrustedSessionChecked(" in src
    assert "I32 GPUResetScrubRunPostTrustedSessionChecked(" in src
    assert "GPU_RESET_SCRUB_ERR_POLICY_GUARD" in src
    assert "GPU_RESET_SCRUB_ERR_SEQUENCE_GUARD" in src


def test_pre_then_post_session_sequence_updates_counters() -> None:
    status, ctx = context_init(
        GPU_RESET_SCRUB_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        bot_dma_log_enabled=1,
        bot_mmio_log_enabled=1,
        bot_dispatch_log_enabled=1,
    )
    assert status == GPU_RESET_SCRUB_OK
    assert ctx is not None

    status, pre_steps, pre_blocks, pre_bytes, pre_seq = run_pre(
        ctx, partition_nbytes=4096, scrub_chunk_nbytes=1024, scrub_pass_count=2
    )
    assert status == GPU_RESET_SCRUB_OK
    assert pre_steps == 9
    assert pre_blocks == 8
    assert pre_bytes == 8192
    assert pre_seq == 9
    assert ctx.session_active == 1

    status, post_steps, post_blocks, post_bytes, post_seq = run_post(
        ctx, partition_nbytes=4096, scrub_chunk_nbytes=1024, scrub_pass_count=2
    )
    assert status == GPU_RESET_SCRUB_OK
    assert post_steps == 9
    assert post_blocks == 8
    assert post_bytes == 8192
    assert post_seq == 18

    assert ctx.session_active == 0
    assert ctx.total_resets == 2
    assert ctx.total_scrub_passes == 4
    assert ctx.total_scrub_blocks == 16
    assert ctx.total_scrub_bytes == 16384


def test_post_without_active_session_is_rejected() -> None:
    status, ctx = context_init(
        GPU_RESET_SCRUB_PROFILE_DEV_LOCAL,
        iommu_enabled=1,
        bot_dma_log_enabled=1,
        bot_mmio_log_enabled=1,
        bot_dispatch_log_enabled=1,
    )
    assert status == GPU_RESET_SCRUB_OK
    assert ctx is not None

    status, *_ = run_post(ctx, partition_nbytes=1024, scrub_chunk_nbytes=256, scrub_pass_count=1)
    assert status == GPU_RESET_SCRUB_ERR_SEQUENCE_GUARD


def test_missing_iommu_or_book_hooks_fails_closed() -> None:
    status, ctx = context_init(
        GPU_RESET_SCRUB_PROFILE_SECURE_LOCAL,
        iommu_enabled=0,
        bot_dma_log_enabled=1,
        bot_mmio_log_enabled=1,
        bot_dispatch_log_enabled=1,
    )
    assert status == GPU_RESET_SCRUB_OK
    assert ctx is not None

    status, *_ = run_pre(ctx, partition_nbytes=2048, scrub_chunk_nbytes=256, scrub_pass_count=1)
    assert status == GPU_RESET_SCRUB_ERR_POLICY_GUARD


def test_plan_rejects_invalid_and_overflow_inputs() -> None:
    status, *_ = plan_checked(partition_nbytes=4096, scrub_chunk_nbytes=0, scrub_pass_count=1)
    assert status == GPU_RESET_SCRUB_ERR_BAD_PARAM

    status, *_ = plan_checked(
        partition_nbytes=I64_MAX,
        scrub_chunk_nbytes=2,
        scrub_pass_count=2,
    )
    assert status == GPU_RESET_SCRUB_ERR_OVERFLOW


def test_sequence_guard_blocks_double_pre_without_post() -> None:
    status, ctx = context_init(
        GPU_RESET_SCRUB_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        bot_dma_log_enabled=1,
        bot_mmio_log_enabled=1,
        bot_dispatch_log_enabled=1,
    )
    assert status == GPU_RESET_SCRUB_OK
    assert ctx is not None

    status, *_ = run_pre(ctx, partition_nbytes=1024, scrub_chunk_nbytes=128, scrub_pass_count=1)
    assert status == GPU_RESET_SCRUB_OK

    status, *_ = run_pre(ctx, partition_nbytes=1024, scrub_chunk_nbytes=128, scrub_pass_count=1)
    assert status == GPU_RESET_SCRUB_ERR_SEQUENCE_GUARD


if __name__ == "__main__":
    test_source_contains_iq1262_symbols()
    test_pre_then_post_session_sequence_updates_counters()
    test_post_without_active_session_is_rejected()
    test_missing_iommu_or_book_hooks_fails_closed()
    test_plan_rejects_invalid_and_overflow_inputs()
    test_sequence_guard_blocks_double_pre_without_post()
    print("ok")
