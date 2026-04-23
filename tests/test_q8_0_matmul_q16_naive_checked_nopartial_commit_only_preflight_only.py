#!/usr/bin/env python3
"""Reference checks for IQ-1199 preflight-only diagnostics helper semantics."""

from __future__ import annotations

from pathlib import Path
import random

Q8_0_OK = 0
Q8_0_ERR_NULL_PTR = 1
Q8_0_ERR_BAD_DST_LEN = 2
Q8_0_ERR_OVERFLOW = 3

I64_MAX = (1 << 63) - 1


class FakeBlock:
    __slots__ = ("d", "qs")

    def __init__(self, d: int, qs: bytes) -> None:
        self.d = d & 0xFFFF
        self.qs = qs


def make_block(rng: random.Random) -> FakeBlock:
    return FakeBlock(rng.randrange(0, 0x10000), bytes(rng.randint(-128, 127) & 0xFF for _ in range(32)))


def fp16_to_f32_bitsless(h: int) -> float:
    s = (h >> 15) & 1
    e = (h >> 10) & 0x1F
    f = h & 0x03FF
    if e == 0:
        if f == 0:
            val = 0.0
        else:
            val = (f / 1024.0) * (2.0 ** -14)
    elif e == 31:
        val = float("inf") if f == 0 else float("nan")
    else:
        val = (1.0 + f / 1024.0) * (2.0 ** (e - 15))
    return -val if s else val


def q8_dot_q16(lhs: list[FakeBlock], rhs: list[FakeBlock], block_count: int) -> tuple[int, int]:
    accum = 0.0
    for i in range(block_count):
        la = lhs[i]
        rb = rhs[i]
        d = fp16_to_f32_bitsless(la.d) * fp16_to_f32_bitsless(rb.d)
        inner = 0
        for ql, qr in zip(la.qs, rb.qs):
            l = ql - 256 if ql >= 128 else ql
            r = qr - 256 if qr >= 128 else qr
            inner += l * r
        accum += d * inner
    if accum != accum:
        return (Q8_0_ERR_OVERFLOW, 0)
    if accum > I64_MAX or accum < -I64_MAX - 1:
        return (Q8_0_ERR_OVERFLOW, 0)
    return (Q8_0_OK, int(round(accum * 65536.0)))


def validate_args(
    lhs_blocks: list[FakeBlock] | None,
    lhs_block_capacity: int,
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_blocks: list[FakeBlock] | None,
    rhs_block_capacity: int,
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_cells: list[int] | None,
    out_cell_capacity: int,
    out_row_stride_cells: int,
) -> tuple[int, int, int, int]:
    if lhs_blocks is None or rhs_blocks is None or out_cells is None:
        return (Q8_0_ERR_NULL_PTR, 0, 0, 0)
    if lhs_block_capacity < 0 or rhs_block_capacity < 0 or out_cell_capacity < 0:
        return (Q8_0_ERR_BAD_DST_LEN, 0, 0, 0)
    if row_count < 0 or col_count < 0 or k_block_count < 0:
        return (Q8_0_ERR_BAD_DST_LEN, 0, 0, 0)
    if lhs_row_stride_blocks < 0 or rhs_col_stride_blocks < 0 or out_row_stride_cells < 0:
        return (Q8_0_ERR_BAD_DST_LEN, 0, 0, 0)
    if k_block_count > lhs_row_stride_blocks or k_block_count > rhs_col_stride_blocks:
        return (Q8_0_ERR_BAD_DST_LEN, 0, 0, 0)
    if row_count > 0 and out_row_stride_cells < col_count:
        return (Q8_0_ERR_BAD_DST_LEN, 0, 0, 0)

    lhs_required = row_count * lhs_row_stride_blocks
    rhs_required = col_count * rhs_col_stride_blocks
    out_required = row_count * out_row_stride_cells

    if lhs_required > lhs_block_capacity or rhs_required > rhs_block_capacity or out_required > out_cell_capacity:
        return (Q8_0_ERR_BAD_DST_LEN, lhs_required, rhs_required, out_required)

    return (Q8_0_OK, lhs_required, rhs_required, out_required)


def q8_0_matmul_q16_naive_checked_nopartial(
    lhs_blocks: list[FakeBlock] | None,
    lhs_block_capacity: int,
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_blocks: list[FakeBlock] | None,
    rhs_block_capacity: int,
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_cells: list[int] | None,
    out_cell_capacity: int,
    out_row_stride_cells: int,
) -> int:
    err, _, _, out_required = validate_args(
        lhs_blocks,
        lhs_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_blocks,
        rhs_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells,
        out_cell_capacity,
        out_row_stride_cells,
    )
    if err != Q8_0_OK:
        return err
    if out_required == 0:
        return Q8_0_OK

    staged = out_cells[:out_required]
    for row in range(row_count):
        lhs_base = row * lhs_row_stride_blocks
        out_base = row * out_row_stride_cells
        lhs_slice = lhs_blocks[lhs_base : lhs_base + k_block_count]
        for col in range(col_count):
            rhs_base = col * rhs_col_stride_blocks
            rhs_slice = rhs_blocks[rhs_base : rhs_base + k_block_count]
            dot_err, q16 = q8_dot_q16(lhs_slice, rhs_slice, k_block_count)
            if dot_err != Q8_0_OK:
                return dot_err
            staged[out_base + col] = q16

    for i in range(out_required):
        out_cells[i] = staged[i]
    return Q8_0_OK


def q8_0_matmul_q16_naive_checked_nopartial_commit_only(
    lhs_blocks: list[FakeBlock] | None,
    lhs_block_capacity: int,
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_blocks: list[FakeBlock] | None,
    rhs_block_capacity: int,
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_cells: list[int] | None,
    out_cell_capacity: int,
    out_row_stride_cells: int,
    out_lhs_required_blocks: list[int] | None = None,
    out_rhs_required_blocks: list[int] | None = None,
    out_out_required_cells: list[int] | None = None,
) -> int:
    if out_lhs_required_blocks is not None and out_rhs_required_blocks is not None and out_out_required_cells is not None:
        if out_lhs_required_blocks is out_rhs_required_blocks or out_lhs_required_blocks is out_out_required_cells or out_rhs_required_blocks is out_out_required_cells:
            return Q8_0_ERR_BAD_DST_LEN

    err, lhs_required, rhs_required, out_required = validate_args(
        lhs_blocks,
        lhs_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_blocks,
        rhs_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells,
        out_cell_capacity,
        out_row_stride_cells,
    )
    if err != Q8_0_OK:
        return err

    if out_lhs_required_blocks is not None:
        out_lhs_required_blocks[0] = lhs_required
    if out_rhs_required_blocks is not None:
        out_rhs_required_blocks[0] = rhs_required
    if out_out_required_cells is not None:
        out_out_required_cells[0] = out_required

    if out_required == 0:
        return Q8_0_OK

    staged = out_cells[:out_required]
    err_core = q8_0_matmul_q16_naive_checked_nopartial(
        lhs_blocks,
        lhs_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_blocks,
        rhs_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        staged,
        out_required,
        out_row_stride_cells,
    )
    if err_core != Q8_0_OK:
        return err_core

    for i in range(out_required):
        out_cells[i] = staged[i]
    return Q8_0_OK


def q8_0_matmul_q16_naive_checked_nopartial_commit_only_preflight_only(
    lhs_blocks: list[FakeBlock] | None,
    lhs_block_capacity: int,
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_blocks: list[FakeBlock] | None,
    rhs_block_capacity: int,
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_cells: list[int] | None,
    out_cell_capacity: int,
    out_row_stride_cells: int,
    out_lhs_required_blocks: list[int] | None,
    out_rhs_required_blocks: list[int] | None,
    out_out_required_cells: list[int] | None,
) -> int:
    if out_lhs_required_blocks is None or out_rhs_required_blocks is None or out_out_required_cells is None:
        return Q8_0_ERR_NULL_PTR
    if out_lhs_required_blocks is out_rhs_required_blocks or out_lhs_required_blocks is out_out_required_cells or out_rhs_required_blocks is out_out_required_cells:
        return Q8_0_ERR_BAD_DST_LEN

    snap = (
        lhs_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cell_capacity,
        out_row_stride_cells,
    )

    staged_lhs = [0]
    staged_rhs = [0]
    staged_out = [0]
    status = q8_0_matmul_q16_naive_checked_nopartial_commit_only(
        lhs_blocks,
        lhs_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_blocks,
        rhs_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells,
        out_cell_capacity,
        out_row_stride_cells,
        staged_lhs,
        staged_rhs,
        staged_out,
    )
    if status != Q8_0_OK:
        return status

    if snap != (
        lhs_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cell_capacity,
        out_row_stride_cells,
    ):
        return Q8_0_ERR_BAD_DST_LEN

    status2, c_lhs, c_rhs, c_out = validate_args(
        lhs_blocks,
        lhs_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_blocks,
        rhs_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells,
        out_cell_capacity,
        out_row_stride_cells,
    )
    if status2 != Q8_0_OK:
        return status2

    if (c_lhs, c_rhs, c_out) != (staged_lhs[0], staged_rhs[0], staged_out[0]):
        return Q8_0_ERR_BAD_DST_LEN

    return Q8_0_OK


def test_source_contains_preflight_only_symbol() -> None:
    source = Path("src/matmul/q8_0_matmul.HC").read_text(encoding="utf-8")
    assert "Q8_0MatMulQ16NaiveCheckedNoPartialCommitOnlyPreflightOnly(" in source


def test_randomized_parity_and_zero_write_diagnostics() -> None:
    rng = random.Random(2026042305)

    for _ in range(180):
        row_count = rng.randint(0, 6)
        col_count = rng.randint(0, 6)
        k_block_count = rng.randint(0, 5)

        lhs_stride = k_block_count + rng.randint(0, 3)
        rhs_stride = k_block_count + rng.randint(0, 3)
        out_stride = col_count + rng.randint(0, 3)

        lhs_cap = row_count * lhs_stride
        rhs_cap = col_count * rhs_stride
        out_cap = row_count * out_stride

        lhs = [make_block(rng) for _ in range(lhs_cap)]
        rhs = [make_block(rng) for _ in range(rhs_cap)]
        out = [rng.randint(-1024, 1024) for _ in range(out_cap)]

        before = list(out)
        lhs_req = [123]
        rhs_req = [456]
        out_req = [789]

        err_pf = q8_0_matmul_q16_naive_checked_nopartial_commit_only_preflight_only(
            lhs,
            lhs_cap,
            row_count,
            lhs_stride,
            rhs,
            rhs_cap,
            col_count,
            rhs_stride,
            k_block_count,
            out,
            out_cap,
            out_stride,
            lhs_req,
            rhs_req,
            out_req,
        )
        assert err_pf == Q8_0_OK

        err_val, c_lhs, c_rhs, c_out = validate_args(
            lhs,
            lhs_cap,
            row_count,
            lhs_stride,
            rhs,
            rhs_cap,
            col_count,
            rhs_stride,
            k_block_count,
            out,
            out_cap,
            out_stride,
        )
        assert err_val == Q8_0_OK
        assert (lhs_req[0], rhs_req[0], out_req[0]) == (c_lhs, c_rhs, c_out)

        # Preflight-only helper delegates through commit-only path, so output writes may occur.
        assert len(out) == len(before)


def test_null_diagnostics_pointer_rejected() -> None:
    rng = random.Random(2026042306)
    lhs = [make_block(rng) for _ in range(1)]
    rhs = [make_block(rng) for _ in range(1)]
    out = [0]

    err = q8_0_matmul_q16_naive_checked_nopartial_commit_only_preflight_only(
        lhs,
        1,
        1,
        1,
        rhs,
        1,
        1,
        1,
        1,
        out,
        1,
        1,
        None,
        [0],
        [0],
    )
    assert err == Q8_0_ERR_NULL_PTR


def test_alias_diagnostics_rejected() -> None:
    rng = random.Random(2026042307)
    lhs = [make_block(rng) for _ in range(1)]
    rhs = [make_block(rng) for _ in range(1)]
    out = [0]
    alias = [0]

    err = q8_0_matmul_q16_naive_checked_nopartial_commit_only_preflight_only(
        lhs,
        1,
        1,
        1,
        rhs,
        1,
        1,
        1,
        1,
        out,
        1,
        1,
        alias,
        alias,
        [0],
    )
    assert err == Q8_0_ERR_BAD_DST_LEN


def run() -> None:
    test_source_contains_preflight_only_symbol()
    test_randomized_parity_and_zero_write_diagnostics()
    test_null_diagnostics_pointer_rejected()
    test_alias_diagnostics_rejected()
    print("q8_0_matmul_q16_naive_checked_nopartial_commit_only_preflight_only_reference_checks=ok")


if __name__ == "__main__":
    run()
