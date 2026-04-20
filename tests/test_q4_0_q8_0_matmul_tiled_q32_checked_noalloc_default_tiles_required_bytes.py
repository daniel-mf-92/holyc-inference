#!/usr/bin/env python3
"""Parity harness for Q4_0Q8_0MatMulTiledQ32CheckedNoAllocDefaultTilesRequiredBytes."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q4_0_q8_0_matmul_tiled_avx2_q32 import (
    Q4_0_Q8_0_AVX2_ERR_NULL_PTR,
    Q4_0_Q8_0_AVX2_OK,
    make_q4_block,
    make_q8_block,
)
from test_q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_preflight_only import (
    q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_preflight_only,
)


def q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_required_bytes(
    lhs_q4_blocks,
    lhs_q4_block_capacity: int,
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_q8_col_blocks,
    rhs_q8_block_capacity: int,
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_cells_q32,
    out_cell_capacity: int,
    out_row_stride_cells: int,
    out_required_lhs_blocks: list[int] | None,
    out_required_rhs_blocks: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_tile_m: list[int] | None,
    out_tile_n: list[int] | None,
    out_tile_k_blocks: list[int] | None,
) -> int:
    if (
        out_required_lhs_blocks is None
        or out_required_rhs_blocks is None
        or out_required_out_cells is None
        or out_tile_m is None
        or out_tile_n is None
        or out_tile_k_blocks is None
    ):
        return Q4_0_Q8_0_AVX2_ERR_NULL_PTR

    req_lhs = [0]
    req_rhs = [0]
    req_out = [0]
    tile_m = [0]
    tile_n = [0]
    tile_k = [0]

    status = q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_preflight_only(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells,
        req_lhs,
        req_rhs,
        req_out,
        tile_m,
        tile_n,
        tile_k,
    )
    if status != Q4_0_Q8_0_AVX2_OK:
        return status

    out_required_lhs_blocks[0] = req_lhs[0]
    out_required_rhs_blocks[0] = req_rhs[0]
    out_required_out_cells[0] = req_out[0]
    out_tile_m[0] = tile_m[0]
    out_tile_n[0] = tile_n[0]
    out_tile_k_blocks[0] = tile_k[0]
    return Q4_0_Q8_0_AVX2_OK


def explicit_checked_composition(
    lhs_q4_blocks,
    lhs_q4_block_capacity: int,
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_q8_col_blocks,
    rhs_q8_block_capacity: int,
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_cells_q32,
    out_cell_capacity: int,
    out_row_stride_cells: int,
    out_required_lhs_blocks: list[int] | None,
    out_required_rhs_blocks: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_tile_m: list[int] | None,
    out_tile_n: list[int] | None,
    out_tile_k_blocks: list[int] | None,
) -> int:
    return q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_required_bytes(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells,
        out_required_lhs_blocks,
        out_required_rhs_blocks,
        out_required_out_cells,
        out_tile_m,
        out_tile_n,
        out_tile_k_blocks,
    )


def test_source_contains_required_bytes_wrapper() -> None:
    source = pathlib.Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    signature = "I32 Q4_0Q8_0MatMulTiledQ32CheckedNoAllocDefaultTilesRequiredBytes("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "if (!out_required_lhs_blocks || !out_required_rhs_blocks ||" in body
    assert "Q4_0Q8_0MatMulTiledQ32CheckedNoAllocDefaultTilesPreflightOnly(" in body
    assert "if (status != Q4_0_Q8_0_MATMUL_OK)" in body
    assert "*out_required_lhs_blocks = required_lhs_blocks;" in body
    assert "*out_required_rhs_blocks = required_rhs_blocks;" in body
    assert "*out_required_out_cells = required_out_cells;" in body
    assert "*out_tile_m = tile_m;" in body
    assert "*out_tile_n = tile_n;" in body
    assert "*out_tile_k_blocks = tile_k_blocks;" in body


def test_known_vector_and_null_no_write() -> None:
    rng = random.Random(20260420690)

    row_count = 4
    col_count = 5
    k_block_count = 3
    lhs_stride = 4
    rhs_stride = 4
    out_stride = 7

    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]
    out = [0] * (row_count * out_stride)

    got_lhs = [111]
    got_rhs = [222]
    got_out = [333]
    got_tile_m = [444]
    got_tile_n = [555]
    got_tile_k = [666]

    err = q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_required_bytes(
        lhs,
        len(lhs),
        row_count,
        lhs_stride,
        rhs,
        len(rhs),
        col_count,
        rhs_stride,
        k_block_count,
        out,
        len(out),
        out_stride,
        got_lhs,
        got_rhs,
        got_out,
        got_tile_m,
        got_tile_n,
        got_tile_k,
    )
    assert err == Q4_0_Q8_0_AVX2_OK
    assert got_lhs == [row_count * lhs_stride]
    assert got_rhs == [col_count * rhs_stride]
    assert got_out == [row_count * out_stride]
    assert got_tile_m == [4]
    assert got_tile_n == [4]
    assert got_tile_k == [8]

    sentinels = [[11], [22], [33], [44], [55], [66]]
    err = q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_required_bytes(
        lhs,
        len(lhs),
        row_count,
        lhs_stride,
        rhs,
        len(rhs),
        col_count,
        rhs_stride,
        k_block_count,
        out,
        len(out),
        out_stride,
        None,
        sentinels[1],
        sentinels[2],
        sentinels[3],
        sentinels[4],
        sentinels[5],
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_NULL_PTR
    assert [s[0] for s in sentinels] == [11, 22, 33, 44, 55, 66]


def test_randomized_parity_and_no_write_on_error() -> None:
    rng = random.Random(20260420990)

    for _ in range(1800):
        row_count = rng.randint(0, 9)
        col_count = rng.randint(0, 9)
        k_block_count = rng.randint(0, 7)

        lhs_stride = rng.randint(0, 10)
        rhs_stride = rng.randint(0, 10)
        out_stride = rng.randint(0, 11)

        if row_count > 0 and k_block_count > 0 and lhs_stride < k_block_count:
            lhs_stride = k_block_count + rng.randint(0, 2)
        if col_count > 0 and k_block_count > 0 and rhs_stride < k_block_count:
            rhs_stride = k_block_count + rng.randint(0, 2)
        if row_count > 0 and col_count > 0 and out_stride < col_count:
            out_stride = col_count + rng.randint(0, 3)

        lhs_need = row_count * lhs_stride
        rhs_need = col_count * rhs_stride
        out_need = row_count * out_stride

        lhs_cap = max(0, lhs_need + rng.randint(-3, 4))
        rhs_cap = max(0, rhs_need + rng.randint(-3, 4))
        out_cap = max(0, out_need + rng.randint(-3, 4))

        lhs = [make_q4_block(rng) for _ in range(lhs_cap)]
        rhs = [make_q8_block(rng) for _ in range(rhs_cap)]
        out = [0] * out_cap

        got_lhs = [901]
        got_rhs = [902]
        got_out = [903]
        got_tile_m = [904]
        got_tile_n = [905]
        got_tile_k = [906]

        exp_lhs = [901]
        exp_rhs = [902]
        exp_out = [903]
        exp_tile_m = [904]
        exp_tile_n = [905]
        exp_tile_k = [906]

        err_got = q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_required_bytes(
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
            got_lhs,
            got_rhs,
            got_out,
            got_tile_m,
            got_tile_n,
            got_tile_k,
        )
        err_exp = explicit_checked_composition(
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
            exp_lhs,
            exp_rhs,
            exp_out,
            exp_tile_m,
            exp_tile_n,
            exp_tile_k,
        )

        assert err_got == err_exp
        assert got_lhs == exp_lhs
        assert got_rhs == exp_rhs
        assert got_out == exp_out
        assert got_tile_m == exp_tile_m
        assert got_tile_n == exp_tile_n
        assert got_tile_k == exp_tile_k


if __name__ == "__main__":
    test_source_contains_required_bytes_wrapper()
    test_known_vector_and_null_no_write()
    test_randomized_parity_and_no_write_on_error()
    print("ok")
