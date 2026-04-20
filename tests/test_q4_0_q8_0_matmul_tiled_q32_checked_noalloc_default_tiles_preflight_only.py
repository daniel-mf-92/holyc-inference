#!/usr/bin/env python3
"""Parity harness for Q4_0Q8_0MatMulTiledQ32CheckedNoAllocDefaultTilesPreflightOnly."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q4_0_q8_0_matmul_tiled_avx2_q32 import (
    I64_MAX,
    Q4_0_Q8_0_AVX2_ERR_BAD_LEN,
    Q4_0_Q8_0_AVX2_ERR_NULL_PTR,
    Q4_0_Q8_0_AVX2_ERR_OVERFLOW,
    Q4_0_Q8_0_AVX2_OK,
    dot_blocks_avx2_q32_checked,
    make_q4_block,
    make_q8_block,
)


def try_add_i64_nonneg(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs > I64_MAX - rhs:
        return False, 0
    return True, lhs + rhs


def try_mul_i64_nonneg(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs == 0 or rhs == 0:
        return True, 0
    if lhs > I64_MAX // rhs:
        return False, 0
    return True, lhs * rhs


def q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_preflight_only(
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

    if lhs_q4_blocks is None or rhs_q8_col_blocks is None or out_cells_q32 is None:
        return Q4_0_Q8_0_AVX2_ERR_NULL_PTR

    if lhs_q4_block_capacity < 0 or rhs_q8_block_capacity < 0 or out_cell_capacity < 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if row_count < 0 or col_count < 0 or k_block_count < 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if lhs_row_stride_blocks < 0 or rhs_col_stride_blocks < 0 or out_row_stride_cells < 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    if k_block_count > lhs_row_stride_blocks or k_block_count > rhs_col_stride_blocks:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if row_count > 0 and out_row_stride_cells < col_count:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    ok, lhs_required_blocks = try_mul_i64_nonneg(row_count, lhs_row_stride_blocks)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    ok, rhs_required_blocks = try_mul_i64_nonneg(col_count, rhs_col_stride_blocks)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    ok, out_required_cells = try_mul_i64_nonneg(row_count, out_row_stride_cells)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

    if lhs_required_blocks > lhs_q4_block_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if rhs_required_blocks > rhs_q8_block_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if out_required_cells > out_cell_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    out_required_lhs_blocks[0] = lhs_required_blocks
    out_required_rhs_blocks[0] = rhs_required_blocks
    out_required_out_cells[0] = out_required_cells
    out_tile_m[0] = 4
    out_tile_n[0] = 4
    out_tile_k_blocks[0] = 8

    # Preflight-only no-partial policy: validate every dot path, no writes.
    row_tile_start = 0
    while row_tile_start < row_count:
        ok, row_tile_end = try_add_i64_nonneg(row_tile_start, 4)
        if not ok:
            return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
        row_tile_end = min(row_tile_end, row_count)

        col_tile_start = 0
        while col_tile_start < col_count:
            ok, col_tile_end = try_add_i64_nonneg(col_tile_start, 4)
            if not ok:
                return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
            col_tile_end = min(col_tile_end, col_count)

            for row_index in range(row_tile_start, row_tile_end):
                ok, lhs_row_base = try_mul_i64_nonneg(row_index, lhs_row_stride_blocks)
                if not ok:
                    return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

                for col_index in range(col_tile_start, col_tile_end):
                    ok, rhs_col_base = try_mul_i64_nonneg(col_index, rhs_col_stride_blocks)
                    if not ok:
                        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

                    err, _ = dot_blocks_avx2_q32_checked(
                        lhs_blocks=lhs_q4_blocks[lhs_row_base : lhs_row_base + k_block_count],
                        rhs_blocks=rhs_q8_col_blocks[rhs_col_base : rhs_col_base + k_block_count],
                        block_count=k_block_count,
                    )
                    if err != Q4_0_Q8_0_AVX2_OK:
                        return err

            col_tile_start = col_tile_end

        row_tile_start = row_tile_end

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
    return q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_preflight_only(
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


def test_source_contains_preflight_only_signature() -> None:
    source = pathlib.Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    signature = "I32 Q4_0Q8_0MatMulTiledQ32CheckedNoAllocDefaultTilesPreflightOnly("
    assert signature in source
    body = source.split(signature, 1)[1]
    assert "*out_required_lhs_blocks = lhs_required_blocks;" in body
    assert "*out_required_rhs_blocks = rhs_required_blocks;" in body
    assert "*out_required_out_cells = out_required_cells;" in body
    assert "*out_tile_m = Q4_0_Q8_0_MATMUL_DEFAULT_TILE_M;" in body
    assert "*out_tile_n = Q4_0_Q8_0_MATMUL_DEFAULT_TILE_N;" in body
    assert "*out_tile_k_blocks = Q4_0_Q8_0_MATMUL_DEFAULT_TILE_K;" in body
    assert "return Q4_0_Q8_0_MATMUL_OK;" in body


def test_known_vector_preflight_outputs_and_no_writes() -> None:
    rng = random.Random(2026042003)

    row_count = 3
    col_count = 4
    k_block_count = 2
    lhs_stride = 3
    rhs_stride = 3
    out_stride = 6

    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]

    out_capacity = row_count * out_stride
    out_buffer = [111111] * out_capacity
    before = list(out_buffer)

    req_lhs = [0]
    req_rhs = [0]
    req_out = [0]
    tile_m = [0]
    tile_n = [0]
    tile_k = [0]

    err = q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_preflight_only(
        lhs,
        len(lhs),
        row_count,
        lhs_stride,
        rhs,
        len(rhs),
        col_count,
        rhs_stride,
        k_block_count,
        out_buffer,
        out_capacity,
        out_stride,
        req_lhs,
        req_rhs,
        req_out,
        tile_m,
        tile_n,
        tile_k,
    )
    assert err == Q4_0_Q8_0_AVX2_OK
    assert req_lhs[0] == row_count * lhs_stride
    assert req_rhs[0] == col_count * rhs_stride
    assert req_out[0] == row_count * out_stride
    assert tile_m[0] == 4
    assert tile_n[0] == 4
    assert tile_k[0] == 8
    assert out_buffer == before


def test_error_parity_adversarial_vectors() -> None:
    rng = random.Random(2026042004)

    for _ in range(300):
        row_count = rng.randint(0, 7)
        col_count = rng.randint(0, 7)
        k_block_count = rng.randint(0, 5)

        lhs_stride = rng.randint(k_block_count, k_block_count + 3)
        rhs_stride = rng.randint(k_block_count, k_block_count + 3)
        out_stride = rng.randint(col_count, col_count + 3) if row_count > 0 else rng.randint(0, 3)

        lhs_capacity = row_count * lhs_stride
        rhs_capacity = col_count * rhs_stride
        out_capacity = row_count * out_stride

        lhs = [make_q4_block(rng) for _ in range(max(lhs_capacity, 1))]
        rhs = [make_q8_block(rng) for _ in range(max(rhs_capacity, 1))]
        out = [777] * max(out_capacity, 1)

        if rng.random() < 0.2:
            out_capacity = max(0, out_capacity - rng.randint(1, 2))
        if rng.random() < 0.2:
            lhs_capacity = max(0, lhs_capacity - rng.randint(1, 2))
        if rng.random() < 0.2:
            rhs_capacity = max(0, rhs_capacity - rng.randint(1, 2))

        req_lhs_a = [123]
        req_rhs_a = [123]
        req_out_a = [123]
        tile_m_a = [123]
        tile_n_a = [123]
        tile_k_a = [123]

        req_lhs_b = [123]
        req_rhs_b = [123]
        req_out_b = [123]
        tile_m_b = [123]
        tile_n_b = [123]
        tile_k_b = [123]

        err_a = q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_preflight_only(
            lhs,
            lhs_capacity,
            row_count,
            lhs_stride,
            rhs,
            rhs_capacity,
            col_count,
            rhs_stride,
            k_block_count,
            out,
            out_capacity,
            out_stride,
            req_lhs_a,
            req_rhs_a,
            req_out_a,
            tile_m_a,
            tile_n_a,
            tile_k_a,
        )
        err_b = explicit_checked_composition(
            lhs,
            lhs_capacity,
            row_count,
            lhs_stride,
            rhs,
            rhs_capacity,
            col_count,
            rhs_stride,
            k_block_count,
            out,
            out_capacity,
            out_stride,
            req_lhs_b,
            req_rhs_b,
            req_out_b,
            tile_m_b,
            tile_n_b,
            tile_k_b,
        )

        assert err_a == err_b
        if err_a == Q4_0_Q8_0_AVX2_OK:
            assert req_lhs_a[0] == req_lhs_b[0]
            assert req_rhs_a[0] == req_rhs_b[0]
            assert req_out_a[0] == req_out_b[0]
            assert tile_m_a[0] == tile_m_b[0] == 4
            assert tile_n_a[0] == tile_n_b[0] == 4
            assert tile_k_a[0] == tile_k_b[0] == 8


def test_null_output_contract_pointers() -> None:
    rng = random.Random(2026042005)
    lhs = [make_q4_block(rng)]
    rhs = [make_q8_block(rng)]
    out = [0]

    req_lhs = [0]
    req_rhs = [0]
    req_out = [0]
    tile_m = [0]
    tile_n = [0]
    tile_k = [0]

    for reqs in [
        (None, req_rhs, req_out, tile_m, tile_n, tile_k),
        (req_lhs, None, req_out, tile_m, tile_n, tile_k),
        (req_lhs, req_rhs, None, tile_m, tile_n, tile_k),
        (req_lhs, req_rhs, req_out, None, tile_n, tile_k),
        (req_lhs, req_rhs, req_out, tile_m, None, tile_k),
        (req_lhs, req_rhs, req_out, tile_m, tile_n, None),
    ]:
        err = q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_preflight_only(
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
            reqs[0],
            reqs[1],
            reqs[2],
            reqs[3],
            reqs[4],
            reqs[5],
        )
        assert err == Q4_0_Q8_0_AVX2_ERR_NULL_PTR


def main() -> None:
    test_source_contains_preflight_only_signature()
    test_known_vector_preflight_outputs_and_no_writes()
    test_error_parity_adversarial_vectors()
    test_null_output_contract_pointers()
    print("ok")


if __name__ == "__main__":
    main()
