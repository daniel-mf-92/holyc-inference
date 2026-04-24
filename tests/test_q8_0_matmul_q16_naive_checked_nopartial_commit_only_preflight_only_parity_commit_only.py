#!/usr/bin/env python3
"""Reference checks for IQ-1277 parity-commit-only wrapper semantics."""

from pathlib import Path
import importlib.util
import sys

_BASE = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _BASE / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ref_core = _load(
    "q8_0_matmul_core",
    "test_q8_0_matmul_q16_naive_checked_nopartial_commit_only_preflight_only.py",
)

Q8_0_OK = ref_core.Q8_0_OK
Q8_0_ERR_BAD_DST_LEN = ref_core.Q8_0_ERR_BAD_DST_LEN
Q8_0_ERR_NULL_PTR = 1


def q8_0_matmul_q16_nopartial_commit_only_preflight_only_parity_commit_only(
    lhs_blocks,
    lhs_block_capacity,
    row_count,
    lhs_row_stride_blocks,
    rhs_col_blocks,
    rhs_block_capacity,
    col_count,
    rhs_col_stride_blocks,
    k_block_count,
    out_cells_q16,
    out_cell_capacity,
    out_row_stride_cells,
    out_lhs_required_blocks,
    out_rhs_required_blocks,
    out_out_required_cells,
):
    if (
        out_lhs_required_blocks is None
        or out_rhs_required_blocks is None
        or out_out_required_cells is None
    ):
        return Q8_0_ERR_NULL_PTR

    if (
        out_lhs_required_blocks is out_rhs_required_blocks
        or out_lhs_required_blocks is out_out_required_cells
        or out_rhs_required_blocks is out_out_required_cells
    ):
        return Q8_0_ERR_BAD_DST_LEN

    snapshot = (
        lhs_blocks,
        lhs_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_col_blocks,
        rhs_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q16,
        out_cell_capacity,
        out_row_stride_cells,
        out_lhs_required_blocks,
        out_rhs_required_blocks,
        out_out_required_cells,
    )

    staged_parity_lhs = [0x4444444444444444]
    staged_parity_rhs = [0x5555555555555555]
    staged_parity_out = [0x6666666666666666]
    parity_status = ref_core.q8_0_matmul_q16_nopartial_commit_only_preflight_only(
        lhs_blocks,
        lhs_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_col_blocks,
        rhs_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q16,
        out_cell_capacity,
        out_row_stride_cells,
        staged_parity_lhs,
        staged_parity_rhs,
        staged_parity_out,
    )

    commit_status = ref_core.q8_0_matmul_q16_nopartial_commit_only(
        lhs_blocks,
        lhs_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_col_blocks,
        rhs_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q16,
        out_cell_capacity,
        out_row_stride_cells,
    )

    if snapshot != (
        lhs_blocks,
        lhs_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_col_blocks,
        rhs_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q16,
        out_cell_capacity,
        out_row_stride_cells,
        out_lhs_required_blocks,
        out_rhs_required_blocks,
        out_out_required_cells,
    ):
        return Q8_0_ERR_BAD_DST_LEN

    if commit_status != parity_status:
        return Q8_0_ERR_BAD_DST_LEN

    if commit_status != Q8_0_OK:
        return commit_status

    err, canonical_lhs_required, canonical_rhs_required, canonical_out_required = ref_core.compute_required_capacities_checked(
        row_count,
        lhs_row_stride_blocks,
        col_count,
        rhs_col_stride_blocks,
        out_row_stride_cells,
    )
    if err != Q8_0_OK:
        return Q8_0_ERR_BAD_DST_LEN

    if (
        canonical_lhs_required != staged_parity_lhs[0]
        or canonical_rhs_required != staged_parity_rhs[0]
        or canonical_out_required != staged_parity_out[0]
    ):
        return Q8_0_ERR_BAD_DST_LEN

    out_lhs_required_blocks[0] = staged_parity_lhs[0]
    out_rhs_required_blocks[0] = staged_parity_rhs[0]
    out_out_required_cells[0] = staged_parity_out[0]
    return Q8_0_OK


def _make_valid_case():
    row_count = 3
    col_count = 4
    k_block_count = 2
    lhs_row_stride_blocks = k_block_count + 1
    rhs_col_stride_blocks = k_block_count + 2
    out_row_stride_cells = col_count + 2

    err, lhs_required, rhs_required, out_required = ref_core.compute_required_capacities_checked(
        row_count,
        lhs_row_stride_blocks,
        col_count,
        rhs_col_stride_blocks,
        out_row_stride_cells,
    )
    assert err == Q8_0_OK

    lhs_blocks = [ref_core.make_block(ref_core.random.Random(1000 + i)) for i in range(lhs_required + 3)]
    rhs_blocks = [ref_core.make_block(ref_core.random.Random(2000 + i)) for i in range(rhs_required + 3)]
    out_cells = [0] * (out_required + 7)

    return {
        "lhs_blocks": lhs_blocks,
        "lhs_block_capacity": len(lhs_blocks),
        "row_count": row_count,
        "lhs_row_stride_blocks": lhs_row_stride_blocks,
        "rhs_col_blocks": rhs_blocks,
        "rhs_block_capacity": len(rhs_blocks),
        "col_count": col_count,
        "rhs_col_stride_blocks": rhs_col_stride_blocks,
        "k_block_count": k_block_count,
        "out_cells_q16": out_cells,
        "out_cell_capacity": len(out_cells),
        "out_row_stride_cells": out_row_stride_cells,
    }


def test_success_parity_and_atomic_publish() -> None:
    case = _make_valid_case()

    err, expected_lhs_required, expected_rhs_required, expected_out_required = ref_core.compute_required_capacities_checked(
        case["row_count"],
        case["lhs_row_stride_blocks"],
        case["col_count"],
        case["rhs_col_stride_blocks"],
        case["out_row_stride_cells"],
    )
    assert err == Q8_0_OK

    out_lhs = [0x1111111111111111]
    out_rhs = [0x2222222222222222]
    out_out = [0x3333333333333333]
    status = q8_0_matmul_q16_nopartial_commit_only_preflight_only_parity_commit_only(
        **case,
        out_lhs_required_blocks=out_lhs,
        out_rhs_required_blocks=out_rhs,
        out_out_required_cells=out_out,
    )

    assert status == Q8_0_OK
    assert out_lhs[0] == expected_lhs_required
    assert out_rhs[0] == expected_rhs_required
    assert out_out[0] == expected_out_required


def test_error_passthrough_and_no_partial_publish() -> None:
    case = _make_valid_case()
    case["lhs_block_capacity"] = 0

    out_lhs = [777]
    out_rhs = [888]
    out_out = [999]
    status = q8_0_matmul_q16_nopartial_commit_only_preflight_only_parity_commit_only(
        **case,
        out_lhs_required_blocks=out_lhs,
        out_rhs_required_blocks=out_rhs,
        out_out_required_cells=out_out,
    )

    assert status == Q8_0_ERR_BAD_DST_LEN
    assert out_lhs[0] == 777
    assert out_rhs[0] == 888
    assert out_out[0] == 999


def test_rejects_aliasing_output_pointers() -> None:
    case = _make_valid_case()

    shared = [123]
    distinct = [456]
    status = q8_0_matmul_q16_nopartial_commit_only_preflight_only_parity_commit_only(
        **case,
        out_lhs_required_blocks=shared,
        out_rhs_required_blocks=shared,
        out_out_required_cells=distinct,
    )
    assert status == Q8_0_ERR_BAD_DST_LEN


def test_holyc_symbol_present() -> None:
    source = Path("src/matmul/q8_0_matmul.HC").read_text(encoding="utf-8")
    symbol = "Q8_0MatMulQ16NaiveCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly"
    assert symbol in source


def run() -> None:
    test_success_parity_and_atomic_publish()
    test_error_passthrough_and_no_partial_publish()
    test_rejects_aliasing_output_pointers()
    test_holyc_symbol_present()
    print("q8_0_matmul_q16_naive_checked_nopartial_commit_only_preflight_only_parity_commit_only=ok")


if __name__ == "__main__":
    run()
