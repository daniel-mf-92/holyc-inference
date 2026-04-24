from dataclasses import dataclass

Q8_0_MATMUL_OK = 0
Q8_0_MATMUL_ERR_NULL_PTR = 1
Q8_0_MATMUL_ERR_BAD_DST_LEN = 2
Q8_0_MATMUL_ERR_OVERFLOW = 3
Q8_0_MATMUL_I64_MAX = 0x7FFFFFFFFFFFFFFF


class Ptr:
    def __init__(self, value=None):
        self.value = value


@dataclass
class MatrixCase:
    row_count: int
    col_count: int
    k_block_count: int
    lhs_row_stride_blocks: int
    rhs_col_stride_blocks: int
    out_row_stride_cells: int
    lhs_block_capacity: int
    rhs_block_capacity: int
    out_cell_capacity: int


def _try_mul_nonneg(lhs: int, rhs: int):
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs == 0 or rhs == 0:
        return True, 0
    if lhs > Q8_0_MATMUL_I64_MAX // rhs:
        return False, 0
    return True, lhs * rhs


def _validate(case: MatrixCase) -> int:
    if case.row_count < 0 or case.col_count < 0:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if case.k_block_count < 0:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if case.lhs_row_stride_blocks < 0 or case.rhs_col_stride_blocks < 0:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if case.out_row_stride_cells < 0:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if case.lhs_block_capacity < 0 or case.rhs_block_capacity < 0 or case.out_cell_capacity < 0:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if case.lhs_row_stride_blocks < case.k_block_count:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if case.rhs_col_stride_blocks < case.k_block_count:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if case.out_row_stride_cells < case.col_count:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    return Q8_0_MATMUL_OK


def _required(case: MatrixCase):
    ok, lhs_req = _try_mul_nonneg(case.row_count, case.lhs_row_stride_blocks)
    if not ok:
        return Q8_0_MATMUL_ERR_OVERFLOW, None
    ok, rhs_req = _try_mul_nonneg(case.col_count, case.rhs_col_stride_blocks)
    if not ok:
        return Q8_0_MATMUL_ERR_OVERFLOW, None
    ok, out_req = _try_mul_nonneg(case.row_count, case.out_row_stride_cells)
    if not ok:
        return Q8_0_MATMUL_ERR_OVERFLOW, None
    return Q8_0_MATMUL_OK, (lhs_req, rhs_req, out_req)


def commit_only(case: MatrixCase):
    st = _validate(case)
    if st != Q8_0_MATMUL_OK:
        return st, None
    st, req = _required(case)
    if st != Q8_0_MATMUL_OK:
        return st, None
    lhs_req, rhs_req, out_req = req
    if lhs_req > case.lhs_block_capacity:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN, None
    if rhs_req > case.rhs_block_capacity:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN, None
    if out_req > case.out_cell_capacity:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN, None
    return Q8_0_MATMUL_OK, req


def preflight_only_parity(case: MatrixCase, out_lhs: Ptr, out_rhs: Ptr, out_out: Ptr):
    if out_lhs is None or out_rhs is None or out_out is None:
        return Q8_0_MATMUL_ERR_NULL_PTR
    if out_lhs is out_rhs or out_lhs is out_out or out_rhs is out_out:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    st_commit, diag_commit = commit_only(case)
    st_can, req = _required(case) if _validate(case) == Q8_0_MATMUL_OK else (_validate(case), None)

    if st_can != st_commit:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if st_commit != Q8_0_MATMUL_OK:
        return st_commit

    lhs_req, rhs_req, out_req = req
    if diag_commit != (lhs_req, rhs_req, out_req):
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    out_lhs.value = lhs_req
    out_rhs.value = rhs_req
    out_out.value = out_req
    return Q8_0_MATMUL_OK


def preflight_only_parity_commit_only(case: MatrixCase, out_lhs: Ptr, out_rhs: Ptr, out_out: Ptr):
    if out_lhs is None or out_rhs is None or out_out is None:
        return Q8_0_MATMUL_ERR_NULL_PTR
    if out_lhs is out_rhs or out_lhs is out_out or out_rhs is out_out:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    snapshot = (
        case.row_count,
        case.col_count,
        case.k_block_count,
        case.lhs_row_stride_blocks,
        case.rhs_col_stride_blocks,
        case.out_row_stride_cells,
        case.lhs_block_capacity,
        case.rhs_block_capacity,
        case.out_cell_capacity,
        out_lhs,
        out_rhs,
        out_out,
    )

    stage_lhs = Ptr()
    stage_rhs = Ptr()
    stage_out = Ptr()
    st_parity = preflight_only_parity(case, stage_lhs, stage_rhs, stage_out)

    st_commit, diag_commit = commit_only(case)

    if snapshot != (
        case.row_count,
        case.col_count,
        case.k_block_count,
        case.lhs_row_stride_blocks,
        case.rhs_col_stride_blocks,
        case.out_row_stride_cells,
        case.lhs_block_capacity,
        case.rhs_block_capacity,
        case.out_cell_capacity,
        out_lhs,
        out_rhs,
        out_out,
    ):
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    if st_parity != st_commit:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if st_commit != Q8_0_MATMUL_OK:
        return st_commit

    if diag_commit is None:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    if stage_lhs.value != diag_commit[0] or stage_rhs.value != diag_commit[1] or stage_out.value != diag_commit[2]:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    out_lhs.value = stage_lhs.value
    out_rhs.value = stage_rhs.value
    out_out.value = stage_out.value
    return Q8_0_MATMUL_OK


def test_commit_only_parity_commit_only_ok():
    case = MatrixCase(
        row_count=3,
        col_count=4,
        k_block_count=2,
        lhs_row_stride_blocks=2,
        rhs_col_stride_blocks=2,
        out_row_stride_cells=4,
        lhs_block_capacity=16,
        rhs_block_capacity=16,
        out_cell_capacity=32,
    )
    out_lhs, out_rhs, out_out = Ptr(-1), Ptr(-1), Ptr(-1)
    st = preflight_only_parity_commit_only(case, out_lhs, out_rhs, out_out)
    assert st == Q8_0_MATMUL_OK
    assert (out_lhs.value, out_rhs.value, out_out.value) == (6, 8, 12)


def test_commit_only_parity_commit_only_alias_rejected():
    case = MatrixCase(
        row_count=1,
        col_count=1,
        k_block_count=1,
        lhs_row_stride_blocks=1,
        rhs_col_stride_blocks=1,
        out_row_stride_cells=1,
        lhs_block_capacity=1,
        rhs_block_capacity=1,
        out_cell_capacity=1,
    )
    shared = Ptr(123)
    other = Ptr(456)
    st = preflight_only_parity_commit_only(case, shared, shared, other)
    assert st == Q8_0_MATMUL_ERR_BAD_DST_LEN
    assert shared.value == 123 and other.value == 456


def test_commit_only_parity_commit_only_capacity_error_no_publish():
    case = MatrixCase(
        row_count=2,
        col_count=2,
        k_block_count=2,
        lhs_row_stride_blocks=2,
        rhs_col_stride_blocks=2,
        out_row_stride_cells=2,
        lhs_block_capacity=3,
        rhs_block_capacity=4,
        out_cell_capacity=4,
    )
    out_lhs, out_rhs, out_out = Ptr(11), Ptr(22), Ptr(33)
    st = preflight_only_parity_commit_only(case, out_lhs, out_rhs, out_out)
    assert st == Q8_0_MATMUL_ERR_BAD_DST_LEN
    assert (out_lhs.value, out_rhs.value, out_out.value) == (11, 22, 33)
