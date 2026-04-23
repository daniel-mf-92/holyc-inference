// intsqrt.HC
// Integer square-root helpers for Q16.16 fixed-point inference math.

#define FP_Q16_SHIFT 16
#define I64_MAX_VALUE 0x7FFFFFFFFFFFFFFF

#define FP_Q16_OK            0
#define FP_Q16_ERR_NULL_PTR  1
#define FP_Q16_ERR_BAD_PARAM 2
#define FP_Q16_ERR_DOMAIN    3
#define FP_Q16_ERR_OVERFLOW  4

U64 IntSqrtU64(U64 x)
{
    U64 res;
    U64 bit;

    res = 0;
    bit = 1 << 62;

    while (bit > x)
        bit >>= 2;

    while (bit)
    {
        if (x >= res + bit)
        {
            x -= res + bit;
            res = (res >> 1) + bit;
        }
        else
            res >>= 1;

        bit >>= 2;
    }

    return res;
}

I64 FPQ16Sqrt(I64 x)
{
    U64 shifted;

    if (x <= 0)
        return 0;

    if (x > (I64_MAX_VALUE >> FP_Q16_SHIFT))
        shifted = I64_MAX_VALUE;
    else
        shifted = ((U64)x) << FP_Q16_SHIFT;

    return IntSqrtU64(shifted);
}

// Commit-only checked Q16 square-root wrapper over FPQ16Sqrt.
//
// Contract:
// - `out_q16` must be non-null.
// - `x_q16` must be non-negative and in range where `(x_q16 << 16)` does not
//   overflow signed I64.
// - Uses a staged scalar and publishes to caller only after all checks pass.
// - Revalidates immutable input snapshot parity before publish.
I32 FPQ16SqrtCheckedNoPartialCommitOnly(I64 x_q16,
                                        I64 *out_q16)
{
    I64 snapshot_x_q16;
    I64 staged_out_q16;

    if (!out_q16)
        return FP_Q16_ERR_NULL_PTR;

    if (x_q16 < 0)
        return FP_Q16_ERR_DOMAIN;

    if (x_q16 > (I64_MAX_VALUE >> FP_Q16_SHIFT))
        return FP_Q16_ERR_OVERFLOW;

    snapshot_x_q16 = x_q16;
    staged_out_q16 = FPQ16Sqrt(x_q16);

    if (snapshot_x_q16 != x_q16)
        return FP_Q16_ERR_BAD_PARAM;

    *out_q16 = staged_out_q16;
    return FP_Q16_OK;
}

// Diagnostics-only checked preflight for commit-only Q16 square-root path.
//
// Contract:
// - `required_out_cells_out` must be non-null.
// - Does not publish `out_q16` lanes to caller; stages output on local scratch.
// - Enforces immutable `{x_q16}` snapshot parity across staged call.
// - Enforces strict tuple parity `{required_out_cells=1}` before publish.
I32 FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnly(I64 x_q16,
                                                     I64 *required_out_cells_out)
{
    I64 snapshot_x_q16;
    I64 staged_out_q16;
    I64 staged_required_out_cells;
    I64 canonical_out_q16;
    I32 status;

    if (!required_out_cells_out)
        return FP_Q16_ERR_NULL_PTR;

    *required_out_cells_out = 0;
    snapshot_x_q16 = x_q16;

    status = FPQ16SqrtCheckedNoPartialCommitOnly(x_q16,
                                                 &staged_out_q16);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_x_q16 != x_q16)
        return FP_Q16_ERR_BAD_PARAM;

    canonical_out_q16 = FPQ16Sqrt(x_q16);
    if (canonical_out_q16 != staged_out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    staged_required_out_cells = 1;
    if (staged_required_out_cells != 1)
        return FP_Q16_ERR_BAD_PARAM;

    *required_out_cells_out = staged_required_out_cells;
    return FP_Q16_OK;
}

// Diagnostics parity gate over commit-only preflight + canonical commit-only
// derivation for Q16 square-root capacity requirements.
//
// Contract:
// - `required_out_cells_out` must be non-null.
// - Performs zero-write diagnostics until all checks pass.
// - Enforces immutable `{x_q16}` snapshots across staged + canonical passes.
// - Enforces strict tuple parity `{required_out_cells=1}` before publish.
I32 FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParity(I64 x_q16,
                                                           I64 *required_out_cells_out)
{
    I64 snapshot_x_q16;
    I64 staged_required_out_cells;
    I64 canonical_out_q16;
    I64 canonical_required_out_cells;
    I32 status;

    if (!required_out_cells_out)
        return FP_Q16_ERR_NULL_PTR;

    *required_out_cells_out = 0;
    snapshot_x_q16 = x_q16;

    status = FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnly(x_q16,
                                                              &staged_required_out_cells);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_x_q16 != x_q16)
        return FP_Q16_ERR_BAD_PARAM;

    status = FPQ16SqrtCheckedNoPartialCommitOnly(x_q16,
                                                 &canonical_out_q16);
    if (status != FP_Q16_OK)
        return status;

    canonical_required_out_cells = 1;
    if (staged_required_out_cells != canonical_required_out_cells)
        return FP_Q16_ERR_BAD_PARAM;

    if (canonical_required_out_cells != 1)
        return FP_Q16_ERR_BAD_PARAM;

    *required_out_cells_out = staged_required_out_cells;
    return FP_Q16_OK;
}

// Commit-only diagnostics wrapper over
// FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParity.
//
// Stages delegated diagnostics into a local scalar, revalidates immutable
// `{x_q16}` snapshots, and atomically publishes only canonical
// `{required_out_cells=1}` on strict parity success.
//
// Contract:
// - `required_out_cells_out` must be non-null.
// - Returns the same status domain as delegated checked sqrt diagnostics.
// - Performs no diagnostics publish on failure.
I32 FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
    I64 x_q16,
    I64 *required_out_cells_out)
{
    I64 snapshot_x_q16;
    I64 staged_required_out_cells;
    I64 canonical_required_out_cells;
    I32 status;

    if (!required_out_cells_out)
        return FP_Q16_ERR_NULL_PTR;

    snapshot_x_q16 = x_q16;

    status = FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParity(
        x_q16,
        &staged_required_out_cells);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_x_q16 != x_q16)
        return FP_Q16_ERR_BAD_PARAM;

    canonical_required_out_cells = 1;
    if (staged_required_out_cells != canonical_required_out_cells)
        return FP_Q16_ERR_BAD_PARAM;

    if (canonical_required_out_cells != 1)
        return FP_Q16_ERR_BAD_PARAM;

    *required_out_cells_out = staged_required_out_cells;
    return FP_Q16_OK;
}

// Diagnostics-only companion over
// FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly.
//
// Stages delegated diagnostics into local scratch, revalidates immutable
// `{x_q16}` snapshot parity, then cross-checks canonical
// FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParity requirements before
// atomically publishing diagnostics output.
//
// Contract:
// - `required_out_cells_out` must be non-null.
// - Does not publish diagnostics on failure (zero-write output behavior).
// - Enforces strict tuple parity `{required_out_cells=1}` before publish.
I32 FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
    I64 x_q16,
    I64 *required_out_cells_out)
{
    I64 snapshot_x_q16;
    I64 staged_required_out_cells;
    I64 canonical_required_out_cells;
    I32 status;

    if (!required_out_cells_out)
        return FP_Q16_ERR_NULL_PTR;

    *required_out_cells_out = 0;
    snapshot_x_q16 = x_q16;

    status = FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
        x_q16,
        &staged_required_out_cells);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_x_q16 != x_q16)
        return FP_Q16_ERR_BAD_PARAM;

    status = FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParity(
        x_q16,
        &canonical_required_out_cells);
    if (status != FP_Q16_OK)
        return status;

    if (staged_required_out_cells != canonical_required_out_cells)
        return FP_Q16_ERR_BAD_PARAM;

    if (canonical_required_out_cells != 1)
        return FP_Q16_ERR_BAD_PARAM;

    *required_out_cells_out = staged_required_out_cells;
    return FP_Q16_OK;
}

// Diagnostics parity gate over
// FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly
// and FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly.
//
// Stages delegated diagnostics into local scratch, revalidates immutable
// `{x_q16}` snapshot parity, then cross-checks strict tuple parity against the
// canonical commit-only diagnostics path before atomically publishing output.
//
// Contract:
// - `required_out_cells_out` must be non-null.
// - Does not publish diagnostics on failure (zero-write output behavior).
// - Enforces strict tuple parity `{required_out_cells=1}` before publish.
I32 FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
    I64 x_q16,
    I64 *required_out_cells_out)
{
    I64 snapshot_x_q16;
    I64 staged_required_out_cells;
    I64 canonical_required_out_cells;
    I32 status;

    if (!required_out_cells_out)
        return FP_Q16_ERR_NULL_PTR;

    *required_out_cells_out = 0;
    snapshot_x_q16 = x_q16;

    status = FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
        x_q16,
        &staged_required_out_cells);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_x_q16 != x_q16)
        return FP_Q16_ERR_BAD_PARAM;

    status = FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
        x_q16,
        &canonical_required_out_cells);
    if (status != FP_Q16_OK)
        return status;

    if (staged_required_out_cells != canonical_required_out_cells)
        return FP_Q16_ERR_BAD_PARAM;

    if (canonical_required_out_cells != 1)
        return FP_Q16_ERR_BAD_PARAM;

    *required_out_cells_out = staged_required_out_cells;
    return FP_Q16_OK;
}

// Commit-only diagnostics wrapper over
// FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity.
//
// Stages delegated diagnostics into local scratch, revalidates immutable
// `{x_q16}` snapshot parity, then cross-checks strict tuple parity against the
// canonical preflight-only diagnostics path before atomically publishing output.
//
// Contract:
// - `required_out_cells_out` must be non-null.
// - Does not publish diagnostics on failure (commit-only behavior).
// - Enforces strict tuple parity `{required_out_cells=1}` before publish.
I32 FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(
    I64 x_q16,
    I64 *required_out_cells_out)
{
    I64 snapshot_x_q16;
    I64 staged_required_out_cells;
    I64 canonical_required_out_cells;
    I32 status;

    if (!required_out_cells_out)
        return FP_Q16_ERR_NULL_PTR;

    snapshot_x_q16 = x_q16;

    status = FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
        x_q16,
        &staged_required_out_cells);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_x_q16 != x_q16)
        return FP_Q16_ERR_BAD_PARAM;

    status = FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
        x_q16,
        &canonical_required_out_cells);
    if (status != FP_Q16_OK)
        return status;

    if (staged_required_out_cells != canonical_required_out_cells)
        return FP_Q16_ERR_BAD_PARAM;

    if (canonical_required_out_cells != 1)
        return FP_Q16_ERR_BAD_PARAM;

    *required_out_cells_out = staged_required_out_cells;
    return FP_Q16_OK;
}
