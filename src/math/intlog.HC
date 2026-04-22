// intlog.HC
// Integer-only Q16.16 natural-log approximation for TempleOS HolyC inference.

#define FP_Q16_SHIFT 16
#define FP_Q16_ONE   (1 << FP_Q16_SHIFT)

#define LOG_Q16_LN2  45426 // round(ln(2) * 65536)
#define LOG_Q16_INV_LN2 94549 // round((1/ln(2)) * 65536)
#define LOG_Q16_HALF (1 << (FP_Q16_SHIFT - 1))
#define LOG_Q16_SPLIT ((FP_Q16_ONE * 3) / 2) // 1.5 in Q16

#define I64_MAX_VALUE  0x7FFFFFFFFFFFFFFF
#define I64_MIN_VALUE -0x8000000000000000

I64 FPQ16Mul(I64 a, I64 b);

I64 FPQ16Ln1pPoly(I64 y_q16)
{
    I64 y2;
    I64 y3;
    I64 y4;
    I64 y5;

    // 5th-order alternating series:
    // ln(1+y) ~= y - y^2/2 + y^3/3 - y^4/4 + y^5/5
    // Accurate and monotonic on y in [-0.25, 0.5].
    y2 = FPQ16Mul(y_q16, y_q16);
    y3 = FPQ16Mul(y2, y_q16);
    y4 = FPQ16Mul(y3, y_q16);
    y5 = FPQ16Mul(y4, y_q16);

    return y_q16 - (y2 / 2) + (y3 / 3) - (y4 / 4) + (y5 / 5);
}

I64 FPQ16LnReduce(I64 x_q16, I64 *k_out)
{
    U64 m_q16;
    I64 k;

    if (!k_out)
        return 0;
    if (x_q16 <= 0)
    {
        *k_out = 0;
        return 0;
    }

    // Normalize x = m * 2^k where m in [1, 2) in Q16.
    m_q16 = (U64)x_q16;
    k = 0;

    while (m_q16 >= (((U64)FP_Q16_ONE) << 1))
    {
        m_q16 >>= 1;
        k++;
    }
    while (m_q16 < (U64)FP_Q16_ONE)
    {
        m_q16 <<= 1;
        k--;
    }

    *k_out = k;
    return (I64)m_q16;
}

I64 FPQ16Ln(I64 x_q16)
{
    I64 k;
    I64 m_q16;
    I64 y_q16;
    I64 poly_q16;
    I64 base_q16;
    I64 ln_q16;

    // ln(x) is undefined for x <= 0. Saturate to a deterministic floor.
    if (x_q16 <= 0)
        return I64_MIN_VALUE;

    m_q16 = FPQ16LnReduce(x_q16, &k);

    // Piecewise range split for monotonic polynomial accuracy:
    // branch A: y in [0, 0.5), branch B: y in [-0.25, 0).
    if (m_q16 >= LOG_Q16_SPLIT)
    {
        m_q16 = (m_q16 + 1) >> 1;
        k++;
    }

    y_q16 = m_q16 - FP_Q16_ONE;
    poly_q16 = FPQ16Ln1pPoly(y_q16);

    base_q16 = k * LOG_Q16_LN2;

    if (base_q16 > 0 && poly_q16 > I64_MAX_VALUE - base_q16)
        return I64_MAX_VALUE;
    if (base_q16 < 0 && poly_q16 < I64_MIN_VALUE - base_q16)
        return I64_MIN_VALUE;

    ln_q16 = base_q16 + poly_q16;
    return ln_q16;
}

I64 FPQ16Log2(I64 x_q16)
{
    I64 ln_q16;
    I64 abs_ln;
    I64 abs_inv;
    I64 max_abs_ln;
    I64 prod;

    ln_q16 = FPQ16Ln(x_q16);

    // Preserve ln(x<=0) deterministic domain floor.
    if (ln_q16 == I64_MIN_VALUE)
        return I64_MIN_VALUE;

    // log2(x) = ln(x) * (1/ln(2)).
    // Keep overflow checks explicit before multiply.
    abs_ln = ln_q16;
    if (abs_ln < 0)
        abs_ln = -abs_ln;

    abs_inv = LOG_Q16_INV_LN2;
    max_abs_ln = I64_MAX_VALUE / abs_inv;
    if (abs_ln > max_abs_ln)
    {
        if (ln_q16 < 0)
            return I64_MIN_VALUE;
        return I64_MAX_VALUE;
    }

    prod = ln_q16 * LOG_Q16_INV_LN2;

    // Q16*Q16 -> Q32. Round to Q16 with sign-aware half-up policy.
    if (prod >= 0)
        return (prod + LOG_Q16_HALF) >> FP_Q16_SHIFT;
    return -(((-prod) + LOG_Q16_HALF) >> FP_Q16_SHIFT);
}

I64 FPQ16LnRatio(I64 num_q16, I64 den_q16)
{
    I64 num_k;
    I64 den_k;
    I64 num_m_q16;
    I64 den_m_q16;
    I64 num_y_q16;
    I64 den_y_q16;
    I64 num_poly_q16;
    I64 den_poly_q16;
    I64 poly_delta_q16;
    I64 k_delta;
    I64 abs_k_delta;
    I64 max_k_delta;
    I64 base_q16;
    I64 ln_ratio_q16;

    // ln(num/den) is undefined when either side is non-positive.
    if (num_q16 <= 0 || den_q16 <= 0)
        return I64_MIN_VALUE;

    num_m_q16 = FPQ16LnReduce(num_q16, &num_k);
    den_m_q16 = FPQ16LnReduce(den_q16, &den_k);

    // Mirror FPQ16Ln split to keep polynomial range tightly bounded.
    if (num_m_q16 >= LOG_Q16_SPLIT)
    {
        num_m_q16 = (num_m_q16 + 1) >> 1;
        num_k++;
    }
    if (den_m_q16 >= LOG_Q16_SPLIT)
    {
        den_m_q16 = (den_m_q16 + 1) >> 1;
        den_k++;
    }

    num_y_q16 = num_m_q16 - FP_Q16_ONE;
    den_y_q16 = den_m_q16 - FP_Q16_ONE;
    num_poly_q16 = FPQ16Ln1pPoly(num_y_q16);
    den_poly_q16 = FPQ16Ln1pPoly(den_y_q16);

    // poly_delta = ln(m_num) - ln(m_den) with explicit overflow guards.
    if (den_poly_q16 < 0 && num_poly_q16 > I64_MAX_VALUE + den_poly_q16)
        return I64_MAX_VALUE;
    if (den_poly_q16 > 0 && num_poly_q16 < I64_MIN_VALUE + den_poly_q16)
        return I64_MIN_VALUE;
    poly_delta_q16 = num_poly_q16 - den_poly_q16;

    // base = (k_num - k_den) * ln(2) in Q16.
    if (den_k < 0 && num_k > I64_MAX_VALUE + den_k)
        return I64_MAX_VALUE;
    if (den_k > 0 && num_k < I64_MIN_VALUE + den_k)
        return I64_MIN_VALUE;
    k_delta = num_k - den_k;

    abs_k_delta = k_delta;
    if (abs_k_delta < 0)
        abs_k_delta = -abs_k_delta;
    max_k_delta = I64_MAX_VALUE / LOG_Q16_LN2;
    if (abs_k_delta > max_k_delta)
    {
        if (k_delta < 0)
            return I64_MIN_VALUE;
        return I64_MAX_VALUE;
    }
    base_q16 = k_delta * LOG_Q16_LN2;

    if (base_q16 > 0 && poly_delta_q16 > I64_MAX_VALUE - base_q16)
        return I64_MAX_VALUE;
    if (base_q16 < 0 && poly_delta_q16 < I64_MIN_VALUE - base_q16)
        return I64_MIN_VALUE;

    ln_ratio_q16 = base_q16 + poly_delta_q16;
    return ln_ratio_q16;
}

// Checked no-partial array ln() helpers.
// These helpers keep the scalar FPQ16Ln math path as the single source of
// truth while adding explicit geometry contracts for bulk tensor lanes.

#define INTLOG_STATUS_OK 0
#define INTLOG_STATUS_BADPTR -1
#define INTLOG_STATUS_BADCOUNT -2
#define INTLOG_STATUS_OVERFLOW -3
#define INTLOG_STATUS_DOMAIN -4

I32 FPQ16LnArrayCheckedNoPartialPreflight(I64 *input_q16,
                                          I64 count,
                                          I64 *out_q16,
                                          I64 *required_bytes_out)
{
    I64 bytes;

    if (!required_bytes_out)
        return INTLOG_STATUS_BADPTR;

    *required_bytes_out = 0;

    if (count < 0)
        return INTLOG_STATUS_BADCOUNT;
    if (!input_q16 || !out_q16)
        return INTLOG_STATUS_BADPTR;

    // Each lane writes one Q16 I64 output cell.
    if (count == 0)
        bytes = 0;
    else if (count > (I64_MAX_VALUE / 8))
        return INTLOG_STATUS_OVERFLOW;
    else
        bytes = count * 8;

    *required_bytes_out = bytes;
    return INTLOG_STATUS_OK;
}

I32 FPQ16LnArrayCheckedNoPartial(I64 *input_q16,
                                 I64 count,
                                 I64 *out_q16,
                                 I64 out_capacity_bytes)
{
    I64 required_bytes;
    I64 i;
    I32 status;

    status = FPQ16LnArrayCheckedNoPartialPreflight(input_q16,
                                                   count,
                                                   out_q16,
                                                   &required_bytes);
    if (status != INTLOG_STATUS_OK)
        return status;
    if (out_capacity_bytes < required_bytes)
        return INTLOG_STATUS_BADCOUNT;

    // All preconditions are validated up-front, so the publish loop is
    // deterministic and no-partial on hard failures.
    for (i = 0; i < count; i++)
        out_q16[i] = FPQ16Ln(input_q16[i]);

    return INTLOG_STATUS_OK;
}

// Commit-only wrapper for ln() array lanes.
//
// This path computes all output lanes into staged scratch first, then
// atomically publishes to caller output only after every lane is available.
// That keeps no-partial semantics explicit even when future checked ln()
// kernels gain failure modes beyond pure scalar FPQ16Ln.
I32 FPQ16LnArrayCheckedNoPartialCommitOnly(I64 *input_q16,
                                           I64 count,
                                           I64 *out_q16,
                                           I64 out_capacity_bytes)
{
    I64 required_bytes;
    I64 i;
    I64 *staged_out_q16;
    I32 status;

    status = FPQ16LnArrayCheckedNoPartialPreflight(input_q16,
                                                   count,
                                                   out_q16,
                                                   &required_bytes);
    if (status != INTLOG_STATUS_OK)
        return status;
    if (out_capacity_bytes < required_bytes)
        return INTLOG_STATUS_BADCOUNT;

    if (!count)
        return INTLOG_STATUS_OK;

    staged_out_q16 = MAlloc(required_bytes);
    if (!staged_out_q16)
        return INTLOG_STATUS_BADPTR;

    for (i = 0; i < count; i++)
        staged_out_q16[i] = FPQ16Ln(input_q16[i]);

    for (i = 0; i < count; i++)
        out_q16[i] = staged_out_q16[i];

    Free(staged_out_q16);
    return INTLOG_STATUS_OK;
}

// Diagnostics-only preflight wrapper for commit-only ln() array path.
//
// Contracts:
//   - Does not write out_q16 lanes.
//   - Publishes required bytes only when geometry/capacity checks pass.
//   - Verifies required-byte geometry parity against canonical lane sizing.
I32 FPQ16LnArrayCheckedNoPartialCommitOnlyPreflightOnly(I64 *input_q16,
                                                        I64 count,
                                                        I64 *out_q16,
                                                        I64 out_capacity_bytes,
                                                        I64 *required_bytes_out)
{
    I64 preflight_required_bytes;
    I64 geometry_required_bytes;
    I32 status;

    if (!required_bytes_out)
        return INTLOG_STATUS_BADPTR;

    *required_bytes_out = 0;

    status = FPQ16LnArrayCheckedNoPartialPreflight(input_q16,
                                                   count,
                                                   out_q16,
                                                   &preflight_required_bytes);
    if (status != INTLOG_STATUS_OK)
        return status;

    if (out_capacity_bytes < preflight_required_bytes)
        return INTLOG_STATUS_BADCOUNT;

    if (!count)
        geometry_required_bytes = 0;
    else if (count > (I64_MAX_VALUE / 8))
        return INTLOG_STATUS_OVERFLOW;
    else
        geometry_required_bytes = count * 8;

    if (geometry_required_bytes != preflight_required_bytes)
        return INTLOG_STATUS_OVERFLOW;

    *required_bytes_out = preflight_required_bytes;
    return INTLOG_STATUS_OK;
}

// Diagnostics-only parity gate over
// FPQ16LnArrayCheckedNoPartialCommitOnlyPreflightOnly and canonical
// FPQ16LnArrayCheckedNoPartial staged on scratch.
//
// Contract:
// - Does not publish any lane to caller out_q16.
// - Revalidates immutable {input_q16,count,out_q16,out_capacity_bytes} snapshots.
// - Requires exact tuple parity on {required_bytes} before publish.
I32 FPQ16LnArrayCheckedNoPartialCommitOnlyPreflightOnlyParity(I64 *input_q16,
                                                              I64 count,
                                                              I64 *out_q16,
                                                              I64 out_capacity_bytes,
                                                              I64 *required_bytes_out)
{
    I64 *snapshot_input_q16;
    I64 snapshot_count;
    I64 *snapshot_out_q16;
    I64 snapshot_out_capacity_bytes;
    I64 staged_required_bytes;
    I64 canonical_required_bytes;
    I64 *scratch_out_q16;
    I64 i;
    I32 status;

    if (!required_bytes_out)
        return INTLOG_STATUS_BADPTR;

    *required_bytes_out = 0;

    snapshot_input_q16 = input_q16;
    snapshot_count = count;
    snapshot_out_q16 = out_q16;
    snapshot_out_capacity_bytes = out_capacity_bytes;

    status = FPQ16LnArrayCheckedNoPartialCommitOnlyPreflightOnly(input_q16,
                                                                 count,
                                                                 out_q16,
                                                                 out_capacity_bytes,
                                                                 &staged_required_bytes);
    if (status != INTLOG_STATUS_OK)
        return status;

    if (snapshot_input_q16 != input_q16 ||
        snapshot_count != count ||
        snapshot_out_q16 != out_q16 ||
        snapshot_out_capacity_bytes != out_capacity_bytes)
        return INTLOG_STATUS_BADCOUNT;

    if (!count)
        canonical_required_bytes = 0;
    else if (count > (I64_MAX_VALUE / 8))
        return INTLOG_STATUS_OVERFLOW;
    else
        canonical_required_bytes = count * 8;

    if (staged_required_bytes != canonical_required_bytes)
        return INTLOG_STATUS_OVERFLOW;

    if (!count)
    {
        *required_bytes_out = staged_required_bytes;
        return INTLOG_STATUS_OK;
    }

    scratch_out_q16 = MAlloc(canonical_required_bytes);
    if (!scratch_out_q16)
        return INTLOG_STATUS_BADPTR;

    status = FPQ16LnArrayCheckedNoPartial(input_q16,
                                          count,
                                          scratch_out_q16,
                                          canonical_required_bytes);
    if (status != INTLOG_STATUS_OK)
    {
        Free(scratch_out_q16);
        return status;
    }

    if (snapshot_input_q16 != input_q16 ||
        snapshot_count != count ||
        snapshot_out_q16 != out_q16 ||
        snapshot_out_capacity_bytes != out_capacity_bytes)
    {
        Free(scratch_out_q16);
        return INTLOG_STATUS_BADCOUNT;
    }

    // Canonical lane parity: every scratch lane must match scalar FPQ16Ln.
    for (i = 0; i < count; i++)
    {
        if (scratch_out_q16[i] != FPQ16Ln(input_q16[i]))
        {
            Free(scratch_out_q16);
            return INTLOG_STATUS_DOMAIN;
        }
    }

    Free(scratch_out_q16);
    *required_bytes_out = staged_required_bytes;
    return INTLOG_STATUS_OK;
}

I32 FPQ16LnArrayCheckedNoPartialDefault(I64 *input_q16,
                                        I64 count,
                                        I64 *out_q16)
{
    I64 required_bytes;
    I32 status;

    status = FPQ16LnArrayCheckedNoPartialPreflight(input_q16,
                                                   count,
                                                   out_q16,
                                                   &required_bytes);
    if (status != INTLOG_STATUS_OK)
        return status;

    return FPQ16LnArrayCheckedNoPartial(input_q16,
                                        count,
                                        out_q16,
                                        required_bytes);
}

I32 FPQ16Log2ArrayCheckedNoPartialPreflight(I64 *input_q16,
                                            I64 count,
                                            I64 *out_q16,
                                            I64 *required_bytes_out)
{
    I64 bytes;

    if (!required_bytes_out)
        return INTLOG_STATUS_BADPTR;

    *required_bytes_out = 0;

    if (count < 0)
        return INTLOG_STATUS_BADCOUNT;
    if (!input_q16 || !out_q16)
        return INTLOG_STATUS_BADPTR;

    // Each lane writes one Q16 I64 output cell.
    if (count == 0)
        bytes = 0;
    else if (count > (I64_MAX_VALUE / 8))
        return INTLOG_STATUS_OVERFLOW;
    else
        bytes = count * 8;

    *required_bytes_out = bytes;
    return INTLOG_STATUS_OK;
}

I32 FPQ16Log2ArrayCheckedNoPartial(I64 *input_q16,
                                   I64 count,
                                   I64 *out_q16,
                                   I64 out_capacity_bytes)
{
    I64 required_bytes;
    I64 i;
    I32 status;

    status = FPQ16Log2ArrayCheckedNoPartialPreflight(input_q16,
                                                     count,
                                                     out_q16,
                                                     &required_bytes);
    if (status != INTLOG_STATUS_OK)
        return status;
    if (out_capacity_bytes < required_bytes)
        return INTLOG_STATUS_BADCOUNT;

    // All preconditions are validated up-front, so the publish loop is
    // deterministic and no-partial on hard failures.
    for (i = 0; i < count; i++)
        out_q16[i] = FPQ16Log2(input_q16[i]);

    return INTLOG_STATUS_OK;
}

I32 FPQ16Log2ArrayCheckedNoPartialDefault(I64 *input_q16,
                                          I64 count,
                                          I64 *out_q16)
{
    I64 required_bytes;
    I32 status;

    status = FPQ16Log2ArrayCheckedNoPartialPreflight(input_q16,
                                                     count,
                                                     out_q16,
                                                     &required_bytes);
    if (status != INTLOG_STATUS_OK)
        return status;

    return FPQ16Log2ArrayCheckedNoPartial(input_q16,
                                          count,
                                          out_q16,
                                          required_bytes);
}

I32 FPQ16LnRatioArrayCheckedNoPartialPreflight(I64 *num_q16,
                                               I64 *den_q16,
                                               I64 count,
                                               I64 *out_q16,
                                               I64 *required_bytes_out)
{
    I64 bytes;
    I64 i;

    if (!required_bytes_out)
        return INTLOG_STATUS_BADPTR;

    *required_bytes_out = 0;

    if (count < 0)
        return INTLOG_STATUS_BADCOUNT;
    if (!num_q16 || !den_q16 || !out_q16)
        return INTLOG_STATUS_BADPTR;

    if (count == 0)
    {
        *required_bytes_out = 0;
        return INTLOG_STATUS_OK;
    }
    if (count > (I64_MAX_VALUE / 8))
        return INTLOG_STATUS_OVERFLOW;

    bytes = count * 8;

    // Strict ln(num/den) domain: every lane must be strictly positive.
    // This scan runs before any output publication to keep no-partial writes.
    for (i = 0; i < count; i++)
    {
        if (num_q16[i] <= 0 || den_q16[i] <= 0)
            return INTLOG_STATUS_DOMAIN;
    }

    *required_bytes_out = bytes;
    return INTLOG_STATUS_OK;
}

I32 FPQ16LnRatioArrayCheckedNoPartial(I64 *num_q16,
                                      I64 *den_q16,
                                      I64 count,
                                      I64 *out_q16,
                                      I64 out_capacity_bytes)
{
    I64 required_bytes;
    I64 i;
    I32 status;

    status = FPQ16LnRatioArrayCheckedNoPartialPreflight(num_q16,
                                                        den_q16,
                                                        count,
                                                        out_q16,
                                                        &required_bytes);
    if (status != INTLOG_STATUS_OK)
        return status;
    if (out_capacity_bytes < required_bytes)
        return INTLOG_STATUS_BADCOUNT;

    // Domain and geometry were fully validated up-front. Publish is
    // deterministic and all-or-nothing on failure paths.
    for (i = 0; i < count; i++)
        out_q16[i] = FPQ16LnRatio(num_q16[i], den_q16[i]);

    return INTLOG_STATUS_OK;
}

I32 FPQ16LnRatioArrayCheckedNoPartialDefault(I64 *num_q16,
                                             I64 *den_q16,
                                             I64 count,
                                             I64 *out_q16)
{
    I64 required_bytes;
    I32 status;

    status = FPQ16LnRatioArrayCheckedNoPartialPreflight(num_q16,
                                                        den_q16,
                                                        count,
                                                        out_q16,
                                                        &required_bytes);
    if (status != INTLOG_STATUS_OK)
        return status;

    return FPQ16LnRatioArrayCheckedNoPartial(num_q16,
                                             den_q16,
                                             count,
                                             out_q16,
                                             required_bytes);
}
