// q8_0_avx2.HC
// AVX2-preparatory lane packing helpers for Q8_0 kernels.
//
// Why this exists:
// - Upcoming tiled AVX2 matmul kernels need signed 8-bit quant lanes widened to
//   signed 16-bit lanes before pairwise multiply-accumulate.
// - This helper defines the exact widening contract in auditable HolyC first.
// - Hot loops can later replace the scalar widening body with AVX2 asm
//   (`vpmovsxbw`) without changing call-site semantics.

#define Q8_0_AVX2_VALUES_PER_BLOCK   32
#define Q8_0_AVX2_PAIR_COUNT         16

#define Q8_0_AVX2_OK                 0
#define Q8_0_AVX2_ERR_NULL_PTR       1
#define Q8_0_AVX2_ERR_BAD_LEN        2
#define Q8_0_AVX2_ERR_OVERFLOW       3

#define Q8_0_AVX2_I64_MAX            0x7FFFFFFFFFFFFFFF
#define Q8_0_AVX2_I64_MIN            (-0x7FFFFFFFFFFFFFFF - 1)

class Q8_0Block
{
    U16 d_fp16;
    I8 qs[Q8_0_AVX2_VALUES_PER_BLOCK];
};

I32 Q8_0Pack32ToI16LanesAVX2(I8 *src_q8, I64 src_len, I16 *dst_i16, I64 dst_len)
{
    I64 lane;

    if (!src_q8 || !dst_i16)
        return Q8_0_AVX2_ERR_NULL_PTR;

    if (src_len < Q8_0_AVX2_VALUES_PER_BLOCK)
        return Q8_0_AVX2_ERR_BAD_LEN;
    if (dst_len < Q8_0_AVX2_VALUES_PER_BLOCK)
        return Q8_0_AVX2_ERR_BAD_LEN;

    // Lane contract:
    //   dst_i16[i] = SignExtend16(src_q8[i]) for i in [0, 31]
    // This exact order matches the two AVX2 16-lane halves that would be
    // produced by widening bytes 0..15 then 16..31.
    for (lane = 0; lane < Q8_0_AVX2_VALUES_PER_BLOCK; ++lane)
        dst_i16[lane] = (I16)src_q8[lane];

    return Q8_0_AVX2_OK;
}

I32 Q8_0PackBlockToI16LanesAVX2(Q8_0Block *block, I16 *dst_i16, I64 dst_len)
{
    if (!block || !dst_i16)
        return Q8_0_AVX2_ERR_NULL_PTR;

    return Q8_0Pack32ToI16LanesAVX2(block->qs,
                                    Q8_0_AVX2_VALUES_PER_BLOCK,
                                    dst_i16,
                                    dst_len);
}

I32 Q8_0MulI16LanesToI32PairsAVX2(I16 *lhs_i16,
                                  I64 lhs_len,
                                  I16 *rhs_i16,
                                  I64 rhs_len,
                                  I32 *dst_i32,
                                  I64 dst_len)
{
    I64 pair_index;
    I64 lane_base;
    I32 product0;
    I32 product1;

    if (!lhs_i16 || !rhs_i16 || !dst_i32)
        return Q8_0_AVX2_ERR_NULL_PTR;

    if (lhs_len < Q8_0_AVX2_VALUES_PER_BLOCK)
        return Q8_0_AVX2_ERR_BAD_LEN;
    if (rhs_len < Q8_0_AVX2_VALUES_PER_BLOCK)
        return Q8_0_AVX2_ERR_BAD_LEN;
    if (dst_len < Q8_0_AVX2_PAIR_COUNT)
        return Q8_0_AVX2_ERR_BAD_LEN;

    // Contract mirrors AVX2 `vpmaddwd` lane behavior used by quantized dot kernels:
    // - Multiply lane pairs independently in signed 16-bit domain.
    // - Accumulate each adjacent pair into one signed 32-bit output.
    // - Preserve lane order so output pair k maps to input lanes [2k, 2k+1].
    for (pair_index = 0; pair_index < Q8_0_AVX2_PAIR_COUNT; ++pair_index)
    {
        lane_base = pair_index << 1;

        product0 = lhs_i16[lane_base] * rhs_i16[lane_base];
        product1 = lhs_i16[lane_base + 1] * rhs_i16[lane_base + 1];

        dst_i32[pair_index] = product0 + product1;
    }

    return Q8_0_AVX2_OK;
}

I32 Q8_0HSumI32PairsAVX2(I32 *pairs_i32,
                         I64 pairs_len,
                         I64 *out_sum_i64)
{
    I64 pair_index;
    I64 running_sum;
    I64 term;

    if (!pairs_i32 || !out_sum_i64)
        return Q8_0_AVX2_ERR_NULL_PTR;
    if (pairs_len < Q8_0_AVX2_PAIR_COUNT)
        return Q8_0_AVX2_ERR_BAD_LEN;

    // Horizontal reduction contract:
    // - Sum all 16 pairwise I32 partial products into one signed I64 scalar.
    // - Preserve deterministic lane order (0..15) for auditability parity
    //   against the scalar reference path.
    // - Explicit overflow checks keep reduction safe if this helper is reused
    //   for wider accumulation streams in future kernels.
    running_sum = 0;
    for (pair_index = 0; pair_index < Q8_0_AVX2_PAIR_COUNT; ++pair_index)
    {
        term = (I64)pairs_i32[pair_index];

        if (term > 0 && running_sum > Q8_0_AVX2_I64_MAX - term)
            return Q8_0_AVX2_ERR_OVERFLOW;
        if (term < 0 && running_sum < Q8_0_AVX2_I64_MIN - term)
            return Q8_0_AVX2_ERR_OVERFLOW;

        running_sum += term;
    }

    *out_sum_i64 = running_sum;
    return Q8_0_AVX2_OK;
}

I32 Q8_0DotI16LanesAVX2(I16 *lhs_i16,
                        I64 lhs_len,
                        I16 *rhs_i16,
                        I64 rhs_len,
                        I64 *out_dot_i64)
{
    I32 pairs_i32[Q8_0_AVX2_PAIR_COUNT];
    I64 dot_i64;
    I32 status;

    if (!lhs_i16 || !rhs_i16 || !out_dot_i64)
        return Q8_0_AVX2_ERR_NULL_PTR;

    // Compose the lane-dot from the two audited AVX2-shape primitives:
    // 1) `vpmaddwd`-equivalent pairwise multiply-add over 32 I16 lanes.
    // 2) Horizontal sum over 16 I32 pair partials into one I64 scalar.
    // Keeping this composition explicit preserves bit-level auditability while
    // allowing future inline-asm replacement of each stage independently.
    status = Q8_0MulI16LanesToI32PairsAVX2(lhs_i16,
                                           lhs_len,
                                           rhs_i16,
                                           rhs_len,
                                           pairs_i32,
                                           Q8_0_AVX2_PAIR_COUNT);
    if (status != Q8_0_AVX2_OK)
        return status;

    status = Q8_0HSumI32PairsAVX2(pairs_i32,
                                  Q8_0_AVX2_PAIR_COUNT,
                                  &dot_i64);
    if (status != Q8_0_AVX2_OK)
        return status;

    *out_dot_i64 = dot_i64;
    return Q8_0_AVX2_OK;
}
