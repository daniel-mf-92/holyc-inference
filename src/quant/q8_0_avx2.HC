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

#define Q8_0_AVX2_OK                 0
#define Q8_0_AVX2_ERR_NULL_PTR       1
#define Q8_0_AVX2_ERR_BAD_LEN        2

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
