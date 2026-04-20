// sampling.HC
// Integer-only sampling helpers for HolyC inference.

#define SAMPLING_Q16_OK            0
#define SAMPLING_Q16_ERR_NULL_PTR  1
#define SAMPLING_Q16_ERR_BAD_PARAM 2
#define SAMPLING_Q16_ERR_OVERFLOW  4
#define SAMPLING_Q16_ERR_DOMAIN    5

#define SAMPLING_Q16_SHIFT 16
#define SAMPLING_Q16_ONE   (1 << SAMPLING_Q16_SHIFT)

I32 FPQ16MulDivRoundedChecked(I64 a_q16,
                              I64 b_q16,
                              I64 d_q16,
                              I64 *out_q16);
I32 FPQ16TopKSelectIndicesChecked(I64 *logits_q16,
                                  I64 lane_count,
                                  I64 k,
                                  I64 *out_indices,
                                  I64 out_index_capacity);

// Temperature scaling for sampling logits (Q16 lanes):
//   out_logits_q16[i] = round_half_up(logits_q16[i] / temperature_q16)
//                      = round_half_up((logits_q16[i] * 1.0_q16) / temperature_q16)
//
// Contracts:
// - temperature_q16 must be strictly positive.
// - capacities/counts are in I64 lanes and must be non-negative.
// - lane_count lanes are read from logits and written to out.
// - no-partial-write: all lanes are preflighted before commit writes begin.
I32 SamplingQ16ApplyTemperatureChecked(I64 *logits_q16,
                                       I64 logits_capacity,
                                       I64 lane_count,
                                       I64 temperature_q16,
                                       I64 *out_logits_q16,
                                       I64 out_capacity)
{
    I64 lane_index;
    I64 last_index;
    I64 last_byte_offset;
    I64 scaled_lane_q16;
    I32 status;

    if (!logits_q16 || !out_logits_q16)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (logits_capacity < 0 || out_capacity < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (lane_count < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (temperature_q16 <= 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (!lane_count)
        return SAMPLING_Q16_OK;

    if (lane_count > logits_capacity || lane_count > out_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    // Guard span/index math before per-lane pointer indexing.
    last_index = lane_count - 1;
    if (last_index > 0x0FFFFFFFFFFFFFFF)
        return SAMPLING_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)logits_q16 > (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
        return SAMPLING_Q16_ERR_OVERFLOW;
    if ((U64)out_logits_q16 > (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
        return SAMPLING_Q16_ERR_OVERFLOW;

    // Preflight every lane so overflow/domain failures never partially update
    // caller-visible output buffers.
    for (lane_index = 0; lane_index < lane_count; lane_index++)
    {
        status = FPQ16MulDivRoundedChecked(logits_q16[lane_index],
                                           SAMPLING_Q16_ONE,
                                           temperature_q16,
                                           &scaled_lane_q16);
        if (status)
            return status;
    }

    for (lane_index = 0; lane_index < lane_count; lane_index++)
    {
        status = FPQ16MulDivRoundedChecked(logits_q16[lane_index],
                                           SAMPLING_Q16_ONE,
                                           temperature_q16,
                                           &out_logits_q16[lane_index]);
        if (status)
            return status;
    }

    return SAMPLING_Q16_OK;
}

// Checked deterministic top-k index selection for sampling.
//
// Purpose:
//   Select the top-k token indices from a vocab-sized logit vector using
//   integer-only ordering and deterministic stable tie-break semantics.
//
// Ordering contract:
//   - Primary key: larger logit wins.
//   - Tie-break: lower token id wins.
//
// Checked contracts:
//   - logits_q16 and out_indices are non-null.
//   - logits_capacity, vocab_size, k, out_index_capacity are non-negative.
//   - vocab_size <= logits_capacity (only caller-owned initialized lanes read).
//   - k <= vocab_size and k <= out_index_capacity.
//   - No partial writes on all validation failures.
//
// Notes:
//   - The wrapper enforces sampling-specific vocabulary bounds, then delegates
//     ordering and stable tie-break selection to the canonical softmax helper
//     so top-k math exists in one audited place.
I32 SamplingTopKSelectIndicesChecked(I64 *logits_q16,
                                     I64 logits_capacity,
                                     I64 vocab_size,
                                     I64 k,
                                     I64 *out_indices,
                                     I64 out_index_capacity)
{
    if (!logits_q16 || !out_indices)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (logits_capacity < 0 || vocab_size < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (k < 0 || out_index_capacity < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (vocab_size > logits_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (k > vocab_size)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (k > out_index_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    // Canonical stable top-k implementation (descending logit, lower-id tie
    // break) is centralized in softmax.HC for single-source determinism.
    return FPQ16TopKSelectIndicesChecked(logits_q16,
                                         vocab_size,
                                         k,
                                         out_indices,
                                         out_index_capacity);
}
