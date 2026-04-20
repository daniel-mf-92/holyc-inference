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
