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

// Checked deterministic top-p cutoff selector over descending probability
// lanes.
//
// Output contract:
//   - Writes exactly one cutoff index to *out_cutoff_index on success.
//   - Returned index is the first crossing i where
//       sum(probs_q16[0..i]) >= top_p_q16.
//   - If cumulative mass never crosses due finite-lane residue, result clamps
//     deterministically to lane_count - 1.
//   - No partial writes on all failure paths.
//
// Checked guards:
//   - probs_q16 and out_cutoff_index are non-null.
//   - probs_capacity/lane_count are non-negative.
//   - lane_count > 0 and lane_count <= probs_capacity.
//   - top_p_q16 in (0, 1.0_q16].
//   - Pointer-span arithmetic for probs_q16[lane_count] cannot overflow U64.
//   - Each probability lane is in [0, 1.0_q16].
//   - Probabilities are monotone non-increasing.
//   - Cumulative mass is monotone, cannot overflow I64, never exceeds 1.0_q16,
//     and total mass must equal 1.0_q16.
I32 SamplingTopPSelectCutoffChecked(I64 *probs_q16,
                                    I64 probs_capacity,
                                    I64 lane_count,
                                    I64 top_p_q16,
                                    I64 *out_cutoff_index)
{
    I64 lane_index;
    I64 last_index;
    I64 last_byte_offset;
    I64 lane_prob_q16;
    I64 prev_prob_q16;
    I64 cumulative_q16;
    I64 prev_cumulative_q16;
    I64 cutoff_index;

    if (!probs_q16 || !out_cutoff_index)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (probs_capacity < 0 || lane_count < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (!lane_count)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (lane_count > probs_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (top_p_q16 <= 0 || top_p_q16 > SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    // Guard span/index arithmetic before lane loads.
    last_index = lane_count - 1;
    if (last_index > 0x0FFFFFFFFFFFFFFF)
        return SAMPLING_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)probs_q16 > (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
        return SAMPLING_Q16_ERR_OVERFLOW;

    cumulative_q16 = 0;
    prev_cumulative_q16 = 0;
    prev_prob_q16 = SAMPLING_Q16_ONE;
    cutoff_index = -1;

    for (lane_index = 0; lane_index < lane_count; lane_index++)
    {
        lane_prob_q16 = probs_q16[lane_index];
        if (lane_prob_q16 < 0 || lane_prob_q16 > SAMPLING_Q16_ONE)
            return SAMPLING_Q16_ERR_BAD_PARAM;
        if (lane_prob_q16 > prev_prob_q16)
            return SAMPLING_Q16_ERR_BAD_PARAM;
        if (lane_prob_q16 > (0x7FFFFFFFFFFFFFFF - cumulative_q16))
            return SAMPLING_Q16_ERR_OVERFLOW;

        cumulative_q16 += lane_prob_q16;
        if (cumulative_q16 < prev_cumulative_q16)
            return SAMPLING_Q16_ERR_OVERFLOW;
        if (cumulative_q16 > SAMPLING_Q16_ONE)
            return SAMPLING_Q16_ERR_BAD_PARAM;

        if (cutoff_index < 0 && cumulative_q16 >= top_p_q16)
            cutoff_index = lane_index;

        prev_prob_q16 = lane_prob_q16;
        prev_cumulative_q16 = cumulative_q16;
    }

    if (cumulative_q16 != SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (cutoff_index < 0)
        cutoff_index = lane_count - 1;

    *out_cutoff_index = cutoff_index;
    return SAMPLING_Q16_OK;
}

// Checked in-place repetition-penalty pass over logits using token history.
//
// Integer penalty rule (Q16):
//   - For positive logits: divide by penalty (reduces favored repeats).
//   - For negative logits: multiply by penalty (pushes further negative).
//
// Determinism / no-partial semantics:
//   - Each token id present in history is penalized at most once.
//   - Duplicate history entries are skipped after first occurrence.
//   - Full preflight pass validates pointers, bounds, history token ids, and
//     per-token fixed-point arithmetic before any in-place commit writes.
I32 SamplingApplyRepetitionPenaltyChecked(I64 *logits_q16,
                                          I64 logits_capacity,
                                          I64 vocab_size,
                                          I64 *token_history,
                                          I64 token_history_capacity,
                                          I64 token_history_count,
                                          I64 penalty_q16)
{
    I64 history_index;
    I64 prior_index;
    I64 token_id;
    I64 last_index;
    I64 last_byte_offset;
    I64 adjusted_logit_q16;
    I64 base_logit_q16;
    I32 status;
    Bool already_seen;

    if (!logits_q16)
        return SAMPLING_Q16_ERR_NULL_PTR;
    if (token_history_count > 0 && !token_history)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (logits_capacity < 0 || vocab_size < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (token_history_capacity < 0 || token_history_count < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (vocab_size > logits_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (token_history_count > token_history_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (penalty_q16 <= 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (!vocab_size || !token_history_count)
        return SAMPLING_Q16_OK;

    // Guard logits pointer span for vocab-sized indexed loads/stores.
    last_index = vocab_size - 1;
    if (last_index > 0x0FFFFFFFFFFFFFFF)
        return SAMPLING_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)logits_q16 > (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
        return SAMPLING_Q16_ERR_OVERFLOW;

    // Guard token-history pointer span for indexed reads.
    last_index = token_history_count - 1;
    if (last_index > 0x0FFFFFFFFFFFFFFF)
        return SAMPLING_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)token_history > (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
        return SAMPLING_Q16_ERR_OVERFLOW;

    // Preflight: validate token ids, dedup policy, and arithmetic viability.
    for (history_index = 0; history_index < token_history_count; history_index++)
    {
        token_id = token_history[history_index];
        if (token_id < 0 || token_id >= vocab_size)
            return SAMPLING_Q16_ERR_BAD_PARAM;

        already_seen = FALSE;
        for (prior_index = 0; prior_index < history_index; prior_index++)
        {
            if (token_history[prior_index] == token_id)
            {
                already_seen = TRUE;
                break;
            }
        }

        if (already_seen)
            continue;

        base_logit_q16 = logits_q16[token_id];
        if (base_logit_q16 < 0)
        {
            status = FPQ16MulDivRoundedChecked(base_logit_q16,
                                               penalty_q16,
                                               SAMPLING_Q16_ONE,
                                               &adjusted_logit_q16);
        }
        else
        {
            status = FPQ16MulDivRoundedChecked(base_logit_q16,
                                               SAMPLING_Q16_ONE,
                                               penalty_q16,
                                               &adjusted_logit_q16);
        }

        if (status)
            return status;
    }

    // Commit: same dedup/arith path, now writing in place.
    for (history_index = 0; history_index < token_history_count; history_index++)
    {
        token_id = token_history[history_index];

        already_seen = FALSE;
        for (prior_index = 0; prior_index < history_index; prior_index++)
        {
            if (token_history[prior_index] == token_id)
            {
                already_seen = TRUE;
                break;
            }
        }

        if (already_seen)
            continue;

        base_logit_q16 = logits_q16[token_id];
        if (base_logit_q16 < 0)
        {
            status = FPQ16MulDivRoundedChecked(base_logit_q16,
                                               penalty_q16,
                                               SAMPLING_Q16_ONE,
                                               &adjusted_logit_q16);
        }
        else
        {
            status = FPQ16MulDivRoundedChecked(base_logit_q16,
                                               SAMPLING_Q16_ONE,
                                               penalty_q16,
                                               &adjusted_logit_q16);
        }

        if (status)
            return status;
        logits_q16[token_id] = adjusted_logit_q16;
    }

    return SAMPLING_Q16_OK;
}
