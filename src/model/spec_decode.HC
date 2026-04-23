// spec_decode.HC
// HolyC speculative decode coordinator (integer-only, deterministic).

#define SPEC_DECODE_OK                0
#define SPEC_DECODE_ERR_NULL_PTR      1
#define SPEC_DECODE_ERR_BAD_PARAM     2
#define SPEC_DECODE_ERR_OVERFLOW      4

#define SPEC_DECODE_VERIFY_ACCEPTED   1
#define SPEC_DECODE_VERIFY_REJECTED   0

I32 SpecDecodeTryAddI64Checked(I64 lhs, I64 rhs, I64 *out_sum)
{
    if (!out_sum)
        return SPEC_DECODE_ERR_NULL_PTR;

    if (rhs > 0)
    {
        if (lhs > (0x7FFFFFFFFFFFFFFF - rhs))
            return SPEC_DECODE_ERR_OVERFLOW;
    }
    else if (rhs < 0)
    {
        if (lhs < (-0x8000000000000000 - rhs))
            return SPEC_DECODE_ERR_OVERFLOW;
    }

    *out_sum = lhs + rhs;
    return SPEC_DECODE_OK;
}

I32 SpecDecodeTryMulI64Checked(I64 lhs, I64 rhs, I64 *out_prod)
{
    if (!out_prod)
        return SPEC_DECODE_ERR_NULL_PTR;

    if (!lhs || !rhs)
    {
        *out_prod = 0;
        return SPEC_DECODE_OK;
    }

    if (lhs > 0)
    {
        if (rhs > 0)
        {
            if (lhs > (0x7FFFFFFFFFFFFFFF / rhs))
                return SPEC_DECODE_ERR_OVERFLOW;
        }
        else
        {
            if (rhs < (-0x8000000000000000 / lhs))
                return SPEC_DECODE_ERR_OVERFLOW;
        }
    }
    else
    {
        if (rhs > 0)
        {
            if (lhs < (-0x8000000000000000 / rhs))
                return SPEC_DECODE_ERR_OVERFLOW;
        }
        else
        {
            if (lhs != 0 && rhs < (0x7FFFFFFFFFFFFFFF / lhs))
                return SPEC_DECODE_ERR_OVERFLOW;
        }
    }

    *out_prod = lhs * rhs;
    return SPEC_DECODE_OK;
}

I32 SpecDecodeValidateWindowChecked(I64 draft_window,
                                    I64 token_capacity,
                                    I64 used_tokens,
                                    I64 *out_max_window)
{
    I64 slack;
    I32 status;

    if (!out_max_window)
        return SPEC_DECODE_ERR_NULL_PTR;

    if (draft_window <= 0 || token_capacity <= 0 || used_tokens < 0)
        return SPEC_DECODE_ERR_BAD_PARAM;

    if (used_tokens > token_capacity)
        return SPEC_DECODE_ERR_BAD_PARAM;

    status = SpecDecodeTryAddI64Checked(token_capacity,
                                        -used_tokens,
                                        &slack);
    if (status != SPEC_DECODE_OK)
        return status;

    if (slack <= 0)
        return SPEC_DECODE_ERR_BAD_PARAM;

    *out_max_window = draft_window;
    if (draft_window > slack)
        *out_max_window = slack;

    return SPEC_DECODE_OK;
}

I32 SpecDecodeDraftTokensChecked(I64 *draft_tokens,
                                 I64 draft_capacity,
                                 I64 draft_start_token,
                                 I64 draft_window,
                                 I64 vocab_size,
                                 I64 seed,
                                 I64 *out_drafted_count)
{
    I64 idx;
    I64 raw;
    I64 token;
    I64 drafted;
    I32 status;

    if (!draft_tokens || !out_drafted_count)
        return SPEC_DECODE_ERR_NULL_PTR;

    if (draft_capacity <= 0 || draft_window <= 0 || vocab_size <= 1)
        return SPEC_DECODE_ERR_BAD_PARAM;

    if (draft_window > draft_capacity)
        return SPEC_DECODE_ERR_BAD_PARAM;

    if (draft_start_token < 0)
        return SPEC_DECODE_ERR_BAD_PARAM;

    idx = 0;
    drafted = 0;
    while (idx < draft_window)
    {
        status = SpecDecodeTryMulI64Checked(draft_start_token + idx + 1,
                                            1103515245,
                                            &raw);
        if (status != SPEC_DECODE_OK)
            return status;

        status = SpecDecodeTryAddI64Checked(raw,
                                            seed + 12345,
                                            &raw);
        if (status != SPEC_DECODE_OK)
            return status;

        if (raw < 0)
            raw = -raw;

        token = raw % vocab_size;
        draft_tokens[idx] = token;
        drafted++;
        idx++;
    }

    *out_drafted_count = drafted;
    return SPEC_DECODE_OK;
}

I32 SpecDecodeVerifyDraftChecked(I64 *draft_tokens,
                                 I64 drafted_count,
                                 I64 accepted_target_count,
                                 I64 *out_accepted_count,
                                 I64 *out_reject_index)
{
    I64 idx;

    if (!draft_tokens || !out_accepted_count || !out_reject_index)
        return SPEC_DECODE_ERR_NULL_PTR;

    if (drafted_count < 0 || accepted_target_count < 0)
        return SPEC_DECODE_ERR_BAD_PARAM;

    if (accepted_target_count > drafted_count)
        return SPEC_DECODE_ERR_BAD_PARAM;

    idx = 0;
    while (idx < accepted_target_count)
    {
        if (draft_tokens[idx] < 0)
            return SPEC_DECODE_ERR_BAD_PARAM;
        idx++;
    }

    *out_accepted_count = accepted_target_count;
    if (accepted_target_count == drafted_count)
        *out_reject_index = -1;
    else
        *out_reject_index = accepted_target_count;

    return SPEC_DECODE_OK;
}

I32 SpecDecodeCommitAcceptedChecked(I64 *token_ids,
                                    I64 token_capacity,
                                    I64 *used_tokens,
                                    I64 *draft_tokens,
                                    I64 drafted_count,
                                    I64 accepted_count,
                                    I64 *out_committed_count)
{
    I64 idx;
    I64 write_index;

    if (!token_ids || !used_tokens || !draft_tokens || !out_committed_count)
        return SPEC_DECODE_ERR_NULL_PTR;

    if (token_capacity <= 0 || *used_tokens < 0)
        return SPEC_DECODE_ERR_BAD_PARAM;

    if (*used_tokens > token_capacity)
        return SPEC_DECODE_ERR_BAD_PARAM;

    if (drafted_count < 0 || accepted_count < 0 || accepted_count > drafted_count)
        return SPEC_DECODE_ERR_BAD_PARAM;

    if (accepted_count == 0)
    {
        *out_committed_count = 0;
        return SPEC_DECODE_OK;
    }

    if (*used_tokens + accepted_count > token_capacity)
        return SPEC_DECODE_ERR_BAD_PARAM;

    idx = 0;
    while (idx < accepted_count)
    {
        write_index = *used_tokens + idx;
        token_ids[write_index] = draft_tokens[idx];
        idx++;
    }

    *used_tokens += accepted_count;
    *out_committed_count = accepted_count;
    return SPEC_DECODE_OK;
}

I32 SpecDecodeCoordinatorStepChecked(I64 *token_ids,
                                     I64 token_capacity,
                                     I64 *used_tokens,
                                     I64 *draft_tokens,
                                     I64 draft_capacity,
                                     I64 draft_window,
                                     I64 accepted_target_count,
                                     I64 vocab_size,
                                     I64 seed,
                                     I64 *out_drafted_count,
                                     I64 *out_accepted_count,
                                     I64 *out_reject_index,
                                     I64 *out_committed_count)
{
    I64 max_window;
    I64 drafted_count;
    I64 accepted_count;
    I64 reject_index;
    I64 committed_count;
    I32 status;

    if (!token_ids || !used_tokens || !draft_tokens || !out_drafted_count ||
        !out_accepted_count || !out_reject_index || !out_committed_count)
        return SPEC_DECODE_ERR_NULL_PTR;

    status = SpecDecodeValidateWindowChecked(draft_window,
                                             token_capacity,
                                             *used_tokens,
                                             &max_window);
    if (status != SPEC_DECODE_OK)
        return status;

    if (max_window > draft_capacity)
        return SPEC_DECODE_ERR_BAD_PARAM;

    status = SpecDecodeDraftTokensChecked(draft_tokens,
                                          draft_capacity,
                                          *used_tokens,
                                          max_window,
                                          vocab_size,
                                          seed,
                                          &drafted_count);
    if (status != SPEC_DECODE_OK)
        return status;

    status = SpecDecodeVerifyDraftChecked(draft_tokens,
                                          drafted_count,
                                          accepted_target_count,
                                          &accepted_count,
                                          &reject_index);
    if (status != SPEC_DECODE_OK)
        return status;

    status = SpecDecodeCommitAcceptedChecked(token_ids,
                                             token_capacity,
                                             used_tokens,
                                             draft_tokens,
                                             drafted_count,
                                             accepted_count,
                                             &committed_count);
    if (status != SPEC_DECODE_OK)
        return status;

    *out_drafted_count = drafted_count;
    *out_accepted_count = accepted_count;
    *out_reject_index = reject_index;
    *out_committed_count = committed_count;
    return SPEC_DECODE_OK;
}
