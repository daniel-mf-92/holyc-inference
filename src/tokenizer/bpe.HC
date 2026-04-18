// bpe.HC
// UTF-8 primitives for HolyC tokenizer paths.

#define TOKENIZER_UTF8_OK 0
#define TOKENIZER_UTF8_ERR_NULL_PTR 1
#define TOKENIZER_UTF8_ERR_BAD_PARAM 2
#define TOKENIZER_UTF8_ERR_OVERFLOW 3
#define TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS 4
#define TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE 5
#define TOKENIZER_UTF8_ERR_BAD_CONTINUATION 6
#define TOKENIZER_UTF8_ERR_BAD_CODEPOINT 7
#define TOKENIZER_UTF8_ERR_TRUNCATED 8

#define TOKENIZER_BPE_OK 0
#define TOKENIZER_BPE_ERR_NULL_PTR 101
#define TOKENIZER_BPE_ERR_BAD_PARAM 102
#define TOKENIZER_BPE_ERR_OVERFLOW 103

Bool TokenizerUTF8IsContinuationByte(U8 byte)
{
    // Continuation bytes are exactly the 10xxxxxx lane.
    return (byte & 0xC0) == 0x80;
}

I32 TokenizerUTF8ExpectedLengthFromLeadByteChecked(U8 lead,
                                                   U8 *out_expected_length)
{
    U8 need;

    if (!out_expected_length)
        return TOKENIZER_UTF8_ERR_NULL_PTR;

    // Valid UTF-8 lead-byte lanes (with C0/C1 and F5..FF rejected).
    if (lead <= 0x7F)
        need = 1;
    else if (lead >= 0xC2 && lead <= 0xDF)
        need = 2;
    else if (lead >= 0xE0 && lead <= 0xEF)
        need = 3;
    else if (lead >= 0xF0 && lead <= 0xF4)
        need = 4;
    else
        return TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE;

    // Commit output only after classification succeeds.
    *out_expected_length = need;
    return TOKENIZER_UTF8_OK;
}

I32 TokenizerUTF8ValidateCodepointScalarChecked(U32 codepoint,
                                                U32 *out_codepoint)
{
    if (!out_codepoint)
        return TOKENIZER_UTF8_ERR_NULL_PTR;

    // Unicode scalar values exclude surrogate halves.
    if (codepoint >= 0xD800 && codepoint <= 0xDFFF)
        return TOKENIZER_UTF8_ERR_BAD_CODEPOINT;

    // Unicode range is capped at U+10FFFF.
    if (codepoint > 0x10FFFF)
        return TOKENIZER_UTF8_ERR_BAD_CODEPOINT;

    // Commit output only after all scalar checks pass.
    *out_codepoint = codepoint;
    return TOKENIZER_UTF8_OK;
}

I32 TokenizerUTF8ValidateContinuationByteChecked(U8 *bytes,
                                                 U64 byte_len,
                                                 U64 *io_cursor,
                                                 U8 *out_continuation_byte)
{
    U64 cursor;
    U8 continuation;
    U64 next_cursor;

    if (!bytes || !io_cursor || !out_continuation_byte)
        return TOKENIZER_UTF8_ERR_NULL_PTR;

    // HolyC integer safety: keep lengths inside signed-positive I64 domain.
    if (byte_len > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_UTF8_ERR_OVERFLOW;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_UTF8_ERR_BAD_PARAM;
    if (cursor == byte_len)
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS;

    continuation = bytes[cursor];
    if (!TokenizerUTF8IsContinuationByte(continuation))
        return TOKENIZER_UTF8_ERR_BAD_CONTINUATION;

    // Deterministic single-byte advance: exactly one byte consumed on success.
    next_cursor = cursor + 1;
    if (next_cursor <= cursor || next_cursor > byte_len)
        return TOKENIZER_UTF8_ERR_OVERFLOW;

    // No-partial semantics: commit cursor and output byte only after all checks.
    *out_continuation_byte = continuation;
    *io_cursor = next_cursor;
    return TOKENIZER_UTF8_OK;
}

I32 TokenizerUTF8ValidateContinuationSpanChecked(U8 *bytes,
                                                 U64 byte_len,
                                                 U64 *io_cursor,
                                                 U64 continuation_count,
                                                 U64 *out_bytes_consumed)
{
    U64 cursor;
    U64 staged_cursor;
    U64 consumed;
    U64 next_consumed;
    U8 continuation;
    I32 err;

    if (!bytes || !io_cursor || !out_bytes_consumed)
        return TOKENIZER_UTF8_ERR_NULL_PTR;

    if (byte_len > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_UTF8_ERR_OVERFLOW;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_UTF8_ERR_BAD_PARAM;

    // Entire continuation run must fit from the caller's current cursor.
    if (continuation_count > byte_len - cursor)
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS;

    staged_cursor = cursor;
    consumed = 0;

    // Compose the single-byte checked validator for each continuation lane.
    while (consumed < continuation_count) {
        err = TokenizerUTF8ValidateContinuationByteChecked(bytes,
                                                           byte_len,
                                                           &staged_cursor,
                                                           &continuation);
        if (err != TOKENIZER_UTF8_OK)
            return err;

        next_consumed = consumed + 1;
        if (next_consumed <= consumed)
            return TOKENIZER_UTF8_ERR_OVERFLOW;
        consumed = next_consumed;
    }

    // All-or-nothing commit: cursor/count update only after full span success.
    *out_bytes_consumed = consumed;
    *io_cursor = staged_cursor;
    return TOKENIZER_UTF8_OK;
}

I32 TokenizerUTF8NextCodepointChecked(U8 *bytes,
                                      U64 byte_len,
                                      U64 *io_cursor,
                                      U32 *out_codepoint,
                                      U8 *out_bytes_consumed)
{
    U64 cursor;
    U8 lead;
    U8 need;
    U32 codepoint;
    U8 cont;
    U64 next_cursor;
    U64 continuation_cursor;
    I32 err;

    if (!bytes || !io_cursor || !out_codepoint || !out_bytes_consumed)
        return TOKENIZER_UTF8_ERR_NULL_PTR;

    // HolyC signed math is I64-backed; keep checked arithmetic inside range.
    if (byte_len > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_UTF8_ERR_OVERFLOW;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_UTF8_ERR_BAD_PARAM;
    if (cursor == byte_len)
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS;

    lead = bytes[cursor];
    codepoint = 0;

    // UTF-8 lead-byte classification with overlong-leading-byte rejection.
    err = TokenizerUTF8ExpectedLengthFromLeadByteChecked(lead, &need);
    if (err != TOKENIZER_UTF8_OK)
        return err;

    if (need == 1)
        codepoint = lead;
    else if (need == 2)
        codepoint = lead & 0x1F;
    else if (need == 3)
        codepoint = lead & 0x0F;
    else
        codepoint = lead & 0x07;

    if (byte_len - cursor < need)
        return TOKENIZER_UTF8_ERR_TRUNCATED;

    continuation_cursor = cursor + 1;
    if (need > 1 && continuation_cursor <= cursor)
        return TOKENIZER_UTF8_ERR_OVERFLOW;

    // Decode continuation payload bits with strict 10xxxxxx checks.
    if (need > 1) {
        err = TokenizerUTF8ValidateContinuationByteChecked(bytes,
                                                           byte_len,
                                                           &continuation_cursor,
                                                           &cont);
        if (err != TOKENIZER_UTF8_OK)
            return err;
        codepoint = (codepoint << 6) | (cont & 0x3F);
    }

    if (need > 2) {
        err = TokenizerUTF8ValidateContinuationByteChecked(bytes,
                                                           byte_len,
                                                           &continuation_cursor,
                                                           &cont);
        if (err != TOKENIZER_UTF8_OK)
            return err;
        codepoint = (codepoint << 6) | (cont & 0x3F);
    }

    if (need > 3) {
        err = TokenizerUTF8ValidateContinuationByteChecked(bytes,
                                                           byte_len,
                                                           &continuation_cursor,
                                                           &cont);
        if (err != TOKENIZER_UTF8_OK)
            return err;
        codepoint = (codepoint << 6) | (cont & 0x3F);
    }

    if (need > 1 && continuation_cursor != cursor + need)
        return TOKENIZER_UTF8_ERR_BAD_PARAM;

    // Reject overlong forms before scalar validation helper.
    if (need == 2 && codepoint < 0x80)
        return TOKENIZER_UTF8_ERR_BAD_CODEPOINT;

    if (need == 3 && codepoint < 0x800)
        return TOKENIZER_UTF8_ERR_BAD_CODEPOINT;

    if (need == 4 && codepoint < 0x10000)
        return TOKENIZER_UTF8_ERR_BAD_CODEPOINT;

    // Centralized scalar-domain check (surrogates + max bound).
    err = TokenizerUTF8ValidateCodepointScalarChecked(codepoint, &codepoint);
    if (err != TOKENIZER_UTF8_OK)
        return err;

    next_cursor = cursor + need;
    if (next_cursor < cursor || next_cursor > byte_len)
        return TOKENIZER_UTF8_ERR_OVERFLOW;

    // Commit only after all checks pass (no partial state updates on failure).
    *out_codepoint = codepoint;
    *out_bytes_consumed = need;
    *io_cursor = next_cursor;
    return TOKENIZER_UTF8_OK;
}

I32 TokenizerUTF8SequenceSpanValidChecked(U8 *bytes,
                                          U64 byte_len,
                                          U64 *io_cursor,
                                          U64 span_nbytes,
                                          Bool *out_is_valid)
{
    U64 cursor;
    U64 span_end;
    U64 scan;
    U8 lead;
    U8 need;
    U32 codepoint;
    U32 scalar_checked;
    U8 cont;
    U64 next_scan;
    U64 continuation_cursor;
    I32 err;

    if (!bytes || !io_cursor || !out_is_valid)
        return TOKENIZER_UTF8_ERR_NULL_PTR;

    // HolyC signed integer runtime: keep lengths in signed-positive range.
    if (byte_len > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_UTF8_ERR_OVERFLOW;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_UTF8_ERR_BAD_PARAM;

    // Caller selects the exact byte span to validate from the current cursor.
    if (span_nbytes > byte_len - cursor)
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS;

    span_end = cursor + span_nbytes;
    if (span_end < cursor || span_end > byte_len)
        return TOKENIZER_UTF8_ERR_OVERFLOW;

    scan = cursor;
    while (scan < span_end) {
        lead = bytes[scan];

        // Lead-byte lane selection (1/2/3/4-byte sequence shape).
        err = TokenizerUTF8ExpectedLengthFromLeadByteChecked(lead, &need);
        if (err != TOKENIZER_UTF8_OK)
            return err;

        if (span_end - scan < need)
            return TOKENIZER_UTF8_ERR_TRUNCATED;

        if (need == 1)
            codepoint = lead;
        else if (need == 2)
            codepoint = lead & 0x1F;
        else if (need == 3)
            codepoint = lead & 0x0F;
        else
            codepoint = lead & 0x07;

        continuation_cursor = scan + 1;
        if (need > 1 && continuation_cursor <= scan)
            return TOKENIZER_UTF8_ERR_OVERFLOW;

        // Continuation lane checks and payload-bit assembly.
        if (need > 1) {
            err = TokenizerUTF8ValidateContinuationByteChecked(bytes,
                                                               span_end,
                                                               &continuation_cursor,
                                                               &cont);
            if (err != TOKENIZER_UTF8_OK)
                return err;
            codepoint = (codepoint << 6) | (cont & 0x3F);
        }

        if (need > 2) {
            err = TokenizerUTF8ValidateContinuationByteChecked(bytes,
                                                               span_end,
                                                               &continuation_cursor,
                                                               &cont);
            if (err != TOKENIZER_UTF8_OK)
                return err;
            codepoint = (codepoint << 6) | (cont & 0x3F);
        }

        if (need > 3) {
            err = TokenizerUTF8ValidateContinuationByteChecked(bytes,
                                                               span_end,
                                                               &continuation_cursor,
                                                               &cont);
            if (err != TOKENIZER_UTF8_OK)
                return err;
            codepoint = (codepoint << 6) | (cont & 0x3F);
        }

        if (need > 1 && continuation_cursor != scan + need)
            return TOKENIZER_UTF8_ERR_BAD_PARAM;

        // Reject overlong forms before Unicode scalar-domain validation.
        if (need == 2 && codepoint < 0x80)
            return TOKENIZER_UTF8_ERR_BAD_CODEPOINT;

        if (need == 3 && codepoint < 0x800)
            return TOKENIZER_UTF8_ERR_BAD_CODEPOINT;

        if (need == 4 && codepoint < 0x10000)
            return TOKENIZER_UTF8_ERR_BAD_CODEPOINT;

        err = TokenizerUTF8ValidateCodepointScalarChecked(codepoint,
                                                          &scalar_checked);
        if (err != TOKENIZER_UTF8_OK)
            return err;

        next_scan = scan + need;
        if (next_scan < scan || next_scan > span_end)
            return TOKENIZER_UTF8_ERR_OVERFLOW;

        scan = next_scan;
    }

    // No-partial semantics: only here do we commit cursor/output state.
    *out_is_valid = TRUE;
    *io_cursor = span_end;
    return TOKENIZER_UTF8_OK;
}

I32 TokenizerUTF8SequenceSpanValidCheckedDefault(U8 *bytes,
                                                 U64 byte_len,
                                                 U64 *io_cursor,
                                                 Bool *out_is_valid)
{
    U64 cursor;
    U64 span_nbytes;

    if (!bytes || !io_cursor || !out_is_valid)
        return TOKENIZER_UTF8_ERR_NULL_PTR;

    // Keep default-span derivation in the same signed-positive I64 envelope.
    if (byte_len > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_UTF8_ERR_OVERFLOW;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_UTF8_ERR_BAD_PARAM;

    // Canonical full-tail span: validate from cursor to end-of-buffer.
    span_nbytes = byte_len - cursor;

    // Delegate all UTF-8 sequence validation and no-partial state semantics
    // to the checked explicit-span core implementation.
    return TokenizerUTF8SequenceSpanValidChecked(bytes,
                                                 byte_len,
                                                 io_cursor,
                                                 span_nbytes,
                                                 out_is_valid);
}

I32 TokenizerUTF8DecodeSpanToCodepointsChecked(U8 *bytes,
                                               U64 byte_len,
                                               U64 *io_cursor,
                                               U64 in_nbytes,
                                               U32 *out_codepoints,
                                               U64 out_codepoint_capacity,
                                               U64 *out_codepoint_count)
{
    U64 cursor;
    U64 span_end;
    U64 scan;
    U64 decoded_count;
    U32 codepoint;
    U8 bytes_used;
    I32 err;
    U64 write_i;

    if (!bytes || !io_cursor || !out_codepoints || !out_codepoint_count)
        return TOKENIZER_UTF8_ERR_NULL_PTR;

    // HolyC signed runtime uses I64 for checked arithmetic contracts.
    if (byte_len > 0x7FFFFFFFFFFFFFFF || out_codepoint_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_UTF8_ERR_OVERFLOW;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_UTF8_ERR_BAD_PARAM;

    if (in_nbytes > byte_len - cursor)
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS;

    span_end = cursor + in_nbytes;
    if (span_end < cursor || span_end > byte_len)
        return TOKENIZER_UTF8_ERR_OVERFLOW;

    // Pass 1: validate and count decoded codepoints.
    // No output/cursor writes are committed during this pass.
    scan = cursor;
    decoded_count = 0;
    while (scan < span_end) {
        err = TokenizerUTF8NextCodepointChecked(bytes,
                                                span_end,
                                                &scan,
                                                &codepoint,
                                                &bytes_used);
        if (err != TOKENIZER_UTF8_OK)
            return err;

        if (decoded_count == out_codepoint_capacity)
            return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS;

        decoded_count++;
        if (decoded_count == 0)
            return TOKENIZER_UTF8_ERR_OVERFLOW;
    }

    if (scan != span_end)
        return TOKENIZER_UTF8_ERR_BAD_PARAM;

    // Pass 2: decode again and materialize codepoints to output buffer.
    scan = cursor;
    write_i = 0;
    while (scan < span_end) {
        err = TokenizerUTF8NextCodepointChecked(bytes,
                                                span_end,
                                                &scan,
                                                &codepoint,
                                                &bytes_used);
        if (err != TOKENIZER_UTF8_OK)
            return err;

        if (write_i >= decoded_count)
            return TOKENIZER_UTF8_ERR_OVERFLOW;

        out_codepoints[write_i] = codepoint;
        write_i++;
    }

    if (write_i != decoded_count)
        return TOKENIZER_UTF8_ERR_BAD_PARAM;

    // Commit cursor and count only after full decode succeeds.
    *out_codepoint_count = decoded_count;
    *io_cursor = span_end;
    return TOKENIZER_UTF8_OK;
}

I32 TokenizerUTF8DecodeSpanToCodepointsCheckedDefaultCapacity(U8 *bytes,
                                                              U64 byte_len,
                                                              U64 *io_cursor,
                                                              U64 in_nbytes,
                                                              U32 *out_codepoints,
                                                              U64 *out_codepoint_count)
{
    U64 derived_capacity;

    if (!bytes || !io_cursor || !out_codepoints || !out_codepoint_count)
        return TOKENIZER_UTF8_ERR_NULL_PTR;

    // Canonical default-capacity rule: each UTF-8 byte can decode to at most
    // one Unicode scalar, so in_nbytes is a strict upper bound on output count.
    if (in_nbytes > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_UTF8_ERR_OVERFLOW;

    derived_capacity = in_nbytes;

    // Preserve checked-core semantics exactly (including no-partial writes).
    return TokenizerUTF8DecodeSpanToCodepointsChecked(bytes,
                                                      byte_len,
                                                      io_cursor,
                                                      in_nbytes,
                                                      out_codepoints,
                                                      derived_capacity,
                                                      out_codepoint_count);
}

I32 TokenizerUTF8CountCodepointsChecked(U8 *bytes,
                                        U64 byte_len,
                                        U64 *io_cursor,
                                        U64 in_nbytes,
                                        U64 *out_codepoint_count)
{
    U64 cursor;
    U64 span_end;
    U64 scan;
    U64 decoded_count;
    U32 codepoint;
    U8 bytes_used;
    I32 err;

    if (!bytes || !io_cursor || !out_codepoint_count)
        return TOKENIZER_UTF8_ERR_NULL_PTR;

    // Keep arithmetic inside the signed-positive I64 safety envelope.
    if (byte_len > 0x7FFFFFFFFFFFFFFF || in_nbytes > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_UTF8_ERR_OVERFLOW;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_UTF8_ERR_BAD_PARAM;

    if (in_nbytes > byte_len - cursor)
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS;

    span_end = cursor + in_nbytes;
    if (span_end < cursor || span_end > byte_len)
        return TOKENIZER_UTF8_ERR_OVERFLOW;

    scan = cursor;
    decoded_count = 0;
    while (scan < span_end) {
        // Compose through the checked UTF-8 decoder so lead-byte, continuation,
        // overlong, and scalar-domain rules stay centralized in one path.
        err = TokenizerUTF8NextCodepointChecked(bytes,
                                                span_end,
                                                &scan,
                                                &codepoint,
                                                &bytes_used);
        if (err != TOKENIZER_UTF8_OK)
            return err;

        decoded_count++;
        if (decoded_count == 0)
            return TOKENIZER_UTF8_ERR_OVERFLOW;
    }

    if (scan != span_end)
        return TOKENIZER_UTF8_ERR_BAD_PARAM;

    // No-partial semantics: commit count/cursor only after full-span success.
    *out_codepoint_count = decoded_count;
    *io_cursor = span_end;
    return TOKENIZER_UTF8_OK;
}

I32 TokenizerUTF8CountCodepointsCheckedDefaultSpan(U8 *bytes,
                                                   U64 byte_len,
                                                   U64 *io_cursor,
                                                   U64 *out_codepoint_count)
{
    U64 cursor;
    U64 in_nbytes;

    if (!bytes || !io_cursor || !out_codepoint_count)
        return TOKENIZER_UTF8_ERR_NULL_PTR;

    // Keep default-span arithmetic inside signed-positive I64 range.
    if (byte_len > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_UTF8_ERR_OVERFLOW;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_UTF8_ERR_BAD_PARAM;

    // Canonical wrapper span: count from current cursor to end-of-buffer.
    in_nbytes = byte_len - cursor;

    // Delegate exact cursor/count commit and UTF-8 validation semantics
    // to the explicit-span checked core helper.
    return TokenizerUTF8CountCodepointsChecked(bytes,
                                               byte_len,
                                               io_cursor,
                                               in_nbytes,
                                               out_codepoint_count);
}

I32 TokenizerBPEMergePairPriorityFindChecked(I32 *token_ids,
                                             U64 token_count,
                                             U64 token_capacity,
                                             I32 *rank_left_tokens,
                                             I32 *rank_right_tokens,
                                             U32 *rank_values,
                                             U64 rank_table_count,
                                             U64 rank_table_capacity,
                                             U64 *out_left_index,
                                             U32 *out_rank,
                                             Bool *out_found)
{
    U64 i;
    U64 j;
    U64 next_i;
    Bool found;
    Bool pair_rank_found;
    U64 best_left_index;
    U32 best_rank;
    I32 left;
    I32 right;
    U32 pair_rank;

    if (!token_ids || !out_left_index || !out_rank || !out_found)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (token_capacity > 0x7FFFFFFFFFFFFFFF || rank_table_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (token_count > token_capacity || rank_table_count > rank_table_capacity)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (rank_table_count > 0 && (!rank_left_tokens || !rank_right_tokens || !rank_values))
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (token_count < 2) {
        *out_found = FALSE;
        return TOKENIZER_BPE_OK;
    }

    found = FALSE;
    best_left_index = 0;
    best_rank = 0;

    // Scan adjacent token pairs and choose the lowest-rank merge.
    // Deterministic tie-break: first pair index wins when ranks are equal.
    for (i = 0; i + 1 < token_count; i++) {
        next_i = i + 1;
        if (next_i <= i || next_i >= token_count)
            return TOKENIZER_BPE_ERR_OVERFLOW;

        left = token_ids[i];
        right = token_ids[next_i];

        pair_rank_found = FALSE;
        pair_rank = 0;

        // Search ranked merge table for this adjacent pair.
        // Duplicate entries are resolved to the lowest rank.
        for (j = 0; j < rank_table_count; j++) {
            if (rank_left_tokens[j] != left)
                continue;
            if (rank_right_tokens[j] != right)
                continue;

            if (!pair_rank_found || rank_values[j] < pair_rank) {
                pair_rank = rank_values[j];
                pair_rank_found = TRUE;
            }
        }

        if (!pair_rank_found)
            continue;

        if (!found || pair_rank < best_rank) {
            best_rank = pair_rank;
            best_left_index = i;
            found = TRUE;
        }
    }

    // No-partial-write contract: commit output state only after full scan.
    if (!found) {
        *out_found = FALSE;
        return TOKENIZER_BPE_OK;
    }

    *out_left_index = best_left_index;
    *out_rank = best_rank;
    *out_found = TRUE;
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEMergePairPriorityFindCheckedNoPartial(I32 *token_ids,
                                                      U64 token_count,
                                                      U64 token_capacity,
                                                      I32 *rank_left_tokens,
                                                      I32 *rank_right_tokens,
                                                      U32 *rank_values,
                                                      U64 rank_table_count,
                                                      U64 rank_table_capacity,
                                                      U64 *out_left_index,
                                                      U32 *out_rank,
                                                      Bool *out_found)
{
    U64 staged_left_index;
    U32 staged_rank;
    Bool staged_found;
    I32 err;

    if (!token_ids || !out_left_index || !out_rank || !out_found)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Stage outputs from caller-visible values so "no merge found" success
    // mirrors the checked core exactly (index/rank remain unchanged).
    staged_left_index = *out_left_index;
    staged_rank = *out_rank;
    staged_found = *out_found;

    err = TokenizerBPEMergePairPriorityFindChecked(token_ids,
                                                   token_count,
                                                   token_capacity,
                                                   rank_left_tokens,
                                                   rank_right_tokens,
                                                   rank_values,
                                                   rank_table_count,
                                                   rank_table_capacity,
                                                   &staged_left_index,
                                                   &staged_rank,
                                                   &staged_found);
    if (err != TOKENIZER_BPE_OK)
        return err;

    // All-or-nothing commit: caller outputs change only after checked core
    // fully succeeds.
    *out_left_index = staged_left_index;
    *out_rank = staged_rank;
    *out_found = staged_found;
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPETokenPairRankLookupChecked(I32 left_token,
                                           I32 right_token,
                                           I32 *rank_left_tokens,
                                           I32 *rank_right_tokens,
                                           U32 *rank_values,
                                           U64 rank_table_count,
                                           U64 rank_table_capacity,
                                           U32 *out_rank,
                                           Bool *out_found)
{
    U64 lo;
    U64 hi;
    U64 mid;
    I32 mid_left;
    I32 mid_right;
    U64 first_match;
    U64 scan;
    U64 next_scan;
    U32 best_rank;

    if (!out_rank || !out_found)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (rank_table_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (rank_table_count > rank_table_capacity)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (rank_table_count > 0 && (!rank_left_tokens || !rank_right_tokens || !rank_values))
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (!rank_table_count) {
        *out_found = FALSE;
        return TOKENIZER_BPE_OK;
    }

    // Binary-search lower bound for key (left_token,right_token)
    // in lexicographically sorted rank tables.
    lo = 0;
    hi = rank_table_count;
    while (lo < hi) {
        mid = lo + ((hi - lo) >> 1);

        mid_left = rank_left_tokens[mid];
        mid_right = rank_right_tokens[mid];

        if (mid_left < left_token || (mid_left == left_token && mid_right < right_token)) {
            lo = mid + 1;
            continue;
        }

        hi = mid;
    }

    first_match = lo;
    if (first_match >= rank_table_count) {
        *out_found = FALSE;
        return TOKENIZER_BPE_OK;
    }

    if (rank_left_tokens[first_match] != left_token ||
        rank_right_tokens[first_match] != right_token) {
        *out_found = FALSE;
        return TOKENIZER_BPE_OK;
    }

    // Deterministic duplicate-key handling: consume contiguous equal-key run
    // and keep the lowest rank value for this pair.
    best_rank = rank_values[first_match];

    scan = first_match + 1;
    if (scan < first_match)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    while (scan < rank_table_count) {
        if (rank_left_tokens[scan] != left_token)
            break;
        if (rank_right_tokens[scan] != right_token)
            break;

        if (rank_values[scan] < best_rank)
            best_rank = rank_values[scan];

        next_scan = scan + 1;
        if (next_scan <= scan)
            return TOKENIZER_BPE_ERR_OVERFLOW;
        scan = next_scan;
    }

    // No-partial-write contract: commit output state only after full lookup.
    *out_rank = best_rank;
    *out_found = TRUE;
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEMergePairRankLookupCheckedDefault(I32 left_token,
                                                  I32 right_token,
                                                  I32 *rank_left_tokens,
                                                  I32 *rank_right_tokens,
                                                  U32 *rank_values,
                                                  U64 rank_table_count,
                                                  U32 *out_rank,
                                                  Bool *out_found)
{
    U64 derived_capacity;

    if (!out_rank || !out_found)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Canonical default-capacity rule: table is assumed tightly packed, so
    // the effective storage capacity equals the active row count.
    if (rank_table_count > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    derived_capacity = rank_table_count;

    // Delegate lookup semantics and no-partial-write contracts to checked core.
    return TokenizerBPETokenPairRankLookupChecked(left_token,
                                                  right_token,
                                                  rank_left_tokens,
                                                  rank_right_tokens,
                                                  rank_values,
                                                  rank_table_count,
                                                  derived_capacity,
                                                  out_rank,
                                                  out_found);
}

I32 TokenizerBPEMergeApplyAtIndexChecked(I32 *token_ids,
                                         U64 token_count,
                                         U64 token_capacity,
                                         U64 left_index,
                                         I32 merged_token,
                                         U64 *out_token_count)
{
    U64 right_index;
    U64 src_index;
    U64 next_src_index;
    U64 dst_index;
    U64 next_count;

    if (!token_ids || !out_token_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (token_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (token_count > token_capacity)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    // Merge needs two adjacent tokens: [left_index] and [left_index+1].
    if (token_count < 2)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (left_index >= token_count)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    right_index = left_index + 1;
    if (right_index <= left_index)
        return TOKENIZER_BPE_ERR_OVERFLOW;
    if (right_index >= token_count)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    // No-partial-write contract: all checks complete above this line.
    token_ids[left_index] = merged_token;

    // Collapse right token by shifting tail one slot left.
    src_index = right_index + 1;
    if (src_index < right_index)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    while (src_index < token_count) {
        dst_index = src_index - 1;
        if (dst_index >= token_capacity)
            return TOKENIZER_BPE_ERR_OVERFLOW;

        token_ids[dst_index] = token_ids[src_index];

        next_src_index = src_index + 1;
        if (next_src_index <= src_index)
            return TOKENIZER_BPE_ERR_OVERFLOW;
        src_index = next_src_index;
    }

    next_count = token_count - 1;
    *out_token_count = next_count;
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEMergePairTokenLookupChecked(I32 left_token,
                                            I32 right_token,
                                            I32 *rank_left_tokens,
                                            I32 *rank_right_tokens,
                                            U32 *rank_values,
                                            I32 *rank_merged_tokens,
                                            U64 rank_table_count,
                                            U64 rank_table_capacity,
                                            I32 *out_merged_token,
                                            U32 *out_rank,
                                            Bool *out_found)
{
    U64 lo;
    U64 hi;
    U64 mid;
    I32 mid_left;
    I32 mid_right;
    U64 first_match;
    U64 scan;
    U64 next_scan;
    Bool found;
    U32 best_rank;
    I32 best_merged;

    if (!out_merged_token || !out_rank || !out_found)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (rank_table_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (rank_table_count > rank_table_capacity)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (rank_table_count > 0 &&
        (!rank_left_tokens || !rank_right_tokens || !rank_values || !rank_merged_tokens))
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (!rank_table_count) {
        *out_found = FALSE;
        return TOKENIZER_BPE_OK;
    }

    // Binary-search lower bound for key (left_token,right_token)
    // in lexicographically sorted rank tables.
    lo = 0;
    hi = rank_table_count;
    while (lo < hi) {
        mid = lo + ((hi - lo) >> 1);

        mid_left = rank_left_tokens[mid];
        mid_right = rank_right_tokens[mid];

        if (mid_left < left_token || (mid_left == left_token && mid_right < right_token)) {
            lo = mid + 1;
            continue;
        }

        hi = mid;
    }

    first_match = lo;
    if (first_match >= rank_table_count) {
        *out_found = FALSE;
        return TOKENIZER_BPE_OK;
    }

    if (rank_left_tokens[first_match] != left_token ||
        rank_right_tokens[first_match] != right_token) {
        *out_found = FALSE;
        return TOKENIZER_BPE_OK;
    }

    // Deterministic duplicate-key handling: keep the lowest rank and its
    // first-seen merged token for that rank.
    scan = first_match;
    found = FALSE;
    best_rank = 0;
    best_merged = 0;

    while (scan < rank_table_count) {
        if (rank_left_tokens[scan] != left_token)
            break;
        if (rank_right_tokens[scan] != right_token)
            break;

        if (!found || rank_values[scan] < best_rank) {
            best_rank = rank_values[scan];
            best_merged = rank_merged_tokens[scan];
            found = TRUE;
        }

        next_scan = scan + 1;
        if (next_scan <= scan)
            return TOKENIZER_BPE_ERR_OVERFLOW;
        scan = next_scan;
    }

    if (!found) {
        *out_found = FALSE;
        return TOKENIZER_BPE_OK;
    }

    // No-partial-write contract: commit output state only after full lookup.
    *out_merged_token = best_merged;
    *out_rank = best_rank;
    *out_found = TRUE;
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEMergeApplyBestPriorityChecked(I32 *token_ids,
                                              U64 token_count,
                                              U64 token_capacity,
                                              I32 *rank_left_tokens,
                                              I32 *rank_right_tokens,
                                              U32 *rank_values,
                                              I32 *rank_merged_tokens,
                                              U64 rank_table_count,
                                              U64 rank_table_capacity,
                                              I32 *out_merged_token,
                                              U32 *out_rank,
                                              Bool *out_found,
                                              U64 *out_token_count)
{
    I32 *staged_tokens;
    U64 i;
    U64 best_left_index;
    U64 best_right_index;
    U32 best_rank;
    Bool best_found;
    I32 merged_token;
    U32 pair_rank;
    Bool pair_found;
    U64 next_token_count;
    I32 err;

    if (!token_ids ||
        !out_merged_token ||
        !out_rank ||
        !out_found ||
        !out_token_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (token_capacity > 0x7FFFFFFFFFFFFFFF ||
        rank_table_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (token_count > token_capacity || rank_table_count > rank_table_capacity)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (rank_table_count > 0 &&
        (!rank_left_tokens || !rank_right_tokens || !rank_values || !rank_merged_tokens))
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (token_count < 2) {
        *out_found = FALSE;
        *out_token_count = token_count;
        return TOKENIZER_BPE_OK;
    }

    // Step 1 (greedy): resolve the best-ranked adjacent pair.
    best_left_index = 0;
    best_rank = 0;
    best_found = FALSE;
    err = TokenizerBPEMergePairPriorityFindChecked(token_ids,
                                                   token_count,
                                                   token_capacity,
                                                   rank_left_tokens,
                                                   rank_right_tokens,
                                                   rank_values,
                                                   rank_table_count,
                                                   rank_table_capacity,
                                                   &best_left_index,
                                                   &best_rank,
                                                   &best_found);
    if (err != TOKENIZER_BPE_OK)
        return err;

    if (!best_found) {
        *out_found = FALSE;
        *out_token_count = token_count;
        return TOKENIZER_BPE_OK;
    }

    best_right_index = best_left_index + 1;
    if (best_right_index <= best_left_index || best_right_index >= token_count)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    // Resolve merged token for the selected pair and verify rank agreement.
    err = TokenizerBPEMergePairTokenLookupChecked(token_ids[best_left_index],
                                                  token_ids[best_right_index],
                                                  rank_left_tokens,
                                                  rank_right_tokens,
                                                  rank_values,
                                                  rank_merged_tokens,
                                                  rank_table_count,
                                                  rank_table_capacity,
                                                  &merged_token,
                                                  &pair_rank,
                                                  &pair_found);
    if (err != TOKENIZER_BPE_OK)
        return err;

    if (!pair_found || pair_rank != best_rank)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    // Step 2 (all-or-nothing): stage the token span, apply merge on staged
    // memory, and commit only if the merge operation fully succeeds.
    staged_tokens = MAlloc(sizeof(I32) * token_count);
    if (!staged_tokens)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    for (i = 0; i < token_count; i++)
        staged_tokens[i] = token_ids[i];

    err = TokenizerBPEMergeApplyAtIndexChecked(staged_tokens,
                                               token_count,
                                               token_capacity,
                                               best_left_index,
                                               merged_token,
                                               &next_token_count);
    if (err != TOKENIZER_BPE_OK) {
        Free(staged_tokens);
        return err;
    }

    for (i = 0; i < next_token_count; i++)
        token_ids[i] = staged_tokens[i];

    Free(staged_tokens);
    *out_merged_token = merged_token;
    *out_rank = best_rank;
    *out_found = TRUE;
    *out_token_count = next_token_count;
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEMergeApplyBestPriorityCheckedNoPartial(I32 *token_ids,
                                                       U64 token_count,
                                                       U64 token_capacity,
                                                       I32 *rank_left_tokens,
                                                       I32 *rank_right_tokens,
                                                       U32 *rank_values,
                                                       I32 *rank_merged_tokens,
                                                       U64 rank_table_count,
                                                       U64 rank_table_capacity,
                                                       I32 *out_merged_token,
                                                       U32 *out_rank,
                                                       Bool *out_found,
                                                       U64 *out_token_count)
{
    I32 staged_merged_token;
    U32 staged_rank;
    Bool staged_found;
    U64 staged_token_count;
    I32 err;

    if (!token_ids ||
        !out_merged_token ||
        !out_rank ||
        !out_found ||
        !out_token_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Stage caller-visible outputs first so any checked-core failure path keeps
    // the original output state unchanged.
    staged_merged_token = *out_merged_token;
    staged_rank = *out_rank;
    staged_found = *out_found;
    staged_token_count = *out_token_count;

    err = TokenizerBPEMergeApplyBestPriorityChecked(token_ids,
                                                    token_count,
                                                    token_capacity,
                                                    rank_left_tokens,
                                                    rank_right_tokens,
                                                    rank_values,
                                                    rank_merged_tokens,
                                                    rank_table_count,
                                                    rank_table_capacity,
                                                    &staged_merged_token,
                                                    &staged_rank,
                                                    &staged_found,
                                                    &staged_token_count);
    if (err != TOKENIZER_BPE_OK)
        return err;

    // All-or-nothing output commit after the checked core has finished.
    *out_merged_token = staged_merged_token;
    *out_rank = staged_rank;
    *out_found = staged_found;
    *out_token_count = staged_token_count;
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEEncodeMergeLoopChecked(I32 *token_ids,
                                       U64 token_count,
                                       U64 token_capacity,
                                       I32 *rank_left_tokens,
                                       I32 *rank_right_tokens,
                                       U32 *rank_values,
                                       I32 *rank_merged_tokens,
                                       U64 rank_table_count,
                                       U64 rank_table_capacity,
                                       U64 *out_token_count)
{
    I32 *staged_tokens;
    U64 staged_count;
    U64 i;
    I32 err;
    Bool best_found;
    U64 best_left_index;
    U64 right_index;
    U32 best_rank;
    U32 checked_pair_rank;
    U32 pair_rank;
    I32 best_merged;
    Bool checked_pair_found;
    Bool pair_found;
    U64 next_count;

    if (!token_ids || !out_token_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (token_capacity > 0x7FFFFFFFFFFFFFFF || rank_table_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (token_count > token_capacity || rank_table_count > rank_table_capacity)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (rank_table_count > 0 &&
        (!rank_left_tokens || !rank_right_tokens || !rank_values || !rank_merged_tokens))
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (token_count < 2) {
        *out_token_count = token_count;
        return TOKENIZER_BPE_OK;
    }

    // Stage the token span and commit only after the full merge loop succeeds.
    staged_tokens = MAlloc(sizeof(I32) * token_count);
    if (!staged_tokens)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    for (i = 0; i < token_count; i++)
        staged_tokens[i] = token_ids[i];
    staged_count = token_count;

    // Deterministic merge loop: find best adjacent rank, materialize merged id,
    // apply in-place compaction, repeat until fixpoint.
    while (staged_count >= 2) {
        best_found = FALSE;
        best_left_index = 0;
        best_rank = 0;

        err = TokenizerBPEMergePairPriorityFindChecked(staged_tokens,
                                                       staged_count,
                                                       token_capacity,
                                                       rank_left_tokens,
                                                       rank_right_tokens,
                                                       rank_values,
                                                       rank_table_count,
                                                       rank_table_capacity,
                                                       &best_left_index,
                                                       &best_rank,
                                                       &best_found);
        if (err != TOKENIZER_BPE_OK) {
            Free(staged_tokens);
            return err;
        }

        if (!best_found)
            break;

        right_index = best_left_index + 1;
        if (right_index <= best_left_index || right_index >= staged_count) {
            Free(staged_tokens);
            return TOKENIZER_BPE_ERR_OVERFLOW;
        }

        err = TokenizerBPETokenPairRankLookupChecked(staged_tokens[best_left_index],
                                                     staged_tokens[right_index],
                                                     rank_left_tokens,
                                                     rank_right_tokens,
                                                     rank_values,
                                                     rank_table_count,
                                                     rank_table_capacity,
                                                     &checked_pair_rank,
                                                     &checked_pair_found);
        if (err != TOKENIZER_BPE_OK) {
            Free(staged_tokens);
            return err;
        }

        // Priority scan and pair-rank lookup must agree on rank existence/value.
        if (!checked_pair_found || checked_pair_rank != best_rank) {
            Free(staged_tokens);
            return TOKENIZER_BPE_ERR_BAD_PARAM;
        }

        err = TokenizerBPEMergePairTokenLookupChecked(staged_tokens[best_left_index],
                                                      staged_tokens[right_index],
                                                      rank_left_tokens,
                                                      rank_right_tokens,
                                                      rank_values,
                                                      rank_merged_tokens,
                                                      rank_table_count,
                                                      rank_table_capacity,
                                                      &best_merged,
                                                      &pair_rank,
                                                      &pair_found);
        if (err != TOKENIZER_BPE_OK) {
            Free(staged_tokens);
            return err;
        }

        // Priority-find and pair-lookup must resolve the same pair/rank.
        if (!pair_found || pair_rank != best_rank) {
            Free(staged_tokens);
            return TOKENIZER_BPE_ERR_BAD_PARAM;
        }

        err = TokenizerBPEMergeApplyAtIndexChecked(staged_tokens,
                                                   staged_count,
                                                   token_capacity,
                                                   best_left_index,
                                                   best_merged,
                                                   &next_count);
        if (err != TOKENIZER_BPE_OK) {
            Free(staged_tokens);
            return err;
        }

        staged_count = next_count;
    }

    for (i = 0; i < staged_count; i++)
        token_ids[i] = staged_tokens[i];

    *out_token_count = staged_count;
    Free(staged_tokens);
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEEncodeSpanChecked(U8 *bytes,
                                  U64 byte_len,
                                  U64 *io_cursor,
                                  U64 span_nbytes,
                                  I32 *rank_left_tokens,
                                  I32 *rank_right_tokens,
                                  U32 *rank_values,
                                  I32 *rank_merged_tokens,
                                  U64 rank_table_count,
                                  U64 rank_table_capacity,
                                  I32 *out_token_ids,
                                  U64 out_token_capacity,
                                  U64 *out_token_count)
{
    U64 cursor;
    U64 span_end;
    U64 valid_cursor;
    Bool span_is_valid;
    I32 err;
    I32 *working_tokens;
    U64 working_count;
    U64 working_capacity;
    U64 i;

    if (!bytes || !io_cursor || !out_token_ids || !out_token_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (byte_len > 0x7FFFFFFFFFFFFFFF ||
        rank_table_capacity > 0x7FFFFFFFFFFFFFFF ||
        out_token_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (rank_table_count > rank_table_capacity)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (rank_table_count > 0 &&
        (!rank_left_tokens || !rank_right_tokens || !rank_values || !rank_merged_tokens))
        return TOKENIZER_BPE_ERR_NULL_PTR;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (span_nbytes > byte_len - cursor)
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS;

    span_end = cursor + span_nbytes;
    if (span_end < cursor || span_end > byte_len)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    // Input span is pretokenized UTF-8; reject malformed sequences up-front.
    valid_cursor = cursor;
    span_is_valid = FALSE;
    err = TokenizerUTF8SequenceSpanValidChecked(bytes,
                                                byte_len,
                                                &valid_cursor,
                                                span_nbytes,
                                                &span_is_valid);
    if (err != TOKENIZER_UTF8_OK)
        return err;
    if (!span_is_valid || valid_cursor != span_end)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    // Base pretokenization: one token per byte before merge passes.
    if (span_nbytes > out_token_capacity)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (!span_nbytes) {
        *out_token_count = 0;
        *io_cursor = span_end;
        return TOKENIZER_BPE_OK;
    }

    working_capacity = span_nbytes;
    working_tokens = MAlloc(sizeof(I32) * working_capacity);
    if (!working_tokens)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    for (i = 0; i < span_nbytes; i++)
        working_tokens[i] = bytes[cursor + i];
    working_count = span_nbytes;

    err = TokenizerBPEEncodeMergeLoopChecked(working_tokens,
                                             working_count,
                                             working_capacity,
                                             rank_left_tokens,
                                             rank_right_tokens,
                                             rank_values,
                                             rank_merged_tokens,
                                             rank_table_count,
                                             rank_table_capacity,
                                             &working_count);
    if (err != TOKENIZER_BPE_OK) {
        Free(working_tokens);
        return err;
    }

    if (working_count > out_token_capacity) {
        Free(working_tokens);
        return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    // No-partial-write output contract: materialize output only on full success.
    for (i = 0; i < working_count; i++)
        out_token_ids[i] = working_tokens[i];

    *out_token_count = working_count;
    *io_cursor = span_end;
    Free(working_tokens);
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEEncodeSpanCheckedNoPartial(U8 *bytes,
                                           U64 byte_len,
                                           U64 *io_cursor,
                                           U64 span_nbytes,
                                           I32 *rank_left_tokens,
                                           I32 *rank_right_tokens,
                                           U32 *rank_values,
                                           I32 *rank_merged_tokens,
                                           U64 rank_table_count,
                                           U64 rank_table_capacity,
                                           I32 *out_token_ids,
                                           U64 out_token_capacity,
                                           U64 *out_token_count)
{
    U64 staged_cursor;
    U64 staged_token_count;
    I32 *staged_tokens;
    U64 staged_token_capacity;
    I32 err;
    U64 i;

    if (!bytes || !io_cursor || !out_token_ids || !out_token_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (out_token_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    // Stage caller-visible cursor/count first so failures preserve state.
    staged_cursor = *io_cursor;
    staged_token_count = *out_token_count;

    // Always pass non-null output storage to the checked core.
    staged_token_capacity = out_token_capacity;
    if (!staged_token_capacity)
        staged_token_capacity = 1;

    staged_tokens = MAlloc(sizeof(I32) * staged_token_capacity);
    if (!staged_tokens)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    err = TokenizerBPEEncodeSpanChecked(bytes,
                                        byte_len,
                                        &staged_cursor,
                                        span_nbytes,
                                        rank_left_tokens,
                                        rank_right_tokens,
                                        rank_values,
                                        rank_merged_tokens,
                                        rank_table_count,
                                        rank_table_capacity,
                                        staged_tokens,
                                        out_token_capacity,
                                        &staged_token_count);
    if (err != TOKENIZER_BPE_OK) {
        Free(staged_tokens);
        return err;
    }

    if (staged_token_count > out_token_capacity) {
        Free(staged_tokens);
        return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    // All-or-nothing output commit: copy staged tokens only on success.
    for (i = 0; i < staged_token_count; i++)
        out_token_ids[i] = staged_tokens[i];

    *out_token_count = staged_token_count;
    *io_cursor = staged_cursor;
    Free(staged_tokens);
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEEncodeSpanCheckedDefaultCapacity(U8 *bytes,
                                                 U64 byte_len,
                                                 U64 *io_cursor,
                                                 U64 span_nbytes,
                                                 I32 *rank_left_tokens,
                                                 I32 *rank_right_tokens,
                                                 U32 *rank_values,
                                                 I32 *rank_merged_tokens,
                                                 U64 rank_table_count,
                                                 U64 rank_table_capacity,
                                                 I32 *out_token_ids,
                                                 U64 *out_token_count)
{
    U64 derived_capacity;

    if (!bytes || !io_cursor || !out_token_ids || !out_token_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Canonical default-capacity rule: byte-level pretokenization yields at
    // most one token per input byte before merge passes shrink the sequence.
    if (span_nbytes > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    derived_capacity = span_nbytes;

    // Delegate all checked bounds/UTF-8/no-partial semantics to the core path.
    return TokenizerBPEEncodeSpanChecked(bytes,
                                         byte_len,
                                         io_cursor,
                                         span_nbytes,
                                         rank_left_tokens,
                                         rank_right_tokens,
                                         rank_values,
                                         rank_merged_tokens,
                                         rank_table_count,
                                         rank_table_capacity,
                                         out_token_ids,
                                         derived_capacity,
                                         out_token_count);
}

I32 TokenizerBPEMergeCandidatesBuildChecked(I32 *token_ids,
                                            U64 token_count,
                                            U64 token_capacity,
                                            I32 *rank_left_tokens,
                                            I32 *rank_right_tokens,
                                            U32 *rank_values,
                                            U64 rank_table_count,
                                            U64 rank_table_capacity,
                                            I32 *out_left_tokens,
                                            I32 *out_right_tokens,
                                            U64 *out_left_indices,
                                            U32 *out_ranks,
                                            U64 out_candidate_capacity,
                                            U64 *out_candidate_count)
{
    U64 i;
    U64 next_i;
    U64 candidate_count;
    I32 left_token;
    I32 right_token;
    U32 rank;
    Bool found;
    I32 err;

    if (!token_ids ||
        !out_left_tokens ||
        !out_right_tokens ||
        !out_left_indices ||
        !out_ranks ||
        !out_candidate_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (token_capacity > 0x7FFFFFFFFFFFFFFF ||
        rank_table_capacity > 0x7FFFFFFFFFFFFFFF ||
        out_candidate_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (token_count > token_capacity || rank_table_count > rank_table_capacity)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (rank_table_count > 0 && (!rank_left_tokens || !rank_right_tokens || !rank_values))
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Zero/one-token spans have no adjacent pair candidates.
    if (token_count < 2) {
        *out_candidate_count = 0;
        return TOKENIZER_BPE_OK;
    }

    // Pass 1: count rank-resolved adjacent pairs with no output writes.
    candidate_count = 0;
    for (i = 0; i + 1 < token_count; i++) {
        next_i = i + 1;
        if (next_i <= i || next_i >= token_count)
            return TOKENIZER_BPE_ERR_OVERFLOW;

        left_token = token_ids[i];
        right_token = token_ids[next_i];

        err = TokenizerBPETokenPairRankLookupChecked(left_token,
                                                     right_token,
                                                     rank_left_tokens,
                                                     rank_right_tokens,
                                                     rank_values,
                                                     rank_table_count,
                                                     rank_table_capacity,
                                                     &rank,
                                                     &found);
        if (err != TOKENIZER_BPE_OK)
            return err;

        if (!found)
            continue;

        if (candidate_count == out_candidate_capacity)
            return TOKENIZER_BPE_ERR_BAD_PARAM;

        candidate_count++;
        if (candidate_count == 0)
            return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    // Pass 2: materialize candidates only after pass-1 capacity checks succeed.
    candidate_count = 0;
    for (i = 0; i + 1 < token_count; i++) {
        next_i = i + 1;
        if (next_i <= i || next_i >= token_count)
            return TOKENIZER_BPE_ERR_OVERFLOW;

        left_token = token_ids[i];
        right_token = token_ids[next_i];

        err = TokenizerBPETokenPairRankLookupChecked(left_token,
                                                     right_token,
                                                     rank_left_tokens,
                                                     rank_right_tokens,
                                                     rank_values,
                                                     rank_table_count,
                                                     rank_table_capacity,
                                                     &rank,
                                                     &found);
        if (err != TOKENIZER_BPE_OK)
            return err;

        if (!found)
            continue;

        if (candidate_count >= out_candidate_capacity)
            return TOKENIZER_BPE_ERR_OVERFLOW;

        out_left_tokens[candidate_count] = left_token;
        out_right_tokens[candidate_count] = right_token;
        out_left_indices[candidate_count] = i;
        out_ranks[candidate_count] = rank;

        candidate_count++;
    }

    *out_candidate_count = candidate_count;
    return TOKENIZER_BPE_OK;
}

#define TOKENIZER_BPE_ASCII_CLASS_WORD 1
#define TOKENIZER_BPE_ASCII_CLASS_DIGIT 2
#define TOKENIZER_BPE_ASCII_CLASS_WHITESPACE 3
#define TOKENIZER_BPE_ASCII_CLASS_PUNCT 4

U8 TokenizerBPEASCIIClass(U8 byte)
{
    if ((byte >= 'a' && byte <= 'z') ||
        (byte >= 'A' && byte <= 'Z') ||
        byte == '_')
        return TOKENIZER_BPE_ASCII_CLASS_WORD;

    if (byte >= '0' && byte <= '9')
        return TOKENIZER_BPE_ASCII_CLASS_DIGIT;

    if (byte == ' ' ||
        byte == '\t' ||
        byte == '\n' ||
        byte == '\r' ||
        byte == '\f' ||
        byte == '\v')
        return TOKENIZER_BPE_ASCII_CLASS_WHITESPACE;

    return TOKENIZER_BPE_ASCII_CLASS_PUNCT;
}

I32 TokenizerBPEEncodePromptChecked(U8 *bytes,
                                    U64 byte_len,
                                    U64 *io_cursor,
                                    U64 prompt_nbytes,
                                    I32 *rank_left_tokens,
                                    I32 *rank_right_tokens,
                                    U32 *rank_values,
                                    I32 *rank_merged_tokens,
                                    U64 rank_table_count,
                                    U64 rank_table_capacity,
                                    I32 *out_token_ids,
                                    U64 out_token_capacity,
                                    U64 *out_token_count)
{
    U64 cursor;
    U64 prompt_end;
    U64 scan;
    U64 span_start;
    U64 span_nbytes;
    U8 lead;
    U8 span_class;
    U8 next_class;
    U64 span_cursor;
    U64 span_token_count;
    U64 remaining_capacity;
    U64 staged_count;
    I32 *staged_tokens;
    U64 staged_capacity;
    U64 i;
    U32 staged_codepoint;
    U8 staged_used;
    I32 err;

    if (!bytes || !io_cursor || !out_token_ids || !out_token_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (byte_len > 0x7FFFFFFFFFFFFFFF ||
        prompt_nbytes > 0x7FFFFFFFFFFFFFFF ||
        rank_table_capacity > 0x7FFFFFFFFFFFFFFF ||
        out_token_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (rank_table_count > rank_table_capacity)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (rank_table_count > 0 &&
        (!rank_left_tokens || !rank_right_tokens || !rank_values || !rank_merged_tokens))
        return TOKENIZER_BPE_ERR_NULL_PTR;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (prompt_nbytes > byte_len - cursor)
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS;

    prompt_end = cursor + prompt_nbytes;
    if (prompt_end < cursor || prompt_end > byte_len)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    // Stage whole-prompt token output so malformed UTF-8 or capacity failures
    // never partially mutate caller output buffers or counters.
    staged_capacity = out_token_capacity;
    if (!staged_capacity)
        staged_capacity = 1;

    staged_tokens = MAlloc(sizeof(I32) * staged_capacity);
    if (!staged_tokens)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    scan = cursor;
    staged_count = 0;

    while (scan < prompt_end) {
        span_start = scan;
        lead = bytes[scan];

        if (lead <= 0x7F) {
            // ASCII pretokenization classes: words, digits, whitespace, and
            // punctuation all form independent contiguous merge spans.
            span_class = TokenizerBPEASCIIClass(lead);
            scan++;

            while (scan < prompt_end) {
                if (bytes[scan] > 0x7F)
                    break;

                next_class = TokenizerBPEASCIIClass(bytes[scan]);
                if (next_class != span_class)
                    break;

                scan++;
            }
        } else {
            // Non-ASCII lanes are advanced by checked UTF-8 decode so split
            // points always align to full codepoints (never inside sequences).
            err = TokenizerUTF8NextCodepointChecked(bytes,
                                                    prompt_end,
                                                    &scan,
                                                    &staged_codepoint,
                                                    &staged_used);
            if (err != TOKENIZER_UTF8_OK) {
                Free(staged_tokens);
                return err;
            }

            // Group contiguous non-ASCII scalars into one span. This mirrors
            // byte-level BPE behavior while preserving UTF-8 boundary safety.
            while (scan < prompt_end && bytes[scan] > 0x7F) {
                err = TokenizerUTF8NextCodepointChecked(bytes,
                                                        prompt_end,
                                                        &scan,
                                                        &staged_codepoint,
                                                        &staged_used);
                if (err != TOKENIZER_UTF8_OK) {
                    Free(staged_tokens);
                    return err;
                }
            }
        }

        span_nbytes = scan - span_start;
        if (!span_nbytes) {
            Free(staged_tokens);
            return TOKENIZER_BPE_ERR_BAD_PARAM;
        }

        if (staged_count > out_token_capacity) {
            Free(staged_tokens);
            return TOKENIZER_BPE_ERR_BAD_PARAM;
        }

        remaining_capacity = out_token_capacity - staged_count;
        span_cursor = span_start;
        span_token_count = 0;

        err = TokenizerBPEEncodeSpanChecked(bytes,
                                            byte_len,
                                            &span_cursor,
                                            span_nbytes,
                                            rank_left_tokens,
                                            rank_right_tokens,
                                            rank_values,
                                            rank_merged_tokens,
                                            rank_table_count,
                                            rank_table_capacity,
                                            staged_tokens + staged_count,
                                            remaining_capacity,
                                            &span_token_count);
        if (err != TOKENIZER_BPE_OK) {
            Free(staged_tokens);
            return err;
        }

        if (span_cursor != scan) {
            Free(staged_tokens);
            return TOKENIZER_BPE_ERR_BAD_PARAM;
        }

        if (span_token_count > remaining_capacity) {
            Free(staged_tokens);
            return TOKENIZER_BPE_ERR_BAD_PARAM;
        }

        staged_count += span_token_count;
        if (staged_count < span_token_count || staged_count > out_token_capacity) {
            Free(staged_tokens);
            return TOKENIZER_BPE_ERR_OVERFLOW;
        }
    }

    if (scan != prompt_end) {
        Free(staged_tokens);
        return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    for (i = 0; i < staged_count; i++)
        out_token_ids[i] = staged_tokens[i];

    *out_token_count = staged_count;
    *io_cursor = prompt_end;
    Free(staged_tokens);
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEEncodePromptCheckedNoPartial(U8 *bytes,
                                             U64 byte_len,
                                             U64 *io_cursor,
                                             U64 prompt_nbytes,
                                             I32 *rank_left_tokens,
                                             I32 *rank_right_tokens,
                                             U32 *rank_values,
                                             I32 *rank_merged_tokens,
                                             U64 rank_table_count,
                                             U64 rank_table_capacity,
                                             I32 *out_token_ids,
                                             U64 out_token_capacity,
                                             U64 *out_token_count)
{
    U64 staged_cursor;
    U64 staged_token_count;
    U64 staged_token_capacity;
    I32 *staged_tokens;
    I32 err;
    U64 i;

    if (!bytes || !io_cursor || !out_token_ids || !out_token_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (out_token_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    // Stage caller-visible state first so failures preserve cursor/count.
    staged_cursor = *io_cursor;
    staged_token_count = *out_token_count;

    // Always provide non-null token storage to the checked prompt core.
    staged_token_capacity = out_token_capacity;
    if (!staged_token_capacity)
        staged_token_capacity = 1;

    staged_tokens = MAlloc(sizeof(I32) * staged_token_capacity);
    if (!staged_tokens)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    err = TokenizerBPEEncodePromptChecked(bytes,
                                          byte_len,
                                          &staged_cursor,
                                          prompt_nbytes,
                                          rank_left_tokens,
                                          rank_right_tokens,
                                          rank_values,
                                          rank_merged_tokens,
                                          rank_table_count,
                                          rank_table_capacity,
                                          staged_tokens,
                                          out_token_capacity,
                                          &staged_token_count);
    if (err != TOKENIZER_BPE_OK) {
        Free(staged_tokens);
        return err;
    }

    if (staged_token_count > out_token_capacity) {
        Free(staged_tokens);
        return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    // Commit output tokens only after successful full-prompt encode.
    for (i = 0; i < staged_token_count; i++)
        out_token_ids[i] = staged_tokens[i];

    *out_token_count = staged_token_count;
    *io_cursor = staged_cursor;
    Free(staged_tokens);
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEEncodePromptCheckedDefaultCapacity(U8 *bytes,
                                                   U64 byte_len,
                                                   U64 *io_cursor,
                                                   U64 prompt_nbytes,
                                                   I32 *rank_left_tokens,
                                                   I32 *rank_right_tokens,
                                                   U32 *rank_values,
                                                   I32 *rank_merged_tokens,
                                                   U64 rank_table_count,
                                                   U64 rank_table_capacity,
                                                   I32 *out_token_ids,
                                                   U64 *out_token_count)
{
    U64 derived_capacity;

    if (!bytes || !io_cursor || !out_token_ids || !out_token_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Canonical default-capacity rule: byte-level pretokenization emits at
    // most one token per prompt byte before merge passes shrink the sequence.
    if (prompt_nbytes > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    derived_capacity = prompt_nbytes;

    // Delegate checked UTF-8 bounds, rank-table, and no-partial semantics
    // to the explicit-capacity prompt encoder core.
    return TokenizerBPEEncodePromptChecked(bytes,
                                           byte_len,
                                           io_cursor,
                                           prompt_nbytes,
                                           rank_left_tokens,
                                           rank_right_tokens,
                                           rank_values,
                                           rank_merged_tokens,
                                           rank_table_count,
                                           rank_table_capacity,
                                           out_token_ids,
                                           derived_capacity,
                                           out_token_count);
}
