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

    // Decode continuation payload bits with strict 10xxxxxx checks.
    if (need > 1) {
        cont = bytes[cursor + 1];
        if (!TokenizerUTF8IsContinuationByte(cont))
            return TOKENIZER_UTF8_ERR_BAD_CONTINUATION;
        codepoint = (codepoint << 6) | (cont & 0x3F);
    }

    if (need > 2) {
        cont = bytes[cursor + 2];
        if (!TokenizerUTF8IsContinuationByte(cont))
            return TOKENIZER_UTF8_ERR_BAD_CONTINUATION;
        codepoint = (codepoint << 6) | (cont & 0x3F);
    }

    if (need > 3) {
        cont = bytes[cursor + 3];
        if (!TokenizerUTF8IsContinuationByte(cont))
            return TOKENIZER_UTF8_ERR_BAD_CONTINUATION;
        codepoint = (codepoint << 6) | (cont & 0x3F);
    }

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

        // Continuation lane checks and payload-bit assembly.
        if (need > 1) {
            cont = bytes[scan + 1];
            if (!TokenizerUTF8IsContinuationByte(cont))
                return TOKENIZER_UTF8_ERR_BAD_CONTINUATION;
            codepoint = (codepoint << 6) | (cont & 0x3F);
        }

        if (need > 2) {
            cont = bytes[scan + 2];
            if (!TokenizerUTF8IsContinuationByte(cont))
                return TOKENIZER_UTF8_ERR_BAD_CONTINUATION;
            codepoint = (codepoint << 6) | (cont & 0x3F);
        }

        if (need > 3) {
            cont = bytes[scan + 3];
            if (!TokenizerUTF8IsContinuationByte(cont))
                return TOKENIZER_UTF8_ERR_BAD_CONTINUATION;
            codepoint = (codepoint << 6) | (cont & 0x3F);
        }

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
