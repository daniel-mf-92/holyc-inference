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

I32 TokenizerUTF8ValidateContinuationSpanCheckedNoPartial(U8 *bytes,
                                                          U64 byte_len,
                                                          U64 *io_cursor,
                                                          U64 continuation_count,
                                                          U64 *out_bytes_consumed)
{
    U64 staged_cursor;
    U64 staged_consumed;
    I32 err;

    if (!bytes || !io_cursor || !out_bytes_consumed)
        return TOKENIZER_UTF8_ERR_NULL_PTR;

    // Stage caller-visible cursor and count before entering checked core so
    // malformed continuation spans never partially update caller outputs.
    staged_cursor = *io_cursor;
    staged_consumed = *out_bytes_consumed;

    err = TokenizerUTF8ValidateContinuationSpanChecked(bytes,
                                                       byte_len,
                                                       &staged_cursor,
                                                       continuation_count,
                                                       &staged_consumed);
    if (err != TOKENIZER_UTF8_OK)
        return err;

    // All-or-nothing wrapper commit: publish staged cursor/count only when
    // the full checked continuation span validation succeeds.
    *io_cursor = staged_cursor;
    *out_bytes_consumed = staged_consumed;
    return TOKENIZER_UTF8_OK;
}

I32 TokenizerUTF8ValidateContinuationSpanCheckedDefault(U8 *bytes,
                                                        U64 byte_len,
                                                        U64 *io_cursor,
                                                        U64 *out_bytes_consumed)
{
    U64 cursor;
    U64 continuation_count;

    if (!bytes || !io_cursor || !out_bytes_consumed)
        return TOKENIZER_UTF8_ERR_NULL_PTR;

    // Keep default-span arithmetic inside signed-positive I64 bounds.
    if (byte_len > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_UTF8_ERR_OVERFLOW;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_UTF8_ERR_BAD_PARAM;

    // Canonical tail-span derivation: validate all remaining bytes from
    // the caller cursor as continuation bytes.
    continuation_count = byte_len - cursor;

    // Delegate no-partial cursor/consumed commit semantics to explicit-count
    // checked implementation.
    return TokenizerUTF8ValidateContinuationSpanChecked(bytes,
                                                        byte_len,
                                                        io_cursor,
                                                        continuation_count,
                                                        out_bytes_consumed);
}

I32 TokenizerUTF8ValidateContinuationSpanCheckedDefaultNoPartial(U8 *bytes,
                                                                 U64 byte_len,
                                                                 U64 *io_cursor,
                                                                 U64 *out_bytes_consumed)
{
    U64 cursor;
    U64 continuation_count;
    U64 staged_cursor;
    U64 staged_consumed;
    I32 err;

    if (!bytes || !io_cursor || !out_bytes_consumed)
        return TOKENIZER_UTF8_ERR_NULL_PTR;

    // Keep default-span derivation and staged-call arithmetic inside signed
    // positive I64 domain used by HolyC runtime checks.
    if (byte_len > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_UTF8_ERR_OVERFLOW;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_UTF8_ERR_BAD_PARAM;

    // Canonical default count: validate full suffix from caller cursor.
    continuation_count = byte_len - cursor;

    // Stage caller-visible outputs so malformed suffixes keep cursor/count
    // unchanged (strict all-or-nothing no-partial semantics).
    staged_cursor = cursor;
    staged_consumed = *out_bytes_consumed;

    err = TokenizerUTF8ValidateContinuationSpanChecked(bytes,
                                                       byte_len,
                                                       &staged_cursor,
                                                       continuation_count,
                                                       &staged_consumed);
    if (err != TOKENIZER_UTF8_OK)
        return err;

    *io_cursor = staged_cursor;
    *out_bytes_consumed = staged_consumed;
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

I32 TokenizerUTF8CountCodepointsCheckedDefaultSpanNoPartial(U8 *bytes,
                                                            U64 byte_len,
                                                            U64 *io_cursor,
                                                            U64 *out_codepoint_count)
{
    U64 cursor;
    U64 in_nbytes;
    U64 staged_cursor;
    U64 staged_count;
    I32 err;

    if (!bytes || !io_cursor || !out_codepoint_count)
        return TOKENIZER_UTF8_ERR_NULL_PTR;

    // Keep default-span derivation inside signed-positive I64 range.
    if (byte_len > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_UTF8_ERR_OVERFLOW;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_UTF8_ERR_BAD_PARAM;

    // Canonical suffix span starts at caller cursor and extends to byte_len.
    in_nbytes = byte_len - cursor;

    // Stage caller-visible cursor/count so malformed UTF-8 suffixes cannot
    // partially commit decode progress.
    staged_cursor = cursor;
    staged_count = *out_codepoint_count;

    err = TokenizerUTF8CountCodepointsChecked(bytes,
                                              byte_len,
                                              &staged_cursor,
                                              in_nbytes,
                                              &staged_count);
    if (err != TOKENIZER_UTF8_OK)
        return err;

    // All-or-nothing commit after checked full-suffix count success.
    *io_cursor = staged_cursor;
    *out_codepoint_count = staged_count;
    return TOKENIZER_UTF8_OK;
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

I32 TokenizerBPEMergePairPriorityFindCheckedDefault(I32 *token_ids,
                                                    U64 token_count,
                                                    I32 *rank_left_tokens,
                                                    I32 *rank_right_tokens,
                                                    U32 *rank_values,
                                                    U64 rank_table_count,
                                                    U64 *out_left_index,
                                                    U32 *out_rank,
                                                    Bool *out_found)
{
    U64 derived_token_capacity;
    U64 derived_rank_table_capacity;

    if (!token_ids || !out_left_index || !out_rank || !out_found)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Canonical default-capacity rule: active spans are tightly packed.
    if (token_count > 0x7FFFFFFFFFFFFFFF || rank_table_count > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    derived_token_capacity = token_count;
    derived_rank_table_capacity = rank_table_count;

    // Delegate pair scan, rank selection, and no-partial-write contracts to
    // the explicit-capacity checked core helper.
    return TokenizerBPEMergePairPriorityFindChecked(token_ids,
                                                    token_count,
                                                    derived_token_capacity,
                                                    rank_left_tokens,
                                                    rank_right_tokens,
                                                    rank_values,
                                                    rank_table_count,
                                                    derived_rank_table_capacity,
                                                    out_left_index,
                                                    out_rank,
                                                    out_found);
}

I32 TokenizerBPEMergePairPriorityFindCheckedDefaultNoPartial(I32 *token_ids,
                                                             U64 token_count,
                                                             I32 *rank_left_tokens,
                                                             I32 *rank_right_tokens,
                                                             U32 *rank_values,
                                                             U64 rank_table_count,
                                                             U64 *out_left_index,
                                                             U32 *out_rank,
                                                             Bool *out_found)
{
    U64 derived_token_capacity;
    U64 derived_rank_table_capacity;
    U64 staged_left_index;
    U32 staged_rank;
    Bool staged_found;
    I32 err;

    if (!token_ids || !out_left_index || !out_rank || !out_found)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Canonical default-capacity contract: active token/rank spans are packed.
    if (token_count > 0x7FFFFFFFFFFFFFFF || rank_table_count > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    derived_token_capacity = token_count;
    derived_rank_table_capacity = rank_table_count;

    // Stage caller-visible outputs so this wrapper stays all-or-nothing even
    // when the checked core fails on malformed rank tables.
    staged_left_index = *out_left_index;
    staged_rank = *out_rank;
    staged_found = *out_found;

    err = TokenizerBPEMergePairPriorityFindChecked(token_ids,
                                                   token_count,
                                                   derived_token_capacity,
                                                   rank_left_tokens,
                                                   rank_right_tokens,
                                                   rank_values,
                                                   rank_table_count,
                                                   derived_rank_table_capacity,
                                                   &staged_left_index,
                                                   &staged_rank,
                                                   &staged_found);
    if (err != TOKENIZER_BPE_OK)
        return err;

    // Commit outputs only after default-capacity derivation + checked scan
    // both complete successfully.
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

I32 TokenizerBPEMergePairRankLookupCheckedNoPartial(I32 left_token,
                                                    I32 right_token,
                                                    I32 *rank_left_tokens,
                                                    I32 *rank_right_tokens,
                                                    U32 *rank_values,
                                                    U64 rank_table_count,
                                                    U64 rank_table_capacity,
                                                    U32 *out_rank,
                                                    Bool *out_found)
{
    U32 staged_rank;
    Bool staged_found;
    I32 err;

    if (!out_rank || !out_found)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Stage caller-visible outputs so malformed-capacity or pointer faults
    // in the checked core cannot partially overwrite the caller state.
    staged_rank = *out_rank;
    staged_found = *out_found;

    err = TokenizerBPETokenPairRankLookupChecked(left_token,
                                                 right_token,
                                                 rank_left_tokens,
                                                 rank_right_tokens,
                                                 rank_values,
                                                 rank_table_count,
                                                 rank_table_capacity,
                                                 &staged_rank,
                                                 &staged_found);
    if (err != TOKENIZER_BPE_OK)
        return err;

    // All-or-nothing commit after full checked lookup succeeds.
    *out_rank = staged_rank;
    *out_found = staged_found;
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

I32 TokenizerBPEMergePairRankLookupCheckedDefaultNoPartial(I32 left_token,
                                                           I32 right_token,
                                                           I32 *rank_left_tokens,
                                                           I32 *rank_right_tokens,
                                                           U32 *rank_values,
                                                           U64 rank_table_count,
                                                           U32 *out_rank,
                                                           Bool *out_found)
{
    U64 derived_capacity;
    U32 staged_rank;
    Bool staged_found;
    I32 err;

    if (!out_rank || !out_found)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Canonical default-capacity rule: rank tables are tightly packed, so
    // active row count equals effective storage capacity.
    if (rank_table_count > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    derived_capacity = rank_table_count;

    // Stage caller-visible outputs so malformed rank-table pointers/capacity
    // faults in the checked core cannot partially overwrite caller state.
    staged_rank = *out_rank;
    staged_found = *out_found;

    err = TokenizerBPETokenPairRankLookupChecked(left_token,
                                                 right_token,
                                                 rank_left_tokens,
                                                 rank_right_tokens,
                                                 rank_values,
                                                 rank_table_count,
                                                 derived_capacity,
                                                 &staged_rank,
                                                 &staged_found);
    if (err != TOKENIZER_BPE_OK)
        return err;

    // All-or-nothing commit after full checked lookup succeeds.
    *out_rank = staged_rank;
    *out_found = staged_found;
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEMergePairRankLookupCheckedDefaultValidateRanks(
                                                           I32 left_token,
                                                           I32 right_token,
                                                           I32 *rank_left_tokens,
                                                           I32 *rank_right_tokens,
                                                           U32 *rank_values,
                                                           U64 rank_table_count,
                                                           U32 *out_rank,
                                                           Bool *out_found)
{
    U64 i;
    U64 run_start;
    U64 run_scan;
    U64 next_scan;
    I32 key_left;
    I32 key_right;
    Bool min_rank_set;
    U32 min_rank;

    if (!out_rank || !out_found)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Keep default-capacity derivation and validation loops inside signed I64
    // range to match checked-core overflow contracts.
    if (rank_table_count > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (rank_table_count > 0 &&
        (!rank_left_tokens || !rank_right_tokens || !rank_values))
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Binary-search rank lookup requires lexicographically monotonic
    // (left,right) ordering across active rows.
    for (i = 1; i < rank_table_count; i++) {
        if (rank_left_tokens[i - 1] > rank_left_tokens[i])
            return TOKENIZER_BPE_ERR_BAD_PARAM;

        if (rank_left_tokens[i - 1] == rank_left_tokens[i] &&
            rank_right_tokens[i - 1] > rank_right_tokens[i])
            return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    // Deterministic duplicate-pair guard: each equal-key run must contain at
    // least one row so "minimum rank wins" semantics are always defined.
    run_start = 0;
    while (run_start < rank_table_count) {
        key_left = rank_left_tokens[run_start];
        key_right = rank_right_tokens[run_start];

        min_rank_set = FALSE;
        min_rank = 0;
        run_scan = run_start;
        while (run_scan < rank_table_count) {
            if (rank_left_tokens[run_scan] != key_left)
                break;
            if (rank_right_tokens[run_scan] != key_right)
                break;

            if (!min_rank_set || rank_values[run_scan] < min_rank) {
                min_rank = rank_values[run_scan];
                min_rank_set = TRUE;
            }

            next_scan = run_scan + 1;
            if (next_scan <= run_scan)
                return TOKENIZER_BPE_ERR_OVERFLOW;
            run_scan = next_scan;
        }

        if (!min_rank_set)
            return TOKENIZER_BPE_ERR_BAD_PARAM;

        run_start = run_scan;
    }

    // After strict sorted-table and duplicate-run validation, dispatch to the
    // canonical default-capacity wrapper for rank lookup.
    return TokenizerBPEMergePairRankLookupCheckedDefault(left_token,
                                                         right_token,
                                                         rank_left_tokens,
                                                         rank_right_tokens,
                                                         rank_values,
                                                         rank_table_count,
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

I32 TokenizerBPEMergePairTokenLookupCheckedDefault(I32 left_token,
                                                   I32 right_token,
                                                   I32 *rank_left_tokens,
                                                   I32 *rank_right_tokens,
                                                   U32 *rank_values,
                                                   I32 *rank_merged_tokens,
                                                   U64 rank_table_count,
                                                   I32 *out_merged_token,
                                                   U32 *out_rank,
                                                   Bool *out_found)
{
    U64 derived_capacity;

    if (!out_merged_token || !out_rank || !out_found)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Canonical default-capacity rule: BPE rank rows are tightly packed, so
    // the active row count is also the effective capacity.
    if (rank_table_count > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    derived_capacity = rank_table_count;

    // Delegate duplicate-rank resolution and no-partial-write behavior to the
    // checked core by passing the canonical derived capacity.
    return TokenizerBPEMergePairTokenLookupChecked(left_token,
                                                   right_token,
                                                   rank_left_tokens,
                                                   rank_right_tokens,
                                                   rank_values,
                                                   rank_merged_tokens,
                                                   rank_table_count,
                                                   derived_capacity,
                                                   out_merged_token,
                                                   out_rank,
                                                   out_found);
}

I32 TokenizerBPEMergePairTokenLookupCheckedDefaultValidateRanks(
                                                   I32 left_token,
                                                   I32 right_token,
                                                   I32 *rank_left_tokens,
                                                   I32 *rank_right_tokens,
                                                   U32 *rank_values,
                                                   I32 *rank_merged_tokens,
                                                   U64 rank_table_count,
                                                   I32 *out_merged_token,
                                                   U32 *out_rank,
                                                   Bool *out_found)
{
    U64 i;

    if (!out_merged_token || !out_rank || !out_found)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Keep default-capacity derivation and table scans within signed-positive
    // I64 bounds to match checked-core overflow contracts.
    if (rank_table_count > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (rank_table_count > 0 &&
        (!rank_left_tokens || !rank_right_tokens || !rank_values || !rank_merged_tokens))
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Guard: binary-search lookup requires lexicographically monotonic
    // (left,right) ordering across the active rank table.
    for (i = 1; i < rank_table_count; i++) {
        if (rank_left_tokens[i - 1] > rank_left_tokens[i])
            return TOKENIZER_BPE_ERR_BAD_PARAM;

        if (rank_left_tokens[i - 1] == rank_left_tokens[i] &&
            rank_right_tokens[i - 1] > rank_right_tokens[i])
            return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    // After monotonic validation, delegate duplicate-pair minimum-rank
    // selection and merged-token resolution to the canonical default wrapper.
    return TokenizerBPEMergePairTokenLookupCheckedDefault(left_token,
                                                          right_token,
                                                          rank_left_tokens,
                                                          rank_right_tokens,
                                                          rank_values,
                                                          rank_merged_tokens,
                                                          rank_table_count,
                                                          out_merged_token,
                                                          out_rank,
                                                          out_found);
}

I32 TokenizerBPEMergePairTokenLookupCheckedDefaultNoPartial(I32 left_token,
                                                            I32 right_token,
                                                            I32 *rank_left_tokens,
                                                            I32 *rank_right_tokens,
                                                            U32 *rank_values,
                                                            I32 *rank_merged_tokens,
                                                            U64 rank_table_count,
                                                            I32 *out_merged_token,
                                                            U32 *out_rank,
                                                            Bool *out_found)
{
    U64 derived_capacity;
    I32 staged_merged_token;
    U32 staged_rank;
    Bool staged_found;
    I32 err;

    if (!out_merged_token || !out_rank || !out_found)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Canonical default-capacity rule: BPE rank rows are tightly packed, so
    // active count is the effective capacity exposed to checked core helpers.
    if (rank_table_count > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    derived_capacity = rank_table_count;

    // Stage caller-visible outputs so malformed rank-table pointers/capacity
    // faults in the checked core cannot partially overwrite caller state.
    staged_merged_token = *out_merged_token;
    staged_rank = *out_rank;
    staged_found = *out_found;

    err = TokenizerBPEMergePairTokenLookupChecked(left_token,
                                                  right_token,
                                                  rank_left_tokens,
                                                  rank_right_tokens,
                                                  rank_values,
                                                  rank_merged_tokens,
                                                  rank_table_count,
                                                  derived_capacity,
                                                  &staged_merged_token,
                                                  &staged_rank,
                                                  &staged_found);
    if (err != TOKENIZER_BPE_OK)
        return err;

    // Commit only after successful checked lookup to preserve all-or-nothing
    // wrapper semantics under adversarial malformed-table inputs.
    *out_merged_token = staged_merged_token;
    *out_rank = staged_rank;
    *out_found = staged_found;
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEMergePairTokenLookupCheckedDefaultNoPartialValidateRanks(
                                                            I32 left_token,
                                                            I32 right_token,
                                                            I32 *rank_left_tokens,
                                                            I32 *rank_right_tokens,
                                                            U32 *rank_values,
                                                            I32 *rank_merged_tokens,
                                                            U64 rank_table_count,
                                                            I32 *out_merged_token,
                                                            U32 *out_rank,
                                                            Bool *out_found)
{
    U64 derived_capacity;
    U64 i;
    U64 run_start;
    U64 run_scan;
    U64 next_scan;
    I32 key_left;
    I32 key_right;
    Bool min_rank_set;
    U32 min_rank;
    I32 staged_merged_token;
    U32 staged_rank;
    Bool staged_found;
    I32 err;

    if (!out_merged_token || !out_rank || !out_found)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Canonical default-capacity derivation uses the active row count as the
    // effective checked-core capacity; keep arithmetic inside signed I64 bounds.
    if (rank_table_count > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (rank_table_count > 0 &&
        (!rank_left_tokens || !rank_right_tokens || !rank_values || !rank_merged_tokens))
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Validate strict lexicographic monotonicity required by binary-search
    // lookup over (left,right) merge-pair key rows.
    for (i = 1; i < rank_table_count; i++) {
        if (rank_left_tokens[i - 1] > rank_left_tokens[i])
            return TOKENIZER_BPE_ERR_BAD_PARAM;

        if (rank_left_tokens[i - 1] == rank_left_tokens[i] &&
            rank_right_tokens[i - 1] > rank_right_tokens[i])
            return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    // Validate each duplicate-key run has at least one rank row so deterministic
    // "lowest rank wins" duplicate handling in the checked core is well-defined.
    run_start = 0;
    while (run_start < rank_table_count) {
        key_left = rank_left_tokens[run_start];
        key_right = rank_right_tokens[run_start];

        min_rank_set = FALSE;
        min_rank = 0;
        run_scan = run_start;
        while (run_scan < rank_table_count) {
            if (rank_left_tokens[run_scan] != key_left)
                break;
            if (rank_right_tokens[run_scan] != key_right)
                break;

            if (!min_rank_set || rank_values[run_scan] < min_rank) {
                min_rank = rank_values[run_scan];
                min_rank_set = TRUE;
            }

            next_scan = run_scan + 1;
            if (next_scan <= run_scan)
                return TOKENIZER_BPE_ERR_OVERFLOW;
            run_scan = next_scan;
        }

        if (!min_rank_set)
            return TOKENIZER_BPE_ERR_BAD_PARAM;

        run_start = run_scan;
    }

    derived_capacity = rank_table_count;

    // Stage outputs so malformed-rank failures never partially overwrite
    // caller-visible merged token, rank, or found state.
    staged_merged_token = *out_merged_token;
    staged_rank = *out_rank;
    staged_found = *out_found;

    err = TokenizerBPEMergePairTokenLookupChecked(left_token,
                                                  right_token,
                                                  rank_left_tokens,
                                                  rank_right_tokens,
                                                  rank_values,
                                                  rank_merged_tokens,
                                                  rank_table_count,
                                                  derived_capacity,
                                                  &staged_merged_token,
                                                  &staged_rank,
                                                  &staged_found);
    if (err != TOKENIZER_BPE_OK)
        return err;

    *out_merged_token = staged_merged_token;
    *out_rank = staged_rank;
    *out_found = staged_found;
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

I32 TokenizerBPEMergeApplyBestPriorityCheckedNoPartialValidateRanks(
                                                       I32 *token_ids,
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
    U64 i;
    U64 run_start;
    U64 run_scan;
    I32 key_left;
    I32 key_right;
    U32 min_rank;
    Bool min_rank_set;
    Bool min_merged_set;
    I32 min_rank_merged_token;

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

    // Guard 1: rank table key order must be monotonic in lexicographic
    // (left,right) order, otherwise binary-search based lookup is undefined.
    for (i = 1; i < rank_table_count; i++) {
        if (rank_left_tokens[i - 1] > rank_left_tokens[i])
            return TOKENIZER_BPE_ERR_BAD_PARAM;

        if (rank_left_tokens[i - 1] == rank_left_tokens[i] &&
            rank_right_tokens[i - 1] > rank_right_tokens[i])
            return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    // Guard 2: for duplicate keys, every minimum-rank row must map to the same
    // merged token so min-rank tie resolution stays deterministic.
    run_start = 0;
    while (run_start < rank_table_count) {
        key_left = rank_left_tokens[run_start];
        key_right = rank_right_tokens[run_start];
        min_rank_set = FALSE;
        min_merged_set = FALSE;
        run_scan = run_start;

        while (run_scan < rank_table_count) {
            if (rank_left_tokens[run_scan] != key_left)
                break;
            if (rank_right_tokens[run_scan] != key_right)
                break;

            if (!min_rank_set || rank_values[run_scan] < min_rank) {
                min_rank = rank_values[run_scan];
                min_rank_merged_token = rank_merged_tokens[run_scan];
                min_rank_set = TRUE;
                min_merged_set = TRUE;
            } else if (rank_values[run_scan] == min_rank) {
                if (!min_merged_set) {
                    min_rank_merged_token = rank_merged_tokens[run_scan];
                    min_merged_set = TRUE;
                } else if (rank_merged_tokens[run_scan] != min_rank_merged_token) {
                    return TOKENIZER_BPE_ERR_BAD_PARAM;
                }
            }

            run_scan++;
            if (run_scan == 0)
                return TOKENIZER_BPE_ERR_OVERFLOW;
        }

        run_start = run_scan;
    }

    // Delegate merge selection + application to checked no-partial core after
    // rank-table monotonic/determinism guards pass.
    return TokenizerBPEMergeApplyBestPriorityCheckedNoPartial(token_ids,
                                                              token_count,
                                                              token_capacity,
                                                              rank_left_tokens,
                                                              rank_right_tokens,
                                                              rank_values,
                                                              rank_merged_tokens,
                                                              rank_table_count,
                                                              rank_table_capacity,
                                                              out_merged_token,
                                                              out_rank,
                                                              out_found,
                                                              out_token_count);
}

I32 TokenizerBPEMergeApplyBestPriorityCheckedDefault(I32 *token_ids,
                                                     U64 token_count,
                                                     I32 *rank_left_tokens,
                                                     I32 *rank_right_tokens,
                                                     U32 *rank_values,
                                                     I32 *rank_merged_tokens,
                                                     U64 rank_table_count,
                                                     I32 *out_merged_token,
                                                     U32 *out_rank,
                                                     Bool *out_found,
                                                     U64 *out_token_count)
{
    U64 derived_token_capacity;
    U64 derived_rank_table_capacity;

    if (!token_ids ||
        !out_merged_token ||
        !out_rank ||
        !out_found ||
        !out_token_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Canonical checked-default contract: capacities are exactly logical counts.
    if (token_count > 0x7FFFFFFFFFFFFFFF ||
        rank_table_count > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    derived_token_capacity = token_count;
    derived_rank_table_capacity = rank_table_count;

    // Delegate merge selection, merge application, and no-partial semantics.
    return TokenizerBPEMergeApplyBestPriorityChecked(token_ids,
                                                     token_count,
                                                     derived_token_capacity,
                                                     rank_left_tokens,
                                                     rank_right_tokens,
                                                     rank_values,
                                                     rank_merged_tokens,
                                                     rank_table_count,
                                                     derived_rank_table_capacity,
                                                     out_merged_token,
                                                     out_rank,
                                                     out_found,
                                                     out_token_count);
}

I32 TokenizerBPEMergeApplyBestPriorityCheckedDefaultNoPartial(I32 *token_ids,
                                                              U64 token_count,
                                                              I32 *rank_left_tokens,
                                                              I32 *rank_right_tokens,
                                                              U32 *rank_values,
                                                              I32 *rank_merged_tokens,
                                                              U64 rank_table_count,
                                                              I32 *out_merged_token,
                                                              U32 *out_rank,
                                                              Bool *out_found,
                                                              U64 *out_token_count)
{
    U64 derived_token_capacity;
    U64 derived_rank_table_capacity;

    if (!token_ids ||
        !out_merged_token ||
        !out_rank ||
        !out_found ||
        !out_token_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (token_count > 0x7FFFFFFFFFFFFFFF ||
        rank_table_count > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    // Canonical default-capacity contract: active counts equal storage extents.
    derived_token_capacity = token_count;
    derived_rank_table_capacity = rank_table_count;

    // Delegate staged all-or-nothing output commit to the checked no-partial
    // helper while preserving canonical default-capacity derivation.
    return TokenizerBPEMergeApplyBestPriorityCheckedNoPartial(token_ids,
                                                              token_count,
                                                              derived_token_capacity,
                                                              rank_left_tokens,
                                                              rank_right_tokens,
                                                              rank_values,
                                                              rank_merged_tokens,
                                                              rank_table_count,
                                                              derived_rank_table_capacity,
                                                              out_merged_token,
                                                              out_rank,
                                                              out_found,
                                                              out_token_count);
}

I32 TokenizerBPEMergeApplyBestPriorityCheckedDefaultNoPartialValidateRanks(
                                                              I32 *token_ids,
                                                              U64 token_count,
                                                              I32 *rank_left_tokens,
                                                              I32 *rank_right_tokens,
                                                              U32 *rank_values,
                                                              I32 *rank_merged_tokens,
                                                              U64 rank_table_count,
                                                              I32 *out_merged_token,
                                                              U32 *out_rank,
                                                              Bool *out_found,
                                                              U64 *out_token_count)
{
    U64 derived_token_capacity;
    U64 derived_rank_table_capacity;

    if (!token_ids ||
        !out_merged_token ||
        !out_rank ||
        !out_found ||
        !out_token_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (token_count > 0x7FFFFFFFFFFFFFFF ||
        rank_table_count > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    // Canonical default-capacity contract: caller logical counts define
    // storage extents for token and rank tables.
    derived_token_capacity = token_count;
    derived_rank_table_capacity = rank_table_count;

    // Compose rank-table monotonic-key + duplicate minimum-rank deterministic
    // merge-token validation with no-partial greedy best-priority merge apply.
    return TokenizerBPEMergeApplyBestPriorityCheckedNoPartialValidateRanks(
                                                              token_ids,
                                                              token_count,
                                                              derived_token_capacity,
                                                              rank_left_tokens,
                                                              rank_right_tokens,
                                                              rank_values,
                                                              rank_merged_tokens,
                                                              rank_table_count,
                                                              derived_rank_table_capacity,
                                                              out_merged_token,
                                                              out_rank,
                                                              out_found,
                                                              out_token_count);
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

I32 TokenizerBPEMergeCandidatesBuildCheckedDefaultCapacity(
                                                I32 *token_ids,
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
                                                U64 *out_candidate_count)
{
    U64 derived_capacity;

    if (!token_ids ||
        !out_left_tokens ||
        !out_right_tokens ||
        !out_left_indices ||
        !out_ranks ||
        !out_candidate_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (token_capacity > 0x7FFFFFFFFFFFFFFF ||
        rank_table_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    // Canonical default-capacity rule for adjacent pair candidates:
    // a token span of N entries has at most N-1 adjacent pairs.
    if (token_count < 2)
        derived_capacity = 0;
    else
        derived_capacity = token_count - 1;

    // Delegate exact checked lookup/writes/no-partial semantics to core helper.
    return TokenizerBPEMergeCandidatesBuildChecked(token_ids,
                                                   token_count,
                                                   token_capacity,
                                                   rank_left_tokens,
                                                   rank_right_tokens,
                                                   rank_values,
                                                   rank_table_count,
                                                   rank_table_capacity,
                                                   out_left_tokens,
                                                   out_right_tokens,
                                                   out_left_indices,
                                                   out_ranks,
                                                   derived_capacity,
                                                   out_candidate_count);
}

I32 TokenizerBPEMergeCandidatesBuildCheckedDefault(I32 *token_ids,
                                                   U64 token_count,
                                                   I32 *rank_left_tokens,
                                                   I32 *rank_right_tokens,
                                                   U32 *rank_values,
                                                   U64 rank_table_count,
                                                   I32 *out_left_tokens,
                                                   I32 *out_right_tokens,
                                                   U64 *out_left_indices,
                                                   U32 *out_ranks,
                                                   U64 *out_candidate_count)
{
    U64 derived_token_capacity;
    U64 derived_rank_table_capacity;
    U64 derived_candidate_capacity;

    if (!token_ids ||
        !out_left_tokens ||
        !out_right_tokens ||
        !out_left_indices ||
        !out_ranks ||
        !out_candidate_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Canonical checked-default contract: capacities equal logical counts.
    if (token_count > 0x7FFFFFFFFFFFFFFF ||
        rank_table_count > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    derived_token_capacity = token_count;
    derived_rank_table_capacity = rank_table_count;

    // Adjacent-pair bound: N tokens can emit at most N-1 merge candidates.
    if (token_count < 2)
        derived_candidate_capacity = 0;
    else
        derived_candidate_capacity = token_count - 1;

    // Delegate exact lookup, bounds, and no-partial output semantics.
    return TokenizerBPEMergeCandidatesBuildChecked(token_ids,
                                                   token_count,
                                                   derived_token_capacity,
                                                   rank_left_tokens,
                                                   rank_right_tokens,
                                                   rank_values,
                                                   rank_table_count,
                                                   derived_rank_table_capacity,
                                                   out_left_tokens,
                                                   out_right_tokens,
                                                   out_left_indices,
                                                   out_ranks,
                                                   derived_candidate_capacity,
                                                   out_candidate_count);
}

I32 TokenizerBPEMergeCandidatesBuildCheckedDefaultValidateRanks(
                                                   I32 *token_ids,
                                                   U64 token_count,
                                                   I32 *rank_left_tokens,
                                                   I32 *rank_right_tokens,
                                                   U32 *rank_values,
                                                   U64 rank_table_count,
                                                   I32 *out_left_tokens,
                                                   I32 *out_right_tokens,
                                                   U64 *out_left_indices,
                                                   U32 *out_ranks,
                                                   U64 *out_candidate_count)
{
    U64 i;
    U64 run_start;
    U64 run_scan;
    I32 key_left;
    I32 key_right;
    U32 min_rank;
    Bool min_rank_set;

    if (!token_ids ||
        !out_left_tokens ||
        !out_right_tokens ||
        !out_left_indices ||
        !out_ranks ||
        !out_candidate_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (token_count > 0x7FFFFFFFFFFFFFFF ||
        rank_table_count > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (rank_table_count > 0 &&
        (!rank_left_tokens || !rank_right_tokens || !rank_values))
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Guard 1: rank-table keys must be monotonic in lexicographic
    // (left,right) order so binary-search-based rank lookup remains valid.
    for (i = 1; i < rank_table_count; i++) {
        if (rank_left_tokens[i - 1] > rank_left_tokens[i])
            return TOKENIZER_BPE_ERR_BAD_PARAM;

        if (rank_left_tokens[i - 1] == rank_left_tokens[i] &&
            rank_right_tokens[i - 1] > rank_right_tokens[i])
            return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    // Guard 2: explicitly walk duplicate-key runs and derive the minimal rank
    // for each run so duplicate-key determinism is validated before dispatch.
    run_start = 0;
    while (run_start < rank_table_count) {
        key_left = rank_left_tokens[run_start];
        key_right = rank_right_tokens[run_start];
        min_rank_set = FALSE;
        run_scan = run_start;

        while (run_scan < rank_table_count) {
            if (rank_left_tokens[run_scan] != key_left)
                break;
            if (rank_right_tokens[run_scan] != key_right)
                break;

            if (!min_rank_set || rank_values[run_scan] < min_rank) {
                min_rank = rank_values[run_scan];
                min_rank_set = TRUE;
            }

            run_scan++;
            if (run_scan == 0)
                return TOKENIZER_BPE_ERR_OVERFLOW;
        }

        if (!min_rank_set)
            return TOKENIZER_BPE_ERR_BAD_PARAM;

        run_start = run_scan;
    }

    // Delegate all checked default-capacity candidate materialization to the
    // canonical checked-default wrapper after rank-table guards pass.
    return TokenizerBPEMergeCandidatesBuildCheckedDefault(token_ids,
                                                          token_count,
                                                          rank_left_tokens,
                                                          rank_right_tokens,
                                                          rank_values,
                                                          rank_table_count,
                                                          out_left_tokens,
                                                          out_right_tokens,
                                                          out_left_indices,
                                                          out_ranks,
                                                          out_candidate_count);
}

I32 TokenizerBPEMergeCandidatesBuildCheckedDefaultCapacityNoPartial(
                                                I32 *token_ids,
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
                                                U64 *out_candidate_count)
{
    U64 derived_capacity;
    U64 staged_capacity;
    U64 staged_i32_bytes;
    U64 staged_u64_bytes;
    U64 staged_u32_bytes;
    I32 *staged_left_tokens;
    I32 *staged_right_tokens;
    U64 *staged_left_indices;
    U32 *staged_ranks;
    U64 staged_count;
    U64 i;
    I32 err;

    if (!token_ids ||
        !out_left_tokens ||
        !out_right_tokens ||
        !out_left_indices ||
        !out_ranks ||
        !out_candidate_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (token_capacity > 0x7FFFFFFFFFFFFFFF ||
        rank_table_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    // Canonical adjacent-pair default: N tokens can emit at most N-1
    // merge candidates, with underflow clamped for empty/single-token spans.
    if (token_count < 2)
        derived_capacity = 0;
    else
        derived_capacity = token_count - 1;

    // Stage output tables so any checked-core error leaves caller arrays and
    // count unchanged (all-or-nothing no-partial-write wrapper contract).
    staged_capacity = derived_capacity;
    if (!staged_capacity)
        staged_capacity = 1;

    if (staged_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(I32)))
        return TOKENIZER_BPE_ERR_OVERFLOW;
    staged_i32_bytes = sizeof(I32) * staged_capacity;
    if (!staged_i32_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (staged_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(U64)))
        return TOKENIZER_BPE_ERR_OVERFLOW;
    staged_u64_bytes = sizeof(U64) * staged_capacity;
    if (!staged_u64_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (staged_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(U32)))
        return TOKENIZER_BPE_ERR_OVERFLOW;
    staged_u32_bytes = sizeof(U32) * staged_capacity;
    if (!staged_u32_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_left_tokens = MAlloc(staged_i32_bytes);
    if (!staged_left_tokens)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_right_tokens = MAlloc(staged_i32_bytes);
    if (!staged_right_tokens) {
        Free(staged_left_tokens);
        return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    staged_left_indices = MAlloc(staged_u64_bytes);
    if (!staged_left_indices) {
        Free(staged_right_tokens);
        Free(staged_left_tokens);
        return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    staged_ranks = MAlloc(staged_u32_bytes);
    if (!staged_ranks) {
        Free(staged_left_indices);
        Free(staged_right_tokens);
        Free(staged_left_tokens);
        return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    for (i = 0; i < derived_capacity; i++) {
        staged_left_tokens[i] = out_left_tokens[i];
        staged_right_tokens[i] = out_right_tokens[i];
        staged_left_indices[i] = out_left_indices[i];
        staged_ranks[i] = out_ranks[i];
    }

    staged_count = *out_candidate_count;

    err = TokenizerBPEMergeCandidatesBuildChecked(token_ids,
                                                  token_count,
                                                  token_capacity,
                                                  rank_left_tokens,
                                                  rank_right_tokens,
                                                  rank_values,
                                                  rank_table_count,
                                                  rank_table_capacity,
                                                  staged_left_tokens,
                                                  staged_right_tokens,
                                                  staged_left_indices,
                                                  staged_ranks,
                                                  derived_capacity,
                                                  &staged_count);
    if (err != TOKENIZER_BPE_OK) {
        Free(staged_ranks);
        Free(staged_left_indices);
        Free(staged_right_tokens);
        Free(staged_left_tokens);
        return err;
    }

    // Commit staged candidate tables and count only after checked core returns
    // success so wrappers never leak partial writes on malformed inputs.
    for (i = 0; i < derived_capacity; i++) {
        out_left_tokens[i] = staged_left_tokens[i];
        out_right_tokens[i] = staged_right_tokens[i];
        out_left_indices[i] = staged_left_indices[i];
        out_ranks[i] = staged_ranks[i];
    }

    *out_candidate_count = staged_count;

    Free(staged_ranks);
    Free(staged_left_indices);
    Free(staged_right_tokens);
    Free(staged_left_tokens);
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEMergeCandidatesBuildCheckedDefaultNoPartial(
                                                   I32 *token_ids,
                                                   U64 token_count,
                                                   I32 *rank_left_tokens,
                                                   I32 *rank_right_tokens,
                                                   U32 *rank_values,
                                                   U64 rank_table_count,
                                                   I32 *out_left_tokens,
                                                   I32 *out_right_tokens,
                                                   U64 *out_left_indices,
                                                   U32 *out_ranks,
                                                   U64 *out_candidate_count)
{
    U64 derived_token_capacity;
    U64 derived_rank_table_capacity;
    U64 derived_candidate_capacity;
    U64 staged_capacity;
    U64 staged_i32_bytes;
    U64 staged_u64_bytes;
    U64 staged_u32_bytes;
    I32 *staged_left_tokens;
    I32 *staged_right_tokens;
    U64 *staged_left_indices;
    U32 *staged_ranks;
    U64 staged_count;
    U64 i;
    I32 err;

    if (!token_ids ||
        !out_left_tokens ||
        !out_right_tokens ||
        !out_left_indices ||
        !out_ranks ||
        !out_candidate_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (token_count > 0x7FFFFFFFFFFFFFFF ||
        rank_table_count > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    // Canonical checked-default contract: capacities equal logical counts.
    derived_token_capacity = token_count;
    derived_rank_table_capacity = rank_table_count;

    // Canonical adjacent-pair default: N tokens can emit at most N-1
    // merge candidates, with underflow clamped for empty/single-token spans.
    if (token_count < 2)
        derived_candidate_capacity = 0;
    else
        derived_candidate_capacity = token_count - 1;

    // Stage output tables so any checked-core error leaves caller arrays and
    // count unchanged (all-or-nothing no-partial-write wrapper contract).
    staged_capacity = derived_candidate_capacity;
    if (!staged_capacity)
        staged_capacity = 1;

    if (staged_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(I32)))
        return TOKENIZER_BPE_ERR_OVERFLOW;
    staged_i32_bytes = sizeof(I32) * staged_capacity;
    if (!staged_i32_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (staged_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(U64)))
        return TOKENIZER_BPE_ERR_OVERFLOW;
    staged_u64_bytes = sizeof(U64) * staged_capacity;
    if (!staged_u64_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (staged_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(U32)))
        return TOKENIZER_BPE_ERR_OVERFLOW;
    staged_u32_bytes = sizeof(U32) * staged_capacity;
    if (!staged_u32_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_left_tokens = MAlloc(staged_i32_bytes);
    if (!staged_left_tokens)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_right_tokens = MAlloc(staged_i32_bytes);
    if (!staged_right_tokens) {
        Free(staged_left_tokens);
        return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    staged_left_indices = MAlloc(staged_u64_bytes);
    if (!staged_left_indices) {
        Free(staged_right_tokens);
        Free(staged_left_tokens);
        return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    staged_ranks = MAlloc(staged_u32_bytes);
    if (!staged_ranks) {
        Free(staged_left_indices);
        Free(staged_right_tokens);
        Free(staged_left_tokens);
        return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    for (i = 0; i < derived_candidate_capacity; i++) {
        staged_left_tokens[i] = out_left_tokens[i];
        staged_right_tokens[i] = out_right_tokens[i];
        staged_left_indices[i] = out_left_indices[i];
        staged_ranks[i] = out_ranks[i];
    }

    staged_count = *out_candidate_count;

    err = TokenizerBPEMergeCandidatesBuildChecked(token_ids,
                                                  token_count,
                                                  derived_token_capacity,
                                                  rank_left_tokens,
                                                  rank_right_tokens,
                                                  rank_values,
                                                  rank_table_count,
                                                  derived_rank_table_capacity,
                                                  staged_left_tokens,
                                                  staged_right_tokens,
                                                  staged_left_indices,
                                                  staged_ranks,
                                                  derived_candidate_capacity,
                                                  &staged_count);
    if (err != TOKENIZER_BPE_OK) {
        Free(staged_ranks);
        Free(staged_left_indices);
        Free(staged_right_tokens);
        Free(staged_left_tokens);
        return err;
    }

    // Commit staged candidate tables and count only after checked core returns
    // success so wrappers never leak partial writes on malformed inputs.
    for (i = 0; i < derived_candidate_capacity; i++) {
        out_left_tokens[i] = staged_left_tokens[i];
        out_right_tokens[i] = staged_right_tokens[i];
        out_left_indices[i] = staged_left_indices[i];
        out_ranks[i] = staged_ranks[i];
    }

    *out_candidate_count = staged_count;

    Free(staged_ranks);
    Free(staged_left_indices);
    Free(staged_right_tokens);
    Free(staged_left_tokens);
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEMergeCandidatesBuildCheckedDefaultNoPartialValidateRanks(
                                                   I32 *token_ids,
                                                   U64 token_count,
                                                   I32 *rank_left_tokens,
                                                   I32 *rank_right_tokens,
                                                   U32 *rank_values,
                                                   U64 rank_table_count,
                                                   I32 *out_left_tokens,
                                                   I32 *out_right_tokens,
                                                   U64 *out_left_indices,
                                                   U32 *out_ranks,
                                                   U64 *out_candidate_count)
{
    U64 i;
    U64 run_start;
    U64 run_scan;
    I32 key_left;
    I32 key_right;
    U32 min_rank;
    Bool min_rank_set;

    if (!token_ids ||
        !out_left_tokens ||
        !out_right_tokens ||
        !out_left_indices ||
        !out_ranks ||
        !out_candidate_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (token_count > 0x7FFFFFFFFFFFFFFF ||
        rank_table_count > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (rank_table_count > 0 &&
        (!rank_left_tokens || !rank_right_tokens || !rank_values))
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Guard 1: rank-table keys must be monotonic in lexicographic
    // (left,right) order so binary-search-based rank lookup remains valid.
    for (i = 1; i < rank_table_count; i++) {
        if (rank_left_tokens[i - 1] > rank_left_tokens[i])
            return TOKENIZER_BPE_ERR_BAD_PARAM;

        if (rank_left_tokens[i - 1] == rank_left_tokens[i] &&
            rank_right_tokens[i - 1] > rank_right_tokens[i])
            return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    // Guard 2: explicitly walk every duplicate-key run and compute its minimal
    // rank. This enforces deterministic duplicate-key minimal-rank semantics
    // before candidate-table build dispatch.
    run_start = 0;
    while (run_start < rank_table_count) {
        key_left = rank_left_tokens[run_start];
        key_right = rank_right_tokens[run_start];
        min_rank_set = FALSE;
        run_scan = run_start;

        while (run_scan < rank_table_count) {
            if (rank_left_tokens[run_scan] != key_left)
                break;
            if (rank_right_tokens[run_scan] != key_right)
                break;

            if (!min_rank_set || rank_values[run_scan] < min_rank) {
                min_rank = rank_values[run_scan];
                min_rank_set = TRUE;
            }

            run_scan++;
            if (run_scan == 0)
                return TOKENIZER_BPE_ERR_OVERFLOW;
        }

        if (!min_rank_set)
            return TOKENIZER_BPE_ERR_BAD_PARAM;

        run_start = run_scan;
    }

    // Delegate all-or-nothing staged output commits to the checked default
    // no-partial wrapper after rank-table guards pass.
    return TokenizerBPEMergeCandidatesBuildCheckedDefaultNoPartial(token_ids,
                                                                   token_count,
                                                                   rank_left_tokens,
                                                                   rank_right_tokens,
                                                                   rank_values,
                                                                   rank_table_count,
                                                                   out_left_tokens,
                                                                   out_right_tokens,
                                                                   out_left_indices,
                                                                   out_ranks,
                                                                   out_candidate_count);
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

#define TOKENIZER_BPE_ASCII_CLASS_NON_ASCII 5

I32 TokenizerBPEPromptSpanScanChecked(U8 *bytes,
                                      U64 byte_len,
                                      U64 *io_cursor,
                                      U64 prompt_nbytes,
                                      U64 *out_span_starts,
                                      U64 *out_span_lengths,
                                      U8 *out_span_classes,
                                      U64 out_span_capacity,
                                      U64 *out_span_count)
{
    U64 cursor;
    U64 prompt_end;
    U64 scan;
    U64 span_start;
    U64 span_nbytes;
    U8 lead;
    U8 span_class;
    U8 next_class;
    U64 staged_count;
    U64 staged_capacity;
    U64 staged_starts_bytes;
    U64 staged_lengths_bytes;
    U64 staged_classes_bytes;
    U64 *staged_starts;
    U64 *staged_lengths;
    U8 *staged_classes;
    U64 i;
    U32 staged_codepoint;
    U8 staged_used;
    I32 err;

    if (!bytes || !io_cursor || !out_span_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (byte_len > 0x7FFFFFFFFFFFFFFF ||
        prompt_nbytes > 0x7FFFFFFFFFFFFFFF ||
        out_span_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (out_span_capacity > 0 &&
        (!out_span_starts || !out_span_lengths || !out_span_classes))
        return TOKENIZER_BPE_ERR_NULL_PTR;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (prompt_nbytes > byte_len - cursor)
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS;

    prompt_end = cursor + prompt_nbytes;
    if (prompt_end < cursor || prompt_end > byte_len)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    // Empty prompt is valid and commits deterministic empty span output.
    if (!prompt_nbytes) {
        *out_span_count = 0;
        *io_cursor = cursor;
        return TOKENIZER_BPE_OK;
    }

    // Non-empty prompts require storage for at least one span.
    if (!out_span_capacity)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    staged_capacity = out_span_capacity;

    // Stage span tables privately so malformed UTF-8 never partially mutates
    // caller output buffers or cursor.
    if (staged_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(U64)))
        return TOKENIZER_BPE_ERR_OVERFLOW;
    staged_starts_bytes = sizeof(U64) * staged_capacity;
    staged_lengths_bytes = sizeof(U64) * staged_capacity;

    if (staged_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(U8)))
        return TOKENIZER_BPE_ERR_OVERFLOW;
    staged_classes_bytes = sizeof(U8) * staged_capacity;

    if (!staged_starts_bytes || !staged_lengths_bytes || !staged_classes_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_starts = MAlloc(staged_starts_bytes);
    if (!staged_starts)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_lengths = MAlloc(staged_lengths_bytes);
    if (!staged_lengths) {
        Free(staged_starts);
        return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    staged_classes = MAlloc(staged_classes_bytes);
    if (!staged_classes) {
        Free(staged_lengths);
        Free(staged_starts);
        return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    scan = cursor;
    staged_count = 0;

    while (scan < prompt_end) {
        span_start = scan;
        lead = bytes[scan];

        if (lead <= 0x7F) {
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
            span_class = TOKENIZER_BPE_ASCII_CLASS_NON_ASCII;

            err = TokenizerUTF8NextCodepointChecked(bytes,
                                                    prompt_end,
                                                    &scan,
                                                    &staged_codepoint,
                                                    &staged_used);
            if (err != TOKENIZER_UTF8_OK) {
                Free(staged_classes);
                Free(staged_lengths);
                Free(staged_starts);
                return err;
            }

            while (scan < prompt_end && bytes[scan] > 0x7F) {
                err = TokenizerUTF8NextCodepointChecked(bytes,
                                                        prompt_end,
                                                        &scan,
                                                        &staged_codepoint,
                                                        &staged_used);
                if (err != TOKENIZER_UTF8_OK) {
                    Free(staged_classes);
                    Free(staged_lengths);
                    Free(staged_starts);
                    return err;
                }
            }
        }

        span_nbytes = scan - span_start;
        if (!span_nbytes) {
            Free(staged_classes);
            Free(staged_lengths);
            Free(staged_starts);
            return TOKENIZER_BPE_ERR_BAD_PARAM;
        }

        if (staged_count >= staged_capacity) {
            Free(staged_classes);
            Free(staged_lengths);
            Free(staged_starts);
            return TOKENIZER_BPE_ERR_BAD_PARAM;
        }

        staged_starts[staged_count] = span_start;
        staged_lengths[staged_count] = span_nbytes;
        staged_classes[staged_count] = span_class;
        staged_count++;

        if (!staged_count || staged_count > staged_capacity) {
            Free(staged_classes);
            Free(staged_lengths);
            Free(staged_starts);
            return TOKENIZER_BPE_ERR_OVERFLOW;
        }
    }

    if (scan != prompt_end) {
        Free(staged_classes);
        Free(staged_lengths);
        Free(staged_starts);
        return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    for (i = 0; i < staged_count; i++) {
        out_span_starts[i] = staged_starts[i];
        out_span_lengths[i] = staged_lengths[i];
        out_span_classes[i] = staged_classes[i];
    }

    *out_span_count = staged_count;
    *io_cursor = prompt_end;

    Free(staged_classes);
    Free(staged_lengths);
    Free(staged_starts);
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEPromptSpanScanCheckedDefaultCapacity(U8 *bytes,
                                                     U64 byte_len,
                                                     U64 *io_cursor,
                                                     U64 prompt_nbytes,
                                                     U64 *out_span_starts,
                                                     U64 *out_span_lengths,
                                                     U8 *out_span_classes,
                                                     U64 *out_span_count)
{
    U64 cursor;
    U64 derived_capacity;

    if (!bytes || !io_cursor || !out_span_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (byte_len > 0x7FFFFFFFFFFFFFFF ||
        prompt_nbytes > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (prompt_nbytes > byte_len - cursor)
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS;

    // Canonical default-capacity contract:
    // prompt byte span of N bytes can never split into more than N spans.
    derived_capacity = prompt_nbytes;

    // Keep explicit output-pointer contract at this wrapper boundary so
    // callers get deterministic NULL_PTR surfaces even if core preflight
    // changes in future refactors.
    if (derived_capacity > 0 &&
        (!out_span_starts || !out_span_lengths || !out_span_classes))
        return TOKENIZER_BPE_ERR_NULL_PTR;

    return TokenizerBPEPromptSpanScanChecked(bytes,
                                             byte_len,
                                             io_cursor,
                                             prompt_nbytes,
                                             out_span_starts,
                                             out_span_lengths,
                                             out_span_classes,
                                             derived_capacity,
                                             out_span_count);
}

I32 TokenizerBPEPromptSpanScanCheckedDefault(U8 *bytes,
                                             U64 byte_len,
                                             U64 *io_cursor,
                                             U64 prompt_nbytes,
                                             U64 *out_span_starts,
                                             U64 *out_span_lengths,
                                             U8 *out_span_classes,
                                             U64 *out_span_count)
{
    U64 cursor;
    U64 derived_capacity;

    if (!bytes || !io_cursor || !out_span_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (byte_len > 0x7FFFFFFFFFFFFFFF ||
        prompt_nbytes > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (prompt_nbytes > byte_len - cursor)
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS;

    // Canonical default wrapper keeps explicit capacity derivation at call
    // boundary so metadata-absent callers always use deterministic span limits.
    derived_capacity = prompt_nbytes;

    return TokenizerBPEPromptSpanScanChecked(bytes,
                                             byte_len,
                                             io_cursor,
                                             prompt_nbytes,
                                             out_span_starts,
                                             out_span_lengths,
                                             out_span_classes,
                                             derived_capacity,
                                             out_span_count);
}

I32 TokenizerBPEPromptSpanScanCheckedDefaultValidateCursor(
    U8 *bytes,
    U64 byte_len,
    U64 *io_cursor,
    U64 prompt_nbytes,
    U64 *out_span_starts,
    U64 *out_span_lengths,
    U8 *out_span_classes,
    U64 *out_span_count)
{
    U64 cursor;
    U64 derived_capacity;

    if (!bytes || !io_cursor || !out_span_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (byte_len > 0x7FFFFFFFFFFFFFFF ||
        prompt_nbytes > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    cursor = *io_cursor;

    // Strict monotonic cursor contract: cursor must point within the source
    // byte buffer before any prompt-window arithmetic occurs.
    if (cursor > byte_len)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    // Strict window contract: requested prompt span must fit entirely inside
    // [cursor, byte_len], guarding underflow in (byte_len - cursor).
    if (prompt_nbytes > byte_len - cursor)
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS;

    // Canonical default-capacity derivation: an N-byte prompt can produce at
    // most N spans in the class-partition scanner.
    derived_capacity = prompt_nbytes;

    return TokenizerBPEPromptSpanScanChecked(bytes,
                                             byte_len,
                                             io_cursor,
                                             prompt_nbytes,
                                             out_span_starts,
                                             out_span_lengths,
                                             out_span_classes,
                                             derived_capacity,
                                             out_span_count);
}

I32 TokenizerBPEPromptSpanScanCheckedDefaultCapacityNoPartial(
    U8 *bytes,
    U64 byte_len,
    U64 *io_cursor,
    U64 prompt_nbytes,
    U64 *out_span_starts,
    U64 *out_span_lengths,
    U8 *out_span_classes,
    U64 *out_span_count)
{
    U64 cursor;
    U64 staged_cursor;
    U64 staged_count;
    U64 derived_capacity;
    U64 staged_alloc_capacity;
    U64 staged_starts_bytes;
    U64 staged_lengths_bytes;
    U64 staged_classes_bytes;
    U64 *staged_starts;
    U64 *staged_lengths;
    U8 *staged_classes;
    I32 err;
    U64 i;

    if (!bytes || !io_cursor || !out_span_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (byte_len > 0x7FFFFFFFFFFFFFFF ||
        prompt_nbytes > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (prompt_nbytes > byte_len - cursor)
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS;

    // Canonical default-capacity contract:
    // prompt byte span of N bytes can never split into more than N spans.
    derived_capacity = prompt_nbytes;

    if (derived_capacity > 0 &&
        (!out_span_starts || !out_span_lengths || !out_span_classes))
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Stage caller-visible cursor/span-count so checked-core failures do not
    // partially mutate visible state.
    staged_cursor = cursor;
    staged_count = *out_span_count;

    staged_alloc_capacity = derived_capacity;
    if (!staged_alloc_capacity)
        staged_alloc_capacity = 1;

    if (staged_alloc_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(U64)))
        return TOKENIZER_BPE_ERR_OVERFLOW;
    staged_starts_bytes = sizeof(U64) * staged_alloc_capacity;
    staged_lengths_bytes = sizeof(U64) * staged_alloc_capacity;

    if (staged_alloc_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(U8)))
        return TOKENIZER_BPE_ERR_OVERFLOW;
    staged_classes_bytes = sizeof(U8) * staged_alloc_capacity;

    if (!staged_starts_bytes || !staged_lengths_bytes || !staged_classes_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_starts = MAlloc(staged_starts_bytes);
    if (!staged_starts)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_lengths = MAlloc(staged_lengths_bytes);
    if (!staged_lengths) {
        Free(staged_starts);
        return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    staged_classes = MAlloc(staged_classes_bytes);
    if (!staged_classes) {
        Free(staged_lengths);
        Free(staged_starts);
        return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    err = TokenizerBPEPromptSpanScanChecked(bytes,
                                            byte_len,
                                            &staged_cursor,
                                            prompt_nbytes,
                                            staged_starts,
                                            staged_lengths,
                                            staged_classes,
                                            derived_capacity,
                                            &staged_count);
    if (err != TOKENIZER_BPE_OK) {
        Free(staged_classes);
        Free(staged_lengths);
        Free(staged_starts);
        return err;
    }

    if (staged_count > derived_capacity) {
        Free(staged_classes);
        Free(staged_lengths);
        Free(staged_starts);
        return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    for (i = 0; i < staged_count; i++) {
        out_span_starts[i] = staged_starts[i];
        out_span_lengths[i] = staged_lengths[i];
        out_span_classes[i] = staged_classes[i];
    }

    *out_span_count = staged_count;
    *io_cursor = staged_cursor;

    Free(staged_classes);
    Free(staged_lengths);
    Free(staged_starts);
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEPromptSpanScanCheckedNoPartial(U8 *bytes,
                                               U64 byte_len,
                                               U64 *io_cursor,
                                               U64 prompt_nbytes,
                                               U64 *out_span_starts,
                                               U64 *out_span_lengths,
                                               U8 *out_span_classes,
                                               U64 out_span_capacity,
                                               U64 *out_span_count)
{
    U64 cursor;
    U64 staged_cursor;
    U64 staged_count;
    U64 staged_alloc_capacity;
    U64 staged_starts_bytes;
    U64 staged_lengths_bytes;
    U64 staged_classes_bytes;
    U64 *staged_starts;
    U64 *staged_lengths;
    U8 *staged_classes;
    I32 err;
    U64 i;

    if (!bytes || !io_cursor || !out_span_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (byte_len > 0x7FFFFFFFFFFFFFFF ||
        prompt_nbytes > 0x7FFFFFFFFFFFFFFF ||
        out_span_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (out_span_capacity > 0 &&
        (!out_span_starts || !out_span_lengths || !out_span_classes))
        return TOKENIZER_BPE_ERR_NULL_PTR;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (prompt_nbytes > byte_len - cursor)
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS;

    // Stage caller-visible cursor/span-count so checked-core failures do not
    // partially mutate visible state.
    staged_cursor = cursor;
    staged_count = *out_span_count;

    staged_alloc_capacity = out_span_capacity;
    if (!staged_alloc_capacity)
        staged_alloc_capacity = 1;

    // Guard allocator byte-size multiplication before private staging writes.
    if (staged_alloc_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(U64)))
        return TOKENIZER_BPE_ERR_OVERFLOW;
    staged_starts_bytes = sizeof(U64) * staged_alloc_capacity;
    staged_lengths_bytes = sizeof(U64) * staged_alloc_capacity;

    if (staged_alloc_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(U8)))
        return TOKENIZER_BPE_ERR_OVERFLOW;
    staged_classes_bytes = sizeof(U8) * staged_alloc_capacity;

    if (!staged_starts_bytes || !staged_lengths_bytes || !staged_classes_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_starts = MAlloc(staged_starts_bytes);
    if (!staged_starts)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_lengths = MAlloc(staged_lengths_bytes);
    if (!staged_lengths) {
        Free(staged_starts);
        return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    staged_classes = MAlloc(staged_classes_bytes);
    if (!staged_classes) {
        Free(staged_lengths);
        Free(staged_starts);
        return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    err = TokenizerBPEPromptSpanScanChecked(bytes,
                                            byte_len,
                                            &staged_cursor,
                                            prompt_nbytes,
                                            staged_starts,
                                            staged_lengths,
                                            staged_classes,
                                            out_span_capacity,
                                            &staged_count);
    if (err != TOKENIZER_BPE_OK) {
        Free(staged_classes);
        Free(staged_lengths);
        Free(staged_starts);
        return err;
    }

    if (staged_count > out_span_capacity) {
        Free(staged_classes);
        Free(staged_lengths);
        Free(staged_starts);
        return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    // Commit staged span tables only after complete checked scan success.
    for (i = 0; i < staged_count; i++) {
        out_span_starts[i] = staged_starts[i];
        out_span_lengths[i] = staged_lengths[i];
        out_span_classes[i] = staged_classes[i];
    }

    *out_span_count = staged_count;
    *io_cursor = staged_cursor;

    Free(staged_classes);
    Free(staged_lengths);
    Free(staged_starts);
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEPromptSpanScanCheckedDefaultNoPartial(U8 *bytes,
                                                      U64 byte_len,
                                                      U64 *io_cursor,
                                                      U64 prompt_nbytes,
                                                      U64 *out_span_starts,
                                                      U64 *out_span_lengths,
                                                      U8 *out_span_classes,
                                                      U64 *out_span_count)
{
    U64 cursor;
    U64 staged_cursor;
    U64 staged_count;
    U64 derived_capacity;
    U64 staged_alloc_capacity;
    U64 staged_starts_bytes;
    U64 staged_lengths_bytes;
    U64 staged_classes_bytes;
    U64 *staged_starts;
    U64 *staged_lengths;
    U8 *staged_classes;
    I32 err;
    U64 i;

    if (!bytes || !io_cursor || !out_span_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (byte_len > 0x7FFFFFFFFFFFFFFF ||
        prompt_nbytes > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (prompt_nbytes > byte_len - cursor)
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS;

    // Canonical default-capacity contract keeps prompt span storage tied to
    // logical byte window size when callers do not provide capacity metadata.
    derived_capacity = prompt_nbytes;

    if (derived_capacity > 0 &&
        (!out_span_starts || !out_span_lengths || !out_span_classes))
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Stage caller-visible cursor/span-count so checked no-partial failures
    // cannot mutate caller-observed state before full success.
    staged_cursor = cursor;
    staged_count = *out_span_count;

    staged_alloc_capacity = derived_capacity;
    if (!staged_alloc_capacity)
        staged_alloc_capacity = 1;

    if (staged_alloc_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(U64)))
        return TOKENIZER_BPE_ERR_OVERFLOW;
    staged_starts_bytes = sizeof(U64) * staged_alloc_capacity;
    staged_lengths_bytes = sizeof(U64) * staged_alloc_capacity;

    if (staged_alloc_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(U8)))
        return TOKENIZER_BPE_ERR_OVERFLOW;
    staged_classes_bytes = sizeof(U8) * staged_alloc_capacity;

    if (!staged_starts_bytes || !staged_lengths_bytes || !staged_classes_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_starts = MAlloc(staged_starts_bytes);
    if (!staged_starts)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_lengths = MAlloc(staged_lengths_bytes);
    if (!staged_lengths) {
        Free(staged_starts);
        return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    staged_classes = MAlloc(staged_classes_bytes);
    if (!staged_classes) {
        Free(staged_lengths);
        Free(staged_starts);
        return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    err = TokenizerBPEPromptSpanScanCheckedNoPartial(bytes,
                                                     byte_len,
                                                     &staged_cursor,
                                                     prompt_nbytes,
                                                     staged_starts,
                                                     staged_lengths,
                                                     staged_classes,
                                                     derived_capacity,
                                                     &staged_count);
    if (err != TOKENIZER_BPE_OK) {
        Free(staged_classes);
        Free(staged_lengths);
        Free(staged_starts);
        return err;
    }

    if (staged_count > derived_capacity) {
        Free(staged_classes);
        Free(staged_lengths);
        Free(staged_starts);
        return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    // Commit staged span table outputs only after full default wrapper pass.
    for (i = 0; i < staged_count; i++) {
        out_span_starts[i] = staged_starts[i];
        out_span_lengths[i] = staged_lengths[i];
        out_span_classes[i] = staged_classes[i];
    }

    *out_span_count = staged_count;
    *io_cursor = staged_cursor;

    Free(staged_classes);
    Free(staged_lengths);
    Free(staged_starts);
    return TOKENIZER_BPE_OK;
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
    U64 staged_alloc_bytes;
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

    // Guard allocator byte-size multiplication before staging writes.
    if (staged_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(I32)))
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_alloc_bytes = sizeof(I32) * staged_capacity;
    if (!staged_alloc_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_tokens = MAlloc(staged_alloc_bytes);
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

I32 TokenizerBPEEncodePromptCheckedViaSpanTable(U8 *bytes,
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
    U64 span_capacity;
    U64 span_alloc_capacity;
    U64 span_starts_bytes;
    U64 span_lengths_bytes;
    U64 span_classes_bytes;
    U64 *span_starts;
    U64 *span_lengths;
    U8 *span_classes;
    U64 span_count;
    U64 span_index;
    U64 expected_span_start;
    U64 span_start;
    U64 span_nbytes;
    U64 span_end;
    U8 span_class;
    U64 token_alloc_capacity;
    U64 token_alloc_bytes;
    I32 *staged_tokens;
    U64 staged_token_count;
    U64 remaining_capacity;
    U64 span_cursor;
    U64 span_token_count;
    U64 i;
    U64 scan_cursor;
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

    span_capacity = prompt_nbytes;
    span_alloc_capacity = span_capacity;
    if (!span_alloc_capacity)
        span_alloc_capacity = 1;

    if (span_alloc_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(U64)))
        return TOKENIZER_BPE_ERR_OVERFLOW;
    span_starts_bytes = sizeof(U64) * span_alloc_capacity;
    span_lengths_bytes = sizeof(U64) * span_alloc_capacity;

    if (span_alloc_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(U8)))
        return TOKENIZER_BPE_ERR_OVERFLOW;
    span_classes_bytes = sizeof(U8) * span_alloc_capacity;

    if (!span_starts_bytes || !span_lengths_bytes || !span_classes_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    span_starts = MAlloc(span_starts_bytes);
    if (!span_starts)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    span_lengths = MAlloc(span_lengths_bytes);
    if (!span_lengths) {
        Free(span_starts);
        return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    span_classes = MAlloc(span_classes_bytes);
    if (!span_classes) {
        Free(span_lengths);
        Free(span_starts);
        return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    scan_cursor = cursor;
    span_count = 0;

    err = TokenizerBPEPromptSpanScanChecked(bytes,
                                            byte_len,
                                            &scan_cursor,
                                            prompt_nbytes,
                                            span_starts,
                                            span_lengths,
                                            span_classes,
                                            span_capacity,
                                            &span_count);
    if (err != TOKENIZER_BPE_OK) {
        Free(span_classes);
        Free(span_lengths);
        Free(span_starts);
        return err;
    }

    if (scan_cursor != prompt_end || span_count > span_capacity) {
        Free(span_classes);
        Free(span_lengths);
        Free(span_starts);
        return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    token_alloc_capacity = out_token_capacity;
    if (!token_alloc_capacity)
        token_alloc_capacity = 1;

    if (token_alloc_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(I32))) {
        Free(span_classes);
        Free(span_lengths);
        Free(span_starts);
        return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    token_alloc_bytes = sizeof(I32) * token_alloc_capacity;
    if (!token_alloc_bytes) {
        Free(span_classes);
        Free(span_lengths);
        Free(span_starts);
        return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    staged_tokens = MAlloc(token_alloc_bytes);
    if (!staged_tokens) {
        Free(span_classes);
        Free(span_lengths);
        Free(span_starts);
        return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    staged_token_count = 0;
    expected_span_start = cursor;

    for (span_index = 0; span_index < span_count; span_index++) {
        span_start = span_starts[span_index];
        span_nbytes = span_lengths[span_index];
        span_class = span_classes[span_index];

        if (span_class < TOKENIZER_BPE_ASCII_CLASS_WORD ||
            span_class > TOKENIZER_BPE_ASCII_CLASS_NON_ASCII) {
            Free(staged_tokens);
            Free(span_classes);
            Free(span_lengths);
            Free(span_starts);
            return TOKENIZER_BPE_ERR_BAD_PARAM;
        }

        if (!span_nbytes || span_start != expected_span_start || span_start >= prompt_end) {
            Free(staged_tokens);
            Free(span_classes);
            Free(span_lengths);
            Free(span_starts);
            return TOKENIZER_BPE_ERR_BAD_PARAM;
        }

        span_end = span_start + span_nbytes;
        if (span_end < span_start || span_end > prompt_end) {
            Free(staged_tokens);
            Free(span_classes);
            Free(span_lengths);
            Free(span_starts);
            return TOKENIZER_BPE_ERR_OVERFLOW;
        }

        if (staged_token_count > out_token_capacity) {
            Free(staged_tokens);
            Free(span_classes);
            Free(span_lengths);
            Free(span_starts);
            return TOKENIZER_BPE_ERR_BAD_PARAM;
        }

        remaining_capacity = out_token_capacity - staged_token_count;
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
                                            staged_tokens + staged_token_count,
                                            remaining_capacity,
                                            &span_token_count);
        if (err != TOKENIZER_BPE_OK) {
            Free(staged_tokens);
            Free(span_classes);
            Free(span_lengths);
            Free(span_starts);
            return err;
        }

        if (span_cursor != span_end || span_token_count > remaining_capacity) {
            Free(staged_tokens);
            Free(span_classes);
            Free(span_lengths);
            Free(span_starts);
            return TOKENIZER_BPE_ERR_BAD_PARAM;
        }

        staged_token_count += span_token_count;
        if (staged_token_count < span_token_count || staged_token_count > out_token_capacity) {
            Free(staged_tokens);
            Free(span_classes);
            Free(span_lengths);
            Free(span_starts);
            return TOKENIZER_BPE_ERR_OVERFLOW;
        }

        expected_span_start = span_end;
    }

    if (expected_span_start != prompt_end) {
        Free(staged_tokens);
        Free(span_classes);
        Free(span_lengths);
        Free(span_starts);
        return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    for (i = 0; i < staged_token_count; i++)
        out_token_ids[i] = staged_tokens[i];

    *out_token_count = staged_token_count;
    *io_cursor = prompt_end;

    Free(staged_tokens);
    Free(span_classes);
    Free(span_lengths);
    Free(span_starts);
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEEncodePromptCheckedViaSpanTableNoPartial(
    U8 *bytes,
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
    U64 cursor;
    U64 staged_cursor;
    U64 staged_token_count;
    U64 derived_out_token_capacity;
    U64 staged_token_capacity;
    U64 staged_alloc_bytes;
    I32 *staged_tokens;
    I32 err;
    U64 i;

    if (!bytes || !io_cursor || !out_token_ids || !out_token_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (byte_len > 0x7FFFFFFFFFFFFFFF ||
        prompt_nbytes > 0x7FFFFFFFFFFFFFFF ||
        rank_table_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (rank_table_count > rank_table_capacity)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (prompt_nbytes > byte_len - cursor)
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS;

    // Canonical default token-capacity rule for prompt encodes: one emitted
    // token per input byte before merges monotonically shrink the sequence.
    derived_out_token_capacity = prompt_nbytes;

    // Stage caller-visible cursor/count/output so any failure path from the
    // span-table core keeps caller buffers unchanged.
    staged_cursor = cursor;
    staged_token_count = *out_token_count;

    staged_token_capacity = derived_out_token_capacity;
    if (!staged_token_capacity)
        staged_token_capacity = 1;

    if (staged_token_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(I32)))
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_alloc_bytes = sizeof(I32) * staged_token_capacity;
    if (!staged_alloc_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_tokens = MAlloc(staged_alloc_bytes);
    if (!staged_tokens)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    err = TokenizerBPEEncodePromptCheckedViaSpanTable(
        bytes,
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
        derived_out_token_capacity,
        &staged_token_count);
    if (err != TOKENIZER_BPE_OK) {
        Free(staged_tokens);
        return err;
    }

    if (staged_token_count > derived_out_token_capacity) {
        Free(staged_tokens);
        return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    for (i = 0; i < staged_token_count; i++)
        out_token_ids[i] = staged_tokens[i];

    *out_token_count = staged_token_count;
    *io_cursor = staged_cursor;
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
    U64 staged_alloc_bytes;
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

    // No-partial wrapper stages into private memory first; validate the
    // backing allocation size exactly like the core prompt encoder.
    if (staged_token_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(I32)))
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_alloc_bytes = sizeof(I32) * staged_token_capacity;
    if (!staged_alloc_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_tokens = MAlloc(staged_alloc_bytes);
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
    U64 cursor;

    if (!bytes || !io_cursor || !out_token_ids || !out_token_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (byte_len > 0x7FFFFFFFFFFFFFFF ||
        rank_table_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (rank_table_count > rank_table_capacity)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (prompt_nbytes > byte_len - cursor)
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS;

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

I32 TokenizerBPEEncodePromptCheckedDefaultCapacityNoPartial(
    U8 *bytes,
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
    U64 cursor;
    U64 staged_cursor;
    U64 staged_token_count;
    U64 derived_capacity;
    U64 staged_token_capacity;
    U64 staged_alloc_bytes;
    I32 *staged_tokens;
    I32 err;
    U64 i;

    if (!bytes || !io_cursor || !out_token_ids || !out_token_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (byte_len > 0x7FFFFFFFFFFFFFFF ||
        prompt_nbytes > 0x7FFFFFFFFFFFFFFF ||
        rank_table_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (rank_table_count > rank_table_capacity)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (prompt_nbytes > byte_len - cursor)
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS;

    // Canonical default-capacity rule: max one emitted pre-merge token
    // per prompt byte, then merges monotonically reduce token count.
    derived_capacity = prompt_nbytes;

    // Stage caller-visible fields so failed encodes never partially mutate
    // cursor, token count, or token IDs.
    staged_cursor = cursor;
    staged_token_count = *out_token_count;

    staged_token_capacity = derived_capacity;
    if (!staged_token_capacity)
        staged_token_capacity = 1;

    // Guard private staging allocation against integer overflow.
    if (staged_token_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(I32)))
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_alloc_bytes = sizeof(I32) * staged_token_capacity;
    if (!staged_alloc_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_tokens = MAlloc(staged_alloc_bytes);
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
                                          derived_capacity,
                                          &staged_token_count);
    if (err != TOKENIZER_BPE_OK) {
        Free(staged_tokens);
        return err;
    }

    if (staged_token_count > derived_capacity) {
        Free(staged_tokens);
        return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    for (i = 0; i < staged_token_count; i++)
        out_token_ids[i] = staged_tokens[i];

    *out_token_count = staged_token_count;
    *io_cursor = staged_cursor;
    Free(staged_tokens);
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEEncodePromptCheckedDefaultCapacityNoPartialValidateRanks(
    U8 *bytes,
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
    U64 i;
    U64 run_start;
    U64 run_scan;
    I32 key_left;
    I32 key_right;
    U32 min_rank;
    Bool min_rank_set;
    Bool min_merged_set;
    I32 min_rank_merged_token;

    if (!bytes || !io_cursor || !out_token_ids || !out_token_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (byte_len > 0x7FFFFFFFFFFFFFFF ||
        prompt_nbytes > 0x7FFFFFFFFFFFFFFF ||
        rank_table_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (rank_table_count > rank_table_capacity)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (rank_table_count > 0 &&
        (!rank_left_tokens || !rank_right_tokens || !rank_values || !rank_merged_tokens))
        return TOKENIZER_BPE_ERR_NULL_PTR;

    // Guard 1: enforce lexicographically sorted (left,right) rank-table keys
    // so binary-search style pair lookup remains deterministic.
    for (i = 1; i < rank_table_count; i++) {
        if (rank_left_tokens[i - 1] > rank_left_tokens[i])
            return TOKENIZER_BPE_ERR_BAD_PARAM;

        if (rank_left_tokens[i - 1] == rank_left_tokens[i] &&
            rank_right_tokens[i - 1] > rank_right_tokens[i])
            return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    // Guard 2: within each duplicate-key run, all rows tying for minimum rank
    // must choose the same merged token to keep merge picks deterministic.
    run_start = 0;
    while (run_start < rank_table_count) {
        key_left = rank_left_tokens[run_start];
        key_right = rank_right_tokens[run_start];
        min_rank_set = FALSE;
        min_merged_set = FALSE;
        run_scan = run_start;

        while (run_scan < rank_table_count) {
            if (rank_left_tokens[run_scan] != key_left)
                break;
            if (rank_right_tokens[run_scan] != key_right)
                break;

            if (!min_rank_set || rank_values[run_scan] < min_rank) {
                min_rank = rank_values[run_scan];
                min_rank_merged_token = rank_merged_tokens[run_scan];
                min_rank_set = TRUE;
                min_merged_set = TRUE;
            } else if (rank_values[run_scan] == min_rank) {
                if (!min_merged_set) {
                    min_rank_merged_token = rank_merged_tokens[run_scan];
                    min_merged_set = TRUE;
                } else if (rank_merged_tokens[run_scan] != min_rank_merged_token) {
                    return TOKENIZER_BPE_ERR_BAD_PARAM;
                }
            }

            run_scan++;
            if (run_scan == 0)
                return TOKENIZER_BPE_ERR_OVERFLOW;
        }

        run_start = run_scan;
    }

    // Delegate all staged no-partial prompt-encode behavior to the canonical
    // default-capacity wrapper once rank validation passes.
    return TokenizerBPEEncodePromptCheckedDefaultCapacityNoPartial(
        bytes,
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
        out_token_count);
}

I32 TokenizerBPEEncodePromptCheckedDefaultNoPartial(U8 *bytes,
                                                    U64 byte_len,
                                                    U64 *io_cursor,
                                                    U64 prompt_nbytes,
                                                    I32 *rank_left_tokens,
                                                    I32 *rank_right_tokens,
                                                    U32 *rank_values,
                                                    I32 *rank_merged_tokens,
                                                    U64 rank_table_count,
                                                    I32 *out_token_ids,
                                                    U64 *out_token_count)
{
    U64 cursor;
    U64 staged_cursor;
    U64 staged_token_count;
    U64 derived_rank_table_capacity;
    U64 derived_out_token_capacity;
    U64 staged_token_capacity;
    U64 staged_alloc_bytes;
    I32 *staged_tokens;
    I32 err;
    U64 i;

    if (!bytes || !io_cursor || !out_token_ids || !out_token_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (byte_len > 0x7FFFFFFFFFFFFFFF ||
        prompt_nbytes > 0x7FFFFFFFFFFFFFFF ||
        rank_table_count > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    cursor = *io_cursor;
    if (cursor > byte_len)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (prompt_nbytes > byte_len - cursor)
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS;

    // Canonical checked-default contract: the active rank table and output
    // token span are tightly packed to logical counts.
    derived_rank_table_capacity = rank_table_count;
    derived_out_token_capacity = prompt_nbytes;

    // Stage caller-visible state so failures from default-derivation guards
    // or the downstream checked encoder keep cursor/count/token IDs unchanged.
    staged_cursor = cursor;
    staged_token_count = *out_token_count;

    staged_token_capacity = derived_out_token_capacity;
    if (!staged_token_capacity)
        staged_token_capacity = 1;

    // Guard the private staging allocation used for all-or-nothing commit.
    if (staged_token_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(I32)))
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_alloc_bytes = sizeof(I32) * staged_token_capacity;
    if (!staged_alloc_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_tokens = MAlloc(staged_alloc_bytes);
    if (!staged_tokens)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    err = TokenizerBPEEncodePromptCheckedNoPartial(bytes,
                                                   byte_len,
                                                   &staged_cursor,
                                                   prompt_nbytes,
                                                   rank_left_tokens,
                                                   rank_right_tokens,
                                                   rank_values,
                                                   rank_merged_tokens,
                                                   rank_table_count,
                                                   derived_rank_table_capacity,
                                                   staged_tokens,
                                                   derived_out_token_capacity,
                                                   &staged_token_count);
    if (err != TOKENIZER_BPE_OK) {
        Free(staged_tokens);
        return err;
    }

    if (staged_token_count > derived_out_token_capacity) {
        Free(staged_tokens);
        return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    for (i = 0; i < staged_token_count; i++)
        out_token_ids[i] = staged_tokens[i];

    *out_token_count = staged_token_count;
    *io_cursor = staged_cursor;
    Free(staged_tokens);
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEDecodeTokenSpanChecked(I32 *token_ids,
                                       U64 token_count,
                                       U64 *io_token_cursor,
                                       U64 span_token_count,
                                       U8 *vocab_piece_bytes,
                                       U64 vocab_piece_bytes_len,
                                       U64 *vocab_piece_offsets,
                                       U64 *vocab_piece_lens,
                                       U64 vocab_piece_count,
                                       U64 vocab_piece_capacity,
                                       U8 *out_bytes,
                                       U64 out_byte_capacity,
                                       U64 *out_byte_count)
{
    U64 token_cursor;
    U64 decode_end;
    U64 scan;
    U64 token_id_u64;
    U64 piece_offset;
    U64 piece_len;
    U64 staged_out_count;
    U64 write_cursor;
    U64 copy_i;

    if (!token_ids || !io_token_cursor || !vocab_piece_bytes ||
        !vocab_piece_offsets || !vocab_piece_lens || !out_bytes ||
        !out_byte_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (token_count > 0x7FFFFFFFFFFFFFFF ||
        span_token_count > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_bytes_len > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_count > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_capacity > 0x7FFFFFFFFFFFFFFF ||
        out_byte_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    token_cursor = *io_token_cursor;
    if (token_cursor > token_count)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (vocab_piece_capacity < vocab_piece_count)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (span_token_count > token_count - token_cursor)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    decode_end = token_cursor + span_token_count;
    if (decode_end < token_cursor || decode_end > token_count)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    // Pass 1: validate the entire token span and accumulate exact output
    // byte count without touching caller outputs (strict no-partial semantics).
    staged_out_count = 0;
    scan = token_cursor;
    while (scan < decode_end) {
        if (token_ids[scan] < 0)
            return TOKENIZER_BPE_ERR_BAD_PARAM;

        token_id_u64 = token_ids[scan];
        if (token_id_u64 >= vocab_piece_count)
            return TOKENIZER_BPE_ERR_BAD_PARAM;

        piece_offset = vocab_piece_offsets[token_id_u64];
        piece_len = vocab_piece_lens[token_id_u64];

        if (piece_offset > vocab_piece_bytes_len)
            return TOKENIZER_BPE_ERR_BAD_PARAM;

        if (piece_len > vocab_piece_bytes_len - piece_offset)
            return TOKENIZER_BPE_ERR_BAD_PARAM;

        if (piece_len > out_byte_capacity - staged_out_count)
            return TOKENIZER_BPE_ERR_BAD_PARAM;

        staged_out_count += piece_len;
        if (staged_out_count > out_byte_capacity)
            return TOKENIZER_BPE_ERR_OVERFLOW;

        scan++;
        if (scan == 0)
            return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    // Pass 2: all bounds validated above, so decode is deterministic and safe.
    write_cursor = 0;
    scan = token_cursor;
    while (scan < decode_end) {
        token_id_u64 = token_ids[scan];
        piece_offset = vocab_piece_offsets[token_id_u64];
        piece_len = vocab_piece_lens[token_id_u64];

        copy_i = 0;
        while (copy_i < piece_len) {
            out_bytes[write_cursor] = vocab_piece_bytes[piece_offset + copy_i];

            write_cursor++;
            if (write_cursor > staged_out_count)
                return TOKENIZER_BPE_ERR_OVERFLOW;

            copy_i++;
            if (copy_i == 0)
                return TOKENIZER_BPE_ERR_OVERFLOW;
        }

        scan++;
        if (scan == 0)
            return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    if (write_cursor != staged_out_count)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    // Commit cursor/count only after full decode succeeds.
    *out_byte_count = staged_out_count;
    *io_token_cursor = decode_end;
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEDecodeTokenSpanCheckedValidateCursorNoPartial(
    I32 *token_ids,
    U64 token_count,
    U64 *io_token_cursor,
    U64 span_token_count,
    U8 *vocab_piece_bytes,
    U64 vocab_piece_bytes_len,
    U64 *vocab_piece_offsets,
    U64 *vocab_piece_lens,
    U64 vocab_piece_count,
    U64 vocab_piece_capacity,
    U8 *out_bytes,
    U64 out_byte_capacity,
    U64 *out_byte_count)
{
    U64 cursor;
    U64 decode_count;
    U64 staged_cursor;
    U64 staged_out_count;
    U64 staged_alloc_capacity;
    U64 staged_alloc_bytes;
    U8 *staged_out_bytes;
    I32 err;
    U64 i;

    if (!token_ids || !io_token_cursor || !vocab_piece_bytes ||
        !vocab_piece_offsets || !vocab_piece_lens || !out_bytes ||
        !out_byte_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (token_count > 0x7FFFFFFFFFFFFFFF ||
        span_token_count > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_bytes_len > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_count > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_capacity > 0x7FFFFFFFFFFFFFFF ||
        out_byte_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    cursor = *io_token_cursor;

    // Strict window contract: decode starts from a valid cursor and consumes
    // at most the remaining token window.
    if (cursor > token_count)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    decode_count = span_token_count;
    if (decode_count > token_count - cursor)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    // Stage caller-visible outputs so wrapper preserves all-or-nothing
    // semantics even when checked decode fails.
    staged_cursor = cursor;
    staged_out_count = *out_byte_count;

    staged_alloc_capacity = out_byte_capacity;
    if (!staged_alloc_capacity)
        staged_alloc_capacity = 1;

    if (staged_alloc_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(U8)))
        return TOKENIZER_BPE_ERR_OVERFLOW;
    staged_alloc_bytes = sizeof(U8) * staged_alloc_capacity;
    if (!staged_alloc_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_out_bytes = MAlloc(staged_alloc_bytes);
    if (!staged_out_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    err = TokenizerBPEDecodeTokenSpanChecked(token_ids,
                                             token_count,
                                             &staged_cursor,
                                             decode_count,
                                             vocab_piece_bytes,
                                             vocab_piece_bytes_len,
                                             vocab_piece_offsets,
                                             vocab_piece_lens,
                                             vocab_piece_count,
                                             vocab_piece_capacity,
                                             staged_out_bytes,
                                             out_byte_capacity,
                                             &staged_out_count);
    if (err != TOKENIZER_BPE_OK) {
        Free(staged_out_bytes);
        return err;
    }

    if (staged_out_count > out_byte_capacity) {
        Free(staged_out_bytes);
        return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    for (i = 0; i < staged_out_count; i++)
        out_bytes[i] = staged_out_bytes[i];

    *out_byte_count = staged_out_count;
    *io_token_cursor = staged_cursor;
    Free(staged_out_bytes);
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEDecodeTokenSpanCheckedDefaultCapacity(
    I32 *token_ids,
    U64 token_count,
    U64 *io_token_cursor,
    U64 span_token_count,
    U8 *vocab_piece_bytes,
    U64 vocab_piece_bytes_len,
    U64 *vocab_piece_offsets,
    U64 *vocab_piece_lens,
    U64 vocab_piece_count,
    U8 *out_bytes,
    U64 out_byte_capacity,
    U64 *out_byte_count)
{
    if (!token_ids || !io_token_cursor || !vocab_piece_bytes ||
        !vocab_piece_offsets || !vocab_piece_lens || !out_bytes ||
        !out_byte_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (token_count > 0x7FFFFFFFFFFFFFFF ||
        span_token_count > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_bytes_len > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_count > 0x7FFFFFFFFFFFFFFF ||
        out_byte_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    // Canonical default-capacity contract for decode vocab tables:
    // logical vocabulary capacity equals the active piece count.
    return TokenizerBPEDecodeTokenSpanChecked(token_ids,
                                              token_count,
                                              io_token_cursor,
                                              span_token_count,
                                              vocab_piece_bytes,
                                              vocab_piece_bytes_len,
                                              vocab_piece_offsets,
                                              vocab_piece_lens,
                                              vocab_piece_count,
                                              vocab_piece_count,
                                              out_bytes,
                                              out_byte_capacity,
                                              out_byte_count);
}

I32 TokenizerBPEDecodeTokenSpanCheckedDefaultNoPartial(
    I32 *token_ids,
    U64 token_count,
    U64 *io_token_cursor,
    U64 span_token_count,
    U8 *vocab_piece_bytes,
    U64 vocab_piece_bytes_len,
    U64 *vocab_piece_offsets,
    U64 *vocab_piece_lens,
    U64 vocab_piece_count,
    U64 vocab_piece_capacity,
    U64 max_piece_bytes,
    U8 *out_bytes,
    U64 *out_byte_count)
{
    U64 derived_out_byte_capacity;
    U64 staged_cursor;
    U64 staged_out_count;
    U64 staged_alloc_capacity;
    U64 staged_alloc_bytes;
    U8 *staged_out_bytes;
    I32 err;
    U64 i;

    if (!token_ids || !io_token_cursor || !vocab_piece_bytes ||
        !vocab_piece_offsets || !vocab_piece_lens || !out_bytes ||
        !out_byte_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (token_count > 0x7FFFFFFFFFFFFFFF ||
        span_token_count > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_bytes_len > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_count > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_capacity > 0x7FFFFFFFFFFFFFFF ||
        max_piece_bytes > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (vocab_piece_count > vocab_piece_capacity)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    // Canonical default-capacity rule: any one decoded token emits at most
    // `max_piece_bytes`, so an input token window of `token_count` fits in
    // `token_count * max_piece_bytes` bytes when overflow-guarded.
    if (token_count && max_piece_bytes > 0x7FFFFFFFFFFFFFFF / token_count)
        return TOKENIZER_BPE_ERR_OVERFLOW;
    derived_out_byte_capacity = token_count * max_piece_bytes;

    // Stage caller-visible cursor/count and decoded bytes so failures from
    // overflow guards or checked decode never partially mutate outputs.
    staged_cursor = *io_token_cursor;
    staged_out_count = *out_byte_count;

    staged_alloc_capacity = derived_out_byte_capacity;
    if (!staged_alloc_capacity)
        staged_alloc_capacity = 1;

    if (staged_alloc_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(U8)))
        return TOKENIZER_BPE_ERR_OVERFLOW;
    staged_alloc_bytes = sizeof(U8) * staged_alloc_capacity;
    if (!staged_alloc_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_out_bytes = MAlloc(staged_alloc_bytes);
    if (!staged_out_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    err = TokenizerBPEDecodeTokenSpanChecked(token_ids,
                                              token_count,
                                              &staged_cursor,
                                              span_token_count,
                                              vocab_piece_bytes,
                                              vocab_piece_bytes_len,
                                              vocab_piece_offsets,
                                              vocab_piece_lens,
                                              vocab_piece_count,
                                              vocab_piece_capacity,
                                              staged_out_bytes,
                                              derived_out_byte_capacity,
                                              &staged_out_count);
    if (err != TOKENIZER_BPE_OK) {
        Free(staged_out_bytes);
        return err;
    }

    if (staged_out_count > derived_out_byte_capacity) {
        Free(staged_out_bytes);
        return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    for (i = 0; i < staged_out_count; i++)
        out_bytes[i] = staged_out_bytes[i];

    *out_byte_count = staged_out_count;
    *io_token_cursor = staged_cursor;
    Free(staged_out_bytes);
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEDecodePromptCheckedDefault(
    I32 *token_ids,
    U64 token_count,
    U64 *io_token_cursor,
    U8 *vocab_piece_bytes,
    U64 vocab_piece_bytes_len,
    U64 *vocab_piece_offsets,
    U64 *vocab_piece_lens,
    U64 vocab_piece_count,
    U64 vocab_piece_capacity,
    U64 max_piece_bytes,
    U8 *out_bytes,
    U64 *out_byte_count)
{
    U64 cursor;
    U64 span_token_count;
    U64 derived_out_byte_capacity;

    if (!token_ids || !io_token_cursor || !vocab_piece_bytes ||
        !vocab_piece_offsets || !vocab_piece_lens || !out_bytes ||
        !out_byte_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (token_count > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_bytes_len > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_count > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_capacity > 0x7FFFFFFFFFFFFFFF ||
        max_piece_bytes > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (vocab_piece_count > vocab_piece_capacity)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    cursor = *io_token_cursor;
    if (cursor > token_count)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    // Full prompt decode consumes the active token window [cursor, token_count).
    span_token_count = token_count - cursor;

    // Canonical default-capacity contract: each token emits at most
    // `max_piece_bytes`, so reserve `token_count * max_piece_bytes` bytes.
    if (token_count && max_piece_bytes > 0x7FFFFFFFFFFFFFFF / token_count)
        return TOKENIZER_BPE_ERR_OVERFLOW;
    derived_out_byte_capacity = token_count * max_piece_bytes;

    return TokenizerBPEDecodeTokenSpanChecked(token_ids,
                                              token_count,
                                              io_token_cursor,
                                              span_token_count,
                                              vocab_piece_bytes,
                                              vocab_piece_bytes_len,
                                              vocab_piece_offsets,
                                              vocab_piece_lens,
                                              vocab_piece_count,
                                              vocab_piece_capacity,
                                              out_bytes,
                                              derived_out_byte_capacity,
                                              out_byte_count);
}

I32 TokenizerBPEDecodePromptCheckedDefaultNoPartial(
    I32 *token_ids,
    U64 token_count,
    U64 *io_token_cursor,
    U8 *vocab_piece_bytes,
    U64 vocab_piece_bytes_len,
    U64 *vocab_piece_offsets,
    U64 *vocab_piece_lens,
    U64 vocab_piece_count,
    U64 vocab_piece_capacity,
    U64 max_piece_bytes,
    U8 *out_bytes,
    U64 *out_byte_count)
{
    U64 cursor;
    U64 span_token_count;
    U64 derived_out_byte_capacity;
    U64 staged_cursor;
    U64 staged_out_count;
    U64 staged_alloc_capacity;
    U64 staged_alloc_bytes;
    U8 *staged_out_bytes;
    I32 err;
    U64 i;

    if (!token_ids || !io_token_cursor || !vocab_piece_bytes ||
        !vocab_piece_offsets || !vocab_piece_lens || !out_bytes ||
        !out_byte_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (token_count > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_bytes_len > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_count > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_capacity > 0x7FFFFFFFFFFFFFFF ||
        max_piece_bytes > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (vocab_piece_count > vocab_piece_capacity)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    cursor = *io_token_cursor;
    if (cursor > token_count)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    // Full-prompt decode means consuming every token from cursor to end.
    span_token_count = token_count - cursor;

    // Canonical default-capacity rule for prompt decode mirrors span decode:
    // reserve `token_count * max_piece_bytes` bytes (overflow-guarded).
    if (token_count && max_piece_bytes > 0x7FFFFFFFFFFFFFFF / token_count)
        return TOKENIZER_BPE_ERR_OVERFLOW;
    derived_out_byte_capacity = token_count * max_piece_bytes;

    // Stage outputs for strict all-or-nothing commit semantics.
    staged_cursor = cursor;
    staged_out_count = *out_byte_count;

    staged_alloc_capacity = derived_out_byte_capacity;
    if (!staged_alloc_capacity)
        staged_alloc_capacity = 1;

    if (staged_alloc_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(U8)))
        return TOKENIZER_BPE_ERR_OVERFLOW;
    staged_alloc_bytes = sizeof(U8) * staged_alloc_capacity;
    if (!staged_alloc_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_out_bytes = MAlloc(staged_alloc_bytes);
    if (!staged_out_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    err = TokenizerBPEDecodeTokenSpanChecked(token_ids,
                                              token_count,
                                              &staged_cursor,
                                              span_token_count,
                                              vocab_piece_bytes,
                                              vocab_piece_bytes_len,
                                              vocab_piece_offsets,
                                              vocab_piece_lens,
                                              vocab_piece_count,
                                              vocab_piece_capacity,
                                              staged_out_bytes,
                                              derived_out_byte_capacity,
                                              &staged_out_count);
    if (err != TOKENIZER_BPE_OK) {
        Free(staged_out_bytes);
        return err;
    }

    if (staged_cursor != token_count || staged_out_count > derived_out_byte_capacity) {
        Free(staged_out_bytes);
        return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    for (i = 0; i < staged_out_count; i++)
        out_bytes[i] = staged_out_bytes[i];

    *out_byte_count = staged_out_count;
    *io_token_cursor = staged_cursor;
    Free(staged_out_bytes);
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEDecodePromptCheckedDefaultNoPartialValidateCursor(
    I32 *token_ids,
    U64 token_count,
    U64 *io_token_cursor,
    U8 *vocab_piece_bytes,
    U64 vocab_piece_bytes_len,
    U64 *vocab_piece_offsets,
    U64 *vocab_piece_lens,
    U64 vocab_piece_count,
    U64 vocab_piece_capacity,
    U64 max_piece_bytes,
    U8 *out_bytes,
    U64 *out_byte_count)
{
    U64 cursor;
    U64 decode_count;
    U64 derived_out_byte_capacity;
    U64 staged_cursor;
    U64 staged_out_count;
    U64 staged_alloc_capacity;
    U64 staged_alloc_bytes;
    U8 *staged_out_bytes;
    I32 err;
    U64 i;

    if (!token_ids || !io_token_cursor || !vocab_piece_bytes ||
        !vocab_piece_offsets || !vocab_piece_lens || !out_bytes ||
        !out_byte_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (token_count > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_bytes_len > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_count > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_capacity > 0x7FFFFFFFFFFFFFFF ||
        max_piece_bytes > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (vocab_piece_count > vocab_piece_capacity)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    cursor = *io_token_cursor;

    // Strict cursor contract: caller cursor must be inside
    // [0, token_count] before deriving any decode window sizes.
    if (cursor > token_count)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    // Prompt decode always consumes from cursor to token_count.
    decode_count = token_count - cursor;

    // Keep explicit window guard here so wrapper contracts remain visible:
    // decode_count must never exceed remaining token window.
    if (decode_count > token_count - cursor)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    // Canonical default-capacity rule: reserve token_count * max_piece_bytes.
    if (token_count && max_piece_bytes > 0x7FFFFFFFFFFFFFFF / token_count)
        return TOKENIZER_BPE_ERR_OVERFLOW;
    derived_out_byte_capacity = token_count * max_piece_bytes;

    // Stage caller-visible outputs for strict all-or-nothing commit semantics.
    staged_cursor = cursor;
    staged_out_count = *out_byte_count;

    staged_alloc_capacity = derived_out_byte_capacity;
    if (!staged_alloc_capacity)
        staged_alloc_capacity = 1;

    if (staged_alloc_capacity > (0x7FFFFFFFFFFFFFFF / sizeof(U8)))
        return TOKENIZER_BPE_ERR_OVERFLOW;
    staged_alloc_bytes = sizeof(U8) * staged_alloc_capacity;
    if (!staged_alloc_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    staged_out_bytes = MAlloc(staged_alloc_bytes);
    if (!staged_out_bytes)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    err = TokenizerBPEDecodeTokenSpanChecked(token_ids,
                                             token_count,
                                             &staged_cursor,
                                             decode_count,
                                             vocab_piece_bytes,
                                             vocab_piece_bytes_len,
                                             vocab_piece_offsets,
                                             vocab_piece_lens,
                                             vocab_piece_count,
                                             vocab_piece_capacity,
                                             staged_out_bytes,
                                             derived_out_byte_capacity,
                                             &staged_out_count);
    if (err != TOKENIZER_BPE_OK) {
        Free(staged_out_bytes);
        return err;
    }

    if (staged_cursor != token_count || staged_out_count > derived_out_byte_capacity) {
        Free(staged_out_bytes);
        return TOKENIZER_BPE_ERR_BAD_PARAM;
    }

    for (i = 0; i < staged_out_count; i++)
        out_bytes[i] = staged_out_bytes[i];

    *out_byte_count = staged_out_count;
    *io_token_cursor = staged_cursor;
    Free(staged_out_bytes);
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEDecodeSingleTokenChecked(I32 token_id,
                                         U8 *vocab_piece_bytes,
                                         U64 vocab_piece_bytes_len,
                                         U64 *vocab_piece_offsets,
                                         U64 *vocab_piece_lens,
                                         U64 vocab_piece_count,
                                         U64 vocab_piece_capacity,
                                         U8 *out_bytes,
                                         U64 out_byte_capacity,
                                         U64 *io_out_cursor,
                                         U64 *out_byte_count)
{
    U64 token_id_u64;
    U64 piece_offset;
    U64 piece_len;
    U64 out_cursor;
    U64 staged_out_count;
    U64 write_cursor;
    U64 i;

    if (!vocab_piece_bytes || !vocab_piece_offsets || !vocab_piece_lens ||
        !out_bytes || !io_out_cursor || !out_byte_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (vocab_piece_bytes_len > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_count > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_capacity > 0x7FFFFFFFFFFFFFFF ||
        out_byte_capacity > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    if (vocab_piece_capacity < vocab_piece_count)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    out_cursor = *io_out_cursor;
    if (out_cursor > out_byte_capacity)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    if (token_id < 0)
        return TOKENIZER_BPE_ERR_BAD_PARAM;
    token_id_u64 = token_id;
    if (token_id_u64 >= vocab_piece_count)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    piece_offset = vocab_piece_offsets[token_id_u64];
    piece_len = vocab_piece_lens[token_id_u64];

    if (piece_offset > vocab_piece_bytes_len)
        return TOKENIZER_BPE_ERR_BAD_PARAM;
    if (piece_len > vocab_piece_bytes_len - piece_offset)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    // Single-token checked decode still obeys caller output cursor/capacity.
    if (piece_len > out_byte_capacity - out_cursor)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    staged_out_count = piece_len;
    write_cursor = out_cursor;

    // All bounds are validated above, so this loop is deterministic and safe.
    for (i = 0; i < piece_len; i++) {
        out_bytes[write_cursor] = vocab_piece_bytes[piece_offset + i];

        write_cursor++;
        if (write_cursor <= out_cursor)
            return TOKENIZER_BPE_ERR_OVERFLOW;
    }

    if (write_cursor != out_cursor + staged_out_count)
        return TOKENIZER_BPE_ERR_BAD_PARAM;

    // No-partial commit: publish count/cursor only after full copy succeeds.
    *out_byte_count = staged_out_count;
    *io_out_cursor = write_cursor;
    return TOKENIZER_BPE_OK;
}

I32 TokenizerBPEDecodeSingleTokenCheckedDefault(
    I32 token_id,
    U8 *vocab_piece_bytes,
    U64 vocab_piece_bytes_len,
    U64 *vocab_piece_offsets,
    U64 *vocab_piece_lens,
    U64 vocab_piece_count,
    U64 max_piece_bytes,
    U8 *out_bytes,
    U64 *io_out_cursor,
    U64 *out_byte_count)
{
    U64 out_cursor;
    U64 derived_out_byte_capacity;

    if (!vocab_piece_bytes || !vocab_piece_offsets || !vocab_piece_lens ||
        !out_bytes || !io_out_cursor || !out_byte_count)
        return TOKENIZER_BPE_ERR_NULL_PTR;

    if (vocab_piece_bytes_len > 0x7FFFFFFFFFFFFFFF ||
        vocab_piece_count > 0x7FFFFFFFFFFFFFFF ||
        max_piece_bytes > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    out_cursor = *io_out_cursor;
    if (out_cursor > 0x7FFFFFFFFFFFFFFF)
        return TOKENIZER_BPE_ERR_OVERFLOW;

    // Canonical default-capacity rule for one-token decode:
    // allow at most `max_piece_bytes` appended at current output cursor.
    if (max_piece_bytes > 0x7FFFFFFFFFFFFFFF - out_cursor)
        return TOKENIZER_BPE_ERR_OVERFLOW;
    derived_out_byte_capacity = out_cursor + max_piece_bytes;

    // Canonical default vocab table-capacity derivation:
    // `vocab_piece_capacity = vocab_piece_count`.
    return TokenizerBPEDecodeSingleTokenChecked(token_id,
                                                vocab_piece_bytes,
                                                vocab_piece_bytes_len,
                                                vocab_piece_offsets,
                                                vocab_piece_lens,
                                                vocab_piece_count,
                                                vocab_piece_count,
                                                out_bytes,
                                                derived_out_byte_capacity,
                                                io_out_cursor,
                                                out_byte_count);
}
