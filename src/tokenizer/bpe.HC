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
    need = 0;
    codepoint = 0;

    // UTF-8 lead-byte classification with overlong-leading-byte rejection.
    if (lead <= 0x7F) {
        need = 1;
        codepoint = lead;
    } else if (lead >= 0xC2 && lead <= 0xDF) {
        need = 2;
        codepoint = lead & 0x1F;
    } else if (lead >= 0xE0 && lead <= 0xEF) {
        need = 3;
        codepoint = lead & 0x0F;
    } else if (lead >= 0xF0 && lead <= 0xF4) {
        need = 4;
        codepoint = lead & 0x07;
    } else {
        return TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE;
    }

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

    // Reject overlong and out-of-range scalar forms.
    if (need == 2 && codepoint < 0x80)
        return TOKENIZER_UTF8_ERR_BAD_CODEPOINT;

    if (need == 3) {
        if (codepoint < 0x800)
            return TOKENIZER_UTF8_ERR_BAD_CODEPOINT;
        if (codepoint >= 0xD800 && codepoint <= 0xDFFF)
            return TOKENIZER_UTF8_ERR_BAD_CODEPOINT;
    }

    if (need == 4) {
        if (codepoint < 0x10000)
            return TOKENIZER_UTF8_ERR_BAD_CODEPOINT;
        if (codepoint > 0x10FFFF)
            return TOKENIZER_UTF8_ERR_BAD_CODEPOINT;
    }

    next_cursor = cursor + need;
    if (next_cursor < cursor || next_cursor > byte_len)
        return TOKENIZER_UTF8_ERR_OVERFLOW;

    // Commit only after all checks pass (no partial state updates on failure).
    *out_codepoint = codepoint;
    *out_bytes_consumed = need;
    *io_cursor = next_cursor;
    return TOKENIZER_UTF8_OK;
}
