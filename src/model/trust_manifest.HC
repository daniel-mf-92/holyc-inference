// trust_manifest.HC
// Trusted model manifest parser + SHA256 verifier for secure-local promotion.
//
// Manifest line format (ASCII, one entry per non-comment line):
//   <sha256_hex_64> <size_bytes_decimal> <relative_model_path>
//
// Example:
//   9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08 123456 models/tinyllama-q4_0.gguf
//
// No-partial contract:
// - Parser preflights the full manifest first.
// - On any error, caller output arrays/counters remain untouched.

#define TRUST_MANIFEST_OK                  0
#define TRUST_MANIFEST_ERR_NULL_PTR        1
#define TRUST_MANIFEST_ERR_BAD_PARAM       2
#define TRUST_MANIFEST_ERR_OVERFLOW        3
#define TRUST_MANIFEST_ERR_MALFORMED       4
#define TRUST_MANIFEST_ERR_CAPACITY        5
#define TRUST_MANIFEST_ERR_ENTRY_NOT_FOUND 6
#define TRUST_MANIFEST_ERR_SIZE_MISMATCH   7
#define TRUST_MANIFEST_ERR_HASH_MISMATCH   8

#define TRUST_MANIFEST_SHA256_HEX_CHARS 64
#define TRUST_MANIFEST_MAX_PATH_CHARS   240
#define TRUST_MANIFEST_SHA256_BYTES      32
#define TRUST_MANIFEST_SHA256_BLOCK_BYTES 64

#define TRUST_U64_MAX_VALUE 0xFFFFFFFFFFFFFFFF

class TrustManifestEntry
{
    U8 sha256_hex[TRUST_MANIFEST_SHA256_HEX_CHARS + 1];
    U64 size_bytes;
    U8 rel_path[TRUST_MANIFEST_MAX_PATH_CHARS + 1];
};

class TrustSHA256Ctx
{
    U32 state[8];
    U64 total_len_bytes;
    U8 block[TRUST_MANIFEST_SHA256_BLOCK_BYTES];
    U32 block_len;
};

U32 g_trust_sha256_k[64] = {
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
};

Bool TrustManifestIsWhitespace(U8 ch)
{
    return (ch == ' ' || ch == '\t' || ch == '\r');
}

Bool TrustManifestIsHexChar(U8 ch)
{
    if (ch >= '0' && ch <= '9')
        return TRUE;
    if (ch >= 'a' && ch <= 'f')
        return TRUE;
    if (ch >= 'A' && ch <= 'F')
        return TRUE;
    return FALSE;
}

I32 TrustManifestHexNibble(U8 ch, U8 *out_nibble)
{
    if (!out_nibble)
        return TRUST_MANIFEST_ERR_NULL_PTR;

    if (ch >= '0' && ch <= '9')
    {
        *out_nibble = ch - '0';
        return TRUST_MANIFEST_OK;
    }
    if (ch >= 'a' && ch <= 'f')
    {
        *out_nibble = 10 + (ch - 'a');
        return TRUST_MANIFEST_OK;
    }
    if (ch >= 'A' && ch <= 'F')
    {
        *out_nibble = 10 + (ch - 'A');
        return TRUST_MANIFEST_OK;
    }

    return TRUST_MANIFEST_ERR_BAD_PARAM;
}

U8 TrustManifestHexCharLower(U8 nibble)
{
    if (nibble < 10)
        return '0' + nibble;
    return 'a' + (nibble - 10);
}

Bool TrustManifestCanReadSpan(U64 total_bytes, U64 offset, U64 need_bytes)
{
    U64 end_offset;

    if (offset > total_bytes)
        return FALSE;
    if (need_bytes > (TRUST_U64_MAX_VALUE - offset))
        return FALSE;

    end_offset = offset + need_bytes;
    if (end_offset > total_bytes)
        return FALSE;

    return TRUE;
}

I32 TrustManifestTryAddU64Checked(U64 lhs, U64 rhs, U64 *out_sum)
{
    if (!out_sum)
        return TRUST_MANIFEST_ERR_NULL_PTR;

    if (rhs > (TRUST_U64_MAX_VALUE - lhs))
        return TRUST_MANIFEST_ERR_OVERFLOW;

    *out_sum = lhs + rhs;
    return TRUST_MANIFEST_OK;
}

I32 TrustManifestTrimmedTokenEnd(U8 *buf,
                                 U64 token_start,
                                 U64 line_end,
                                 U64 *out_trimmed_end)
{
    U64 end_offset;

    if (!buf || !out_trimmed_end)
        return TRUST_MANIFEST_ERR_NULL_PTR;
    if (token_start > line_end)
        return TRUST_MANIFEST_ERR_BAD_PARAM;

    end_offset = line_end;
    while (end_offset > token_start)
    {
        if (!TrustManifestIsWhitespace(buf[end_offset - 1]))
            break;
        end_offset--;
    }

    *out_trimmed_end = end_offset;
    return TRUST_MANIFEST_OK;
}

I32 TrustManifestParseU64DecimalChecked(U8 *buf,
                                        U64 token_start,
                                        U64 token_end,
                                        U64 *out_value)
{
    U64 value;
    U64 i;
    U8 ch;
    U64 next_value;

    if (!buf || !out_value)
        return TRUST_MANIFEST_ERR_NULL_PTR;
    if (token_end <= token_start)
        return TRUST_MANIFEST_ERR_MALFORMED;

    value = 0;
    for (i = token_start; i < token_end; i++)
    {
        ch = buf[i];
        if (ch < '0' || ch > '9')
            return TRUST_MANIFEST_ERR_MALFORMED;

        if (value > (TRUST_U64_MAX_VALUE / 10))
            return TRUST_MANIFEST_ERR_OVERFLOW;

        next_value = value * 10;
        if ((U64)(ch - '0') > (TRUST_U64_MAX_VALUE - next_value))
            return TRUST_MANIFEST_ERR_OVERFLOW;

        value = next_value + (U64)(ch - '0');
    }

    *out_value = value;
    return TRUST_MANIFEST_OK;
}

I32 TrustManifestPathValidateRelativeChecked(U8 *path_buf, U64 path_len)
{
    U64 i;
    U64 segment_start;
    U64 segment_len;

    if (!path_buf)
        return TRUST_MANIFEST_ERR_NULL_PTR;
    if (!path_len)
        return TRUST_MANIFEST_ERR_MALFORMED;
    if (path_len > TRUST_MANIFEST_MAX_PATH_CHARS)
        return TRUST_MANIFEST_ERR_CAPACITY;

    if (path_buf[0] == '/' || path_buf[0] == '\\')
        return TRUST_MANIFEST_ERR_MALFORMED;

    segment_start = 0;
    i = 0;
    while (i <= path_len)
    {
        if (i == path_len || path_buf[i] == '/')
        {
            segment_len = i - segment_start;
            if (!segment_len)
                return TRUST_MANIFEST_ERR_MALFORMED;

            if (segment_len == 1 && path_buf[segment_start] == '.')
                return TRUST_MANIFEST_ERR_MALFORMED;
            if (segment_len == 2 &&
                path_buf[segment_start] == '.' &&
                path_buf[segment_start + 1] == '.')
                return TRUST_MANIFEST_ERR_MALFORMED;

            segment_start = i + 1;
        }

        if (i < path_len)
        {
            if (path_buf[i] == '\\' || path_buf[i] == ':' || path_buf[i] == 0)
                return TRUST_MANIFEST_ERR_MALFORMED;
        }

        i++;
    }

    return TRUST_MANIFEST_OK;
}

I32 TrustManifestParseLineChecked(U8 *manifest_buf,
                                  U64 manifest_nbytes,
                                  U64 line_start,
                                  U64 line_end,
                                  TrustManifestEntry *out_entry,
                                  Bool commit_mode)
{
    U64 cursor;
    U64 token_start;
    U64 hash_start;
    U64 hash_end;
    U64 size_start;
    U64 size_end;
    U64 path_start;
    U64 path_end;
    U64 path_len;
    U64 i;
    I32 status;
    U64 parsed_size;

    if (!manifest_buf)
        return TRUST_MANIFEST_ERR_NULL_PTR;
    if (line_end <= line_start)
        return TRUST_MANIFEST_ERR_MALFORMED;
    if (!TrustManifestCanReadSpan(manifest_nbytes, line_start, line_end - line_start))
        return TRUST_MANIFEST_ERR_BAD_PARAM;

    cursor = line_start;
    while (cursor < line_end && TrustManifestIsWhitespace(manifest_buf[cursor]))
        cursor++;

    // Skip blanks and comments.
    if (cursor >= line_end)
        return TRUST_MANIFEST_OK;
    if (manifest_buf[cursor] == '#')
        return TRUST_MANIFEST_OK;

    // Hash token: exactly 64 hex chars.
    hash_start = cursor;
    while (cursor < line_end && !TrustManifestIsWhitespace(manifest_buf[cursor]))
        cursor++;
    hash_end = cursor;

    if ((hash_end - hash_start) != TRUST_MANIFEST_SHA256_HEX_CHARS)
        return TRUST_MANIFEST_ERR_MALFORMED;

    for (i = hash_start; i < hash_end; i++)
    {
        if (!TrustManifestIsHexChar(manifest_buf[i]))
            return TRUST_MANIFEST_ERR_MALFORMED;
    }

    while (cursor < line_end && TrustManifestIsWhitespace(manifest_buf[cursor]))
        cursor++;

    // Size token: decimal u64.
    size_start = cursor;
    while (cursor < line_end && !TrustManifestIsWhitespace(manifest_buf[cursor]))
        cursor++;
    size_end = cursor;

    status = TrustManifestParseU64DecimalChecked(manifest_buf,
                                                 size_start,
                                                 size_end,
                                                 &parsed_size);
    if (status)
        return status;

    while (cursor < line_end && TrustManifestIsWhitespace(manifest_buf[cursor]))
        cursor++;

    // Path token: rest of line, trimmed right.
    path_start = cursor;
    status = TrustManifestTrimmedTokenEnd(manifest_buf,
                                          path_start,
                                          line_end,
                                          &path_end);
    if (status)
        return status;
    if (path_end <= path_start)
        return TRUST_MANIFEST_ERR_MALFORMED;

    path_len = path_end - path_start;
    status = TrustManifestPathValidateRelativeChecked(manifest_buf + path_start,
                                                      path_len);
    if (status)
        return status;

    if (commit_mode)
    {
        if (!out_entry)
            return TRUST_MANIFEST_ERR_NULL_PTR;

        for (i = 0; i < TRUST_MANIFEST_SHA256_HEX_CHARS; i++)
        {
            U8 nibble;

            status = TrustManifestHexNibble(manifest_buf[hash_start + i], &nibble);
            if (status)
                return status;
            out_entry->sha256_hex[i] = TrustManifestHexCharLower(nibble);
        }
        out_entry->sha256_hex[TRUST_MANIFEST_SHA256_HEX_CHARS] = 0;

        out_entry->size_bytes = parsed_size;

        for (i = 0; i < path_len; i++)
            out_entry->rel_path[i] = manifest_buf[path_start + i];
        out_entry->rel_path[path_len] = 0;
    }

    return TRUST_MANIFEST_OK;
}

I32 TrustManifestParsePassChecked(U8 *manifest_buf,
                                  U64 manifest_nbytes,
                                  TrustManifestEntry *out_entries,
                                  I64 entry_capacity,
                                  Bool commit_mode,
                                  I64 *out_count)
{
    U64 cursor;
    U64 line_start;
    U64 line_end;
    I64 entry_count;
    I32 status;

    if (!manifest_buf || !out_count)
        return TRUST_MANIFEST_ERR_NULL_PTR;
    if (entry_capacity < 0)
        return TRUST_MANIFEST_ERR_BAD_PARAM;

    cursor = 0;
    entry_count = 0;

    while (cursor < manifest_nbytes)
    {
        line_start = cursor;

        while (cursor < manifest_nbytes && manifest_buf[cursor] != '\n')
            cursor++;
        line_end = cursor;

        if (line_end > line_start)
        {
            status = TrustManifestParseLineChecked(manifest_buf,
                                                   manifest_nbytes,
                                                   line_start,
                                                   line_end,
                                                   commit_mode ? (out_entries + entry_count) : NULL,
                                                   commit_mode);
            if (status)
                return status;

            // Count non-comment, non-empty lines as entries.
            {
                U64 probe;
                probe = line_start;
                while (probe < line_end && TrustManifestIsWhitespace(manifest_buf[probe]))
                    probe++;

                if (probe < line_end && manifest_buf[probe] != '#')
                {
                    if (entry_count >= entry_capacity)
                        return TRUST_MANIFEST_ERR_CAPACITY;
                    entry_count++;
                }
            }
        }

        if (cursor < manifest_nbytes && manifest_buf[cursor] == '\n')
            cursor++;
    }

    *out_count = entry_count;
    return TRUST_MANIFEST_OK;
}

I32 TrustManifestParseCheckedNoPartial(U8 *manifest_buf,
                                       U64 manifest_nbytes,
                                       TrustManifestEntry *out_entries,
                                       I64 entry_capacity,
                                       I64 *out_entry_count)
{
    I64 required_entries;
    I64 committed_entries;
    I32 status;

    if (!manifest_buf || !out_entry_count)
        return TRUST_MANIFEST_ERR_NULL_PTR;
    if (entry_capacity < 0)
        return TRUST_MANIFEST_ERR_BAD_PARAM;

    if (entry_capacity && !out_entries)
        return TRUST_MANIFEST_ERR_NULL_PTR;

    // Pass 1: validate everything, derive entry count, do not write outputs.
    status = TrustManifestParsePassChecked(manifest_buf,
                                           manifest_nbytes,
                                           NULL,
                                           0x7FFFFFFFFFFFFFFF,
                                           FALSE,
                                           &required_entries);
    if (status)
        return status;

    if (required_entries > entry_capacity)
        return TRUST_MANIFEST_ERR_CAPACITY;

    // Pass 2: commit after full preflight success.
    status = TrustManifestParsePassChecked(manifest_buf,
                                           manifest_nbytes,
                                           out_entries,
                                           entry_capacity,
                                           TRUE,
                                           &committed_entries);
    if (status)
        return status;

    if (committed_entries != required_entries)
        return TRUST_MANIFEST_ERR_MALFORMED;

    *out_entry_count = committed_entries;
    return TRUST_MANIFEST_OK;
}

I32 TrustManifestCStringLenChecked(U8 *str, U64 max_len, U64 *out_len)
{
    U64 i;

    if (!str || !out_len)
        return TRUST_MANIFEST_ERR_NULL_PTR;

    for (i = 0; i < max_len; i++)
    {
        if (str[i] == 0)
        {
            *out_len = i;
            return TRUST_MANIFEST_OK;
        }
    }

    return TRUST_MANIFEST_ERR_BAD_PARAM;
}

Bool TrustManifestPathEquals(U8 *lhs, U8 *rhs)
{
    U64 i;

    if (!lhs || !rhs)
        return FALSE;

    i = 0;
    while (lhs[i] && rhs[i])
    {
        if (lhs[i] != rhs[i])
            return FALSE;
        i++;
    }

    return (lhs[i] == 0 && rhs[i] == 0);
}

I32 TrustManifestFindEntryByPathChecked(TrustManifestEntry *entries,
                                        I64 entry_count,
                                        U8 *rel_path,
                                        I64 *out_entry_index)
{
    I64 i;
    U64 path_len;
    I32 status;

    if (!entries || !rel_path || !out_entry_index)
        return TRUST_MANIFEST_ERR_NULL_PTR;
    if (entry_count < 0)
        return TRUST_MANIFEST_ERR_BAD_PARAM;

    status = TrustManifestCStringLenChecked(rel_path,
                                            TRUST_MANIFEST_MAX_PATH_CHARS,
                                            &path_len);
    if (status)
        return status;

    status = TrustManifestPathValidateRelativeChecked(rel_path, path_len);
    if (status)
        return status;

    for (i = 0; i < entry_count; i++)
    {
        if (TrustManifestPathEquals(entries[i].rel_path, rel_path))
        {
            *out_entry_index = i;
            return TRUST_MANIFEST_OK;
        }
    }

    return TRUST_MANIFEST_ERR_ENTRY_NOT_FOUND;
}

U32 TrustSHA256RotR(U32 x, U32 bits)
{
    return (x >> bits) | (x << (32 - bits));
}

U32 TrustSHA256Ch(U32 x, U32 y, U32 z)
{
    return (x & y) ^ (~x & z);
}

U32 TrustSHA256Maj(U32 x, U32 y, U32 z)
{
    return (x & y) ^ (x & z) ^ (y & z);
}

U32 TrustSHA256BSig0(U32 x)
{
    return TrustSHA256RotR(x, 2) ^ TrustSHA256RotR(x, 13) ^ TrustSHA256RotR(x, 22);
}

U32 TrustSHA256BSig1(U32 x)
{
    return TrustSHA256RotR(x, 6) ^ TrustSHA256RotR(x, 11) ^ TrustSHA256RotR(x, 25);
}

U32 TrustSHA256SSig0(U32 x)
{
    return TrustSHA256RotR(x, 7) ^ TrustSHA256RotR(x, 18) ^ (x >> 3);
}

U32 TrustSHA256SSig1(U32 x)
{
    return TrustSHA256RotR(x, 17) ^ TrustSHA256RotR(x, 19) ^ (x >> 10);
}

I32 TrustSHA256TransformChecked(TrustSHA256Ctx *ctx, U8 *block64)
{
    U32 w[64];
    U32 a, b, c, d, e, f, g, h;
    U32 t1;
    U32 t2;
    I64 i;

    if (!ctx || !block64)
        return TRUST_MANIFEST_ERR_NULL_PTR;

    for (i = 0; i < 16; i++)
    {
        U64 base;
        base = (U64)i * 4;
        w[i] = ((U32)block64[base] << 24)
             | ((U32)block64[base + 1] << 16)
             | ((U32)block64[base + 2] << 8)
             | ((U32)block64[base + 3]);
    }

    for (i = 16; i < 64; i++)
        w[i] = TrustSHA256SSig1(w[i - 2]) + w[i - 7] + TrustSHA256SSig0(w[i - 15]) + w[i - 16];

    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];
    f = ctx->state[5];
    g = ctx->state[6];
    h = ctx->state[7];

    for (i = 0; i < 64; i++)
    {
        t1 = h + TrustSHA256BSig1(e) + TrustSHA256Ch(e, f, g) + g_trust_sha256_k[i] + w[i];
        t2 = TrustSHA256BSig0(a) + TrustSHA256Maj(a, b, c);

        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;

    return TRUST_MANIFEST_OK;
}

I32 TrustSHA256InitChecked(TrustSHA256Ctx *ctx)
{
    if (!ctx)
        return TRUST_MANIFEST_ERR_NULL_PTR;

    ctx->state[0] = 0x6A09E667;
    ctx->state[1] = 0xBB67AE85;
    ctx->state[2] = 0x3C6EF372;
    ctx->state[3] = 0xA54FF53A;
    ctx->state[4] = 0x510E527F;
    ctx->state[5] = 0x9B05688C;
    ctx->state[6] = 0x1F83D9AB;
    ctx->state[7] = 0x5BE0CD19;

    ctx->total_len_bytes = 0;
    ctx->block_len = 0;
    return TRUST_MANIFEST_OK;
}

I32 TrustSHA256UpdateChecked(TrustSHA256Ctx *ctx, U8 *data, U64 data_len)
{
    U64 i;
    I32 status;

    if (!ctx || !data)
        return TRUST_MANIFEST_ERR_NULL_PTR;

    status = TrustManifestTryAddU64Checked(ctx->total_len_bytes,
                                           data_len,
                                           &ctx->total_len_bytes);
    if (status)
        return status;

    for (i = 0; i < data_len; i++)
    {
        ctx->block[ctx->block_len] = data[i];
        ctx->block_len++;

        if (ctx->block_len == TRUST_MANIFEST_SHA256_BLOCK_BYTES)
        {
            status = TrustSHA256TransformChecked(ctx, ctx->block);
            if (status)
                return status;
            ctx->block_len = 0;
        }
    }

    return TRUST_MANIFEST_OK;
}

I32 TrustSHA256FinalChecked(TrustSHA256Ctx *ctx, U8 *out_hash32)
{
    U64 bit_len;
    I32 status;
    I64 i;

    if (!ctx || !out_hash32)
        return TRUST_MANIFEST_ERR_NULL_PTR;

    if (ctx->total_len_bytes > (TRUST_U64_MAX_VALUE >> 3))
        return TRUST_MANIFEST_ERR_OVERFLOW;

    bit_len = ctx->total_len_bytes << 3;

    ctx->block[ctx->block_len++] = 0x80;

    while (ctx->block_len != 56)
    {
        if (ctx->block_len == 64)
        {
            status = TrustSHA256TransformChecked(ctx, ctx->block);
            if (status)
                return status;
            ctx->block_len = 0;
        }

        ctx->block[ctx->block_len++] = 0;
    }

    for (i = 0; i < 8; i++)
    {
        U8 shift;
        shift = 56 - ((U8)i * 8);
        ctx->block[56 + i] = (U8)(bit_len >> shift);
    }

    status = TrustSHA256TransformChecked(ctx, ctx->block);
    if (status)
        return status;

    for (i = 0; i < 8; i++)
    {
        U32 word;
        word = ctx->state[i];
        out_hash32[i * 4] = (U8)(word >> 24);
        out_hash32[i * 4 + 1] = (U8)(word >> 16);
        out_hash32[i * 4 + 2] = (U8)(word >> 8);
        out_hash32[i * 4 + 3] = (U8)word;
    }

    return TRUST_MANIFEST_OK;
}

I32 TrustManifestSHA256HexComputeChecked(U8 *data,
                                         U64 data_len,
                                         U8 *out_hex)
{
    TrustSHA256Ctx ctx;
    U8 digest[TRUST_MANIFEST_SHA256_BYTES];
    I64 i;
    I32 status;

    if (!data || !out_hex)
        return TRUST_MANIFEST_ERR_NULL_PTR;

    status = TrustSHA256InitChecked(&ctx);
    if (status)
        return status;

    status = TrustSHA256UpdateChecked(&ctx, data, data_len);
    if (status)
        return status;

    status = TrustSHA256FinalChecked(&ctx, digest);
    if (status)
        return status;

    for (i = 0; i < TRUST_MANIFEST_SHA256_BYTES; i++)
    {
        out_hex[i * 2] = TrustManifestHexCharLower((digest[i] >> 4) & 0x0F);
        out_hex[i * 2 + 1] = TrustManifestHexCharLower(digest[i] & 0x0F);
    }
    out_hex[TRUST_MANIFEST_SHA256_HEX_CHARS] = 0;

    return TRUST_MANIFEST_OK;
}

I32 TrustManifestVerifyEntrySHA256Checked(TrustManifestEntry *entry,
                                          U8 *model_buf,
                                          U64 model_nbytes)
{
    U8 hash_hex[TRUST_MANIFEST_SHA256_HEX_CHARS + 1];
    I64 i;
    I32 status;

    if (!entry || !model_buf)
        return TRUST_MANIFEST_ERR_NULL_PTR;

    if (entry->size_bytes != model_nbytes)
        return TRUST_MANIFEST_ERR_SIZE_MISMATCH;

    status = TrustManifestSHA256HexComputeChecked(model_buf,
                                                  model_nbytes,
                                                  hash_hex);
    if (status)
        return status;

    for (i = 0; i < TRUST_MANIFEST_SHA256_HEX_CHARS; i++)
    {
        if (hash_hex[i] != entry->sha256_hex[i])
            return TRUST_MANIFEST_ERR_HASH_MISMATCH;
    }

    return TRUST_MANIFEST_OK;
}

I32 TrustManifestVerifyPathCheckedNoPartial(U8 *manifest_buf,
                                            U64 manifest_nbytes,
                                            TrustManifestEntry *scratch_entries,
                                            I64 scratch_capacity,
                                            U8 *target_rel_path,
                                            U8 *model_buf,
                                            U64 model_nbytes,
                                            I64 *out_entry_index)
{
    I64 entry_count;
    I64 entry_index;
    I32 status;

    if (!manifest_buf || !scratch_entries || !target_rel_path || !model_buf || !out_entry_index)
        return TRUST_MANIFEST_ERR_NULL_PTR;

    status = TrustManifestParseCheckedNoPartial(manifest_buf,
                                                manifest_nbytes,
                                                scratch_entries,
                                                scratch_capacity,
                                                &entry_count);
    if (status)
        return status;

    status = TrustManifestFindEntryByPathChecked(scratch_entries,
                                                 entry_count,
                                                 target_rel_path,
                                                 &entry_index);
    if (status)
        return status;

    status = TrustManifestVerifyEntrySHA256Checked(scratch_entries + entry_index,
                                                   model_buf,
                                                   model_nbytes);
    if (status)
        return status;

    *out_entry_index = entry_index;
    return TRUST_MANIFEST_OK;
}
