// quarantine.HC
// Model quarantine import->verify->promote workflow for secure-local trust gates.
//
// Runtime contract:
// - import stage records an untrusted model candidate path + byte size
// - verify stage enforces manifest hash+size match for the imported path
// - promote stage requires secure-local profile and verified quarantine state
//
// No networking, no external deps, integer-only checks.

#define QUARANTINE_OK                0
#define QUARANTINE_ERR_NULL_PTR      1
#define QUARANTINE_ERR_BAD_PARAM     2
#define QUARANTINE_ERR_OVERFLOW      3
#define QUARANTINE_ERR_MALFORMED     4
#define QUARANTINE_ERR_BAD_STATE     5
#define QUARANTINE_ERR_PROFILE_GUARD 6
#define QUARANTINE_ERR_ENTRY_NOT_FOUND 7
#define QUARANTINE_ERR_SIZE_MISMATCH 8
#define QUARANTINE_ERR_HASH_MISMATCH 9

#define QUARANTINE_STAGE_EMPTY    0
#define QUARANTINE_STAGE_IMPORTED 1
#define QUARANTINE_STAGE_VERIFIED 2
#define QUARANTINE_STAGE_PROMOTED 3

#define QUARANTINE_MAX_PATH_CHARS 240
#define QUARANTINE_SHA256_HEX_CHARS 64
#define QUARANTINE_U64_MAX_VALUE 0xFFFFFFFFFFFFFFFF

class ModelQuarantineState
{
    I64 stage;
    U8 import_rel_path[QUARANTINE_MAX_PATH_CHARS + 1];
    U8 promoted_rel_path[QUARANTINE_MAX_PATH_CHARS + 1];
    U64 imported_model_nbytes;
    I64 verified_manifest_entry;
    U8 verified_hash_hex[QUARANTINE_SHA256_HEX_CHARS + 1];
    I64 verified_profile_id;
};

I32 InferenceProfileStatusChecked(I64 *out_profile_id,
                                  U8 **out_profile_name,
                                  I64 *out_is_secure_default);
I32 TrustManifestSHA256HexComputeChecked(U8 *data,
                                         U64 data_len,
                                         U8 *out_hex);

Bool QuarantineIsSpace(U8 ch)
{
    return (ch == ' ' || ch == '\t' || ch == '\r');
}

Bool QuarantineIsHex(U8 ch)
{
    if (ch >= '0' && ch <= '9')
        return TRUE;
    if (ch >= 'a' && ch <= 'f')
        return TRUE;
    if (ch >= 'A' && ch <= 'F')
        return TRUE;
    return FALSE;
}

U8 QuarantineHexToLower(U8 ch)
{
    if (ch >= 'A' && ch <= 'F')
        return ch + ('a' - 'A');
    return ch;
}

Bool QuarantineCanReadSpan(U64 total, U64 offset, U64 need)
{
    if (offset > total)
        return FALSE;
    if (need > (QUARANTINE_U64_MAX_VALUE - offset))
        return FALSE;
    if (offset + need > total)
        return FALSE;
    return TRUE;
}

I64 QuarantineCStringLenBounded(U8 *buf, I64 max_chars)
{
    I64 i;

    if (!buf || max_chars < 0)
        return -1;

    for (i = 0; i <= max_chars; i++)
    {
        if (buf[i] == 0)
            return i;
    }

    return -1;
}

Bool QuarantinePathHasTraversal(U8 *path)
{
    I64 i;

    for (i = 0; path[i]; i++)
    {
        if (path[i] != '.')
            continue;

        if (i == 0 || path[i - 1] == '/')
        {
            if (path[i + 1] == '.' && (path[i + 2] == 0 || path[i + 2] == '/'))
                return TRUE;
        }
    }

    return FALSE;
}

Bool QuarantinePathIsSafeRelative(U8 *path)
{
    I64 i;
    I64 len;

    len = QuarantineCStringLenBounded(path, QUARANTINE_MAX_PATH_CHARS);
    if (len <= 0)
        return FALSE;

    if (path[0] == '/' || path[0] == '\\')
        return FALSE;

    if (QuarantinePathHasTraversal(path))
        return FALSE;

    for (i = 0; i < len; i++)
    {
        if (path[i] == '\\' || path[i] == ':' || path[i] == '\n')
            return FALSE;
    }

    return TRUE;
}

I32 QuarantineStringCopyChecked(U8 *dst,
                                I64 dst_capacity,
                                U8 *src)
{
    I64 i;

    if (!dst || !src)
        return QUARANTINE_ERR_NULL_PTR;

    if (dst_capacity <= 0)
        return QUARANTINE_ERR_BAD_PARAM;

    for (i = 0; i < dst_capacity - 1; i++)
    {
        dst[i] = src[i];
        if (src[i] == 0)
            return QUARANTINE_OK;
    }

    if (src[i] != 0)
        return QUARANTINE_ERR_BAD_PARAM;

    dst[i] = 0;
    return QUARANTINE_OK;
}

I32 QuarantineParseU64DecimalChecked(U8 *line,
                                     I64 start,
                                     I64 end,
                                     U64 *out_value)
{
    U64 value;
    I64 i;
    U8 ch;

    if (!line || !out_value)
        return QUARANTINE_ERR_NULL_PTR;

    if (start < 0 || end < start)
        return QUARANTINE_ERR_BAD_PARAM;

    if (start == end)
        return QUARANTINE_ERR_MALFORMED;

    value = 0;
    for (i = start; i < end; i++)
    {
        ch = line[i];
        if (ch < '0' || ch > '9')
            return QUARANTINE_ERR_MALFORMED;

        if (value > (QUARANTINE_U64_MAX_VALUE / 10))
            return QUARANTINE_ERR_OVERFLOW;

        value = value * 10;
        if ((U64)(ch - '0') > (QUARANTINE_U64_MAX_VALUE - value))
            return QUARANTINE_ERR_OVERFLOW;

        value += (U64)(ch - '0');
    }

    *out_value = value;
    return QUARANTINE_OK;
}

I32 QuarantineManifestFindMatchChecked(U8 *manifest_buf,
                                       U64 manifest_nbytes,
                                       U8 *target_rel_path,
                                       U64 target_model_nbytes,
                                       U8 *out_hash_hex,
                                       I64 *out_entry_index)
{
    U64 line_start;
    U64 i;
    U64 line_end;
    I64 hash_start;
    I64 hash_end;
    I64 size_start;
    I64 size_end;
    I64 path_start;
    I64 path_end;
    I64 path_len;
    I64 j;
    U64 parsed_size;
    I64 entry_index;
    I32 status;

    if (!manifest_buf || !target_rel_path || !out_hash_hex || !out_entry_index)
        return QUARANTINE_ERR_NULL_PTR;

    line_start = 0;
    entry_index = 0;

    while (line_start < manifest_nbytes)
    {
        line_end = line_start;
        while (line_end < manifest_nbytes && manifest_buf[line_end] != '\n')
            line_end++;

        i = line_start;
        while (i < line_end && QuarantineIsSpace(manifest_buf[i]))
            i++;

        if (i < line_end && manifest_buf[i] != '#')
        {
            hash_start = i;
            while (i < line_end && !QuarantineIsSpace(manifest_buf[i]))
                i++;
            hash_end = i;

            while (i < line_end && QuarantineIsSpace(manifest_buf[i]))
                i++;
            size_start = i;
            while (i < line_end && !QuarantineIsSpace(manifest_buf[i]))
                i++;
            size_end = i;

            while (i < line_end && QuarantineIsSpace(manifest_buf[i]))
                i++;
            path_start = i;
            path_end = line_end;
            while (path_end > path_start && QuarantineIsSpace(manifest_buf[path_end - 1]))
                path_end--;

            if (hash_end - hash_start != QUARANTINE_SHA256_HEX_CHARS)
                return QUARANTINE_ERR_MALFORMED;
            if (size_start == size_end || path_start == path_end)
                return QUARANTINE_ERR_MALFORMED;

            for (j = hash_start; j < hash_end; j++)
            {
                if (!QuarantineIsHex(manifest_buf[j]))
                    return QUARANTINE_ERR_MALFORMED;
            }

            status = QuarantineParseU64DecimalChecked(manifest_buf,
                                                      size_start,
                                                      size_end,
                                                      &parsed_size);
            if (status)
                return status;

            path_len = path_end - path_start;
            if (path_len <= 0 || path_len > QUARANTINE_MAX_PATH_CHARS)
                return QUARANTINE_ERR_MALFORMED;

            if (QuarantineCStringLenBounded(target_rel_path, QUARANTINE_MAX_PATH_CHARS) == path_len)
            {
                for (j = 0; j < path_len; j++)
                {
                    if (target_rel_path[j] != manifest_buf[path_start + j])
                        break;
                }

                if (j == path_len)
                {
                    if (parsed_size != target_model_nbytes)
                        return QUARANTINE_ERR_SIZE_MISMATCH;

                    for (j = 0; j < QUARANTINE_SHA256_HEX_CHARS; j++)
                        out_hash_hex[j] = QuarantineHexToLower(manifest_buf[hash_start + j]);
                    out_hash_hex[QUARANTINE_SHA256_HEX_CHARS] = 0;
                    *out_entry_index = entry_index;
                    return QUARANTINE_OK;
                }
            }

            entry_index++;
        }

        line_start = line_end;
        if (line_start < manifest_nbytes && manifest_buf[line_start] == '\n')
            line_start++;
    }

    return QUARANTINE_ERR_ENTRY_NOT_FOUND;
}

I32 ModelQuarantineStateInitChecked(ModelQuarantineState *state)
{
    if (!state)
        return QUARANTINE_ERR_NULL_PTR;

    state->stage = QUARANTINE_STAGE_EMPTY;
    state->import_rel_path[0] = 0;
    state->promoted_rel_path[0] = 0;
    state->verified_hash_hex[0] = 0;
    state->imported_model_nbytes = 0;
    state->verified_manifest_entry = -1;
    state->verified_profile_id = 0;
    return QUARANTINE_OK;
}

I32 ModelQuarantineImportChecked(ModelQuarantineState *state,
                                 U8 *candidate_rel_path,
                                 U8 *model_buf,
                                 U64 model_nbytes)
{
    I32 status;

    if (!state || !candidate_rel_path || !model_buf)
        return QUARANTINE_ERR_NULL_PTR;

    if (!model_nbytes)
        return QUARANTINE_ERR_BAD_PARAM;

    if (!QuarantinePathIsSafeRelative(candidate_rel_path))
        return QUARANTINE_ERR_BAD_PARAM;

    status = QuarantineStringCopyChecked(state->import_rel_path,
                                         QUARANTINE_MAX_PATH_CHARS + 1,
                                         candidate_rel_path);
    if (status)
        return status;

    state->stage = QUARANTINE_STAGE_IMPORTED;
    state->promoted_rel_path[0] = 0;
    state->verified_hash_hex[0] = 0;
    state->imported_model_nbytes = model_nbytes;
    state->verified_manifest_entry = -1;
    state->verified_profile_id = 0;
    return QUARANTINE_OK;
}

I32 ModelQuarantineVerifyChecked(ModelQuarantineState *state,
                                 U8 *manifest_buf,
                                 U64 manifest_nbytes,
                                 U8 *model_buf,
                                 U64 model_nbytes)
{
    U8 expected_hash[QUARANTINE_SHA256_HEX_CHARS + 1];
    U8 computed_hash[QUARANTINE_SHA256_HEX_CHARS + 1];
    I64 entry_index;
    I32 status;
    I64 profile_id;
    U8 *profile_name;
    I64 is_secure_default;

    if (!state || !manifest_buf || !model_buf)
        return QUARANTINE_ERR_NULL_PTR;

    if (state->stage < QUARANTINE_STAGE_IMPORTED)
        return QUARANTINE_ERR_BAD_STATE;

    if (model_nbytes != state->imported_model_nbytes)
        return QUARANTINE_ERR_SIZE_MISMATCH;

    status = QuarantineManifestFindMatchChecked(manifest_buf,
                                                manifest_nbytes,
                                                state->import_rel_path,
                                                model_nbytes,
                                                expected_hash,
                                                &entry_index);
    if (status)
        return status;

    status = TrustManifestSHA256HexComputeChecked(model_buf,
                                                  model_nbytes,
                                                  computed_hash);
    if (status)
        return status;

    if (!QuarantineCanReadSpan(model_nbytes, 0, model_nbytes))
        return QUARANTINE_ERR_OVERFLOW;

    for (status = 0; status < QUARANTINE_SHA256_HEX_CHARS; status++)
    {
        if (computed_hash[status] != expected_hash[status])
            return QUARANTINE_ERR_HASH_MISMATCH;
    }

    status = InferenceProfileStatusChecked(&profile_id,
                                           &profile_name,
                                           &is_secure_default);
    if (status)
        return QUARANTINE_ERR_PROFILE_GUARD;

    state->verified_manifest_entry = entry_index;
    state->verified_profile_id = profile_id;
    for (status = 0; status < QUARANTINE_SHA256_HEX_CHARS; status++)
        state->verified_hash_hex[status] = computed_hash[status];
    state->verified_hash_hex[QUARANTINE_SHA256_HEX_CHARS] = 0;
    state->stage = QUARANTINE_STAGE_VERIFIED;

    return QUARANTINE_OK;
}

I32 ModelQuarantinePromoteChecked(ModelQuarantineState *state,
                                  U8 *trusted_rel_path)
{
    I64 profile_id;
    U8 *profile_name;
    I64 is_secure_default;
    I32 status;

    if (!state || !trusted_rel_path)
        return QUARANTINE_ERR_NULL_PTR;

    if (state->stage < QUARANTINE_STAGE_VERIFIED)
        return QUARANTINE_ERR_BAD_STATE;

    if (!QuarantinePathIsSafeRelative(trusted_rel_path))
        return QUARANTINE_ERR_BAD_PARAM;

    status = InferenceProfileStatusChecked(&profile_id,
                                           &profile_name,
                                           &is_secure_default);
    if (status)
        return QUARANTINE_ERR_PROFILE_GUARD;

    if (!is_secure_default)
        return QUARANTINE_ERR_PROFILE_GUARD;

    status = QuarantineStringCopyChecked(state->promoted_rel_path,
                                         QUARANTINE_MAX_PATH_CHARS + 1,
                                         trusted_rel_path);
    if (status)
        return status;

    state->stage = QUARANTINE_STAGE_PROMOTED;
    return QUARANTINE_OK;
}

I32 ModelQuarantineImportVerifyPromoteChecked(ModelQuarantineState *state,
                                              U8 *candidate_rel_path,
                                              U8 *trusted_rel_path,
                                              U8 *manifest_buf,
                                              U64 manifest_nbytes,
                                              U8 *model_buf,
                                              U64 model_nbytes)
{
    I32 status;

    if (!state || !candidate_rel_path || !trusted_rel_path ||
        !manifest_buf || !model_buf)
        return QUARANTINE_ERR_NULL_PTR;

    status = ModelQuarantineImportChecked(state,
                                          candidate_rel_path,
                                          model_buf,
                                          model_nbytes);
    if (status)
        return status;

    status = ModelQuarantineVerifyChecked(state,
                                          manifest_buf,
                                          manifest_nbytes,
                                          model_buf,
                                          model_nbytes);
    if (status)
        return status;

    status = ModelQuarantinePromoteChecked(state,
                                           trusted_rel_path);
    if (status)
        return status;

    return QUARANTINE_OK;
}
