// attestation_manifest.HC
// IQ-1271: Attestation evidence emitter for trusted runtime sessions.
// Integer-only, deterministic, no external dependencies.

#define ATTEST_MANIFEST_OK             0
#define ATTEST_MANIFEST_ERR_NULL_PTR   1
#define ATTEST_MANIFEST_ERR_BAD_PARAM  2
#define ATTEST_MANIFEST_ERR_CAPACITY   3
#define ATTEST_MANIFEST_ERR_OVERFLOW   4

#define ATTEST_SESSION_ID_MAX_CHARS    64
#define ATTEST_PROFILE_NAME_MAX_CHARS  31
#define ATTEST_DIGEST_HEX_CHARS        64
#define ATTEST_MAX_LINES               12
#define ATTEST_MAX_LINE_CHARS          160

class InferenceAttestationManifest
{
    U8 session_id[ATTEST_SESSION_ID_MAX_CHARS + 1];
    U8 profile_name[ATTEST_PROFILE_NAME_MAX_CHARS + 1];
    U8 policy_digest_hex[ATTEST_DIGEST_HEX_CHARS + 1];
    I64 profile_id;
    I64 attestation_nonce;
    I64 trusted_model_count;
    I64 quarantine_block_count;
    I64 gpu_dispatch_allowed;
    I64 iommu_active;
    I64 bot_gpu_hooks_active;
    I64 line_count;
    U8 lines[ATTEST_MAX_LINES][ATTEST_MAX_LINE_CHARS + 1];
};

Bool AttestManifestIsSafeTextChar(U8 ch)
{
    if (ch >= 'a' && ch <= 'z')
        return TRUE;
    if (ch >= 'A' && ch <= 'Z')
        return TRUE;
    if (ch >= '0' && ch <= '9')
        return TRUE;
    if (ch == '-' || ch == '_' || ch == '.' || ch == ':' || ch == '/')
        return TRUE;
    return FALSE;
}

I32 AttestManifestU64ToDecimalChecked(U64 value, U8 *out_buf, I64 out_capacity)
{
    U8 scratch[32];
    I64 scratch_len;
    I64 i;

    if (!out_buf)
        return ATTEST_MANIFEST_ERR_NULL_PTR;
    if (out_capacity < 2)
        return ATTEST_MANIFEST_ERR_CAPACITY;

    if (!value)
    {
        out_buf[0] = '0';
        out_buf[1] = 0;
        return ATTEST_MANIFEST_OK;
    }

    scratch_len = 0;
    while (value)
    {
        if (scratch_len >= 31)
            return ATTEST_MANIFEST_ERR_CAPACITY;
        scratch[scratch_len++] = '0' + (U8)(value % 10);
        value /= 10;
    }

    if ((scratch_len + 1) > out_capacity)
        return ATTEST_MANIFEST_ERR_CAPACITY;

    for (i = 0; i < scratch_len; i++)
        out_buf[i] = scratch[scratch_len - 1 - i];
    out_buf[scratch_len] = 0;

    return ATTEST_MANIFEST_OK;
}

I32 AttestManifestCopySanitizedChecked(U8 *dst,
                                       I64 dst_capacity,
                                       U8 *src,
                                       I64 max_src_len)
{
    I64 i;
    U8 ch;

    if (!dst || !src)
        return ATTEST_MANIFEST_ERR_NULL_PTR;
    if (dst_capacity < 1)
        return ATTEST_MANIFEST_ERR_CAPACITY;
    if (max_src_len < 0)
        return ATTEST_MANIFEST_ERR_BAD_PARAM;

    i = 0;
    while (i < max_src_len)
    {
        ch = src[i];
        if (!ch)
            break;
        if (!AttestManifestIsSafeTextChar(ch))
            return ATTEST_MANIFEST_ERR_BAD_PARAM;
        if (i >= (dst_capacity - 1))
            return ATTEST_MANIFEST_ERR_CAPACITY;
        dst[i] = ch;
        i++;
    }

    if (!i)
        return ATTEST_MANIFEST_ERR_BAD_PARAM;

    dst[i] = 0;
    return ATTEST_MANIFEST_OK;
}

I32 AttestManifestAppendLineKVChecked(InferenceAttestationManifest *manifest,
                                      U8 *key,
                                      U8 *value)
{
    I64 key_len;
    I64 value_len;
    I64 i;
    I64 write_idx;
    I64 line_idx;

    if (!manifest || !key || !value)
        return ATTEST_MANIFEST_ERR_NULL_PTR;
    if (manifest->line_count < 0 || manifest->line_count > ATTEST_MAX_LINES)
        return ATTEST_MANIFEST_ERR_BAD_PARAM;
    if (manifest->line_count >= ATTEST_MAX_LINES)
        return ATTEST_MANIFEST_ERR_CAPACITY;

    key_len = 0;
    while (key[key_len])
        key_len++;
    value_len = 0;
    while (value[value_len])
        value_len++;

    if (!key_len)
        return ATTEST_MANIFEST_ERR_BAD_PARAM;
    if ((key_len + 1 + value_len + 1) > (ATTEST_MAX_LINE_CHARS + 1))
        return ATTEST_MANIFEST_ERR_CAPACITY;

    line_idx = manifest->line_count;
    write_idx = 0;

    for (i = 0; i < key_len; i++)
        manifest->lines[line_idx][write_idx++] = key[i];
    manifest->lines[line_idx][write_idx++] = '=';
    for (i = 0; i < value_len; i++)
        manifest->lines[line_idx][write_idx++] = value[i];
    manifest->lines[line_idx][write_idx] = 0;

    manifest->line_count++;
    return ATTEST_MANIFEST_OK;
}

I32 InferenceAttestationManifestInitChecked(InferenceAttestationManifest *manifest,
                                            U8 *session_id,
                                            I64 profile_id,
                                            U8 *profile_name,
                                            U8 *policy_digest_hex,
                                            I64 attestation_nonce)
{
    I32 rc;

    if (!manifest || !session_id || !profile_name || !policy_digest_hex)
        return ATTEST_MANIFEST_ERR_NULL_PTR;
    if (profile_id <= 0)
        return ATTEST_MANIFEST_ERR_BAD_PARAM;
    if (attestation_nonce < 0)
        return ATTEST_MANIFEST_ERR_BAD_PARAM;

    rc = AttestManifestCopySanitizedChecked(manifest->session_id,
                                            ATTEST_SESSION_ID_MAX_CHARS + 1,
                                            session_id,
                                            ATTEST_SESSION_ID_MAX_CHARS);
    if (rc)
        return rc;

    rc = AttestManifestCopySanitizedChecked(manifest->profile_name,
                                            ATTEST_PROFILE_NAME_MAX_CHARS + 1,
                                            profile_name,
                                            ATTEST_PROFILE_NAME_MAX_CHARS);
    if (rc)
        return rc;

    rc = AttestManifestCopySanitizedChecked(manifest->policy_digest_hex,
                                            ATTEST_DIGEST_HEX_CHARS + 1,
                                            policy_digest_hex,
                                            ATTEST_DIGEST_HEX_CHARS);
    if (rc)
        return rc;

    manifest->profile_id = profile_id;
    manifest->attestation_nonce = attestation_nonce;
    manifest->trusted_model_count = 0;
    manifest->quarantine_block_count = 0;
    manifest->gpu_dispatch_allowed = 0;
    manifest->iommu_active = 0;
    manifest->bot_gpu_hooks_active = 0;
    manifest->line_count = 0;

    return ATTEST_MANIFEST_OK;
}

I32 InferenceAttestationManifestSetTrustCountsChecked(InferenceAttestationManifest *manifest,
                                                      I64 trusted_model_count,
                                                      I64 quarantine_block_count)
{
    if (!manifest)
        return ATTEST_MANIFEST_ERR_NULL_PTR;
    if (trusted_model_count < 0 || quarantine_block_count < 0)
        return ATTEST_MANIFEST_ERR_BAD_PARAM;

    manifest->trusted_model_count = trusted_model_count;
    manifest->quarantine_block_count = quarantine_block_count;
    return ATTEST_MANIFEST_OK;
}

I32 InferenceAttestationManifestSetGPUStateChecked(InferenceAttestationManifest *manifest,
                                                   I64 gpu_dispatch_allowed,
                                                   I64 iommu_active,
                                                   I64 bot_gpu_hooks_active)
{
    if (!manifest)
        return ATTEST_MANIFEST_ERR_NULL_PTR;
    if ((gpu_dispatch_allowed != 0 && gpu_dispatch_allowed != 1) ||
        (iommu_active != 0 && iommu_active != 1) ||
        (bot_gpu_hooks_active != 0 && bot_gpu_hooks_active != 1))
        return ATTEST_MANIFEST_ERR_BAD_PARAM;

    manifest->gpu_dispatch_allowed = gpu_dispatch_allowed;
    manifest->iommu_active = iommu_active;
    manifest->bot_gpu_hooks_active = bot_gpu_hooks_active;
    return ATTEST_MANIFEST_OK;
}

I32 InferenceAttestationManifestEmitChecked(InferenceAttestationManifest *manifest)
{
    I32 rc;
    U8 nbuf[32];

    if (!manifest)
        return ATTEST_MANIFEST_ERR_NULL_PTR;

    manifest->line_count = 0;

    rc = AttestManifestAppendLineKVChecked(manifest,
                                           (U8 *)"session_id",
                                           manifest->session_id);
    if (rc)
        return rc;

    rc = AttestManifestAppendLineKVChecked(manifest,
                                           (U8 *)"profile_name",
                                           manifest->profile_name);
    if (rc)
        return rc;

    rc = AttestManifestAppendLineKVChecked(manifest,
                                           (U8 *)"policy_digest",
                                           manifest->policy_digest_hex);
    if (rc)
        return rc;

    rc = AttestManifestU64ToDecimalChecked((U64)manifest->profile_id, nbuf, 32);
    if (rc)
        return rc;
    rc = AttestManifestAppendLineKVChecked(manifest, (U8 *)"profile_id", nbuf);
    if (rc)
        return rc;

    rc = AttestManifestU64ToDecimalChecked((U64)manifest->attestation_nonce, nbuf, 32);
    if (rc)
        return rc;
    rc = AttestManifestAppendLineKVChecked(manifest, (U8 *)"nonce", nbuf);
    if (rc)
        return rc;

    rc = AttestManifestU64ToDecimalChecked((U64)manifest->trusted_model_count, nbuf, 32);
    if (rc)
        return rc;
    rc = AttestManifestAppendLineKVChecked(manifest, (U8 *)"trusted_models", nbuf);
    if (rc)
        return rc;

    rc = AttestManifestU64ToDecimalChecked((U64)manifest->quarantine_block_count, nbuf, 32);
    if (rc)
        return rc;
    rc = AttestManifestAppendLineKVChecked(manifest, (U8 *)"quarantine_blocks", nbuf);
    if (rc)
        return rc;

    rc = AttestManifestU64ToDecimalChecked((U64)manifest->gpu_dispatch_allowed, nbuf, 32);
    if (rc)
        return rc;
    rc = AttestManifestAppendLineKVChecked(manifest, (U8 *)"gpu_dispatch_allowed", nbuf);
    if (rc)
        return rc;

    rc = AttestManifestU64ToDecimalChecked((U64)manifest->iommu_active, nbuf, 32);
    if (rc)
        return rc;
    rc = AttestManifestAppendLineKVChecked(manifest, (U8 *)"iommu_active", nbuf);
    if (rc)
        return rc;

    rc = AttestManifestU64ToDecimalChecked((U64)manifest->bot_gpu_hooks_active, nbuf, 32);
    if (rc)
        return rc;
    rc = AttestManifestAppendLineKVChecked(manifest, (U8 *)"bot_gpu_hooks_active", nbuf);
    if (rc)
        return rc;

    return ATTEST_MANIFEST_OK;
}
