// quant_profile.HC
// Quantization profile selector for throughput worker plane.
//
// Goals:
// - Integer-only policy selection (HolyC runtime purity).
// - secure-local remains default and strict.
// - dev-local is explicit opt-in and may tune throughput knobs,
//   but cannot disable quarantine/hash-manifest gates.

#define QUANT_PROFILE_OK                    0
#define QUANT_PROFILE_ERR_NULL_PTR          1
#define QUANT_PROFILE_ERR_BAD_PARAM         2
#define QUANT_PROFILE_ERR_PROFILE_GUARD     3
#define QUANT_PROFILE_ERR_POLICY_GUARD      4

#define QUANT_PROFILE_SECURE_LOCAL          1
#define QUANT_PROFILE_DEV_LOCAL             2

#define QUANT_MODE_Q4_0                     1
#define QUANT_MODE_Q8_0                     2
#define QUANT_MODE_MIXED_Q4Q8               3

#define QUANT_PROFILE_PREF_BALANCED         1
#define QUANT_PROFILE_PREF_THROUGHPUT       2
#define QUANT_PROFILE_PREF_ACCURACY         3

class QuantProfileSelection
{
    I64 profile_id;
    I64 quant_mode;
    I64 preferred_block_rows;
    I64 prefetch_distance;
    I64 speculative_decode_enabled;
    I64 prefix_cache_enabled;
};

I32 InferenceProfileStatusChecked(I64 *out_profile_id,
                                  U8 **out_profile_name,
                                  I64 *out_is_secure_default);

Bool QuantProfileIsBinary(I64 value)
{
    return (value == 0 || value == 1);
}

Bool QuantProfileIsValidProfileId(I64 profile_id)
{
    return (profile_id == QUANT_PROFILE_SECURE_LOCAL ||
            profile_id == QUANT_PROFILE_DEV_LOCAL);
}

Bool QuantProfileIsValidPreference(I64 preference)
{
    return (preference == QUANT_PROFILE_PREF_BALANCED ||
            preference == QUANT_PROFILE_PREF_THROUGHPUT ||
            preference == QUANT_PROFILE_PREF_ACCURACY);
}

I32 QuantProfileDefaultSelectionChecked(QuantProfileSelection *out_sel)
{
    if (!out_sel)
        return QUANT_PROFILE_ERR_NULL_PTR;

    out_sel->profile_id = QUANT_PROFILE_SECURE_LOCAL;
    out_sel->quant_mode = QUANT_MODE_Q4_0;
    out_sel->preferred_block_rows = 4;
    out_sel->prefetch_distance = 1;
    out_sel->speculative_decode_enabled = 0;
    out_sel->prefix_cache_enabled = 1;
    return QUANT_PROFILE_OK;
}

I32 QuantProfileSelectForProfileChecked(I64 profile_id,
                                        I64 preference,
                                        I64 secure_quarantine_gate_enabled,
                                        I64 secure_manifest_gate_enabled,
                                        QuantProfileSelection *out_sel)
{
    I32 status;

    if (!out_sel)
        return QUANT_PROFILE_ERR_NULL_PTR;

    if (!QuantProfileIsValidProfileId(profile_id) ||
        !QuantProfileIsValidPreference(preference))
        return QUANT_PROFILE_ERR_BAD_PARAM;

    if (!QuantProfileIsBinary(secure_quarantine_gate_enabled) ||
        !QuantProfileIsBinary(secure_manifest_gate_enabled))
        return QUANT_PROFILE_ERR_BAD_PARAM;

    // Hard policy: inference quant-profile dispatch is forbidden unless
    // quarantine + manifest verification gates remain enforced.
    if (!secure_quarantine_gate_enabled || !secure_manifest_gate_enabled)
        return QUANT_PROFILE_ERR_POLICY_GUARD;

    // Start from hardened defaults to preserve deterministic behavior.
    status = QuantProfileDefaultSelectionChecked(out_sel);
    if (status != QUANT_PROFILE_OK)
        return status;

    out_sel->profile_id = profile_id;

    if (profile_id == QUANT_PROFILE_SECURE_LOCAL)
    {
        // secure-local: deterministic and auditable by default.
        if (preference == QUANT_PROFILE_PREF_ACCURACY)
        {
            out_sel->quant_mode = QUANT_MODE_Q8_0;
            out_sel->preferred_block_rows = 2;
        }
        else if (preference == QUANT_PROFILE_PREF_THROUGHPUT)
        {
            out_sel->quant_mode = QUANT_MODE_Q4_0;
            out_sel->preferred_block_rows = 8;
        }
        else
        {
            out_sel->quant_mode = QUANT_MODE_Q4_0;
            out_sel->preferred_block_rows = 4;
        }

        // secure-local keeps speculative decode off for audit simplicity.
        out_sel->speculative_decode_enabled = 0;
        out_sel->prefix_cache_enabled = 1;
        out_sel->prefetch_distance = 1;
        return QUANT_PROFILE_OK;
    }

    if (profile_id == QUANT_PROFILE_DEV_LOCAL)
    {
        // dev-local remains local-only experimentation path. We permit
        // throughput-oriented knobs while retaining trust gates.
        if (preference == QUANT_PROFILE_PREF_ACCURACY)
        {
            out_sel->quant_mode = QUANT_MODE_Q8_0;
            out_sel->preferred_block_rows = 2;
            out_sel->prefetch_distance = 1;
            out_sel->speculative_decode_enabled = 0;
        }
        else if (preference == QUANT_PROFILE_PREF_THROUGHPUT)
        {
            out_sel->quant_mode = QUANT_MODE_MIXED_Q4Q8;
            out_sel->preferred_block_rows = 12;
            out_sel->prefetch_distance = 3;
            out_sel->speculative_decode_enabled = 1;
        }
        else
        {
            out_sel->quant_mode = QUANT_MODE_MIXED_Q4Q8;
            out_sel->preferred_block_rows = 8;
            out_sel->prefetch_distance = 2;
            out_sel->speculative_decode_enabled = 1;
        }

        out_sel->prefix_cache_enabled = 1;
        return QUANT_PROFILE_OK;
    }

    return QUANT_PROFILE_ERR_PROFILE_GUARD;
}

I32 QuantProfileSelectChecked(I64 preference,
                              I64 secure_quarantine_gate_enabled,
                              I64 secure_manifest_gate_enabled,
                              QuantProfileSelection *out_sel)
{
    I64 profile_id;
    U8 *profile_name;
    I64 is_secure_default;
    I32 status;

    if (!out_sel)
        return QUANT_PROFILE_ERR_NULL_PTR;

    status = InferenceProfileStatusChecked(&profile_id,
                                           &profile_name,
                                           &is_secure_default);
    if (status != 0)
        return QUANT_PROFILE_ERR_PROFILE_GUARD;

    return QuantProfileSelectForProfileChecked(profile_id,
                                               preference,
                                               secure_quarantine_gate_enabled,
                                               secure_manifest_gate_enabled,
                                               out_sel);
}
