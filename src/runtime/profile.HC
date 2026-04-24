// profile.HC
// Runtime security profile state for inference execution.
// secure-local is the mandatory default; dev-local requires explicit opt-in.

#define INFERENCE_PROFILE_OK            0
#define INFERENCE_PROFILE_ERR_NULL_PTR  1
#define INFERENCE_PROFILE_ERR_BAD_PARAM 2
#define INFERENCE_PROFILE_ERR_POLICY    3

#define INFERENCE_PROFILE_SECURE_LOCAL 1
#define INFERENCE_PROFILE_DEV_LOCAL    2

#define Q8_0_DOT_BENCH_SECURE_GATE_OK                      0
#define Q8_0_DOT_BENCH_SECURE_GATE_ERR_NULL_PTR            1
#define Q8_0_DOT_BENCH_SECURE_GATE_ERR_BAD_PARAM           2
#define Q8_0_DOT_BENCH_SECURE_GATE_ERR_PROFILE_NOT_SECURE  3
#define Q8_0_DOT_BENCH_SECURE_GATE_ERR_AUDIT_HOOKS_INACTIVE 4

// Global runtime profile state defaults to secure-local.
I64 g_inference_profile_mode = INFERENCE_PROFILE_SECURE_LOCAL;

I64 InferenceProfileGetCurrent()
{
    return g_inference_profile_mode;
}

U8 *InferenceProfileNameFromId(I64 profile_id)
{
    if (profile_id == INFERENCE_PROFILE_SECURE_LOCAL)
        return (U8 *)"secure-local";
    if (profile_id == INFERENCE_PROFILE_DEV_LOCAL)
        return (U8 *)"dev-local";
    return (U8 *)"unknown";
}

I32 InferenceProfileTrySetChecked(I64 requested_profile_id)
{
    if (requested_profile_id != INFERENCE_PROFILE_SECURE_LOCAL &&
        requested_profile_id != INFERENCE_PROFILE_DEV_LOCAL)
        return INFERENCE_PROFILE_ERR_BAD_PARAM;

    g_inference_profile_mode = requested_profile_id;
    return INFERENCE_PROFILE_OK;
}

// Explicit opt-in path for experimentation profile.
I32 InferenceProfileSetDevLocalChecked()
{
    return InferenceProfileTrySetChecked(INFERENCE_PROFILE_DEV_LOCAL);
}

// Return to hardened default profile.
I32 InferenceProfileSetSecureLocalChecked()
{
    return InferenceProfileTrySetChecked(INFERENCE_PROFILE_SECURE_LOCAL);
}

I32 InferenceProfileStatusChecked(I64 *out_profile_id,
                                  U8 **out_profile_name,
                                  I64 *out_is_secure_default)
{
    I64 active_profile;

    if (!out_profile_id || !out_profile_name || !out_is_secure_default)
        return INFERENCE_PROFILE_ERR_NULL_PTR;

    active_profile = InferenceProfileGetCurrent();

    if (active_profile != INFERENCE_PROFILE_SECURE_LOCAL &&
        active_profile != INFERENCE_PROFILE_DEV_LOCAL)
        return INFERENCE_PROFILE_ERR_BAD_PARAM;

    *out_profile_id = active_profile;
    *out_profile_name = InferenceProfileNameFromId(active_profile);
    *out_is_secure_default =
        (active_profile == INFERENCE_PROFILE_SECURE_LOCAL);

    return INFERENCE_PROFILE_OK;
}

// TempleOS CLI helper: `InferenceProfileStatus;`
// Returns the active profile id (1=secure-local, 2=dev-local).
I64 InferenceProfileStatus()
{
    return InferenceProfileGetCurrent();
}

I32 Q8_0DotBenchRunDefaultSuiteSecureLocalAuditGate(
    I64 profile_id,
    I64 audit_hooks_active,
    I64 *out_gate_passed,
    I64 *out_policy_error_code)
{
    I64 staged_gate_passed;
    I64 staged_policy_error_code;

    if (!out_gate_passed || !out_policy_error_code)
        return Q8_0_DOT_BENCH_SECURE_GATE_ERR_NULL_PTR;

    if ((profile_id != INFERENCE_PROFILE_SECURE_LOCAL) &&
        (profile_id != INFERENCE_PROFILE_DEV_LOCAL))
        return Q8_0_DOT_BENCH_SECURE_GATE_ERR_BAD_PARAM;

    if ((audit_hooks_active != 0) && (audit_hooks_active != 1))
        return Q8_0_DOT_BENCH_SECURE_GATE_ERR_BAD_PARAM;

    staged_gate_passed = 0;
    staged_policy_error_code = Q8_0_DOT_BENCH_SECURE_GATE_OK;

    if (profile_id != INFERENCE_PROFILE_SECURE_LOCAL)
    {
        staged_policy_error_code =
            Q8_0_DOT_BENCH_SECURE_GATE_ERR_PROFILE_NOT_SECURE;
    }
    else if (!audit_hooks_active)
    {
        staged_policy_error_code =
            Q8_0_DOT_BENCH_SECURE_GATE_ERR_AUDIT_HOOKS_INACTIVE;
    }
    else
    {
        staged_gate_passed = 1;
        staged_policy_error_code = Q8_0_DOT_BENCH_SECURE_GATE_OK;
    }

    *out_gate_passed = staged_gate_passed;
    *out_policy_error_code = staged_policy_error_code;
    return Q8_0_DOT_BENCH_SECURE_GATE_OK;
}
