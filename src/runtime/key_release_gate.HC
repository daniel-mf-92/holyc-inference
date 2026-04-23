// key_release_gate.HC
// IQ-1272: Key-release handshake verifier for trusted runtime unlock.
//
// Contract:
// - secure-local remains default and fail-closed.
// - release is granted only if all three controls are true:
//   1) TempleOS signed approval present
//   2) Attestation evidence valid
//   3) Policy digest parity valid
// - Integer-only HolyC logic; no external dependencies.

#define INFERENCE_KEY_RELEASE_OK               0
#define INFERENCE_KEY_RELEASE_ERR_NULL_PTR     1
#define INFERENCE_KEY_RELEASE_ERR_BAD_PARAM    2
#define INFERENCE_KEY_RELEASE_ERR_POLICY_GUARD 3

#define INFERENCE_PROFILE_SECURE_LOCAL 1
#define INFERENCE_PROFILE_DEV_LOCAL    2

I32 InferenceProfileStatusChecked(I64 *out_profile_id,
                                  U8 **out_profile_name,
                                  I64 *out_is_secure_default);

Bool InferenceKeyReleaseFlagIsBinary(I64 value)
{
    return (value == 0 || value == 1);
}

I32 InferenceKeyReleaseHandshakeVerifyChecked(I64 templeos_signed_approval,
                                              I64 attestation_evidence_valid,
                                              I64 policy_digest_parity_valid,
                                              I64 *out_release_allowed,
                                              I64 *out_failure_bits,
                                              I64 *out_profile_id)
{
    I64 profile_id;
    U8 *profile_name;
    I64 is_secure_default;
    I64 failure_bits;
    I32 status;

    if (!out_release_allowed || !out_failure_bits || !out_profile_id)
        return INFERENCE_KEY_RELEASE_ERR_NULL_PTR;

    *out_release_allowed = 0;
    *out_failure_bits = 0;
    *out_profile_id = 0;

    if (!InferenceKeyReleaseFlagIsBinary(templeos_signed_approval) ||
        !InferenceKeyReleaseFlagIsBinary(attestation_evidence_valid) ||
        !InferenceKeyReleaseFlagIsBinary(policy_digest_parity_valid))
        return INFERENCE_KEY_RELEASE_ERR_BAD_PARAM;

    profile_id = 0;
    profile_name = NULL;
    is_secure_default = 0;

    status = InferenceProfileStatusChecked(&profile_id,
                                           &profile_name,
                                           &is_secure_default);
    if (status != 0)
        return INFERENCE_KEY_RELEASE_ERR_POLICY_GUARD;

    if (profile_id != INFERENCE_PROFILE_SECURE_LOCAL &&
        profile_id != INFERENCE_PROFILE_DEV_LOCAL)
        return INFERENCE_KEY_RELEASE_ERR_POLICY_GUARD;

    if (!InferenceKeyReleaseFlagIsBinary(is_secure_default))
        return INFERENCE_KEY_RELEASE_ERR_POLICY_GUARD;

    // Fail bits are human-auditable and deterministic:
    // bit0: missing signed approval
    // bit1: invalid attestation evidence
    // bit2: policy digest parity mismatch
    // bit3: secure-default/profile drift (must be secure-local default)
    failure_bits = 0;
    failure_bits |= ((!templeos_signed_approval) << 0);
    failure_bits |= ((!attestation_evidence_valid) << 1);
    failure_bits |= ((!policy_digest_parity_valid) << 2);
    failure_bits |= ((!is_secure_default) << 3);

    // Strong gate in secure-local: all three proofs required.
    // dev-local remains explicit opt-in but cannot bypass handshake.
    *out_release_allowed = (failure_bits == 0);
    *out_failure_bits = failure_bits;
    *out_profile_id = profile_id;

    return INFERENCE_KEY_RELEASE_OK;
}

// TempleOS CLI helper: `InferenceKeyReleaseStatus;`
// Returns 1 only when handshake passes strict secure policy checks.
I64 InferenceKeyReleaseStatus(I64 templeos_signed_approval,
                              I64 attestation_evidence_valid,
                              I64 policy_digest_parity_valid)
{
    I64 release_allowed;
    I64 failure_bits;
    I64 profile_id;
    I32 status;

    release_allowed = 0;
    failure_bits = 0;
    profile_id = 0;

    status = InferenceKeyReleaseHandshakeVerifyChecked(templeos_signed_approval,
                                                       attestation_evidence_valid,
                                                       policy_digest_parity_valid,
                                                       &release_allowed,
                                                       &failure_bits,
                                                       &profile_id);
    if (status != INFERENCE_KEY_RELEASE_OK)
        return 0;

    return release_allowed;
}
