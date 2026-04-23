// batch_scheduler.HC
// Continuous batching scheduler for high-throughput inference worker plane.
//
// Design goals:
// - Integer-only deterministic scheduling math.
// - Secure-local policy gate enforced before dispatch planning.
// - No external dependencies; explicit, auditable token accounting.

#define BATCH_SCHED_OK                    0
#define BATCH_SCHED_ERR_NULL_PTR          1
#define BATCH_SCHED_ERR_BAD_PARAM         2
#define BATCH_SCHED_ERR_OVERFLOW          3
#define BATCH_SCHED_ERR_POLICY_GUARD      4

#define BATCH_SCHED_PROFILE_SECURE_LOCAL  1
#define BATCH_SCHED_PROFILE_DEV_LOCAL     2

#define BATCH_SCHED_I64_MAX 0x7FFFFFFFFFFFFFFF

// Declared in src/runtime/profile.HC.
I32 InferenceProfileStatusChecked(I64 *out_profile_id,
                                  U8 **out_profile_name,
                                  I64 *out_is_secure_default);

class InferenceBatchRequest
{
    I64 request_id;
    I64 arrival_seq;
    I64 remaining_prefill_tokens;
    I64 remaining_decode_tokens;
    I64 state_flags;
};

// state_flags bit0: request active (1=eligible for scheduling).

I64 g_batch_sched_require_policy_digest_match = 1;
I64 g_batch_sched_require_attestation_ok = 1;

Bool BatchSchedFlagIsBinary(I64 value)
{
    return (value == 0 || value == 1);
}

Bool BatchSchedTryAddI64NonNeg(I64 lhs, I64 rhs, I64 *out_sum)
{
    if (!out_sum)
        return FALSE;
    if (lhs < 0 || rhs < 0)
        return FALSE;
    if (lhs > BATCH_SCHED_I64_MAX - rhs)
        return FALSE;

    *out_sum = lhs + rhs;
    return TRUE;
}

I32 BatchSchedulerSetSecurityGuardsChecked(I64 require_policy_digest_match,
                                           I64 require_attestation_ok)
{
    if (!BatchSchedFlagIsBinary(require_policy_digest_match) ||
        !BatchSchedFlagIsBinary(require_attestation_ok))
        return BATCH_SCHED_ERR_BAD_PARAM;

    g_batch_sched_require_policy_digest_match = require_policy_digest_match;
    g_batch_sched_require_attestation_ok = require_attestation_ok;
    return BATCH_SCHED_OK;
}

I32 BatchSchedulerPolicyGateChecked(I64 policy_digest_match,
                                    I64 attestation_ok)
{
    I64 profile_id;
    U8 *profile_name;
    I64 is_secure_default;
    I64 require_policy;
    I64 require_attestation;
    I32 status;

    if (!BatchSchedFlagIsBinary(policy_digest_match) ||
        !BatchSchedFlagIsBinary(attestation_ok))
        return BATCH_SCHED_ERR_BAD_PARAM;

    profile_id = 0;
    profile_name = NULL;
    is_secure_default = 0;
    status = InferenceProfileStatusChecked(&profile_id,
                                           &profile_name,
                                           &is_secure_default);
    if (status != 0)
        return BATCH_SCHED_ERR_POLICY_GUARD;

    if (profile_id != BATCH_SCHED_PROFILE_SECURE_LOCAL &&
        profile_id != BATCH_SCHED_PROFILE_DEV_LOCAL)
        return BATCH_SCHED_ERR_POLICY_GUARD;

    if (!BatchSchedFlagIsBinary(is_secure_default))
        return BATCH_SCHED_ERR_POLICY_GUARD;

    // secure-local is fail-closed: both controls must be active.
    require_policy = g_batch_sched_require_policy_digest_match;
    require_attestation = g_batch_sched_require_attestation_ok;
    if (profile_id == BATCH_SCHED_PROFILE_SECURE_LOCAL)
    {
        require_policy = 1;
        require_attestation = 1;
    }

    if ((require_policy && !policy_digest_match) ||
        (require_attestation && !attestation_ok))
        return BATCH_SCHED_ERR_POLICY_GUARD;

    return BATCH_SCHED_OK;
}

I64 BatchRequestHasWork(InferenceBatchRequest *req)
{
    I64 total;

    if (!req)
        return 0;
    if (!(req->state_flags & 1))
        return 0;
    if (req->remaining_prefill_tokens < 0 || req->remaining_decode_tokens < 0)
        return 0;

    if (!BatchSchedTryAddI64NonNeg(req->remaining_prefill_tokens,
                                   req->remaining_decode_tokens,
                                   &total))
        return 0;

    return (total > 0);
}

// Plan one continuous-batching step.
//
// Scheduling policy:
// 1) Select up to max_active_requests in stable FIFO array order.
// 2) Use round-robin prefill allocation (1 token per pass).
// 3) Spend remaining budget on round-robin decode allocation.
//
// This keeps prompt ingestion moving while preserving fairness.
I32 BatchSchedulerPlanStepChecked(
    InferenceBatchRequest *requests,
    I64 request_count,
    I64 max_active_requests,
    I64 token_budget,
    I64 policy_digest_match,
    I64 attestation_ok,
    I64 *out_request_indices,
    I64 *out_token_allocations,
    I64 *out_active_count,
    I64 *out_prefill_tokens,
    I64 *out_decode_tokens)
{
    I64 idx;
    I64 active_count;
    I64 budget_left;
    I64 planned_prefill;
    I64 planned_decode;
    I64 *staged_indices;
    I64 *staged_allocs;
    I64 *staged_prefill_left;
    I64 *staged_decode_left;
    I64 stage_bytes;
    I64 progressed;
    I64 out_i;
    I32 status;

    if (!requests || !out_request_indices || !out_token_allocations ||
        !out_active_count || !out_prefill_tokens || !out_decode_tokens)
        return BATCH_SCHED_ERR_NULL_PTR;

    if (request_count < 0 || max_active_requests < 0 || token_budget < 0)
        return BATCH_SCHED_ERR_BAD_PARAM;

    status = BatchSchedulerPolicyGateChecked(policy_digest_match,
                                             attestation_ok);
    if (status != BATCH_SCHED_OK)
        return status;

    if (max_active_requests > request_count)
        max_active_requests = request_count;

    if (!max_active_requests || !token_budget)
    {
        *out_active_count = 0;
        *out_prefill_tokens = 0;
        *out_decode_tokens = 0;
        return BATCH_SCHED_OK;
    }

    if (!BatchSchedTryAddI64NonNeg(max_active_requests,
                                   sizeof(I64),
                                   &stage_bytes))
        return BATCH_SCHED_ERR_OVERFLOW;

    staged_indices = MAlloc(stage_bytes);
    if (!staged_indices)
        return BATCH_SCHED_ERR_OVERFLOW;

    staged_allocs = MAlloc(stage_bytes);
    if (!staged_allocs)
    {
        Free(staged_indices);
        return BATCH_SCHED_ERR_OVERFLOW;
    }

    staged_prefill_left = MAlloc(stage_bytes);
    if (!staged_prefill_left)
    {
        Free(staged_allocs);
        Free(staged_indices);
        return BATCH_SCHED_ERR_OVERFLOW;
    }

    staged_decode_left = MAlloc(stage_bytes);
    if (!staged_decode_left)
    {
        Free(staged_prefill_left);
        Free(staged_allocs);
        Free(staged_indices);
        return BATCH_SCHED_ERR_OVERFLOW;
    }

    active_count = 0;
    for (idx = 0; idx < request_count; ++idx)
    {
        if (active_count >= max_active_requests)
            break;
        if (!BatchRequestHasWork(&requests[idx]))
            continue;

        staged_indices[active_count] = idx;
        staged_allocs[active_count] = 0;
        staged_prefill_left[active_count] = requests[idx].remaining_prefill_tokens;
        staged_decode_left[active_count] = requests[idx].remaining_decode_tokens;
        ++active_count;
    }

    if (!active_count)
    {
        Free(staged_decode_left);
        Free(staged_prefill_left);
        Free(staged_allocs);
        Free(staged_indices);
        *out_active_count = 0;
        *out_prefill_tokens = 0;
        *out_decode_tokens = 0;
        return BATCH_SCHED_OK;
    }

    budget_left = token_budget;
    planned_prefill = 0;
    planned_decode = 0;

    // Phase 1: prompt ingestion first.
    while (budget_left > 0)
    {
        progressed = 0;
        for (idx = 0; idx < active_count && budget_left > 0; ++idx)
        {
            if (staged_prefill_left[idx] <= 0)
                continue;

            staged_prefill_left[idx]--;
            staged_allocs[idx]++;
            budget_left--;
            planned_prefill++;
            progressed = 1;
        }

        if (!progressed)
            break;
    }

    // Phase 2: decode tokens from remaining budget.
    while (budget_left > 0)
    {
        progressed = 0;
        for (idx = 0; idx < active_count && budget_left > 0; ++idx)
        {
            if (staged_decode_left[idx] <= 0)
                continue;

            staged_decode_left[idx]--;
            staged_allocs[idx]++;
            budget_left--;
            planned_decode++;
            progressed = 1;
        }

        if (!progressed)
            break;
    }

    // Atomic publish of full plan.
    for (out_i = 0; out_i < active_count; ++out_i)
    {
        out_request_indices[out_i] = staged_indices[out_i];
        out_token_allocations[out_i] = staged_allocs[out_i];
    }

    *out_active_count = active_count;
    *out_prefill_tokens = planned_prefill;
    *out_decode_tokens = planned_decode;

    Free(staged_decode_left);
    Free(staged_prefill_left);
    Free(staged_allocs);
    Free(staged_indices);
    return BATCH_SCHED_OK;
}

// Apply scheduler allocations after kernel execution bookkeeping.
// Allocation is consumed from prefill first, then decode.
I32 BatchSchedulerAdvanceRequestsChecked(
    InferenceBatchRequest *requests,
    I64 request_count,
    I64 *request_indices,
    I64 *token_allocations,
    I64 active_count)
{
    I64 i;
    I64 req_idx;
    I64 alloc;
    I64 available;
    I64 consume_prefill;
    I64 consume_decode;

    if (!requests || !request_indices || !token_allocations)
        return BATCH_SCHED_ERR_NULL_PTR;
    if (request_count < 0 || active_count < 0)
        return BATCH_SCHED_ERR_BAD_PARAM;

    // Preflight all tuple bounds before mutating state.
    for (i = 0; i < active_count; ++i)
    {
        req_idx = request_indices[i];
        alloc = token_allocations[i];

        if (req_idx < 0 || req_idx >= request_count || alloc < 0)
            return BATCH_SCHED_ERR_BAD_PARAM;
        if (requests[req_idx].remaining_prefill_tokens < 0 ||
            requests[req_idx].remaining_decode_tokens < 0)
            return BATCH_SCHED_ERR_BAD_PARAM;

        if (!BatchSchedTryAddI64NonNeg(requests[req_idx].remaining_prefill_tokens,
                                       requests[req_idx].remaining_decode_tokens,
                                       &available))
            return BATCH_SCHED_ERR_OVERFLOW;
        if (alloc > available)
            return BATCH_SCHED_ERR_BAD_PARAM;
    }

    for (i = 0; i < active_count; ++i)
    {
        req_idx = request_indices[i];
        alloc = token_allocations[i];

        consume_prefill = alloc;
        if (consume_prefill > requests[req_idx].remaining_prefill_tokens)
            consume_prefill = requests[req_idx].remaining_prefill_tokens;

        requests[req_idx].remaining_prefill_tokens -= consume_prefill;
        consume_decode = alloc - consume_prefill;
        requests[req_idx].remaining_decode_tokens -= consume_decode;
    }

    return BATCH_SCHED_OK;
}
