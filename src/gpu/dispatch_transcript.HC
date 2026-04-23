// dispatch_transcript.HC
// Deterministic GPU dispatch transcript recorder (WS9-29).
//
// Security contract:
// - Every dispatch gets a monotonic sequence ID and deterministic hash.
// - Transcript chain hash detects dropped/reordered/tampered entries.
// - Integer-only checked arithmetic; fail closed on invalid geometry/state.
// - Designed for Book-of-Truth export and offline replay verification.

#define GPU_DISPATCH_TX_OK                    0
#define GPU_DISPATCH_TX_ERR_NULL_PTR          1
#define GPU_DISPATCH_TX_ERR_BAD_PARAM         2
#define GPU_DISPATCH_TX_ERR_BAD_STATE         3
#define GPU_DISPATCH_TX_ERR_CAPACITY          4
#define GPU_DISPATCH_TX_ERR_OVERFLOW          5
#define GPU_DISPATCH_TX_ERR_ALIGNMENT         6
#define GPU_DISPATCH_TX_ERR_SEQUENCE_OVERFLOW 7
#define GPU_DISPATCH_TX_ERR_NOT_FOUND         8

#define GPU_DISPATCH_TX_ALIGN_BYTES 16
#define GPU_DISPATCH_TX_I64_MAX     0x7FFFFFFFFFFFFFFF

class GPUDispatchTranscriptEntry
{
    I64 seq_id;
    I64 queue_id;
    I64 queue_depth_before;
    I64 queue_depth_after;

    I64 descriptor_addr;
    I64 descriptor_bytes;
    I64 descriptor_type;
    I64 descriptor_op;

    I64 cycle_start;
    I64 cycle_end;
    I64 cycle_delta;

    I64 status_code;
    I64 entry_hash;
    I64 chain_hash;
};

class GPUDispatchTranscript
{
    GPUDispatchTranscriptEntry *entries;
    I64 capacity;
    I64 count;
    I64 head;
    I64 next_seq_id;

    I64 session_nonce;
    I64 chain_hash;

    I64 total_descriptor_bytes;
    I64 total_cycle_delta;
};

Bool GPUDispatchTranscriptIsAlignedPow2(I64 value,
                                        I64 alignment)
{
    if (alignment <= 0)
        return FALSE;

    return ((value & (alignment - 1)) == 0);
}

Bool GPUDispatchTranscriptAddChecked(I64 lhs,
                                     I64 rhs,
                                     I64 *out_sum)
{
    if (!out_sum)
        return FALSE;

    if (lhs < 0 || rhs < 0)
        return FALSE;

    if (lhs > (GPU_DISPATCH_TX_I64_MAX - rhs))
        return FALSE;

    *out_sum = lhs + rhs;
    return TRUE;
}

I64 GPUDispatchTranscriptMix64(I64 seed,
                               I64 value)
{
    I64 x;

    // SplitMix64-style integer mixer: deterministic, no floating point,
    // useful for transcript hashes and parity signatures.
    x = seed ^ value;
    x ^= (x >> 30);
    x *= 0xBF58476D1CE4E5B9;
    x ^= (x >> 27);
    x *= 0x94D049BB133111EB;
    x ^= (x >> 31);

    if (x == 0)
        x = 0x9E3779B97F4A7C15;

    return x;
}

I64 GPUDispatchTranscriptComputeEntryHash(I64 session_nonce,
                                          I64 seq_id,
                                          I64 queue_id,
                                          I64 queue_depth_before,
                                          I64 queue_depth_after,
                                          I64 descriptor_addr,
                                          I64 descriptor_bytes,
                                          I64 descriptor_type,
                                          I64 descriptor_op,
                                          I64 cycle_start,
                                          I64 cycle_end,
                                          I64 status_code)
{
    I64 h;

    h = GPUDispatchTranscriptMix64(session_nonce, seq_id);
    h = GPUDispatchTranscriptMix64(h, queue_id);
    h = GPUDispatchTranscriptMix64(h, queue_depth_before);
    h = GPUDispatchTranscriptMix64(h, queue_depth_after);
    h = GPUDispatchTranscriptMix64(h, descriptor_addr);
    h = GPUDispatchTranscriptMix64(h, descriptor_bytes);
    h = GPUDispatchTranscriptMix64(h, descriptor_type);
    h = GPUDispatchTranscriptMix64(h, descriptor_op);
    h = GPUDispatchTranscriptMix64(h, cycle_start);
    h = GPUDispatchTranscriptMix64(h, cycle_end);
    h = GPUDispatchTranscriptMix64(h, status_code);

    return h;
}

Bool GPUDispatchTranscriptStateIsValid(const GPUDispatchTranscript *transcript)
{
    if (!transcript)
        return FALSE;

    if (!transcript->entries)
        return FALSE;

    if (transcript->capacity <= 0)
        return FALSE;

    if (transcript->count < 0 || transcript->count > transcript->capacity)
        return FALSE;

    if (transcript->head < 0 || transcript->head >= transcript->capacity)
        return FALSE;

    if (transcript->next_seq_id <= 0)
        return FALSE;

    if (transcript->total_descriptor_bytes < 0 ||
        transcript->total_cycle_delta < 0)
        return FALSE;

    return TRUE;
}

I32 GPUDispatchTranscriptInit(GPUDispatchTranscript *transcript,
                              GPUDispatchTranscriptEntry *storage,
                              I64 capacity,
                              I64 session_nonce)
{
    I64 i;

    if (!transcript || !storage)
        return GPU_DISPATCH_TX_ERR_NULL_PTR;

    if (capacity <= 0)
        return GPU_DISPATCH_TX_ERR_CAPACITY;

    transcript->entries = storage;
    transcript->capacity = capacity;
    transcript->count = 0;
    transcript->head = 0;
    transcript->next_seq_id = 1;
    transcript->session_nonce = session_nonce;
    transcript->chain_hash = GPUDispatchTranscriptMix64(session_nonce,
                                                        0xD15A7C7001);
    transcript->total_descriptor_bytes = 0;
    transcript->total_cycle_delta = 0;

    for (i = 0; i < capacity; ++i)
    {
        storage[i].seq_id = 0;
        storage[i].queue_id = 0;
        storage[i].queue_depth_before = 0;
        storage[i].queue_depth_after = 0;
        storage[i].descriptor_addr = 0;
        storage[i].descriptor_bytes = 0;
        storage[i].descriptor_type = 0;
        storage[i].descriptor_op = 0;
        storage[i].cycle_start = 0;
        storage[i].cycle_end = 0;
        storage[i].cycle_delta = 0;
        storage[i].status_code = 0;
        storage[i].entry_hash = 0;
        storage[i].chain_hash = 0;
    }

    return GPU_DISPATCH_TX_OK;
}

I32 GPUDispatchTranscriptRecordChecked(GPUDispatchTranscript *transcript,
                                       I64 queue_id,
                                       I64 queue_depth_before,
                                       I64 queue_depth_after,
                                       I64 descriptor_addr,
                                       I64 descriptor_bytes,
                                       I64 descriptor_type,
                                       I64 descriptor_op,
                                       I64 cycle_start,
                                       I64 cycle_end,
                                       I64 status_code,
                                       I64 *out_seq_id,
                                       I64 *out_entry_hash,
                                       I64 *out_chain_hash)
{
    I64 slot_idx;
    I64 seq_id;
    I64 cycle_delta;
    I64 entry_hash;
    I64 chain_hash;
    I64 total_descriptor_bytes;
    I64 total_cycle_delta;
    GPUDispatchTranscriptEntry *slot;

    if (!transcript || !out_seq_id || !out_entry_hash || !out_chain_hash)
        return GPU_DISPATCH_TX_ERR_NULL_PTR;

    *out_seq_id = 0;
    *out_entry_hash = 0;
    *out_chain_hash = 0;

    if (!GPUDispatchTranscriptStateIsValid(transcript))
        return GPU_DISPATCH_TX_ERR_BAD_STATE;

    if (queue_id < 0 || queue_depth_before < 0 || queue_depth_after < 0 ||
        descriptor_addr < 0 || descriptor_bytes <= 0 ||
        descriptor_type <= 0 || descriptor_op <= 0 ||
        cycle_start < 0 || cycle_end < 0 ||
        status_code < 0)
        return GPU_DISPATCH_TX_ERR_BAD_PARAM;

    if (cycle_end < cycle_start)
        return GPU_DISPATCH_TX_ERR_BAD_PARAM;

    if (!GPUDispatchTranscriptIsAlignedPow2(descriptor_addr,
                                            GPU_DISPATCH_TX_ALIGN_BYTES) ||
        !GPUDispatchTranscriptIsAlignedPow2(descriptor_bytes,
                                            GPU_DISPATCH_TX_ALIGN_BYTES))
        return GPU_DISPATCH_TX_ERR_ALIGNMENT;

    cycle_delta = cycle_end - cycle_start;

    if (transcript->next_seq_id == GPU_DISPATCH_TX_I64_MAX)
        return GPU_DISPATCH_TX_ERR_SEQUENCE_OVERFLOW;

    seq_id = transcript->next_seq_id;

    entry_hash = GPUDispatchTranscriptComputeEntryHash(transcript->session_nonce,
                                                       seq_id,
                                                       queue_id,
                                                       queue_depth_before,
                                                       queue_depth_after,
                                                       descriptor_addr,
                                                       descriptor_bytes,
                                                       descriptor_type,
                                                       descriptor_op,
                                                       cycle_start,
                                                       cycle_end,
                                                       status_code);

    chain_hash = GPUDispatchTranscriptMix64(transcript->chain_hash,
                                            entry_hash ^ seq_id);

    if (!GPUDispatchTranscriptAddChecked(transcript->total_descriptor_bytes,
                                         descriptor_bytes,
                                         &total_descriptor_bytes))
        return GPU_DISPATCH_TX_ERR_OVERFLOW;

    if (!GPUDispatchTranscriptAddChecked(transcript->total_cycle_delta,
                                         cycle_delta,
                                         &total_cycle_delta))
        return GPU_DISPATCH_TX_ERR_OVERFLOW;

    slot_idx = transcript->head;
    slot = &transcript->entries[slot_idx];

    // Commit after full preflight/checksum stage.
    slot->seq_id = seq_id;
    slot->queue_id = queue_id;
    slot->queue_depth_before = queue_depth_before;
    slot->queue_depth_after = queue_depth_after;
    slot->descriptor_addr = descriptor_addr;
    slot->descriptor_bytes = descriptor_bytes;
    slot->descriptor_type = descriptor_type;
    slot->descriptor_op = descriptor_op;
    slot->cycle_start = cycle_start;
    slot->cycle_end = cycle_end;
    slot->cycle_delta = cycle_delta;
    slot->status_code = status_code;
    slot->entry_hash = entry_hash;
    slot->chain_hash = chain_hash;

    transcript->next_seq_id += 1;
    transcript->chain_hash = chain_hash;
    transcript->total_descriptor_bytes = total_descriptor_bytes;
    transcript->total_cycle_delta = total_cycle_delta;

    transcript->head += 1;
    if (transcript->head >= transcript->capacity)
        transcript->head = 0;

    if (transcript->count < transcript->capacity)
        transcript->count += 1;

    *out_seq_id = seq_id;
    *out_entry_hash = entry_hash;
    *out_chain_hash = chain_hash;

    return GPU_DISPATCH_TX_OK;
}

I32 GPUDispatchTranscriptSnapshotChecked(const GPUDispatchTranscript *transcript,
                                         I64 *out_count,
                                         I64 *out_head,
                                         I64 *out_next_seq_id,
                                         I64 *out_chain_hash,
                                         I64 *out_total_descriptor_bytes,
                                         I64 *out_total_cycle_delta)
{
    if (!transcript || !out_count || !out_head || !out_next_seq_id ||
        !out_chain_hash || !out_total_descriptor_bytes ||
        !out_total_cycle_delta)
        return GPU_DISPATCH_TX_ERR_NULL_PTR;

    if (!GPUDispatchTranscriptStateIsValid(transcript))
        return GPU_DISPATCH_TX_ERR_BAD_STATE;

    *out_count = transcript->count;
    *out_head = transcript->head;
    *out_next_seq_id = transcript->next_seq_id;
    *out_chain_hash = transcript->chain_hash;
    *out_total_descriptor_bytes = transcript->total_descriptor_bytes;
    *out_total_cycle_delta = transcript->total_cycle_delta;

    return GPU_DISPATCH_TX_OK;
}

I32 GPUDispatchTranscriptGetBySeqChecked(const GPUDispatchTranscript *transcript,
                                         I64 seq_id,
                                         GPUDispatchTranscriptEntry *out_entry)
{
    I64 i;

    if (!transcript || !out_entry)
        return GPU_DISPATCH_TX_ERR_NULL_PTR;

    if (!GPUDispatchTranscriptStateIsValid(transcript))
        return GPU_DISPATCH_TX_ERR_BAD_STATE;

    if (seq_id <= 0)
        return GPU_DISPATCH_TX_ERR_BAD_PARAM;

    for (i = 0; i < transcript->capacity; ++i)
    {
        if (transcript->entries[i].seq_id != seq_id)
            continue;

        *out_entry = transcript->entries[i];
        return GPU_DISPATCH_TX_OK;
    }

    return GPU_DISPATCH_TX_ERR_NOT_FOUND;
}
