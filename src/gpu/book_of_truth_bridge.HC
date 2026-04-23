// book_of_truth_bridge.HC
// GPU -> Book-of-Truth event bridge.
//
// Design goals:
// - Integer-only, deterministic event encoding.
// - No hidden side effects: caller supplies storage and receives exact outputs.
// - Fail closed on invalid type/op/capacity/range state.
//
// This bridge records three security-critical GPU event classes:
// - DMA mapping lifecycle (map/update/unmap)
// - MMIO writes (allowlisted register transactions)
// - Dispatch submissions (descriptor + queue metadata)

#define BOT_GPU_BRIDGE_OK                    0
#define BOT_GPU_BRIDGE_ERR_NULL_PTR          1
#define BOT_GPU_BRIDGE_ERR_BAD_PARAM         2
#define BOT_GPU_BRIDGE_ERR_CAPACITY          3
#define BOT_GPU_BRIDGE_ERR_EVENT_TYPE        4
#define BOT_GPU_BRIDGE_ERR_EVENT_OP          5
#define BOT_GPU_BRIDGE_ERR_SEQUENCE_OVERFLOW 6

#define BOT_GPU_EVENT_DMA      1
#define BOT_GPU_EVENT_MMIO     2
#define BOT_GPU_EVENT_DISPATCH 3

#define BOT_GPU_DMA_MAP    1
#define BOT_GPU_DMA_UPDATE 2
#define BOT_GPU_DMA_UNMAP  3

#define BOT_GPU_MMIO_WRITE 1

#define BOT_GPU_DISPATCH_SUBMIT    1
#define BOT_GPU_DISPATCH_COMPLETE  2
#define BOT_GPU_DISPATCH_TIMEOUT   3

class BOTGPUEvent
{
    I64 seq_id;
    I64 event_type;
    I64 event_op;
    I64 arg0;
    I64 arg1;
    I64 arg2;
    I64 arg3;
};

class BOTGPUBridge
{
    BOTGPUEvent *events;
    I64 capacity;
    I64 count;
    I64 head;
    I64 next_seq_id;
};

Bool BOTGPUBridgeEventTypeIsValid(I64 event_type)
{
    return (event_type == BOT_GPU_EVENT_DMA ||
            event_type == BOT_GPU_EVENT_MMIO ||
            event_type == BOT_GPU_EVENT_DISPATCH);
}

Bool BOTGPUBridgeEventOpIsValid(I64 event_type, I64 event_op)
{
    if (event_type == BOT_GPU_EVENT_DMA)
    {
        return (event_op == BOT_GPU_DMA_MAP ||
                event_op == BOT_GPU_DMA_UPDATE ||
                event_op == BOT_GPU_DMA_UNMAP);
    }

    if (event_type == BOT_GPU_EVENT_MMIO)
        return (event_op == BOT_GPU_MMIO_WRITE);

    if (event_type == BOT_GPU_EVENT_DISPATCH)
    {
        return (event_op == BOT_GPU_DISPATCH_SUBMIT ||
                event_op == BOT_GPU_DISPATCH_COMPLETE ||
                event_op == BOT_GPU_DISPATCH_TIMEOUT);
    }

    return FALSE;
}

I32 BOTGPUBridgeInit(BOTGPUBridge *bridge,
                     BOTGPUEvent *storage,
                     I64 capacity)
{
    if (!bridge || !storage)
        return BOT_GPU_BRIDGE_ERR_NULL_PTR;

    if (capacity <= 0)
        return BOT_GPU_BRIDGE_ERR_CAPACITY;

    bridge->events = storage;
    bridge->capacity = capacity;
    bridge->count = 0;
    bridge->head = 0;
    bridge->next_seq_id = 1;

    return BOT_GPU_BRIDGE_OK;
}

I32 BOTGPUBridgeAppendChecked(BOTGPUBridge *bridge,
                              I64 event_type,
                              I64 event_op,
                              I64 arg0,
                              I64 arg1,
                              I64 arg2,
                              I64 arg3,
                              I64 *out_seq_id)
{
    I64 slot_idx;
    BOTGPUEvent *slot;

    if (!bridge || !out_seq_id)
        return BOT_GPU_BRIDGE_ERR_NULL_PTR;

    *out_seq_id = 0;

    if (!bridge->events)
        return BOT_GPU_BRIDGE_ERR_NULL_PTR;

    if (bridge->capacity <= 0)
        return BOT_GPU_BRIDGE_ERR_CAPACITY;

    if (bridge->count < 0 || bridge->count > bridge->capacity ||
        bridge->head < 0 || bridge->head >= bridge->capacity)
        return BOT_GPU_BRIDGE_ERR_BAD_PARAM;

    if (!BOTGPUBridgeEventTypeIsValid(event_type))
        return BOT_GPU_BRIDGE_ERR_EVENT_TYPE;

    if (!BOTGPUBridgeEventOpIsValid(event_type, event_op))
        return BOT_GPU_BRIDGE_ERR_EVENT_OP;

    if (bridge->next_seq_id <= 0)
        return BOT_GPU_BRIDGE_ERR_SEQUENCE_OVERFLOW;

    slot_idx = bridge->head;
    slot = &bridge->events[slot_idx];

    slot->seq_id = bridge->next_seq_id;
    slot->event_type = event_type;
    slot->event_op = event_op;
    slot->arg0 = arg0;
    slot->arg1 = arg1;
    slot->arg2 = arg2;
    slot->arg3 = arg3;

    *out_seq_id = bridge->next_seq_id;
    bridge->next_seq_id += 1;

    // Ring progression: modulo arithmetic keeps writes bounded to
    // [0, capacity), overwriting the oldest event when full.
    bridge->head += 1;
    if (bridge->head >= bridge->capacity)
        bridge->head = 0;

    if (bridge->count < bridge->capacity)
        bridge->count += 1;

    return BOT_GPU_BRIDGE_OK;
}

I32 BOTGPUBridgeRecordDMA(BOTGPUBridge *bridge,
                          I64 dma_op,
                          I64 lease_id,
                          I64 phys_addr,
                          I64 nbytes,
                          I64 iommu_domain,
                          I64 *out_seq_id)
{
    return BOTGPUBridgeAppendChecked(bridge,
                                     BOT_GPU_EVENT_DMA,
                                     dma_op,
                                     lease_id,
                                     phys_addr,
                                     nbytes,
                                     iommu_domain,
                                     out_seq_id);
}

I32 BOTGPUBridgeRecordMMIOWrite(BOTGPUBridge *bridge,
                                I64 bar_index,
                                I64 reg_offset,
                                I64 value,
                                I64 width_bytes,
                                I64 *out_seq_id)
{
    return BOTGPUBridgeAppendChecked(bridge,
                                     BOT_GPU_EVENT_MMIO,
                                     BOT_GPU_MMIO_WRITE,
                                     bar_index,
                                     reg_offset,
                                     value,
                                     width_bytes,
                                     out_seq_id);
}

I32 BOTGPUBridgeRecordDispatch(BOTGPUBridge *bridge,
                               I64 dispatch_op,
                               I64 queue_id,
                               I64 descriptor_addr,
                               I64 descriptor_bytes,
                               I64 fence_id,
                               I64 *out_seq_id)
{
    return BOTGPUBridgeAppendChecked(bridge,
                                     BOT_GPU_EVENT_DISPATCH,
                                     dispatch_op,
                                     queue_id,
                                     descriptor_addr,
                                     descriptor_bytes,
                                     fence_id,
                                     out_seq_id);
}

