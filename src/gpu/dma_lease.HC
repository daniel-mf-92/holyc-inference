// dma_lease.HC
// DMA lease-token manager for GPU dispatch hardening.
//
// Security goals:
// - Leases are explicit capabilities: {lease_id, lease_token}.
// - Authorize path fails closed on token/domain/range mismatch.
// - Overlapping active physical ranges are rejected.
// - Integer-only checked arithmetic; no partial state publish on failure.

#define GPU_DMA_LEASE_OK                    0
#define GPU_DMA_LEASE_ERR_NULL_PTR          1
#define GPU_DMA_LEASE_ERR_BAD_PARAM         2
#define GPU_DMA_LEASE_ERR_BAD_STATE         3
#define GPU_DMA_LEASE_ERR_CAPACITY          4
#define GPU_DMA_LEASE_ERR_OVERFLOW          5
#define GPU_DMA_LEASE_ERR_NO_FREE_SLOT      6
#define GPU_DMA_LEASE_ERR_RANGE_OVERLAP     7
#define GPU_DMA_LEASE_ERR_NOT_FOUND         8
#define GPU_DMA_LEASE_ERR_TOKEN_MISMATCH    9
#define GPU_DMA_LEASE_ERR_DOMAIN_MISMATCH  10
#define GPU_DMA_LEASE_ERR_RANGE_OOB        11

#define GPU_DMA_LEASE_TOKEN_SALT 0x6A09E667F3BCC909
#define GPU_DMA_LEASE_I64_MAX    0x7FFFFFFFFFFFFFFF

class GPUDMALease
{
    I64 active;
    I64 lease_id;
    I64 lease_token;
    I64 phys_addr;
    I64 nbytes;
    I64 iommu_domain;
    I64 owner_tag;
};

class GPUDMALeaseManager
{
    GPUDMALease *leases;
    I64 capacity;
    I64 count;
    I64 next_lease_id;
};

Bool GPUDMALeaseAddChecked(I64 lhs, I64 rhs, I64 *out_sum)
{
    if (!out_sum)
        return FALSE;

    if (lhs < 0 || rhs < 0)
        return FALSE;

    if (lhs > (GPU_DMA_LEASE_I64_MAX - rhs))
        return FALSE;

    *out_sum = lhs + rhs;
    return TRUE;
}

Bool GPUDMALeaseManagerStateIsValid(const GPUDMALeaseManager *mgr)
{
    if (!mgr)
        return FALSE;

    if (!mgr->leases)
        return FALSE;

    if (mgr->capacity <= 0)
        return FALSE;

    if (mgr->count < 0 || mgr->count > mgr->capacity)
        return FALSE;

    if (mgr->next_lease_id <= 0)
        return FALSE;

    return TRUE;
}

Bool GPUDMALeaseRangesOverlap(I64 lhs_addr,
                              I64 lhs_nbytes,
                              I64 rhs_addr,
                              I64 rhs_nbytes)
{
    I64 lhs_end;
    I64 rhs_end;

    if (!GPUDMALeaseAddChecked(lhs_addr, lhs_nbytes, &lhs_end))
        return TRUE;

    if (!GPUDMALeaseAddChecked(rhs_addr, rhs_nbytes, &rhs_end))
        return TRUE;

    // Half-open interval overlap check:
    // [lhs_addr, lhs_end) intersects [rhs_addr, rhs_end)
    // iff lhs_addr < rhs_end && rhs_addr < lhs_end.
    if (lhs_addr < rhs_end && rhs_addr < lhs_end)
        return TRUE;

    return FALSE;
}

I64 GPUDMALeaseBuildToken(I64 lease_id,
                          I64 owner_tag,
                          I64 phys_addr,
                          I64 nbytes,
                          I64 iommu_domain)
{
    I64 token;

    // Deterministic integer mixing. Caller treats token as opaque capability.
    token = lease_id * 6364136223846793005;
    token ^= (owner_tag * 1442695040888963407);
    token ^= (phys_addr << 7);
    token ^= (nbytes << 17);
    token ^= (iommu_domain << 29);
    token ^= GPU_DMA_LEASE_TOKEN_SALT;

    if (token == 0)
        token = GPU_DMA_LEASE_TOKEN_SALT;

    return token;
}

I32 GPUDMALeaseInit(GPUDMALeaseManager *mgr,
                    GPUDMALease *storage,
                    I64 capacity)
{
    I64 i;

    if (!mgr || !storage)
        return GPU_DMA_LEASE_ERR_NULL_PTR;

    if (capacity <= 0)
        return GPU_DMA_LEASE_ERR_CAPACITY;

    mgr->leases = storage;
    mgr->capacity = capacity;
    mgr->count = 0;
    mgr->next_lease_id = 1;

    for (i = 0; i < capacity; ++i)
    {
        storage[i].active = 0;
        storage[i].lease_id = 0;
        storage[i].lease_token = 0;
        storage[i].phys_addr = 0;
        storage[i].nbytes = 0;
        storage[i].iommu_domain = 0;
        storage[i].owner_tag = 0;
    }

    return GPU_DMA_LEASE_OK;
}

I32 GPUDMALeaseAcquireChecked(GPUDMALeaseManager *mgr,
                              I64 phys_addr,
                              I64 nbytes,
                              I64 iommu_domain,
                              I64 owner_tag,
                              I64 *out_lease_id,
                              I64 *out_lease_token)
{
    I64 i;
    I64 free_idx;
    I64 lease_id;
    I64 lease_token;
    I64 range_end;

    if (!mgr || !out_lease_id || !out_lease_token)
        return GPU_DMA_LEASE_ERR_NULL_PTR;

    *out_lease_id = 0;
    *out_lease_token = 0;

    if (!GPUDMALeaseManagerStateIsValid(mgr))
        return GPU_DMA_LEASE_ERR_BAD_STATE;

    if (phys_addr < 0 || nbytes <= 0 || iommu_domain < 0 || owner_tag < 0)
        return GPU_DMA_LEASE_ERR_BAD_PARAM;

    if (!GPUDMALeaseAddChecked(phys_addr, nbytes, &range_end))
        return GPU_DMA_LEASE_ERR_OVERFLOW;

    free_idx = -1;

    for (i = 0; i < mgr->capacity; ++i)
    {
        if (!mgr->leases[i].active)
        {
            if (free_idx < 0)
                free_idx = i;
            continue;
        }

        if (GPUDMALeaseRangesOverlap(phys_addr,
                                     nbytes,
                                     mgr->leases[i].phys_addr,
                                     mgr->leases[i].nbytes))
            return GPU_DMA_LEASE_ERR_RANGE_OVERLAP;
    }

    if (free_idx < 0)
        return GPU_DMA_LEASE_ERR_NO_FREE_SLOT;

    lease_id = mgr->next_lease_id;
    if (lease_id <= 0)
        return GPU_DMA_LEASE_ERR_OVERFLOW;

    lease_token = GPUDMALeaseBuildToken(lease_id,
                                        owner_tag,
                                        phys_addr,
                                        nbytes,
                                        iommu_domain);

    // Commit after full preflight.
    mgr->leases[free_idx].active = 1;
    mgr->leases[free_idx].lease_id = lease_id;
    mgr->leases[free_idx].lease_token = lease_token;
    mgr->leases[free_idx].phys_addr = phys_addr;
    mgr->leases[free_idx].nbytes = nbytes;
    mgr->leases[free_idx].iommu_domain = iommu_domain;
    mgr->leases[free_idx].owner_tag = owner_tag;

    mgr->count += 1;
    mgr->next_lease_id += 1;

    *out_lease_id = lease_id;
    *out_lease_token = lease_token;
    return GPU_DMA_LEASE_OK;
}

I32 GPUDMALeaseFindActiveIndexChecked(const GPUDMALeaseManager *mgr,
                                      I64 lease_id,
                                      I64 *out_index)
{
    I64 i;

    if (!mgr || !out_index)
        return GPU_DMA_LEASE_ERR_NULL_PTR;

    *out_index = -1;

    if (!GPUDMALeaseManagerStateIsValid(mgr))
        return GPU_DMA_LEASE_ERR_BAD_STATE;

    if (lease_id <= 0)
        return GPU_DMA_LEASE_ERR_BAD_PARAM;

    for (i = 0; i < mgr->capacity; ++i)
    {
        if (!mgr->leases[i].active)
            continue;

        if (mgr->leases[i].lease_id == lease_id)
        {
            *out_index = i;
            return GPU_DMA_LEASE_OK;
        }
    }

    return GPU_DMA_LEASE_ERR_NOT_FOUND;
}

I32 GPUDMALeaseAuthorizeChecked(const GPUDMALeaseManager *mgr,
                                I64 lease_id,
                                I64 lease_token,
                                I64 iommu_domain,
                                I64 req_phys_addr,
                                I64 req_nbytes,
                                I64 *out_slot_index)
{
    I64 slot_idx;
    I64 req_end;
    I64 lease_end;
    GPUDMALease *slot;

    if (!mgr || !out_slot_index)
        return GPU_DMA_LEASE_ERR_NULL_PTR;

    *out_slot_index = -1;

    if (iommu_domain < 0 || req_phys_addr < 0 || req_nbytes <= 0)
        return GPU_DMA_LEASE_ERR_BAD_PARAM;

    if (!GPUDMALeaseAddChecked(req_phys_addr, req_nbytes, &req_end))
        return GPU_DMA_LEASE_ERR_OVERFLOW;

    if (GPUDMALeaseFindActiveIndexChecked(mgr, lease_id, &slot_idx) != GPU_DMA_LEASE_OK)
        return GPU_DMA_LEASE_ERR_NOT_FOUND;

    slot = &mgr->leases[slot_idx];

    if (slot->lease_token != lease_token)
        return GPU_DMA_LEASE_ERR_TOKEN_MISMATCH;

    if (slot->iommu_domain != iommu_domain)
        return GPU_DMA_LEASE_ERR_DOMAIN_MISMATCH;

    if (!GPUDMALeaseAddChecked(slot->phys_addr, slot->nbytes, &lease_end))
        return GPU_DMA_LEASE_ERR_BAD_STATE;

    if (req_phys_addr < slot->phys_addr || req_end > lease_end)
        return GPU_DMA_LEASE_ERR_RANGE_OOB;

    *out_slot_index = slot_idx;
    return GPU_DMA_LEASE_OK;
}

I32 GPUDMALeaseReleaseChecked(GPUDMALeaseManager *mgr,
                              I64 lease_id,
                              I64 lease_token)
{
    I64 slot_idx;

    if (!mgr)
        return GPU_DMA_LEASE_ERR_NULL_PTR;

    if (!GPUDMALeaseManagerStateIsValid(mgr))
        return GPU_DMA_LEASE_ERR_BAD_STATE;

    if (GPUDMALeaseFindActiveIndexChecked(mgr, lease_id, &slot_idx) != GPU_DMA_LEASE_OK)
        return GPU_DMA_LEASE_ERR_NOT_FOUND;

    if (mgr->leases[slot_idx].lease_token != lease_token)
        return GPU_DMA_LEASE_ERR_TOKEN_MISMATCH;

    mgr->leases[slot_idx].active = 0;
    mgr->leases[slot_idx].lease_id = 0;
    mgr->leases[slot_idx].lease_token = 0;
    mgr->leases[slot_idx].phys_addr = 0;
    mgr->leases[slot_idx].nbytes = 0;
    mgr->leases[slot_idx].iommu_domain = 0;
    mgr->leases[slot_idx].owner_tag = 0;

    mgr->count -= 1;
    return GPU_DMA_LEASE_OK;
}
