#!/usr/bin/env python3
"""Parity harness for GPU DMA lease-token manager (IQ-1261)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

GPU_DMA_LEASE_OK = 0
GPU_DMA_LEASE_ERR_NULL_PTR = 1
GPU_DMA_LEASE_ERR_BAD_PARAM = 2
GPU_DMA_LEASE_ERR_BAD_STATE = 3
GPU_DMA_LEASE_ERR_CAPACITY = 4
GPU_DMA_LEASE_ERR_OVERFLOW = 5
GPU_DMA_LEASE_ERR_NO_FREE_SLOT = 6
GPU_DMA_LEASE_ERR_RANGE_OVERLAP = 7
GPU_DMA_LEASE_ERR_NOT_FOUND = 8
GPU_DMA_LEASE_ERR_TOKEN_MISMATCH = 9
GPU_DMA_LEASE_ERR_DOMAIN_MISMATCH = 10
GPU_DMA_LEASE_ERR_RANGE_OOB = 11

GPU_DMA_LEASE_TOKEN_SALT = 0x6A09E667F3BCC909
I64_MAX = 0x7FFFFFFFFFFFFFFF
I64_MASK = 0xFFFFFFFFFFFFFFFF


def _to_i64(value: int) -> int:
    value &= I64_MASK
    if value >= (1 << 63):
        value -= 1 << 64
    return value


def _add_checked(lhs: int, rhs: int) -> int | None:
    if lhs < 0 or rhs < 0:
        return None
    if lhs > I64_MAX - rhs:
        return None
    return lhs + rhs


@dataclass
class Lease:
    active: int = 0
    lease_id: int = 0
    lease_token: int = 0
    phys_addr: int = 0
    nbytes: int = 0
    iommu_domain: int = 0
    owner_tag: int = 0


@dataclass
class Manager:
    leases: list[Lease]
    capacity: int
    count: int = 0
    next_lease_id: int = 1


def _state_is_valid(mgr: Manager) -> bool:
    return (
        mgr is not None
        and mgr.leases is not None
        and mgr.capacity > 0
        and 0 <= mgr.count <= mgr.capacity
        and mgr.next_lease_id > 0
    )


def _ranges_overlap(lhs_addr: int, lhs_nbytes: int, rhs_addr: int, rhs_nbytes: int) -> bool:
    lhs_end = _add_checked(lhs_addr, lhs_nbytes)
    rhs_end = _add_checked(rhs_addr, rhs_nbytes)
    if lhs_end is None or rhs_end is None:
        return True
    return lhs_addr < rhs_end and rhs_addr < lhs_end


def _build_token(lease_id: int, owner_tag: int, phys_addr: int, nbytes: int, iommu_domain: int) -> int:
    token = _to_i64(lease_id * 6364136223846793005)
    token = _to_i64(token ^ _to_i64(owner_tag * 1442695040888963407))
    token = _to_i64(token ^ _to_i64(phys_addr << 7))
    token = _to_i64(token ^ _to_i64(nbytes << 17))
    token = _to_i64(token ^ _to_i64(iommu_domain << 29))
    token = _to_i64(token ^ GPU_DMA_LEASE_TOKEN_SALT)
    return GPU_DMA_LEASE_TOKEN_SALT if token == 0 else token


def lease_init(capacity: int) -> tuple[int, Manager | None]:
    if capacity <= 0:
        return GPU_DMA_LEASE_ERR_CAPACITY, None
    return GPU_DMA_LEASE_OK, Manager(leases=[Lease() for _ in range(capacity)], capacity=capacity)


def lease_acquire(
    mgr: Manager | None,
    phys_addr: int,
    nbytes: int,
    iommu_domain: int,
    owner_tag: int,
) -> tuple[int, int, int]:
    if mgr is None:
        return GPU_DMA_LEASE_ERR_NULL_PTR, 0, 0
    if not _state_is_valid(mgr):
        return GPU_DMA_LEASE_ERR_BAD_STATE, 0, 0
    if phys_addr < 0 or nbytes <= 0 or iommu_domain < 0 or owner_tag < 0:
        return GPU_DMA_LEASE_ERR_BAD_PARAM, 0, 0
    if _add_checked(phys_addr, nbytes) is None:
        return GPU_DMA_LEASE_ERR_OVERFLOW, 0, 0

    free_idx = -1
    for i, lease in enumerate(mgr.leases):
        if not lease.active:
            if free_idx < 0:
                free_idx = i
            continue
        if _ranges_overlap(phys_addr, nbytes, lease.phys_addr, lease.nbytes):
            return GPU_DMA_LEASE_ERR_RANGE_OVERLAP, 0, 0

    if free_idx < 0:
        return GPU_DMA_LEASE_ERR_NO_FREE_SLOT, 0, 0

    lease_id = mgr.next_lease_id
    if lease_id <= 0:
        return GPU_DMA_LEASE_ERR_OVERFLOW, 0, 0

    token = _build_token(lease_id, owner_tag, phys_addr, nbytes, iommu_domain)
    slot = mgr.leases[free_idx]
    slot.active = 1
    slot.lease_id = lease_id
    slot.lease_token = token
    slot.phys_addr = phys_addr
    slot.nbytes = nbytes
    slot.iommu_domain = iommu_domain
    slot.owner_tag = owner_tag

    mgr.count += 1
    mgr.next_lease_id += 1
    return GPU_DMA_LEASE_OK, lease_id, token


def _find_active_index(mgr: Manager | None, lease_id: int) -> tuple[int, int]:
    if mgr is None:
        return GPU_DMA_LEASE_ERR_NULL_PTR, -1
    if not _state_is_valid(mgr):
        return GPU_DMA_LEASE_ERR_BAD_STATE, -1
    if lease_id <= 0:
        return GPU_DMA_LEASE_ERR_BAD_PARAM, -1

    for idx, lease in enumerate(mgr.leases):
        if lease.active and lease.lease_id == lease_id:
            return GPU_DMA_LEASE_OK, idx
    return GPU_DMA_LEASE_ERR_NOT_FOUND, -1


def lease_authorize(
    mgr: Manager | None,
    lease_id: int,
    lease_token: int,
    iommu_domain: int,
    req_phys_addr: int,
    req_nbytes: int,
) -> tuple[int, int]:
    if mgr is None:
        return GPU_DMA_LEASE_ERR_NULL_PTR, -1
    if iommu_domain < 0 or req_phys_addr < 0 or req_nbytes <= 0:
        return GPU_DMA_LEASE_ERR_BAD_PARAM, -1
    req_end = _add_checked(req_phys_addr, req_nbytes)
    if req_end is None:
        return GPU_DMA_LEASE_ERR_OVERFLOW, -1

    status, idx = _find_active_index(mgr, lease_id)
    if status != GPU_DMA_LEASE_OK:
        return status, -1
    slot = mgr.leases[idx]

    if slot.lease_token != lease_token:
        return GPU_DMA_LEASE_ERR_TOKEN_MISMATCH, -1
    if slot.iommu_domain != iommu_domain:
        return GPU_DMA_LEASE_ERR_DOMAIN_MISMATCH, -1

    lease_end = _add_checked(slot.phys_addr, slot.nbytes)
    if lease_end is None:
        return GPU_DMA_LEASE_ERR_BAD_STATE, -1

    if req_phys_addr < slot.phys_addr or req_end > lease_end:
        return GPU_DMA_LEASE_ERR_RANGE_OOB, -1

    return GPU_DMA_LEASE_OK, idx


def lease_release(mgr: Manager | None, lease_id: int, lease_token: int) -> int:
    status, idx = _find_active_index(mgr, lease_id)
    if status != GPU_DMA_LEASE_OK:
        return status
    slot = mgr.leases[idx]

    if slot.lease_token != lease_token:
        return GPU_DMA_LEASE_ERR_TOKEN_MISMATCH

    mgr.leases[idx] = Lease()
    mgr.count -= 1
    return GPU_DMA_LEASE_OK


def _bench_authorize_ns_per_call(mgr: Manager, lease_id: int, token: int, iters: int) -> float:
    start = time.perf_counter_ns()
    for _ in range(iters):
        status, _ = lease_authorize(mgr, lease_id, token, 9, 0x1000, 128)
        if status != GPU_DMA_LEASE_OK:
            raise AssertionError("authorize failed during benchmark")
    elapsed = time.perf_counter_ns() - start
    return elapsed / iters


def test_source_contains_iq1261_symbols() -> None:
    src = Path("src/gpu/dma_lease.HC").read_text(encoding="utf-8")
    assert "class GPUDMALease" in src
    assert "class GPUDMALeaseManager" in src
    assert "I32 GPUDMALeaseAcquireChecked(" in src
    assert "I32 GPUDMALeaseAuthorizeChecked(" in src
    assert "I32 GPUDMALeaseReleaseChecked(" in src
    assert "GPU_DMA_LEASE_ERR_TOKEN_MISMATCH" in src


def test_acquire_authorize_release_happy_path() -> None:
    status, mgr = lease_init(4)
    assert status == GPU_DMA_LEASE_OK
    assert mgr is not None

    status, lease_id, token = lease_acquire(mgr, 0x1000, 0x200, 9, 77)
    assert status == GPU_DMA_LEASE_OK
    assert lease_id == 1
    assert token != 0
    assert mgr.count == 1

    status, idx = lease_authorize(mgr, lease_id, token, 9, 0x1100, 0x80)
    assert status == GPU_DMA_LEASE_OK
    assert idx == 0

    status = lease_release(mgr, lease_id, token)
    assert status == GPU_DMA_LEASE_OK
    assert mgr.count == 0


def test_hardening_rejects_overlap_and_token_mismatch() -> None:
    status, mgr = lease_init(4)
    assert status == GPU_DMA_LEASE_OK
    assert mgr is not None

    status, lease_id, token = lease_acquire(mgr, 0x2000, 0x400, 3, 11)
    assert status == GPU_DMA_LEASE_OK

    status, _, _ = lease_acquire(mgr, 0x2200, 0x100, 5, 12)
    assert status == GPU_DMA_LEASE_ERR_RANGE_OVERLAP

    status, _ = lease_authorize(mgr, lease_id, token ^ 1, 3, 0x2000, 0x100)
    assert status == GPU_DMA_LEASE_ERR_TOKEN_MISMATCH


def test_domain_and_range_guards() -> None:
    status, mgr = lease_init(2)
    assert status == GPU_DMA_LEASE_OK
    assert mgr is not None

    status, lease_id, token = lease_acquire(mgr, 0x3000, 0x300, 8, 91)
    assert status == GPU_DMA_LEASE_OK

    status, _ = lease_authorize(mgr, lease_id, token, 2, 0x3000, 0x40)
    assert status == GPU_DMA_LEASE_ERR_DOMAIN_MISMATCH

    status, _ = lease_authorize(mgr, lease_id, token, 8, 0x2FF0, 0x20)
    assert status == GPU_DMA_LEASE_ERR_RANGE_OOB

    status, _ = lease_authorize(mgr, lease_id, token, 8, 0x3200, 0x200)
    assert status == GPU_DMA_LEASE_ERR_RANGE_OOB


def test_capacity_and_overflow_paths() -> None:
    status, mgr = lease_init(1)
    assert status == GPU_DMA_LEASE_OK
    assert mgr is not None

    status, _, _ = lease_acquire(mgr, I64_MAX - 7, 16, 1, 1)
    assert status == GPU_DMA_LEASE_ERR_OVERFLOW

    status, lease_id, token = lease_acquire(mgr, 0x4000, 0x80, 1, 2)
    assert status == GPU_DMA_LEASE_OK

    status, _, _ = lease_acquire(mgr, 0x5000, 0x80, 1, 3)
    assert status == GPU_DMA_LEASE_ERR_NO_FREE_SLOT

    status = lease_release(mgr, lease_id, token + 1)
    assert status == GPU_DMA_LEASE_ERR_TOKEN_MISMATCH


def test_secure_local_perf_overhead_budget_plan() -> None:
    # Perf-overhead measurement plan for secure-local mode:
    # compare authorized lease checks vs direct no-check baseline and
    # keep policy overhead bounded for per-dispatch DMA validation.
    status, mgr = lease_init(2)
    assert status == GPU_DMA_LEASE_OK
    assert mgr is not None

    status, lease_id, token = lease_acquire(mgr, 0x1000, 0x400, 9, 55)
    assert status == GPU_DMA_LEASE_OK

    loops = 200_000
    ns_secure = _bench_authorize_ns_per_call(mgr, lease_id, token, loops)

    start = time.perf_counter_ns()
    for _ in range(loops):
        idx = 0
        if mgr.leases[idx].lease_id != lease_id:
            raise AssertionError("baseline mismatch")
    ns_baseline = (time.perf_counter_ns() - start) / loops

    assert ns_secure > 0
    assert ns_baseline > 0
    assert ns_secure <= ns_baseline * 40.0


if __name__ == "__main__":
    test_source_contains_iq1261_symbols()
    test_acquire_authorize_release_happy_path()
    test_hardening_rejects_overlap_and_token_mismatch()
    test_domain_and_range_guards()
    test_capacity_and_overflow_paths()
    test_secure_local_perf_overhead_budget_plan()
    print("ok")
