#!/usr/bin/env python3
"""Host-side parity/spec harness for quant_profile.HC policy math.

This script mirrors HolyC logic and validates key policy invariants.
"""

from dataclasses import dataclass

QUANT_PROFILE_OK = 0
QUANT_PROFILE_ERR_BAD_PARAM = 2
QUANT_PROFILE_ERR_POLICY_GUARD = 4

QUANT_PROFILE_SECURE_LOCAL = 1
QUANT_PROFILE_DEV_LOCAL = 2

QUANT_MODE_Q4_0 = 1
QUANT_MODE_Q8_0 = 2
QUANT_MODE_MIXED_Q4Q8 = 3

PREF_BALANCED = 1
PREF_THROUGHPUT = 2
PREF_ACCURACY = 3


@dataclass
class Sel:
    profile_id: int = QUANT_PROFILE_SECURE_LOCAL
    quant_mode: int = QUANT_MODE_Q4_0
    preferred_block_rows: int = 4
    prefetch_distance: int = 1
    speculative_decode_enabled: int = 0
    prefix_cache_enabled: int = 1


def _is_binary(v: int) -> bool:
    return v in (0, 1)


def select_for_profile(profile_id: int, preference: int, quarantine_gate: int, manifest_gate: int):
    if profile_id not in (QUANT_PROFILE_SECURE_LOCAL, QUANT_PROFILE_DEV_LOCAL):
        return QUANT_PROFILE_ERR_BAD_PARAM, None
    if preference not in (PREF_BALANCED, PREF_THROUGHPUT, PREF_ACCURACY):
        return QUANT_PROFILE_ERR_BAD_PARAM, None
    if not _is_binary(quarantine_gate) or not _is_binary(manifest_gate):
        return QUANT_PROFILE_ERR_BAD_PARAM, None
    if quarantine_gate == 0 or manifest_gate == 0:
        return QUANT_PROFILE_ERR_POLICY_GUARD, None

    s = Sel(profile_id=profile_id)

    if profile_id == QUANT_PROFILE_SECURE_LOCAL:
        if preference == PREF_ACCURACY:
            s.quant_mode = QUANT_MODE_Q8_0
            s.preferred_block_rows = 2
        elif preference == PREF_THROUGHPUT:
            s.quant_mode = QUANT_MODE_Q4_0
            s.preferred_block_rows = 8
        else:
            s.quant_mode = QUANT_MODE_Q4_0
            s.preferred_block_rows = 4
        s.speculative_decode_enabled = 0
        s.prefix_cache_enabled = 1
        s.prefetch_distance = 1
        return QUANT_PROFILE_OK, s

    if preference == PREF_ACCURACY:
        s.quant_mode = QUANT_MODE_Q8_0
        s.preferred_block_rows = 2
        s.prefetch_distance = 1
        s.speculative_decode_enabled = 0
    elif preference == PREF_THROUGHPUT:
        s.quant_mode = QUANT_MODE_MIXED_Q4Q8
        s.preferred_block_rows = 12
        s.prefetch_distance = 3
        s.speculative_decode_enabled = 1
    else:
        s.quant_mode = QUANT_MODE_MIXED_Q4Q8
        s.preferred_block_rows = 8
        s.prefetch_distance = 2
        s.speculative_decode_enabled = 1

    s.prefix_cache_enabled = 1
    return QUANT_PROFILE_OK, s


def main() -> None:
    status, sel = select_for_profile(QUANT_PROFILE_SECURE_LOCAL, PREF_BALANCED, 1, 1)
    assert status == QUANT_PROFILE_OK
    assert sel.quant_mode == QUANT_MODE_Q4_0
    assert sel.speculative_decode_enabled == 0

    status, sel = select_for_profile(QUANT_PROFILE_SECURE_LOCAL, PREF_ACCURACY, 1, 1)
    assert status == QUANT_PROFILE_OK
    assert sel.quant_mode == QUANT_MODE_Q8_0
    assert sel.preferred_block_rows == 2

    status, sel = select_for_profile(QUANT_PROFILE_DEV_LOCAL, PREF_THROUGHPUT, 1, 1)
    assert status == QUANT_PROFILE_OK
    assert sel.quant_mode == QUANT_MODE_MIXED_Q4Q8
    assert sel.speculative_decode_enabled == 1
    assert sel.prefetch_distance == 3

    status, _ = select_for_profile(QUANT_PROFILE_DEV_LOCAL, PREF_THROUGHPUT, 0, 1)
    assert status == QUANT_PROFILE_ERR_POLICY_GUARD

    status, _ = select_for_profile(QUANT_PROFILE_DEV_LOCAL, PREF_THROUGHPUT, 1, 0)
    assert status == QUANT_PROFILE_ERR_POLICY_GUARD

    status, _ = select_for_profile(99, PREF_BALANCED, 1, 1)
    assert status == QUANT_PROFILE_ERR_BAD_PARAM

    status, _ = select_for_profile(QUANT_PROFILE_SECURE_LOCAL, 99, 1, 1)
    assert status == QUANT_PROFILE_ERR_BAD_PARAM

    print("ok: quant_profile policy selector vectors")


if __name__ == "__main__":
    main()
