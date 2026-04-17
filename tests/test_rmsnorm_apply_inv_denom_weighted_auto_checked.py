#!/usr/bin/env python3
"""Reference checks for FPQ16RMSNormApplyInvDenomWeightedAutoChecked semantics."""

from __future__ import annotations

import random

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT

I64_MAX_VALUE = (1 << 63) - 1
U64_MAX_VALUE = (1 << 64) - 1

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_DOMAIN = 3
FP_Q16_ERR_OVERFLOW = 4

FP_Q16_RMSNORM_AUTO_PATH_OUT_OF_PLACE = 0
FP_Q16_RMSNORM_AUTO_PATH_IN_PLACE = 1


def fp_abs_to_u64(x: int) -> int:
    if x >= 0:
        return x
    return (-(x + 1)) + 1


def fp_try_apply_sign_from_u64_checked(mag: int, is_negative: bool) -> tuple[int, int]:
    limit = I64_MAX_VALUE
    if is_negative:
        limit = 1 << 63

    if mag > limit:
        return FP_Q16_ERR_OVERFLOW, 0

    if is_negative:
        if mag >= (1 << 63):
            return FP_Q16_OK, -(1 << 63)
        return FP_Q16_OK, -mag

    return FP_Q16_OK, mag


def fpq16_mul_div_rounded_checked(a_q16: int, b_q16: int, d_q16: int) -> tuple[int, int]:
    if d_q16 == 0:
        return FP_Q16_ERR_DOMAIN, 0

    abs_a = fp_abs_to_u64(a_q16)
    abs_b = fp_abs_to_u64(b_q16)
    abs_d = fp_abs_to_u64(d_q16)
    is_negative = (a_q16 < 0) ^ (b_q16 < 0) ^ (d_q16 < 0)

    if abs_a != 0 and abs_b != 0 and abs_a > (U64_MAX_VALUE // abs_b):
        return FP_Q16_ERR_OVERFLOW, 0

    abs_num = abs_a * abs_b
    q = abs_num // abs_d
    r = abs_num % abs_d

    limit = I64_MAX_VALUE
    if is_negative:
        limit = 1 << 63

    if q > limit:
        return FP_Q16_ERR_OVERFLOW, 0

    if r >= ((abs_d + 1) >> 1):
        if q == limit:
            return FP_Q16_ERR_OVERFLOW, 0
        q += 1

    return fp_try_apply_sign_from_u64_checked(q, is_negative)


def fpq16_mul_div_rounded_by_positive_int_checked(a_q16: int, b_q16: int, d_int: int) -> tuple[int, int]:
    if d_int <= 0:
        return FP_Q16_ERR_DOMAIN, 0

    if d_int > (U64_MAX_VALUE >> FP_Q16_SHIFT):
        return FP_Q16_ERR_OVERFLOW, 0

    abs_a = fp_abs_to_u64(a_q16)
    abs_b = fp_abs_to_u64(b_q16)
    is_negative = (a_q16 < 0) ^ (b_q16 < 0)

    if abs_a != 0 and abs_b != 0 and abs_a > (U64_MAX_VALUE // abs_b):
        return FP_Q16_ERR_OVERFLOW, 0

    abs_num = abs_a * abs_b
    abs_den_q16 = d_int << FP_Q16_SHIFT

    q = abs_num // abs_den_q16
    r = abs_num % abs_den_q16

    limit = I64_MAX_VALUE
    if is_negative:
        limit = 1 << 63

    if q > limit:
        return FP_Q16_ERR_OVERFLOW, 0

    if r >= ((abs_den_q16 + 1) >> 1):
        if q == limit:
            return FP_Q16_ERR_OVERFLOW, 0
        q += 1

    return fp_try_apply_sign_from_u64_checked(q, is_negative)


def fpq16_mul_div_rounded_by_positive_int_from_q16_den_checked(a_q16: int, b_q16: int, d_q16: int) -> tuple[int, int]:
    if d_q16 <= 0:
        return FP_Q16_ERR_DOMAIN, 0

    if (d_q16 & (FP_Q16_ONE - 1)) != 0:
        return FP_Q16_ERR_BAD_PARAM, 0

    d_int = d_q16 >> FP_Q16_SHIFT
    if d_int <= 0:
        return FP_Q16_ERR_DOMAIN, 0

    return fpq16_mul_div_rounded_by_positive_int_checked(a_q16, b_q16, d_int)


def fpq16_rmsnorm_apply_inv_denom_weighted_checked(
    input_q16: list[int] | None,
    gamma_q16: list[int] | None,
    count: int,
    inv_denom_q16: int,
    den_q16: int,
) -> tuple[int, list[int]]:
    if input_q16 is None or gamma_q16 is None:
        return FP_Q16_ERR_NULL_PTR, []
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM, []
    if inv_denom_q16 <= 0:
        return FP_Q16_ERR_DOMAIN, []

    out = [0] * max(count, 0)
    for i in range(count):
        err, scaled = fpq16_mul_div_rounded_by_positive_int_from_q16_den_checked(
            input_q16[i],
            inv_denom_q16,
            den_q16,
        )
        if err != FP_Q16_OK:
            return err, []

        err, weighted = fpq16_mul_div_rounded_checked(scaled, gamma_q16[i], FP_Q16_ONE)
        if err != FP_Q16_OK:
            return err, []

        out[i] = weighted

    return FP_Q16_OK, out


def fpq16_rmsnorm_apply_inv_denom_weighted_inplace_checked(
    input_q16: list[int] | None,
    gamma_q16: list[int] | None,
    count: int,
    inv_denom_q16: int,
    den_q16: int,
) -> tuple[int, list[int]]:
    if input_q16 is None or gamma_q16 is None:
        return FP_Q16_ERR_NULL_PTR, []
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM, []
    if inv_denom_q16 <= 0:
        return FP_Q16_ERR_DOMAIN, []

    out = list(input_q16)
    for i in range(count):
        err, scaled = fpq16_mul_div_rounded_by_positive_int_from_q16_den_checked(
            out[i],
            inv_denom_q16,
            den_q16,
        )
        if err != FP_Q16_OK:
            return err, []

        err, weighted = fpq16_mul_div_rounded_checked(scaled, gamma_q16[i], FP_Q16_ONE)
        if err != FP_Q16_OK:
            return err, []

        out[i] = weighted

    return FP_Q16_OK, out


def fpq16_rmsnorm_apply_inv_denom_weighted_auto_validate_checked(
    input_q16: list[int] | None,
    gamma_q16: list[int] | None,
    output_q16: list[int] | None,
    count: int,
    inv_denom_q16: int,
    den_q16: int,
    input_addr: int,
    gamma_addr: int,
    output_addr: int,
) -> int:
    if input_q16 is None or gamma_q16 is None or output_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM
    if inv_denom_q16 <= 0:
        return FP_Q16_ERR_DOMAIN
    if den_q16 <= 0:
        return FP_Q16_ERR_DOMAIN
    if (den_q16 & (FP_Q16_ONE - 1)) != 0:
        return FP_Q16_ERR_BAD_PARAM
    if (den_q16 >> FP_Q16_SHIFT) <= 0:
        return FP_Q16_ERR_DOMAIN

    if count == 0:
        return FP_Q16_OK

    last_index = count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW

    last_byte_offset = last_index << 3

    if input_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW
    if gamma_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW
    if output_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW

    return FP_Q16_OK


def fpq16_rmsnorm_apply_inv_denom_weighted_auto_select_path_checked(
    input_q16: list[int] | None,
    output_q16: list[int] | None,
) -> tuple[int, int]:
    if input_q16 is None or output_q16 is None:
        return FP_Q16_ERR_NULL_PTR, FP_Q16_RMSNORM_AUTO_PATH_OUT_OF_PLACE

    if input_q16 is output_q16:
        return FP_Q16_OK, FP_Q16_RMSNORM_AUTO_PATH_IN_PLACE

    return FP_Q16_OK, FP_Q16_RMSNORM_AUTO_PATH_OUT_OF_PLACE


def fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked(
    input_q16: list[int] | None,
    gamma_q16: list[int] | None,
    output_q16: list[int] | None,
    count: int,
    inv_denom_q16: int,
    den_q16: int,
    input_addr: int = 0x1000,
    gamma_addr: int = 0x2000,
    output_addr: int = 0x3000,
) -> tuple[int, list[int], int]:
    status = fpq16_rmsnorm_apply_inv_denom_weighted_auto_validate_checked(
        input_q16,
        gamma_q16,
        output_q16,
        count,
        inv_denom_q16,
        den_q16,
        input_addr,
        gamma_addr,
        output_addr,
    )
    if status != FP_Q16_OK:
        return status, [], FP_Q16_RMSNORM_AUTO_PATH_OUT_OF_PLACE

    status, selected_path = fpq16_rmsnorm_apply_inv_denom_weighted_auto_select_path_checked(input_q16, output_q16)
    if status != FP_Q16_OK:
        return status, [], FP_Q16_RMSNORM_AUTO_PATH_OUT_OF_PLACE

    if selected_path == FP_Q16_RMSNORM_AUTO_PATH_IN_PLACE:
        status, out = fpq16_rmsnorm_apply_inv_denom_weighted_inplace_checked(
            input_q16,
            gamma_q16,
            count,
            inv_denom_q16,
            den_q16,
        )
        return status, out, selected_path

    if selected_path != FP_Q16_RMSNORM_AUTO_PATH_OUT_OF_PLACE:
        return FP_Q16_ERR_BAD_PARAM, [], selected_path

    status, out = fpq16_rmsnorm_apply_inv_denom_weighted_checked(
        input_q16,
        gamma_q16,
        count,
        inv_denom_q16,
        den_q16,
    )
    return status, out, selected_path


def fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked_with_path_out(
    input_q16: list[int] | None,
    gamma_q16: list[int] | None,
    output_q16: list[int] | None,
    count: int,
    inv_denom_q16: int,
    den_q16: int,
    path_out: list[int] | None,
    input_addr: int = 0x1000,
    gamma_addr: int = 0x2000,
    output_addr: int = 0x3000,
) -> tuple[int, list[int], int]:
    if path_out is None:
        return FP_Q16_ERR_NULL_PTR, [], FP_Q16_RMSNORM_AUTO_PATH_OUT_OF_PLACE

    status = fpq16_rmsnorm_apply_inv_denom_weighted_auto_validate_checked(
        input_q16,
        gamma_q16,
        output_q16,
        count,
        inv_denom_q16,
        den_q16,
        input_addr,
        gamma_addr,
        output_addr,
    )
    if status != FP_Q16_OK:
        return status, [], FP_Q16_RMSNORM_AUTO_PATH_OUT_OF_PLACE

    status, selected_path = fpq16_rmsnorm_apply_inv_denom_weighted_auto_select_path_checked(input_q16, output_q16)
    if status != FP_Q16_OK:
        return status, [], FP_Q16_RMSNORM_AUTO_PATH_OUT_OF_PLACE

    path_out[0] = selected_path

    if selected_path == FP_Q16_RMSNORM_AUTO_PATH_IN_PLACE:
        status, out = fpq16_rmsnorm_apply_inv_denom_weighted_inplace_checked(
            input_q16,
            gamma_q16,
            count,
            inv_denom_q16,
            den_q16,
        )
        return status, out, selected_path

    if selected_path != FP_Q16_RMSNORM_AUTO_PATH_OUT_OF_PLACE:
        return FP_Q16_ERR_BAD_PARAM, [], selected_path

    status, out = fpq16_rmsnorm_apply_inv_denom_weighted_checked(
        input_q16,
        gamma_q16,
        count,
        inv_denom_q16,
        den_q16,
    )
    return status, out, selected_path


def test_contract_surfaces() -> None:
    vec = [1, 2, 3]
    gamma = [FP_Q16_ONE, FP_Q16_ONE, FP_Q16_ONE]

    assert fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked(None, gamma, vec, 3, FP_Q16_ONE, FP_Q16_ONE)[0] == FP_Q16_ERR_NULL_PTR
    assert fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked(vec, None, vec, 3, FP_Q16_ONE, FP_Q16_ONE)[0] == FP_Q16_ERR_NULL_PTR
    assert fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked(vec, gamma, None, 3, FP_Q16_ONE, FP_Q16_ONE)[0] == FP_Q16_ERR_NULL_PTR
    assert fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked(vec, gamma, vec, -1, FP_Q16_ONE, FP_Q16_ONE)[0] == FP_Q16_ERR_BAD_PARAM
    assert fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked(vec, gamma, vec, 3, 0, FP_Q16_ONE)[0] == FP_Q16_ERR_DOMAIN
    assert fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked(vec, gamma, vec, 3, FP_Q16_ONE, 0)[0] == FP_Q16_ERR_DOMAIN
    assert fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked(vec, gamma, vec, 3, FP_Q16_ONE, FP_Q16_ONE + 1)[0] == FP_Q16_ERR_BAD_PARAM


def test_alias_and_non_alias_dispatch_path_ids() -> None:
    vec = [11, -22, 33, -44]
    gamma = [FP_Q16_ONE, FP_Q16_ONE, FP_Q16_ONE, FP_Q16_ONE]

    err_alias, out_alias, path_alias = fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked(
        vec,
        gamma,
        vec,
        len(vec),
        FP_Q16_ONE,
        FP_Q16_ONE,
    )
    assert err_alias == FP_Q16_OK
    assert path_alias == FP_Q16_RMSNORM_AUTO_PATH_IN_PLACE
    assert out_alias == vec

    out_buf = [0] * len(vec)
    err_non_alias, out_non_alias, path_non_alias = fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked(
        vec,
        gamma,
        out_buf,
        len(vec),
        FP_Q16_ONE,
        FP_Q16_ONE,
    )
    assert err_non_alias == FP_Q16_OK
    assert path_non_alias == FP_Q16_RMSNORM_AUTO_PATH_OUT_OF_PLACE
    assert out_non_alias == vec


def test_dispatch_equivalence_vs_explicit_paths() -> None:
    rng = random.Random(20260417_1601)

    for _ in range(5000):
        count = rng.randint(0, 64)
        vec = [rng.randint(-(1 << 27), 1 << 27) for _ in range(max(count, 1))]
        gamma = [rng.randint(-(1 << 17), 1 << 17) for _ in range(max(count, 1))]
        inv = rng.randint(1, 1 << 27)
        den_int = rng.randint(1, 1 << 16)
        den_q16 = den_int << FP_Q16_SHIFT

        vec_slice = vec[:count]
        gamma_slice = gamma[:count]

        err_alias, out_alias, path_alias = fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked(
            vec_slice,
            gamma_slice,
            vec_slice,
            count,
            inv,
            den_q16,
        )
        err_in, out_in = fpq16_rmsnorm_apply_inv_denom_weighted_inplace_checked(
            vec_slice,
            gamma_slice,
            count,
            inv,
            den_q16,
        )
        assert path_alias == FP_Q16_RMSNORM_AUTO_PATH_IN_PLACE
        assert err_alias == err_in
        if err_alias == FP_Q16_OK:
            assert out_alias == out_in

        out_buf = [0] * count
        err_non_alias, out_non_alias, path_non_alias = fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked(
            vec_slice,
            gamma_slice,
            out_buf,
            count,
            inv,
            den_q16,
        )
        err_ref, out_ref = fpq16_rmsnorm_apply_inv_denom_weighted_checked(
            vec_slice,
            gamma_slice,
            count,
            inv,
            den_q16,
        )
        assert path_non_alias == FP_Q16_RMSNORM_AUTO_PATH_OUT_OF_PLACE
        assert err_non_alias == err_ref
        if err_non_alias == FP_Q16_OK:
            assert out_non_alias == out_ref


def test_bad_param_and_overflow_parity_vs_explicit_paths() -> None:
    vec = [FP_Q16_ONE, -(2 * FP_Q16_ONE)]
    gamma = [I64_MAX_VALUE, I64_MAX_VALUE]

    err_alias_bad, _, _ = fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked(
        vec,
        gamma,
        vec,
        2,
        FP_Q16_ONE,
        FP_Q16_ONE + 1,
    )
    err_non_alias_bad, _, _ = fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked(
        vec,
        gamma,
        [0, 0],
        2,
        FP_Q16_ONE,
        FP_Q16_ONE + 1,
    )
    err_in_bad, _ = fpq16_rmsnorm_apply_inv_denom_weighted_inplace_checked(vec, gamma, 2, FP_Q16_ONE, FP_Q16_ONE + 1)
    err_ref_bad, _ = fpq16_rmsnorm_apply_inv_denom_weighted_checked(vec, gamma, 2, FP_Q16_ONE, FP_Q16_ONE + 1)
    assert err_alias_bad == err_in_bad == FP_Q16_ERR_BAD_PARAM
    assert err_non_alias_bad == err_ref_bad == FP_Q16_ERR_BAD_PARAM

    huge = I64_MAX_VALUE
    err_alias_ovf, _, _ = fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked(
        [FP_Q16_ONE],
        [huge],
        [FP_Q16_ONE],
        1,
        huge,
        FP_Q16_ONE,
    )
    err_non_alias_ovf, _, _ = fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked(
        [FP_Q16_ONE],
        [huge],
        [0],
        1,
        huge,
        FP_Q16_ONE,
    )
    err_in_ovf, _ = fpq16_rmsnorm_apply_inv_denom_weighted_inplace_checked([FP_Q16_ONE], [huge], 1, huge, FP_Q16_ONE)
    err_ref_ovf, _ = fpq16_rmsnorm_apply_inv_denom_weighted_checked([FP_Q16_ONE], [huge], 1, huge, FP_Q16_ONE)
    assert err_alias_ovf == err_in_ovf == FP_Q16_ERR_OVERFLOW
    assert err_non_alias_ovf == err_ref_ovf == FP_Q16_ERR_OVERFLOW


def test_validate_pointer_span_overflow_guards() -> None:
    vec = [1]
    gamma = [1]
    out = [0]

    huge_count = (I64_MAX_VALUE >> 3) + 2
    err, _, _ = fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked(
        vec,
        gamma,
        out,
        huge_count,
        FP_Q16_ONE,
        FP_Q16_ONE,
    )
    assert err == FP_Q16_ERR_OVERFLOW

    err_wrap_input, _, _ = fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked(
        vec,
        gamma,
        out,
        4,
        FP_Q16_ONE,
        FP_Q16_ONE,
        input_addr=U64_MAX_VALUE - 7,
        gamma_addr=0x2000,
        output_addr=0x3000,
    )
    assert err_wrap_input == FP_Q16_ERR_OVERFLOW

    err_wrap_gamma, _, _ = fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked(
        vec,
        gamma,
        out,
        4,
        FP_Q16_ONE,
        FP_Q16_ONE,
        input_addr=0x1000,
        gamma_addr=U64_MAX_VALUE - 7,
        output_addr=0x3000,
    )
    assert err_wrap_gamma == FP_Q16_ERR_OVERFLOW

    err_wrap_output, _, _ = fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked(
        vec,
        gamma,
        out,
        4,
        FP_Q16_ONE,
        FP_Q16_ONE,
        input_addr=0x1000,
        gamma_addr=0x2000,
        output_addr=U64_MAX_VALUE - 7,
    )
    assert err_wrap_output == FP_Q16_ERR_OVERFLOW


def test_with_path_out_contract_and_dispatch_parity() -> None:
    vec = [13, -27, 39, -55]
    gamma = [FP_Q16_ONE, FP_Q16_ONE // 2, -(FP_Q16_ONE // 4), FP_Q16_ONE]

    err_null, _, _ = fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked_with_path_out(
        vec,
        gamma,
        [0, 0, 0, 0],
        len(vec),
        FP_Q16_ONE,
        FP_Q16_ONE,
        None,
    )
    assert err_null == FP_Q16_ERR_NULL_PTR

    alias_path_out = [FP_Q16_RMSNORM_AUTO_PATH_OUT_OF_PLACE]
    err_alias, out_alias, path_alias = fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked_with_path_out(
        vec,
        gamma,
        vec,
        len(vec),
        FP_Q16_ONE,
        FP_Q16_ONE,
        alias_path_out,
    )
    assert err_alias == FP_Q16_OK
    assert path_alias == FP_Q16_RMSNORM_AUTO_PATH_IN_PLACE
    assert alias_path_out[0] == FP_Q16_RMSNORM_AUTO_PATH_IN_PLACE
    assert out_alias == fpq16_rmsnorm_apply_inv_denom_weighted_inplace_checked(vec, gamma, len(vec), FP_Q16_ONE, FP_Q16_ONE)[1]

    out_buf = [0] * len(vec)
    non_alias_path_out = [FP_Q16_RMSNORM_AUTO_PATH_IN_PLACE]
    err_non_alias, out_non_alias, path_non_alias = fpq16_rmsnorm_apply_inv_denom_weighted_auto_checked_with_path_out(
        vec,
        gamma,
        out_buf,
        len(vec),
        FP_Q16_ONE,
        FP_Q16_ONE,
        non_alias_path_out,
    )
    assert err_non_alias == FP_Q16_OK
    assert path_non_alias == FP_Q16_RMSNORM_AUTO_PATH_OUT_OF_PLACE
    assert non_alias_path_out[0] == FP_Q16_RMSNORM_AUTO_PATH_OUT_OF_PLACE
    assert out_non_alias == fpq16_rmsnorm_apply_inv_denom_weighted_checked(vec, gamma, len(vec), FP_Q16_ONE, FP_Q16_ONE)[1]


def run() -> None:
    test_contract_surfaces()
    test_alias_and_non_alias_dispatch_path_ids()
    test_dispatch_equivalence_vs_explicit_paths()
    test_bad_param_and_overflow_parity_vs_explicit_paths()
    test_validate_pointer_span_overflow_guards()
    test_with_path_out_contract_and_dispatch_parity()
    print("rmsnorm_apply_inv_denom_weighted_auto_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
