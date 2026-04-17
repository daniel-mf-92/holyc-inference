#!/usr/bin/env python3
"""Parity checks for GGUFMetadataReadI16QuadCheckedDefault."""

from __future__ import annotations

import random

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

I64_MAX = (1 << 63) - 1
U16_MASK = (1 << 16) - 1


def reinterpret_u16_as_i16(value: int) -> int:
    value &= U16_MASK
    if value >= (1 << 15):
        return value - (1 << 16)
    return value


def gguf_metadata_cursor_can_advance_checked(
    cursor: int,
    need: int,
    table_end: int,
) -> tuple[int, int | None]:
    if cursor > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW, None
    if need > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW, None
    if table_end > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW, None
    if cursor > table_end:
        return GGUF_META_TABLE_ERR_BAD_PARAM, None
    if cursor > ((1 << 64) - 1) - need:
        return GGUF_META_TABLE_ERR_OVERFLOW, None

    next_cursor = cursor + need
    if next_cursor > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, None

    return GGUF_META_TABLE_OK, next_cursor


def gguf_metadata_read_u8_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_value_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_value_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR
    if buf_nbytes > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW

    cur = cursor_ref[0]
    err, next_cursor = gguf_metadata_cursor_can_advance_checked(cur, 1, table_end)
    if err != GGUF_META_TABLE_OK:
        return err

    assert next_cursor is not None
    if next_cursor > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    out_value_ref[0] = buf[cur]
    cursor_ref[0] = next_cursor
    return GGUF_META_TABLE_OK


def gguf_metadata_read_u16le_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_value_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_value_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    b0 = [0]
    b1 = [0]

    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, b0)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, b1)
    if err != GGUF_META_TABLE_OK:
        return err

    out_value_ref[0] = b0[0] | (b1[0] << 8)
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_i16le_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_value_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_value_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    raw = [0]

    err = gguf_metadata_read_u16le_checked(buf, buf_nbytes, cur, table_end, raw)
    if err != GGUF_META_TABLE_OK:
        return err

    out_value_ref[0] = reinterpret_u16_as_i16(raw[0])
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_i16_quad_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
    out_third_ref: list[int] | None,
    out_fourth_ref: list[int] | None,
) -> int:
    if (
        buf is None
        or cursor_ref is None
        or out_first_ref is None
        or out_second_ref is None
        or out_third_ref is None
        or out_fourth_ref is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    first = [0]
    second = [0]
    third = [0]
    fourth = [0]

    err = gguf_metadata_read_i16le_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_i16le_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_i16le_checked(buf, buf_nbytes, cur, table_end, third)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_i16le_checked(buf, buf_nbytes, cur, table_end, fourth)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    out_fourth_ref[0] = fourth[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_i16_quad_checked_default(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
    out_third_ref: list[int] | None,
    out_fourth_ref: list[int] | None,
) -> int:
    return gguf_metadata_read_i16_quad_checked(
        buf,
        buf_nbytes,
        cursor_ref,
        buf_nbytes,
        out_first_ref,
        out_second_ref,
        out_third_ref,
        out_fourth_ref,
    )


def _le_u16(value: int) -> list[int]:
    return [value & 0xFF, (value >> 8) & 0xFF]


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [9]
    out_first = [111]
    out_second = [222]
    out_third = [333]
    out_fourth = [444]

    assert (
        gguf_metadata_read_i16_quad_checked_default(
            None,
            32,
            cursor,
            out_first,
            out_second,
            out_third,
            out_fourth,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [9]
    assert out_first == [111]
    assert out_second == [222]
    assert out_third == [333]
    assert out_fourth == [444]


def test_uses_default_end_and_no_commit_on_short_payload() -> None:
    first = reinterpret_u16_as_i16(0x7FFF)
    second = reinterpret_u16_as_i16(0x8000)
    third = reinterpret_u16_as_i16(0xBEEF)
    buf = _le_u16(0x7FFF) + _le_u16(0x8000) + _le_u16(0xBEEF) + [0xAA]

    cursor = [0]
    out_first = [911]
    out_second = [922]
    out_third = [933]
    out_fourth = [944]

    err = gguf_metadata_read_i16_quad_checked_default(
        buf,
        len(buf),
        cursor,
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]
    assert out_first == [911]
    assert out_second == [922]
    assert out_third == [933]
    assert out_fourth == [944]

    # First 3 values are intentionally materialized to lock expected decode path.
    assert [first, second, third] == [32767, -32768, -16657]


def test_success_matches_checked_core_with_default_end() -> None:
    raw_values = [0x0123, 0x89AB, 0x7FFF, 0x8001]
    expected = [reinterpret_u16_as_i16(v) for v in raw_values]

    buf: list[int] = []
    for raw in raw_values:
        buf.extend(_le_u16(raw))

    cursor_default = [0]
    out_default = [[0], [0], [0], [0]]
    err_default = gguf_metadata_read_i16_quad_checked_default(
        buf,
        len(buf),
        cursor_default,
        out_default[0],
        out_default[1],
        out_default[2],
        out_default[3],
    )

    cursor_checked = [0]
    out_checked = [[0], [0], [0], [0]]
    err_checked = gguf_metadata_read_i16_quad_checked(
        buf,
        len(buf),
        cursor_checked,
        len(buf),
        out_checked[0],
        out_checked[1],
        out_checked[2],
        out_checked[3],
    )

    assert err_default == GGUF_META_TABLE_OK
    assert err_default == err_checked
    assert cursor_default == [8]
    assert cursor_default == cursor_checked
    assert [x[0] for x in out_default] == expected
    assert [x[0] for x in out_default] == [x[0] for x in out_checked]


def test_cursor_prebiased_success() -> None:
    prefix = [0xFF, 0xEE, 0xDD]
    raws = [0xA0A0, 0x0B0B, 0x7001, 0x8E00]
    buf = prefix.copy()
    for raw in raws:
        buf.extend(_le_u16(raw))
    buf.extend([0xAA, 0xBB])

    cursor = [len(prefix)]
    out_first = [0]
    out_second = [0]
    out_third = [0]
    out_fourth = [0]

    err = gguf_metadata_read_i16_quad_checked_default(
        buf,
        len(buf),
        cursor,
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_OK
    assert cursor == [len(prefix) + 8]
    assert [out_first[0], out_second[0], out_third[0], out_fourth[0]] == [
        reinterpret_u16_as_i16(v) for v in raws
    ]


def test_randomized_default_wrapper_parity() -> None:
    rng = random.Random(0x1A16D3)

    for _ in range(500):
        buf_len = rng.randint(0, 96)
        buf = [rng.randrange(0, 256) for _ in range(buf_len)]

        cursor_seed = rng.randint(0, buf_len + 8)

        out_seed = [
            rng.randint(-(1 << 15), (1 << 15) - 1),
            rng.randint(-(1 << 15), (1 << 15) - 1),
            rng.randint(-(1 << 15), (1 << 15) - 1),
            rng.randint(-(1 << 15), (1 << 15) - 1),
        ]

        cursor_default = [cursor_seed]
        out_default = [[out_seed[0]], [out_seed[1]], [out_seed[2]], [out_seed[3]]]

        cursor_checked = [cursor_seed]
        out_checked = [[out_seed[0]], [out_seed[1]], [out_seed[2]], [out_seed[3]]]

        err_default = gguf_metadata_read_i16_quad_checked_default(
            buf,
            buf_len,
            cursor_default,
            out_default[0],
            out_default[1],
            out_default[2],
            out_default[3],
        )

        err_checked = gguf_metadata_read_i16_quad_checked(
            buf,
            buf_len,
            cursor_checked,
            buf_len,
            out_checked[0],
            out_checked[1],
            out_checked[2],
            out_checked[3],
        )

        assert err_default == err_checked
        assert cursor_default == cursor_checked
        assert [x[0] for x in out_default] == [x[0] for x in out_checked]


def main() -> None:
    test_null_ptr_and_no_partial_write()
    test_uses_default_end_and_no_commit_on_short_payload()
    test_success_matches_checked_core_with_default_end()
    test_cursor_prebiased_success()
    test_randomized_default_wrapper_parity()
    print("gguf_metadata_read_i16_quad_checked_default: ok")


if __name__ == "__main__":
    main()
