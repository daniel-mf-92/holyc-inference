#!/usr/bin/env python3
"""Parity harness for GGUFTensorInfoReadNameCheckedNoPartial (IQ-996)."""

from __future__ import annotations

import random
import struct
from pathlib import Path

GGUF_TENSOR_PARSE_OK = 0
GGUF_TENSOR_PARSE_ERR_NULL_PTR = 1
GGUF_TENSOR_PARSE_ERR_TRUNCATED = 2
GGUF_TENSOR_PARSE_ERR_BAD_NAME_LEN = 4

GGUF_TENSOR_MAX_NAME_BYTES = 1 << 20
I64_MAX = (1 << 63) - 1


def u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def gguf_name_entry(name_raw: bytes) -> bytes:
    return u64(len(name_raw)) + name_raw


def parse_name_checked_nopartial(
    buf: bytes | None,
    size: int,
    cursor: int,
    out_name: dict | None,
    out_next_cursor: list[int] | None,
) -> int:
    if buf is None or out_name is None or out_next_cursor is None:
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR
    if cursor > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if size > I64_MAX or cursor > I64_MAX:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    c = cursor
    if c + 8 > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    (name_len,) = struct.unpack_from("<Q", buf, c)
    c += 8

    if name_len > GGUF_TENSOR_MAX_NAME_BYTES:
        return GGUF_TENSOR_PARSE_ERR_BAD_NAME_LEN
    if c + name_len > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    name_raw = buf[c : c + name_len]
    c += name_len

    out_name.clear()
    out_name.update({"len": name_len, "data": name_raw})
    out_next_cursor[0] = c
    return GGUF_TENSOR_PARSE_OK


def test_source_contains_iq996_name_parser_and_parseone_composition() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoReadNameCheckedNoPartial(U8 *buf,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I64 GGUFTensorParseOne(", 1)[0]

    assert "status = GGUFTensorReadString(buf," in body
    assert "GGUFTensorTryAddU64(cursor, 8, &checked_end)" in body
    assert "(U64)staged_cursor != checked_end" in body
    assert "*out_name = staged;" in body
    assert "*out_next_cursor = checked_end;" in body

    parse_one_body = source.split("I64 GGUFTensorParseOne(", 1)[1]
    assert "GGUFTensorInfoReadNameCheckedNoPartial(buf," in parse_one_body


def test_null_and_no_partial_publish_on_failure() -> None:
    payload = gguf_name_entry(b"tok_embd")
    out_name = {"sentinel": 123}
    out_next = [77]

    err = parse_name_checked_nopartial(None, len(payload), 0, out_name, out_next)
    assert err == GGUF_TENSOR_PARSE_ERR_NULL_PTR
    assert out_name == {"sentinel": 123}
    assert out_next == [77]

    truncated = payload[:5]
    err = parse_name_checked_nopartial(truncated, len(truncated), 0, out_name, out_next)
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED
    assert out_name == {"sentinel": 123}
    assert out_next == [77]


def test_adversarial_name_length_and_truncation_vectors() -> None:
    out_name = {"old": 1}
    out_next = [9]

    too_long = u64(GGUF_TENSOR_MAX_NAME_BYTES + 1)
    err = parse_name_checked_nopartial(too_long, len(too_long), 0, out_name, out_next)
    assert err == GGUF_TENSOR_PARSE_ERR_BAD_NAME_LEN
    assert out_name == {"old": 1}
    assert out_next == [9]

    bogus_len = u64(128) + b"abc"
    err = parse_name_checked_nopartial(bogus_len, len(bogus_len), 0, out_name, out_next)
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED
    assert out_name == {"old": 1}
    assert out_next == [9]

    payload = gguf_name_entry(b"x")
    err = parse_name_checked_nopartial(payload, len(payload), len(payload) + 1, out_name, out_next)
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED
    assert out_name == {"old": 1}
    assert out_next == [9]


def test_success_and_randomized_cursor_parity() -> None:
    rng = random.Random(20260422996)

    for i in range(400):
        name_len = rng.randint(0, 128)
        name_raw = bytes(rng.randint(0, 255) for _ in range(name_len))

        prefix_len = rng.randint(0, 16)
        prefix = bytes(rng.randint(0, 255) for _ in range(prefix_len))
        payload = prefix + gguf_name_entry(name_raw)

        out_name = {"keep": i}
        out_next = [4444]

        err = parse_name_checked_nopartial(payload, len(payload), prefix_len, out_name, out_next)
        assert err == GGUF_TENSOR_PARSE_OK
        assert out_name["len"] == name_len
        assert out_name["data"] == name_raw
        assert out_next[0] == prefix_len + 8 + name_len


if __name__ == "__main__":
    test_source_contains_iq996_name_parser_and_parseone_composition()
    test_null_and_no_partial_publish_on_failure()
    test_adversarial_name_length_and_truncation_vectors()
    test_success_and_randomized_cursor_parity()
    print("gguf_tensorinfo_read_name_checked_nopartial=ok")

