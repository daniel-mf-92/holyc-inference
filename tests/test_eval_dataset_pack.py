#!/usr/bin/env python3
"""Host-side checks for the offline eval dataset packer."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKER_PATH = ROOT / "bench" / "dataset_pack.py"
spec = importlib.util.spec_from_file_location("dataset_pack", PACKER_PATH)
dataset_pack = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["dataset_pack"] = dataset_pack
spec.loader.exec_module(dataset_pack)


def unpack_records(payload: bytes) -> tuple[dict, list[dict]]:
    magic, version, flags, record_count, metadata_len, source_digest = dataset_pack.HEADER.unpack_from(payload, 0)
    assert magic == dataset_pack.MAGIC
    assert version == dataset_pack.VERSION
    assert flags == 0
    assert len(source_digest) == 32
    cursor = dataset_pack.HEADER.size
    metadata = json.loads(payload[cursor : cursor + metadata_len].decode("utf-8"))
    cursor += metadata_len

    records = []
    for _ in range(record_count):
        id_len, prompt_len, choice_count, answer_index, provenance_len, record_flags = (
            dataset_pack.RECORD_HEADER.unpack_from(payload, cursor)
        )
        cursor += dataset_pack.RECORD_HEADER.size
        record_id = payload[cursor : cursor + id_len].decode("utf-8")
        cursor += id_len
        prompt = payload[cursor : cursor + prompt_len].decode("utf-8")
        cursor += prompt_len
        provenance = payload[cursor : cursor + provenance_len].decode("utf-8")
        cursor += provenance_len
        choices = []
        for _choice in range(choice_count):
            (choice_len,) = struct.unpack_from("<I", payload, cursor)
            cursor += 4
            choices.append(payload[cursor : cursor + choice_len].decode("utf-8"))
            cursor += choice_len
        records.append(
            {
                "answer_index": answer_index,
                "choices": choices,
                "flags": record_flags,
                "id": record_id,
                "prompt": prompt,
                "provenance": provenance,
            }
        )
    assert cursor == len(payload)
    return metadata, records


def test_smoke_shapes_pack_into_deterministic_binary() -> None:
    sample = ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"
    rows = dataset_pack.read_jsonl(sample)
    records = dataset_pack.normalize_records(rows, "smoke-eval", "validation")
    payload_a = dataset_pack.pack_records(records, "smoke-eval", "validation")
    payload_b = dataset_pack.pack_records(records, "smoke-eval", "validation")
    assert payload_a == payload_b

    metadata, packed_records = unpack_records(payload_a)
    assert metadata["format"] == "hceval-mc"
    assert metadata["record_count"] == 3
    assert [record.answer_index for record in records] == [0, 0, 0]
    assert [record["id"] for record in packed_records] == [
        "smoke-hellaswag-1",
        "smoke-arc-1",
        "smoke-truthfulqa-1",
    ]
    assert packed_records[1]["choices"][0] == "thermometer"


def test_cli_writes_binary_and_manifest() -> None:
    sample = ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "sample.hceval"
        manifest = Path(tmp) / "sample.manifest.json"
        assert (
            dataset_pack.main(
                [
                    "--input",
                    str(sample),
                    "--output",
                    str(output),
                    "--manifest",
                    str(manifest),
                    "--dataset",
                    "smoke-eval",
                    "--split",
                    "validation",
                ]
            )
            == 0
        )
        payload = output.read_bytes()
        manifest_json = json.loads(manifest.read_text(encoding="utf-8"))
        assert manifest_json["record_count"] == 3
        assert manifest_json["binary_sha256"] == dataset_pack.hashlib.sha256(payload).hexdigest()
        assert manifest_json["byte_stats"]["max_prompt_bytes"] > 0
        assert manifest_json["byte_stats"]["max_record_payload_bytes"] > 0
        assert manifest_json["record_spans"] == dataset_pack.record_spans(
            dataset_pack.normalize_records(dataset_pack.read_jsonl(sample), "smoke-eval", "validation"),
            "smoke-eval",
            "validation",
        )
        assert manifest_json["record_spans"][0]["offset"] > dataset_pack.HEADER.size
        assert manifest_json["record_spans"][-1]["offset"] + manifest_json["record_spans"][-1]["length"] == len(payload)


def test_cli_size_gates_fail_before_writing() -> None:
    sample = ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "sample.hceval"
        manifest = Path(tmp) / "sample.manifest.json"
        status = dataset_pack.main(
            [
                "--input",
                str(sample),
                "--output",
                str(output),
                "--manifest",
                str(manifest),
                "--max-prompt-bytes",
                "8",
            ]
        )

        assert status == 2
        assert not output.exists()
        assert not manifest.exists()


def test_invalid_answer_fails_fast() -> None:
    bad_row = {
        "id": "bad",
        "prompt": "Pick one",
        "choices": ["one", "two"],
        "answer_index": 7,
    }
    try:
        dataset_pack.normalize_records([bad_row], "bad", "validation")
    except ValueError as exc:
        assert "outside choice range" in str(exc)
    else:
        raise AssertionError("invalid answer index should fail")


def test_hellaswag_ctx_parts_normalize_without_ctx_field() -> None:
    row = {
        "id": "hellaswag-parts",
        "dataset": "hellaswag",
        "split": "validation",
        "ctx_a": "A chef slices onions",
        "ctx_b": "and places them into a pan.",
        "endings": [
            "They begin to sizzle.",
            "The dog flies away.",
            "A book closes itself.",
            "The snow melts upward.",
        ],
        "label": "0",
        "provenance": "synthetic HellaSwag ctx_a ctx_b test",
    }

    record = dataset_pack.normalize_records([row], "eval", "validation")[0]

    assert record.prompt == "A chef slices onions and places them into a pan."
    assert record.choices[0] == "They begin to sizzle."
    assert record.answer_index == 0


if __name__ == "__main__":
    test_smoke_shapes_pack_into_deterministic_binary()
    test_cli_writes_binary_and_manifest()
    test_cli_size_gates_fail_before_writing()
    test_invalid_answer_fails_fast()
    test_hellaswag_ctx_parts_normalize_without_ctx_field()
    print("eval_dataset_pack_tests=ok")
