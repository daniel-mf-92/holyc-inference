#!/usr/bin/env python3
"""CI smoke gate for packed HCEval choice semantics auditing."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import dataset_pack
import hceval_choice_semantics_audit


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def pack_hceval(path: Path, rows: list[dict[str, object]]) -> None:
    source = path.with_suffix(".jsonl")
    manifest = Path(str(path) + ".manifest.json")
    write_jsonl(source, rows)
    records = dataset_pack.normalize_records(dataset_pack.read_jsonl(source), "smoke", "validation")
    dataset_pack.write_outputs(records, path, manifest, "smoke", "validation")


def run_audit(binary: Path, output_dir: Path, stem: str, *, fail_on_findings: bool) -> int:
    return hceval_choice_semantics_audit.main(
        [
            "--input",
            str(binary),
            "--output",
            str(output_dir / f"{stem}.json"),
            "--markdown",
            str(output_dir / f"{stem}.md"),
            "--csv",
            str(output_dir / f"{stem}.csv"),
            "--findings-csv",
            str(output_dir / f"{stem}_findings.csv"),
            "--junit",
            str(output_dir / f"{stem}_junit.xml"),
            "--min-overlap-chars",
            "5",
            *(["--fail-on-findings"] if fail_on_findings else []),
        ]
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        output_dir = tmp_path / "out"
        clean = tmp_path / "clean.hceval"
        bad = tmp_path / "bad.hceval"

        pack_hceval(
            clean,
            [
                {
                    "id": "clean-1",
                    "prompt": "Pick the object used to measure temperature.",
                    "choices": ["thermometer", "ruler", "scale", "compass"],
                    "answer_index": 0,
                    "provenance": "synthetic smoke",
                }
            ],
        )
        pack_hceval(
            bad,
            [
                {
                    "id": "bad-1",
                    "prompt": "The clue already names alpha centauri.",
                    "choices": ["Alpha Centauri", "alpha centauri", "Sirius", "Vega"],
                    "answer_index": 0,
                    "provenance": "synthetic smoke",
                }
            ],
        )

        clean_status = run_audit(clean, output_dir, "clean", fail_on_findings=True)
        bad_status = run_audit(bad, output_dir, "bad", fail_on_findings=True)
        bad_report = json.loads((output_dir / "bad.json").read_text(encoding="utf-8"))

        if clean_status != 0:
            print("clean_hceval_choice_semantics_smoke=fail")
            return 1
        if bad_status == 0:
            print("bad_hceval_choice_semantics_smoke=fail_missing_findings")
            return 1
        kinds = {finding["kind"] for finding in bad_report["findings"]}
        required = {"duplicate_normalized_choice", "answer_choice_alias", "choice_text_in_prompt"}
        if not required <= kinds:
            print(f"bad_hceval_choice_semantics_smoke=fail kinds={sorted(kinds)}")
            return 1

    print("hceval_choice_semantics_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
