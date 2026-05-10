#!/usr/bin/env python3
"""Tests for bench smoke manifest generation."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import bench_smoke_manifest


def write_script(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def parse_args(extra: list[str]) -> object:
    return bench_smoke_manifest.build_parser().parse_args(extra)


def test_manifest_records_paired_smoke_metadata(tmp_path: Path) -> None:
    write_script(tmp_path / "dataset_demo.py", "#!/usr/bin/env python3\n")
    smoke = tmp_path / "dataset_demo_ci_smoke.py"
    write_script(
        smoke,
        "#!/usr/bin/env python3\n"
        '"""CI smoke for dataset_demo."""\n'
        "\n"
        "def main() -> int:\n"
        "    return 0\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n",
    )
    args = parse_args([str(tmp_path), "--require-paired-tools", "--require-shebang"])

    records = [bench_smoke_manifest.load_record(path) for path in bench_smoke_manifest.iter_smoke_files(args.inputs, args.pattern)]
    findings = bench_smoke_manifest.audit(records, args)

    assert findings == []
    assert len(records) == 1
    assert records[0].area == "dataset"
    assert records[0].paired_tool_exists is True
    assert records[0].docstring == "CI smoke for dataset_demo."


def test_manifest_flags_unpaired_or_malformed_smokes(tmp_path: Path) -> None:
    smoke = tmp_path / "qemu_orphan_ci_smoke.py"
    write_script(smoke, "print('no guard')\n")
    args = parse_args([str(smoke), "--require-paired-tools", "--require-shebang"])

    records = [bench_smoke_manifest.load_record(path) for path in bench_smoke_manifest.iter_smoke_files(args.inputs, args.pattern)]
    findings = bench_smoke_manifest.audit(records, args)

    kinds = {finding.kind for finding in findings}
    assert {"missing_main_guard", "missing_shebang", "missing_paired_tool", "too_short"} <= kinds


def test_cli_writes_manifest_outputs(tmp_path: Path) -> None:
    write_script(tmp_path / "eval_demo.py", "#!/usr/bin/env python3\n")
    write_script(
        tmp_path / "eval_demo_ci_smoke.py",
        "#!/usr/bin/env python3\n"
        '"""CI smoke for eval_demo."""\n'
        "\n"
        "def main() -> int:\n"
        "    return 0\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n",
    )
    output_dir = tmp_path / "out"

    status = bench_smoke_manifest.main(
        [str(tmp_path), "--output-dir", str(output_dir), "--output-stem", "smokes", "--require-paired-tools"]
    )

    assert status == 0
    payload = json.loads((output_dir / "smokes.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["areas"]["eval"] == 1
    rows = list(csv.DictReader((output_dir / "smokes.csv").open(encoding="utf-8")))
    assert rows[0]["name"] == "eval_demo_ci_smoke.py"
    assert "No smoke manifest findings." in (output_dir / "smokes.md").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "smokes_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_bench_smoke_manifest"
    assert junit_root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_manifest_records_paired_smoke_metadata(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_manifest_flags_unpaired_or_malformed_smokes(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_manifest_outputs(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
