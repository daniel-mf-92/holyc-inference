#!/usr/bin/env python3
"""Smoke test for air-gap audit artifact generation."""

from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import airgap_audit


def write_report(path: Path, command: list[str], *, recorded_ok: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "profile": "ci-airgap-smoke",
                "benchmarks": [
                    {
                        "prompt": "smoke",
                        "phase": "measured",
                        "command": command,
                        "command_airgap_ok": recorded_ok,
                        "command_has_explicit_nic_none": "-nic" in command and "none" in command,
                        "command_has_legacy_net_none": "-net" in command and "none" in command,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-airgap-audit-ci-") as tmp:
        root = Path(tmp)
        passing = root / "passing.json"
        failing = root / "failing.json"
        write_report(passing, ["qemu-system-x86_64", "-nic", "none", "-serial", "stdio"])
        write_report(failing, ["qemu-system-x86_64", "-net", "none", "-device", "e1000"], recorded_ok=True)

        pass_dir = root / "pass"
        status = airgap_audit.main([str(passing), "--output-dir", str(pass_dir)])
        if status != 0:
            raise AssertionError(f"expected passing airgap audit, got {status}")
        pass_payload = json.loads((pass_dir / "airgap_audit_latest.json").read_text(encoding="utf-8"))
        pass_junit = ET.parse(pass_dir / "airgap_audit_latest_junit.xml").getroot()
        if pass_payload["summary"]["commands"] != 1:
            raise AssertionError("passing command count was not recorded")
        if pass_junit.attrib.get("failures") != "0":
            raise AssertionError("passing junit reported failures")

        fail_dir = root / "fail"
        status = airgap_audit.main([str(failing), "--output-dir", str(fail_dir)])
        if status == 0:
            raise AssertionError("expected failing airgap audit")
        fail_payload = json.loads((fail_dir / "airgap_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_payload["findings"]}
        if "airgap_violation" not in kinds:
            raise AssertionError("missing airgap violation finding")
        if "recorded_airgap_drift" not in kinds:
            raise AssertionError("missing recorded airgap drift finding")
        if "legacy `-net none`" not in (fail_dir / "airgap_audit_latest.md").read_text(encoding="utf-8"):
            raise AssertionError("markdown did not include legacy net finding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
