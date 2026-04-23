#!/usr/bin/env python3
"""Harness for IQ-1265 Trinity policy drift checker."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

GATE_SCRIPT = Path("automation/check-trinity-policy-sync.sh")

REQUIRED_INVARIANTS = {
    "secure-default",
    "dev-local-guard",
    "quarantine-hash",
    "gpu-iommu-bot",
    "attestation-digest",
    "trinity-drift-guard",
}


def _run_gate(extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(GATE_SCRIPT)],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )


def _parse_json_lines(stdout: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def test_source_contains_iq1265_contract() -> None:
    src = GATE_SCRIPT.read_text(encoding="utf-8")

    assert "trinity-policy-sync" in src
    assert "TRINITY_INFERENCE_DOC" in src
    assert "TRINITY_TEMPLE_DOC" in src
    assert "TRINITY_SANHEDRIN_DOC" in src
    assert 'check_pattern "TRI-SEC-05-INF"' in src
    assert 'check_pattern "TRI-SEC-06-SAN"' in src


def test_gate_emits_machine_readable_output_on_repo() -> None:
    result = _run_gate()
    records = _parse_json_lines(result.stdout)
    assert records

    check_records = [rec for rec in records if rec.get("type") == "check"]
    summary_records = [rec for rec in records if rec.get("type") == "summary"]

    assert check_records
    assert len(summary_records) == 1

    seen_invariants = {
        str(rec.get("invariant"))
        for rec in check_records
        if str(rec.get("invariant")) in REQUIRED_INVARIANTS
    }
    assert seen_invariants == REQUIRED_INVARIANTS

    summary = summary_records[0]
    assert summary.get("status") in ("pass", "fail")
    assert summary.get("drift") in ("true", "false")
    assert int(summary.get("passed", -1)) >= 1


def test_gate_passes_on_synthetic_synced_docs(tmp_path: Path) -> None:
    inf = tmp_path / "inference.md"
    tem = tmp_path / "temple.md"
    san = tmp_path / "sanhedrin.md"

    inf.write_text(
        "\n".join(
            [
                "`secure-local` is the default mode and must remain default.",
                "`dev-local` is explicit opt-in and may not disable air-gap or Book of Truth.",
                "Every model is untrusted until quarantine + hash-manifest verification passes.",
                "GPU dispatch requires IOMMU + Book-of-Truth hooks before dispatch.",
                "Never bypass attestation/policy-digest handshake for speed.",
                "Do not land policy changes that create Trinity drift across docs.",
            ]
        ),
        encoding="utf-8",
    )
    tem.write_text(
        "\n".join(
            [
                "`secure-local` (default): Book of Truth always-on.",
                "`dev-local` (explicit opt-in): must remain air-gapped and keep Book of Truth on.",
                "Model quarantine + hash verification mandatory.",
                "GPU disabled unless IOMMU and Book of Truth logging hooks active.",
                "Trusted load requires attestation evidence + policy digest match.",
                "Treat policy drift as a release blocker.",
            ]
        ),
        encoding="utf-8",
    )
    san.write_text(
        "\n".join(
            [
                "default profile is not `secure-local`",
                "secure-local|dev-local|quarantine|Book of Truth|IOMMU|GPU",
                "trusted model load path can bypass quarantine/hash verification",
                "GPU tasks bypass IOMMU or Book-of-Truth audit hooks",
                "attestation + policy digest parity",
                "Treat Trinity policy parity mismatches as CRITICAL",
            ]
        ),
        encoding="utf-8",
    )

    result = _run_gate(
        extra_env={
            "TRINITY_INFERENCE_DOC": str(inf),
            "TRINITY_TEMPLE_DOC": str(tem),
            "TRINITY_SANHEDRIN_DOC": str(san),
        }
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    records = _parse_json_lines(result.stdout)
    summary = [rec for rec in records if rec.get("type") == "summary"]
    assert len(summary) == 1
    assert summary[0].get("status") == "pass"
    assert summary[0].get("drift") == "false"


def test_gate_fails_when_one_doc_drops_attestation_clause(tmp_path: Path) -> None:
    inf = tmp_path / "inference.md"
    tem = tmp_path / "temple.md"
    san = tmp_path / "sanhedrin.md"

    inf.write_text(
        "\n".join(
            [
                "`secure-local` is the default mode and must remain default.",
                "`dev-local` is explicit opt-in and may not disable air-gap or Book of Truth.",
                "Every model is untrusted until quarantine + hash-manifest verification passes.",
                "GPU dispatch requires IOMMU + Book-of-Truth hooks before dispatch.",
                "Never bypass attestation/policy-digest handshake for speed.",
                "Do not land policy changes that create Trinity drift across docs.",
            ]
        ),
        encoding="utf-8",
    )
    tem.write_text(
        "\n".join(
            [
                "`secure-local` (default): Book of Truth always-on.",
                "`dev-local` (explicit opt-in): must remain air-gapped and keep Book of Truth on.",
                "Model quarantine + hash verification mandatory.",
                "GPU disabled unless IOMMU and Book of Truth logging hooks active.",
                "Treat policy drift as a release blocker.",
            ]
        ),
        encoding="utf-8",
    )
    san.write_text(
        "\n".join(
            [
                "default profile is not `secure-local`",
                "secure-local|dev-local|quarantine|Book of Truth|IOMMU|GPU",
                "trusted model load path can bypass quarantine/hash verification",
                "GPU tasks bypass IOMMU or Book-of-Truth audit hooks",
                "attestation + policy digest parity",
                "Treat Trinity policy parity mismatches as CRITICAL",
            ]
        ),
        encoding="utf-8",
    )

    result = _run_gate(
        extra_env={
            "TRINITY_INFERENCE_DOC": str(inf),
            "TRINITY_TEMPLE_DOC": str(tem),
            "TRINITY_SANHEDRIN_DOC": str(san),
        }
    )
    assert result.returncode != 0

    records = _parse_json_lines(result.stdout)
    summary = [rec for rec in records if rec.get("type") == "summary"]
    assert len(summary) == 1
    assert summary[0].get("status") == "fail"
    assert summary[0].get("drift") == "true"


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    test_source_contains_iq1265_contract()
    test_gate_emits_machine_readable_output_on_repo()
    with TemporaryDirectory() as tmp:
        test_gate_passes_on_synthetic_synced_docs(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_gate_fails_when_one_doc_drops_attestation_clause(Path(tmp))
    print("ok")
