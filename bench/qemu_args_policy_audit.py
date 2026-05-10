#!/usr/bin/env python3
"""Audit reusable QEMU argument files for benchmark air-gap policy.

This host-side tool validates QEMU argument fragments before they are handed to
the benchmark launcher. It does not launch QEMU. Fragment files must not enable
networking, add network devices, open remote display sockets, include nested
argument/config files, use legacy `-net none`, or embed a full
`qemu-system-*` command.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ARG_FILE_SUFFIXES = {".args", ".qemuargs", ".qemu-args", ".txt", ".json"}
NETWORK_DEVICE_MARKERS = (
    "dp8393x",
    "e1000",
    "eepro100",
    "ftgmac100",
    "igb",
    "igbvf",
    "i825",
    "lan9118",
    "lance",
    "mcf_fec",
    "ne2k",
    "pcnet",
    "rocker",
    "rtl8139",
    "smc91c111",
    "spapr-vlan",
    "sunhme",
    "sungem",
    "tulip",
    "usb-net",
    "virtio-net",
    "virtio-vsock",
    "vhost-vsock",
    "vmxnet",
    "xgmac",
    "xlnx-ethlite",
    "xen_nic",
)
SOCKET_ENDPOINT_MARKERS = (
    "guestfwd=",
    "hostfwd=",
    "socket,",
    "tcp:",
    "telnet:",
    "tftp=",
    "udp:",
    "unix:",
    "websocket",
)
REMOTE_RESOURCE_MARKERS = (
    "file.driver=ftp",
    "file.driver=http",
    "file.driver=https",
    "file.driver=iscsi",
    "file.driver=nbd",
    "file.driver=sftp",
    "file.driver=ssh",
    "ftp://",
    "ftps://",
    "gluster://",
    "http://",
    "https://",
    "iscsi:",
    "nbd:",
    "rbd:",
    "sftp://",
    "ssh://",
    "tftp://",
    "url=ftp://",
    "url=http://",
    "url=https://",
    "vxhs://",
)
SOCKET_TRANSPORT_OPTIONS = {"-chardev", "-gdb", "-incoming", "-monitor", "-qmp", "-qmp-pretty", "-serial", "-vnc"}
USER_NET_SERVICE_OPTIONS = {"-bootp", "-redir", "-smb", "-tftp"}
REMOTE_DISPLAY_MARKERS = ("spice", "vnc")
TLS_OPTION_MARKERS = ("tls-creds", "tlsauthz", "tls-cipher-suites")
TLS_VALUE_OPTIONS = {
    "-chardev",
    "-display",
    "-monitor",
    "-object",
    "-qmp",
    "-qmp-pretty",
    "-serial",
    "-vnc",
}


@dataclass(frozen=True)
class ArgFileRecord:
    path: str
    exists: bool
    size_bytes: int | None
    sha256: str | None
    format: str
    arg_count: int
    command_like: bool
    explicit_nic_none: bool
    nic_none_count: int
    legacy_net_none: bool
    violation_count: int


@dataclass(frozen=True)
class Finding:
    path: str
    reason: str
    arg_index: int | None
    argument: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_network_device_arg(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in NETWORK_DEVICE_MARKERS)


def is_socket_endpoint_arg(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in SOCKET_ENDPOINT_MARKERS)


def is_remote_resource_arg(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in REMOTE_RESOURCE_MARKERS)


def is_remote_display_arg(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in REMOTE_DISPLAY_MARKERS)


def is_tls_arg(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in TLS_OPTION_MARKERS)


def canonical_qemu_option(arg: str) -> str:
    if arg.startswith("--") and len(arg) > 2:
        return "-" + arg[2:]
    return arg


def has_explicit_nic_none(args: list[str]) -> bool:
    return nic_none_count(args) > 0


def nic_none_count(args: list[str]) -> int:
    count = 0
    for index, arg in enumerate(args):
        option = canonical_qemu_option(arg)
        if option == "-nic" and index + 1 < len(args) and args[index + 1] == "none":
            count += 1
        if option == "-nic=none":
            count += 1
    return count


def has_legacy_net_none(args: list[str]) -> bool:
    for index, arg in enumerate(args):
        option = canonical_qemu_option(arg)
        if option == "-net" and index + 1 < len(args) and args[index + 1] == "none":
            return True
        if option == "-net=none":
            return True
    return False


def is_command_like(args: list[str]) -> bool:
    return bool(args) and Path(args[0].strip("'\"`")).name.lower().startswith("qemu-system")


def parse_args_file(path: Path) -> tuple[list[str], str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise ValueError("JSON QEMU args file must be a string array")
        return list(payload), "json-array"
    return shlex.split(text, comments=True), "shell-fragment"


def iter_input_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in ARG_FILE_SUFFIXES:
                    yield child
        else:
            yield path


def audit_args(path: Path, args: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    disabled_nics = nic_none_count(args)
    if disabled_nics:
        findings.append(
            Finding(
                str(path),
                "fragment includes -nic none",
                None,
                "",
                (
                    "QEMU args fragments must not include explicit `-nic none`; "
                    "the benchmark launcher injects exactly one NIC disablement."
                ),
            )
        )
    if disabled_nics > 1:
        findings.append(
            Finding(
                str(path),
                "duplicate -nic none",
                None,
                "",
                f"QEMU args fragments must not contain more than one explicit `-nic none`; found {disabled_nics}.",
            )
        )
    if is_command_like(args):
        findings.append(
            Finding(
                path=str(path),
                reason="qemu executable embedded in args file",
                arg_index=0,
                argument=args[0],
                detail="QEMU args files must contain fragments only; the benchmark launcher owns the executable and injected `-nic none`.",
            )
        )

    index = 0
    while index < len(args):
        arg = args[index]
        option = canonical_qemu_option(arg)
        next_arg = args[index + 1] if index + 1 < len(args) else ""
        previous_arg = canonical_qemu_option(args[index - 1]) if index else ""

        if option in TLS_VALUE_OPTIONS and is_tls_arg(next_arg):
            findings.append(
                Finding(str(path), "tls option", index, arg, f"`{arg} {next_arg}` violates the no-TLS air-gap policy")
            )
        if previous_arg not in TLS_VALUE_OPTIONS and is_tls_arg(arg):
            findings.append(Finding(str(path), "tls option", index, arg, f"`{arg}` violates the no-TLS air-gap policy"))
        if is_remote_resource_arg(arg):
            findings.append(
                Finding(
                    str(path),
                    "remote resource",
                    index,
                    arg,
                    f"`{arg}` can make QEMU open a network-backed resource and violates the air-gap policy",
                )
            )
        if (
            option in {"-blockdev", "-cdrom", "-drive", "-hda", "-hdb", "-hdc", "-hdd", "-initrd", "-kernel"}
            and is_remote_resource_arg(next_arg)
        ):
            findings.append(
                Finding(
                    str(path),
                    "remote resource",
                    index,
                    arg,
                    f"`{arg} {next_arg}` can make QEMU open a network-backed resource and violates the air-gap policy",
                )
            )

        if arg.startswith("@") and len(arg) > 1:
            findings.append(
                Finding(
                    str(path),
                    "nested qemu args include",
                    index,
                    arg,
                    "QEMU response files are forbidden because they can bypass the audited air-gap fragment.",
                )
            )

        if option == "-readconfig":
            detail = "QEMU config includes are forbidden because they can add devices outside the audited args fragment"
            if next_arg:
                detail = f"`-readconfig {next_arg}` is forbidden; config includes can add devices outside the audited args fragment"
            findings.append(Finding(str(path), "qemu config include", index, arg, detail))
            index += 2
            continue
        if option.startswith("-readconfig="):
            findings.append(
                Finding(
                    str(path),
                    "qemu config include",
                    index,
                    arg,
                    f"`{arg}` is forbidden; config includes can add devices outside the audited args fragment",
                )
            )

        if option == "-nic":
            if next_arg != "none":
                findings.append(
                    Finding(str(path), "non-air-gapped -nic", index, arg, f"`-nic {next_arg}` enables networking")
                )
            index += 2
            continue
        if option.startswith("-nic=") and option != "-nic=none":
            findings.append(Finding(str(path), "non-air-gapped -nic", index, arg, f"`{arg}` enables networking"))

        if option == "-net":
            if next_arg == "none":
                detail = "legacy `-net none` is not allowed in benchmark fragments; rely on injected `-nic none`"
            else:
                detail = f"`-net {next_arg}` enables networking"
            findings.append(Finding(str(path), "networking -net", index, arg, detail))
            index += 2
            continue
        if option.startswith("-net="):
            if option == "-net=none":
                detail = "legacy `-net=none` is not allowed in benchmark fragments; rely on injected `-nic none`"
            else:
                detail = f"`{arg}` enables networking"
            findings.append(Finding(str(path), "networking -net", index, arg, detail))

        if option == "-netdev" or option.startswith("-netdev"):
            findings.append(Finding(str(path), "network backend", index, arg, f"`{arg}` is forbidden by air-gap policy"))
        if option in USER_NET_SERVICE_OPTIONS or any(option.startswith(f"{service}=") for service in USER_NET_SERVICE_OPTIONS):
            findings.append(
                Finding(
                    str(path),
                    "user-mode network service",
                    index,
                    arg,
                    f"`{arg}` is forbidden because it configures QEMU user-mode networking services",
                )
            )
        if option == "-device" and is_network_device_arg(next_arg):
            findings.append(Finding(str(path), "network device", index, arg, f"`{next_arg}` is a network device"))
        if option.startswith("-device=") and is_network_device_arg(option):
            findings.append(Finding(str(path), "network device", index, arg, f"`{arg}` is a network device"))
        if option == "-vnc" and next_arg != "none":
            findings.append(Finding(str(path), "socket endpoint", index, arg, f"`-vnc {next_arg}` opens a remote display socket"))
            index += 2
            continue
        if option.startswith("-vnc=") and option != "-vnc=none":
            findings.append(Finding(str(path), "socket endpoint", index, arg, f"`{arg}` opens a remote display socket"))
        if option == "-display" and is_remote_display_arg(next_arg):
            findings.append(
                Finding(
                    str(path),
                    "remote display socket",
                    index,
                    arg,
                    f"`-display {next_arg}` can open a remote display socket",
                )
            )
            index += 2
            continue
        if option.startswith("-display=") and is_remote_display_arg(option):
            findings.append(
                Finding(str(path), "remote display socket", index, arg, f"`{arg}` can open a remote display socket")
            )
        if option == "-spice":
            findings.append(
                Finding(str(path), "remote display socket", index, arg, f"`-spice {next_arg}` can open a SPICE socket")
            )
            index += 2
            continue
        if option.startswith("-spice"):
            findings.append(Finding(str(path), "remote display socket", index, arg, f"`{arg}` can open a SPICE socket"))
        if option in SOCKET_TRANSPORT_OPTIONS and is_socket_endpoint_arg(next_arg):
            findings.append(Finding(str(path), "socket endpoint", index, arg, f"`{arg} {next_arg}` violates the no-sockets air-gap policy"))
            index += 2
            continue
        if any(option.startswith(f"{socket_option}=") for socket_option in SOCKET_TRANSPORT_OPTIONS) and is_socket_endpoint_arg(option):
            findings.append(Finding(str(path), "socket endpoint", index, arg, f"`{arg}` violates the no-sockets air-gap policy"))
        if is_socket_endpoint_arg(arg) and ("hostfwd=" in arg.lower() or "guestfwd=" in arg.lower()):
            findings.append(Finding(str(path), "socket endpoint", index, arg, f"`{arg}` forwards host/guest sockets"))

        index += 1
    return findings


def audit(paths: Iterable[Path]) -> tuple[list[ArgFileRecord], list[Finding]]:
    records: list[ArgFileRecord] = []
    findings: list[Finding] = []
    for path in iter_input_files(paths):
        if not path.exists():
            findings.append(Finding(str(path), "missing args file", None, "", "input path does not exist"))
            records.append(ArgFileRecord(str(path), False, None, None, "missing", 0, False, False, 0, False, 1))
            continue

        stat = path.stat()
        try:
            args, file_format = parse_args_file(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            findings.append(Finding(str(path), "unparseable args file", None, "", str(exc)))
            records.append(
                ArgFileRecord(str(path), True, stat.st_size, file_sha256(path), "unparseable", 0, False, False, 0, False, 1)
            )
            continue

        path_findings = audit_args(path, args)
        findings.extend(path_findings)
        records.append(
            ArgFileRecord(
                path=str(path),
                exists=True,
                size_bytes=stat.st_size,
                sha256=file_sha256(path),
                format=file_format,
                arg_count=len(args),
                command_like=is_command_like(args),
                explicit_nic_none=has_explicit_nic_none(args),
                nic_none_count=nic_none_count(args),
                legacy_net_none=has_legacy_net_none(args),
                violation_count=len(path_findings),
            )
        )
    return records, findings


def write_json(path: Path, records: list[ArgFileRecord], findings: list[Finding]) -> None:
    report: dict[str, Any] = {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "files_checked": len(records),
            "files_with_findings": sum(1 for record in records if record.violation_count),
            "findings": len(findings),
            "explicit_nic_none_fragments": sum(1 for record in records if record.explicit_nic_none),
            "duplicate_nic_none_fragments": sum(1 for record in records if record.nic_none_count > 1),
            "legacy_net_none_fragments": sum(1 for record in records if record.legacy_net_none),
        },
        "records": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, records: list[ArgFileRecord], findings: list[Finding]) -> None:
    lines = [
        "# QEMU Args Policy Audit",
        "",
        f"Files checked: {len(records)}",
        f"Findings: {len(findings)}",
        "",
    ]
    if findings:
        lines.extend(["| File | Arg | Reason | Detail |", "| --- | ---: | --- | --- |"])
        for finding in findings:
            arg_index = "" if finding.arg_index is None else str(finding.arg_index)
            lines.append(
                "| {path} | {arg_index} | {reason} | {detail} |".format(
                    path=finding.path,
                    arg_index=arg_index,
                    reason=finding.reason.replace("|", "\\|"),
                    detail=finding.detail.replace("|", "\\|"),
                )
            )
    else:
        lines.append("All QEMU argument files passed the host-side air-gap fragment policy.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_records_csv(path: Path, records: list[ArgFileRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ArgFileRecord.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Finding.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_junit(path: Path, findings: list[Finding]) -> None:
    failures_by_path: dict[str, list[Finding]] = {}
    for finding in findings:
        failures_by_path.setdefault(finding.path, []).append(finding)

    test_count = max(1, len(failures_by_path) + (0 if findings else 1))
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_args_policy_audit",
            "tests": str(test_count),
            "failures": str(len(failures_by_path)),
            "errors": "0",
        },
    )
    if not findings:
        ET.SubElement(suite, "testcase", {"classname": "qemu_args_policy_audit", "name": "all_args_files"})
    for path_name, path_findings in sorted(failures_by_path.items()):
        case = ET.SubElement(
            suite,
            "testcase",
            {"classname": "qemu_args_policy_audit", "name": Path(path_name).name},
        )
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "qemu_args_policy_violation",
                "message": "; ".join(finding.reason for finding in path_findings),
            },
        )
        failure.text = "\n".join(f"{finding.arg_index}: {finding.detail}" for finding in path_findings)

    ET.indent(suite)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
    with path.open("ab") as handle:
        handle.write(b"\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="QEMU args files or directories")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_args_policy_audit_latest")
    parser.add_argument("--min-files", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_files < 0:
        parser.error("--min-files must be >= 0")

    records, findings = audit(args.inputs)
    if len(records) < args.min_files:
        findings.append(
            Finding(
                path="",
                reason="insufficient args file coverage",
                arg_index=None,
                argument="",
                detail=f"checked {len(records)} files, required at least {args.min_files}",
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", records, findings)
    write_markdown(args.output_dir / f"{stem}.md", records, findings)
    write_records_csv(args.output_dir / f"{stem}.csv", records)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
