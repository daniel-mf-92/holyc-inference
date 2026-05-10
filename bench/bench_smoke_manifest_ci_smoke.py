#!/usr/bin/env python3
"""CI smoke for bench_smoke_manifest."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_smoke_manifest


def write_script(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        output = root / "out"
        write_script(root / "qemu_demo.py", "#!/usr/bin/env python3\n")
        write_script(
            root / "qemu_demo_ci_smoke.py",
            "#!/usr/bin/env python3\n"
            '"""CI smoke for qemu_demo."""\n'
            "\n"
            "def main() -> int:\n"
            "    return 0\n"
            "\n"
            'if __name__ == "__main__":\n'
            "    raise SystemExit(main())\n",
        )

        ok = bench_smoke_manifest.main(
            [
                str(root),
                "--output-dir",
                str(output / "ok"),
                "--output-stem",
                "smokes",
                "--require-paired-tools",
                "--require-shebang",
                "--min-smokes",
                "1",
            ]
        )
        if rc := require(ok == 0, "manifest_pass_failed"):
            return rc
        payload = json.loads((output / "ok" / "smokes.json").read_text(encoding="utf-8"))
        if rc := require(payload["summary"]["smoke_scripts"] == 1, "wrong_smoke_count"):
            return rc
        if rc := require(payload["summary"]["areas"]["qemu"] == 1, "missing_qemu_area"):
            return rc

        write_script(
            root / "eval_orphan_ci_smoke.py",
            "def main() -> int:\n"
            "    return 0\n"
            "\n"
            'if __name__ == "__main__":\n'
            "    raise SystemExit(main())\n",
        )
        failed = bench_smoke_manifest.main(
            [
                str(root / "eval_orphan_ci_smoke.py"),
                "--output-dir",
                str(output / "failed"),
                "--output-stem",
                "smokes",
                "--require-paired-tools",
                "--require-shebang",
            ]
        )
        if rc := require(failed == 1, "manifest_failure_passed"):
            return rc
        failed_payload = json.loads((output / "failed" / "smokes.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in failed_payload["findings"]}
        if rc := require({"missing_paired_tool", "missing_shebang"} <= kinds, "missing_expected_findings"):
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
