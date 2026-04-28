# Perf Regression Dashboard

Generated: 2026-04-28T10:27:29Z
Status: pass
Records: 352
Regressions: 0
P05 throughput regressions: 0
P95 TTFT regressions: 0
Host overhead regressions: 0
Sample violations: 0
Variability violations: 0
Commit coverage violations: 0
Comparison coverage violations: 0
Prompt-suite drift violations: 0
Telemetry coverage violations: 0

## Regressions

No regressions detected.

## Sample Coverage

Sample coverage requirements satisfied.

## Variability

Variability requirements satisfied.

## Commit Coverage

Commit coverage requirements satisfied.

## Comparison Coverage

Explicit comparison commits were present for all benchmark keys.

## Prompt Suite Drift

Prompt-suite hashes are consistent for comparable benchmark keys.

## Telemetry Coverage

Required telemetry fields are present for every commit point.

## Comparisons

| Key | Baseline | Candidate | Median tok/s Delta | P05 tok/s Delta | Wall tok/s Delta | Memory Delta | Median TTFT Delta | P95 TTFT Delta | Host Overhead Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/ci-short | ci-base | ci-head | -1.00% | -1.00% | -1.05% | 0.00% | -2.00% | -2.00% | -20.00% |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | d1ee8f0158c3 | e4311d1fe5fa | 0.00% | 0.00% | -17.90% | 0.00% | 0.00% | 0.00% | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | d1ee8f0158c3 | e4311d1fe5fa | 0.00% | 0.00% | -68.37% | 0.00% | 0.00% | 0.00% | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | a0126ad2c367 | d1ee8f0158c3 | 0.00% | 0.00% | 11.88% | 0.00% | 0.00% | 0.00% | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | a0126ad2c367 | d1ee8f0158c3 | 0.00% | 0.00% | 27.82% | 0.00% | 0.00% | 0.00% | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | e4b0c6c494ee | d1ee8f0158c3 | 0.00% | 0.00% | 69.44% | 0.00% | 0.00% | 0.00% | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | e4b0c6c494ee | d1ee8f0158c3 | 0.00% | 0.00% | -81.52% | 0.00% | 0.00% | 0.00% | - |

## Commit Points

| Key | Commit | Records | Tok/s Records | Wall Tok/s Records | Memory Records | TTFT Records | Host Overhead Records | P05 tok/s | Median tok/s | Median wall tok/s | Median TTFT us | P95 TTFT us | Median Host Overhead % | Tok/s CV | Max Memory Bytes | Prompt Suite |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/ci-short | ci-base | 1 | 1 | 1 | 1 | 1 | 1 | 100.000 | 100.000 | 95.000 | 50000.0 | 50000.0 | 5.000 | - | 1000000 | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/ci-short | ci-head | 1 | 1 | 1 | 1 | 1 | 1 | 101.000 | 101.000 | 96.000 | 49000.0 | 49000.0 | 4.000 | - | 1000000 | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 21ad93152049 | 6 | 6 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67207168 | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 5bf1fd7158fd | 6 | 6 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67207168 | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 786eee1bfa27 | 6 | 6 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | bf852f741851 | 6 | 6 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | bb8266801970 | 6 | 6 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 1588059f2388 | 6 | 6 | 0 | 6 | 6 | 0 | 160.000 | 160.000 | - | 12400.0 | 12400.0 | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 3b7e701e8a6b | 6 | 6 | 6 | 6 | 6 | 0 | 160.000 | 160.000 | 810.315 | 12400.0 | 12400.0 | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | a0126ad2c367 | 8 | 8 | 8 | 8 | 8 | 8 | 160.000 | 160.000 | 726.883 | 12400.0 | 12400.0 | -77.974 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | a3d460c2276c | 2 | 2 | 2 | 2 | 2 | 2 | 160.000 | 160.000 | 536.198 | 12400.0 | 12400.0 | -70.037 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | d1ee8f0158c3 | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 747.251 | 12400.0 | 12400.0 | -77.613 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | e4311d1fe5fa | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 881.025 | 12400.0 | 12400.0 | -81.838 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 21ad93152049 | 6 | 6 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67174400 | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 5bf1fd7158fd | 6 | 6 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67174400 | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 786eee1bfa27 | 6 | 6 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | bf852f741851 | 6 | 6 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | bb8266801970 | 6 | 6 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 1588059f2388 | 6 | 6 | 0 | 6 | 6 | 0 | 160.000 | 160.000 | - | 11600.0 | 11600.0 | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 3b7e701e8a6b | 6 | 6 | 6 | 6 | 6 | 0 | 160.000 | 160.000 | 653.354 | 11600.0 | 11600.0 | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | a0126ad2c367 | 8 | 8 | 8 | 8 | 8 | 8 | 160.000 | 160.000 | 164.307 | 11600.0 | 11600.0 | 109.419 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | a3d460c2276c | 2 | 2 | 2 | 2 | 2 | 2 | 160.000 | 160.000 | 499.856 | 11600.0 | 11600.0 | -67.991 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | d1ee8f0158c3 | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 340.918 | 11600.0 | 11600.0 | -46.934 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | e4311d1fe5fa | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 574.001 | 11600.0 | 11600.0 | -72.103 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | 21ad93152049 | 6 | 6 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67207168 | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | 5bf1fd7158fd | 6 | 6 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67207168 | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | 786eee1bfa27 | 6 | 6 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | bf852f741851 | 6 | 6 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | bb8266801970 | 6 | 6 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | 1588059f2388 | 6 | 6 | 0 | 6 | 6 | 0 | 160.000 | 160.000 | - | 12400.0 | 12400.0 | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | 3b7e701e8a6b | 6 | 6 | 6 | 6 | 6 | 0 | 160.000 | 160.000 | 574.549 | 12400.0 | 12400.0 | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | a0126ad2c367 | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 659.143 | 12400.0 | 12400.0 | -74.897 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | d1ee8f0158c3 | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 580.842 | 12400.0 | 12400.0 | -71.631 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | 21ad93152049 | 6 | 6 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67174400 | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | 5bf1fd7158fd | 6 | 6 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67174400 | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | 786eee1bfa27 | 6 | 6 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | bf852f741851 | 6 | 6 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | bb8266801970 | 6 | 6 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | 1588059f2388 | 6 | 6 | 0 | 6 | 6 | 0 | 160.000 | 160.000 | - | 11600.0 | 11600.0 | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | 3b7e701e8a6b | 6 | 6 | 6 | 6 | 6 | 0 | 160.000 | 160.000 | 441.891 | 11600.0 | 11600.0 | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | a0126ad2c367 | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 434.410 | 11600.0 | 11600.0 | -58.154 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | d1ee8f0158c3 | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 313.577 | 11600.0 | 11600.0 | -42.998 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 5b98f2570b1c | 3 | 3 | 0 | 0 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | - | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 85cce7c2f224 | 3 | 3 | 0 | 0 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | - | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 3f9df4f0bd7a | 3 | 3 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67207168 | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | e71bfc15a184 | 3 | 3 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67207168 | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 7f007ced1a84 | 3 | 3 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67207168 | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 708be92b565a | 3 | 3 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67207168 | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 786eee1bfa27 | 3 | 3 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | b5c013b1b596 | 3 | 3 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | ea55b6ea98dd | 3 | 3 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 36f90ed16742 | 3 | 3 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | b1e55ed7d43f | 3 | 3 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 4981d42b1a52 | 3 | 3 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 1588059f2388 | 3 | 3 | 0 | 3 | 3 | 0 | 160.000 | 160.000 | - | 12400.0 | 12400.0 | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | ec4b6dde4fd6 | 3 | 3 | 3 | 3 | 3 | 0 | 160.000 | 160.000 | 815.633 | 12400.0 | 12400.0 | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | ba875d39995e | 3 | 3 | 3 | 3 | 3 | 0 | 160.000 | 160.000 | 591.366 | 12400.0 | 12400.0 | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 9305c9617dbe | 3 | 3 | 3 | 3 | 3 | 3 | 160.000 | 160.000 | 1164.681 | 12400.0 | 12400.0 | -86.262 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | cd2ef624364d | 3 | 3 | 3 | 3 | 3 | 3 | 160.000 | 160.000 | 821.355 | 12400.0 | 12400.0 | -80.520 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | e4b0c6c494ee | 3 | 3 | 3 | 3 | 3 | 3 | 160.000 | 160.000 | 819.155 | 12400.0 | 12400.0 | -80.468 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | d1ee8f0158c3 | 3 | 3 | 3 | 3 | 3 | 3 | 160.000 | 160.000 | 250.327 | 12400.0 | 12400.0 | -36.084 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 5b98f2570b1c | 3 | 3 | 0 | 0 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | - | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 85cce7c2f224 | 3 | 3 | 0 | 0 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | - | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 3f9df4f0bd7a | 3 | 3 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67174400 | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | e71bfc15a184 | 3 | 3 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67174400 | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 7f007ced1a84 | 3 | 3 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67174400 | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 708be92b565a | 3 | 3 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67174400 | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 786eee1bfa27 | 3 | 3 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | b5c013b1b596 | 3 | 3 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | ea55b6ea98dd | 3 | 3 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 36f90ed16742 | 3 | 3 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | b1e55ed7d43f | 3 | 3 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 4981d42b1a52 | 3 | 3 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 1588059f2388 | 3 | 3 | 0 | 3 | 3 | 0 | 160.000 | 160.000 | - | 11600.0 | 11600.0 | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | ec4b6dde4fd6 | 3 | 3 | 3 | 3 | 3 | 0 | 160.000 | 160.000 | 365.827 | 11600.0 | 11600.0 | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | ba875d39995e | 3 | 3 | 3 | 3 | 3 | 0 | 160.000 | 160.000 | 401.043 | 11600.0 | 11600.0 | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 9305c9617dbe | 3 | 3 | 3 | 3 | 3 | 3 | 160.000 | 160.000 | 752.534 | 11600.0 | 11600.0 | -78.739 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | cd2ef624364d | 3 | 3 | 3 | 3 | 3 | 3 | 160.000 | 160.000 | 469.552 | 11600.0 | 11600.0 | -65.925 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | e4b0c6c494ee | 3 | 3 | 3 | 3 | 3 | 3 | 160.000 | 160.000 | 320.093 | 11600.0 | 11600.0 | -50.014 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | d1ee8f0158c3 | 3 | 3 | 3 | 3 | 3 | 3 | 160.000 | 160.000 | 581.047 | 11600.0 | 11600.0 | -72.463 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |

## Latest Summary

| Key | Records | Latest Commit | P05 tok/s | Median tok/s | Median wall tok/s | Median TTFT us | P95 TTFT us | Median Host Overhead % | Max Memory Bytes |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/ci-short | 2 | ci-head | 100.050 | 100.500 | 95.500 | 49500.0 | 49950.0 | 4.500 | 1000000 |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 64 | e4311d1fe5fa | 160.000 | 160.000 | 746.245 | 12400.0 | 12400.0 | -78.559 | 67207168 |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 64 | e4311d1fe5fa | 160.000 | 160.000 | 478.248 | 11600.0 | 11600.0 | -54.393 | 67174400 |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | 54 | d1ee8f0158c3 | 160.000 | 160.000 | 609.565 | 12400.0 | 12400.0 | -73.399 | 67207168 |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | 54 | d1ee8f0158c3 | 160.000 | 160.000 | 384.474 | 11600.0 | 11600.0 | -52.562 | 67174400 |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 57 | d1ee8f0158c3 | 160.000 | 160.000 | 805.714 | 12400.0 | 12400.0 | -80.494 | 67207168 |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 57 | d1ee8f0158c3 | 160.000 | 160.000 | 480.311 | 11600.0 | 11600.0 | -70.598 | 67174400 |
