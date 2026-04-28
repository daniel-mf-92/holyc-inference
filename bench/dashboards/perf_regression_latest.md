# Perf Regression Dashboard

Generated: 2026-04-28T14:14:39Z
Status: pass
Records: 471
Regressions: 0
P05 throughput regressions: 0
Token latency regressions: 0
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

| Key | Baseline | Candidate | Median tok/s Delta | P05 tok/s Delta | Wall tok/s Delta | us/token Delta | Wall us/token Delta | Memory Delta | Median TTFT Delta | P95 TTFT Delta | Host Overhead Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bench_result_index_latest/ci-airgap-smoke/synthetic-smoke/Q4_0/- | c8e896a45950 | 6a8bc0e18988 | - | - | - | 0.00% | 688.66% | - | - | - | - |
| bench_result_index_latest/ci-airgap-smoke/synthetic-smoke/Q8_0/- | c8e896a45950 | 6a8bc0e18988 | - | - | - | 0.00% | 433.56% | - | - | - | - |
| bench_result_index_latest/synthetic-airgap-smoke/synthetic-smoke/Q4_0/- | b943c38d923f | fc1728848aad | - | - | - | 0.00% | 15.98% | - | - | - | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 6a8bc0e18988 | f884c29d4b92 | 0.00% | 0.00% | -782.87% | 0.00% | -88.64% | 0.00% | 0.00% | 0.00% | -191.37% |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 6a8bc0e18988 | f884c29d4b92 | 0.00% | 0.00% | -1418.63% | 0.00% | -93.93% | 0.00% | 0.00% | 0.00% | -126.66% |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | 6a8bc0e18988 | f884c29d4b92 | 0.00% | 0.00% | -34.47% | 0.00% | -26.40% | 0.00% | 0.00% | 0.00% | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | 6a8bc0e18988 | f884c29d4b92 | 0.00% | 0.00% | -1853.32% | 0.00% | -94.87% | 0.00% | 0.00% | 0.00% | -111.03% |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | fc1728848aad | d580dff2e3e3 | 0.00% | 0.00% | -39.49% | 0.00% | -28.31% | 0.00% | 0.00% | 0.00% | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | fc1728848aad | d580dff2e3e3 | 0.00% | 0.00% | 19.05% | 0.00% | 23.53% | 0.00% | 0.00% | 0.00% | - |

## Commit Points

| Key | Commit | Records | Tok/s Records | Wall Tok/s Records | us/token Records | Wall us/token Records | Memory Records | TTFT Records | Host Overhead Records | P05 tok/s | Median tok/s | Median wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Median TTFT us | P95 TTFT us | Median Host Overhead % | Tok/s CV | Max Memory Bytes | Prompt Suite |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bench_matrix_latest/ci-airgap-smoke/synthetic-smoke/Q4_0/- | f884c29d4b92 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | - | - | - | 6250.000 | 6250.000 | 1400.276 | 1400.276 | - | - | - | - | - | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| bench_matrix_latest/ci-airgap-smoke/synthetic-smoke/Q8_0/- | f884c29d4b92 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | - | - | - | 6250.000 | 6250.000 | 1817.432 | 1817.432 | - | - | - | - | - | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| bench_result_index_latest/ci-airgap-smoke/synthetic-smoke/Q4_0/- | a3d460c2276c | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | - | - | - | 6250.000 | 6250.000 | 1996.370 | 1996.370 | - | - | - | - | - | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| bench_result_index_latest/ci-airgap-smoke/synthetic-smoke/Q4_0/- | d1ee8f0158c3 | 2 | 0 | 0 | 2 | 2 | 0 | 0 | 0 | - | - | - | 6250.000 | 6250.000 | 1940.141 | 1940.141 | - | - | - | - | - | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| bench_result_index_latest/ci-airgap-smoke/synthetic-smoke/Q4_0/- | e4311d1fe5fa | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | - | - | - | 6250.000 | 6250.000 | 1418.792 | 1418.792 | - | - | - | - | - | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| bench_result_index_latest/ci-airgap-smoke/synthetic-smoke/Q4_0/- | c8e896a45950 | 2 | 0 | 0 | 2 | 2 | 0 | 0 | 0 | - | - | - | 6250.000 | 6250.000 | 1884.995 | 1884.995 | - | - | - | - | - | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| bench_result_index_latest/ci-airgap-smoke/synthetic-smoke/Q4_0/- | 6a8bc0e18988 | 3 | 0 | 0 | 3 | 3 | 0 | 0 | 0 | - | - | - | 6250.000 | 6250.000 | 14866.271 | 14866.271 | - | - | - | - | - | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| bench_result_index_latest/ci-airgap-smoke/synthetic-smoke/Q8_0/- | d1ee8f0158c3 | 2 | 0 | 0 | 2 | 2 | 0 | 0 | 0 | - | - | - | 6250.000 | 6250.000 | 2241.964 | 2241.964 | - | - | - | - | - | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| bench_result_index_latest/ci-airgap-smoke/synthetic-smoke/Q8_0/- | c8e896a45950 | 2 | 0 | 0 | 2 | 2 | 0 | 0 | 0 | - | - | - | 6250.000 | 6250.000 | 4158.052 | 4158.052 | - | - | - | - | - | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| bench_result_index_latest/ci-airgap-smoke/synthetic-smoke/Q8_0/- | 6a8bc0e18988 | 3 | 0 | 0 | 3 | 3 | 0 | 0 | 0 | - | - | - | 6250.000 | 6250.000 | 22185.719 | 22185.719 | - | - | - | - | - | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| bench_result_index_latest/synthetic-airgap-smoke/synthetic-smoke/Q4_0/- | cd2ef624364d | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | - | - | - | 6250.000 | 6250.000 | 1576.130 | 1576.130 | - | - | - | - | - | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| bench_result_index_latest/synthetic-airgap-smoke/synthetic-smoke/Q4_0/- | e4b0c6c494ee | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | - | - | - | 6250.000 | 6250.000 | 2337.615 | 2337.615 | - | - | - | - | - | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| bench_result_index_latest/synthetic-airgap-smoke/synthetic-smoke/Q4_0/- | d1ee8f0158c3 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | - | - | - | 6250.000 | 6250.000 | 1750.328 | 1750.328 | - | - | - | - | - | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| bench_result_index_latest/synthetic-airgap-smoke/synthetic-smoke/Q4_0/- | b943c38d923f | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | - | - | - | 6250.000 | 6250.000 | 1695.453 | 1695.453 | - | - | - | - | - | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| bench_result_index_latest/synthetic-airgap-smoke/synthetic-smoke/Q4_0/- | fc1728848aad | 2 | 0 | 0 | 2 | 2 | 0 | 0 | 0 | - | - | - | 6250.000 | 6250.000 | 1966.359 | 1966.359 | - | - | - | - | - | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 21ad93152049 | 6 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67207168 | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 5bf1fd7158fd | 6 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67207168 | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 786eee1bfa27 | 6 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | bf852f741851 | 6 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | bb8266801970 | 6 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 1588059f2388 | 6 | 6 | 0 | 0 | 0 | 6 | 6 | 0 | 160.000 | 160.000 | - | - | - | - | - | 12400.0 | 12400.0 | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 3b7e701e8a6b | 6 | 6 | 6 | 0 | 0 | 6 | 6 | 0 | 160.000 | 160.000 | 810.315 | - | - | - | - | 12400.0 | 12400.0 | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | a0126ad2c367 | 8 | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 160.000 | 160.000 | 726.883 | - | - | - | - | 12400.0 | 12400.0 | -77.974 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | a3d460c2276c | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 160.000 | 160.000 | 536.198 | 6250.000 | 6250.000 | 1872.698 | 1980.876 | 12400.0 | 12400.0 | -70.037 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | d1ee8f0158c3 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 747.251 | 6250.000 | 6250.000 | 1399.198 | 1691.250 | 12400.0 | 12400.0 | -77.613 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | e4311d1fe5fa | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 160.000 | 160.000 | 881.025 | 6250.000 | 6250.000 | 1135.104 | 1142.679 | 12400.0 | 12400.0 | -81.838 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | c8e896a45950 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 663.876 | 6250.000 | 6250.000 | 1510.229 | 1587.208 | 12400.0 | 12400.0 | -75.836 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 6a8bc0e18988 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 86.403 | 6250.000 | 6250.000 | 11642.104 | 12534.417 | 12400.0 | 12400.0 | 86.274 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | f884c29d4b92 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 762.827 | 6250.000 | 6250.000 | 1323.010 | 1449.521 | 12400.0 | 12400.0 | -78.832 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 21ad93152049 | 6 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67174400 | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 5bf1fd7158fd | 6 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67174400 | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 786eee1bfa27 | 6 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | bf852f741851 | 6 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | bb8266801970 | 6 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 1588059f2388 | 6 | 6 | 0 | 0 | 0 | 6 | 6 | 0 | 160.000 | 160.000 | - | - | - | - | - | 11600.0 | 11600.0 | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 3b7e701e8a6b | 6 | 6 | 6 | 0 | 0 | 6 | 6 | 0 | 160.000 | 160.000 | 653.354 | - | - | - | - | 11600.0 | 11600.0 | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | a0126ad2c367 | 8 | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 160.000 | 160.000 | 164.307 | - | - | - | - | 11600.0 | 11600.0 | 109.419 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | a3d460c2276c | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 160.000 | 160.000 | 499.856 | 6250.000 | 6250.000 | 2000.578 | 2001.239 | 11600.0 | 11600.0 | -67.991 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | d1ee8f0158c3 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 340.918 | 6250.000 | 6250.000 | 3316.609 | 4444.188 | 11600.0 | 11600.0 | -46.934 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | e4311d1fe5fa | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 160.000 | 160.000 | 574.001 | 6250.000 | 6250.000 | 1743.562 | 1788.112 | 11600.0 | 11600.0 | -72.103 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | c8e896a45950 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 414.952 | 6250.000 | 6250.000 | 2436.297 | 2689.812 | 11600.0 | 11600.0 | -61.019 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 6a8bc0e18988 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 45.117 | 6250.000 | 6250.000 | 24180.703 | 31163.281 | 11600.0 | 11600.0 | 286.891 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | f884c29d4b92 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 685.166 | 6250.000 | 6250.000 | 1468.969 | 1586.906 | 11600.0 | 11600.0 | -76.496 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | 21ad93152049 | 6 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67207168 | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | 5bf1fd7158fd | 6 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67207168 | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | 786eee1bfa27 | 6 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | bf852f741851 | 6 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | bb8266801970 | 6 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | 1588059f2388 | 6 | 6 | 0 | 0 | 0 | 6 | 6 | 0 | 160.000 | 160.000 | - | - | - | - | - | 12400.0 | 12400.0 | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | 3b7e701e8a6b | 6 | 6 | 6 | 0 | 0 | 6 | 6 | 0 | 160.000 | 160.000 | 574.549 | - | - | - | - | 12400.0 | 12400.0 | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | a0126ad2c367 | 6 | 6 | 6 | 0 | 0 | 6 | 6 | 6 | 160.000 | 160.000 | 659.143 | - | - | - | - | 12400.0 | 12400.0 | -74.897 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | d1ee8f0158c3 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 580.842 | 6250.000 | 6250.000 | 1773.062 | 2075.021 | 12400.0 | 12400.0 | -71.631 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | c8e896a45950 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 426.830 | 6250.000 | 6250.000 | 3046.656 | 4510.979 | 12400.0 | 12400.0 | -51.253 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | 6a8bc0e18988 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 498.569 | 6250.000 | 6250.000 | 2027.750 | 2239.000 | 12400.0 | 12400.0 | -67.556 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | f884c29d4b92 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 670.445 | 6250.000 | 6250.000 | 1492.500 | 1530.208 | 12400.0 | 12400.0 | -76.120 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | 21ad93152049 | 6 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67174400 | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | 5bf1fd7158fd | 6 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67174400 | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | 786eee1bfa27 | 6 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | bf852f741851 | 6 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | bb8266801970 | 6 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | 1588059f2388 | 6 | 6 | 0 | 0 | 0 | 6 | 6 | 0 | 160.000 | 160.000 | - | - | - | - | - | 11600.0 | 11600.0 | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | 3b7e701e8a6b | 6 | 6 | 6 | 0 | 0 | 6 | 6 | 0 | 160.000 | 160.000 | 441.891 | - | - | - | - | 11600.0 | 11600.0 | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | a0126ad2c367 | 6 | 6 | 6 | 0 | 0 | 6 | 6 | 6 | 160.000 | 160.000 | 434.410 | - | - | - | - | 11600.0 | 11600.0 | -58.154 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | d1ee8f0158c3 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 313.577 | 6250.000 | 6250.000 | 3562.641 | 4716.375 | 11600.0 | 11600.0 | -42.998 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | c8e896a45950 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 195.099 | 6250.000 | 6250.000 | 5827.375 | 7849.625 | 11600.0 | 11600.0 | -6.762 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | 6a8bc0e18988 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 23.296 | 6250.000 | 6250.000 | 42941.891 | 43751.344 | 11600.0 | 11600.0 | 587.070 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | f884c29d4b92 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 160.000 | 160.000 | 455.037 | 6250.000 | 6250.000 | 2201.922 | 2299.188 | 11600.0 | 11600.0 | -64.769 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 5b98f2570b1c | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | - | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 85cce7c2f224 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | - | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 3f9df4f0bd7a | 3 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67207168 | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | e71bfc15a184 | 3 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67207168 | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 7f007ced1a84 | 3 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67207168 | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 708be92b565a | 3 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67207168 | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 786eee1bfa27 | 3 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | b5c013b1b596 | 3 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | ea55b6ea98dd | 3 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 36f90ed16742 | 3 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | b1e55ed7d43f | 3 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 4981d42b1a52 | 3 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 1588059f2388 | 3 | 3 | 0 | 0 | 0 | 3 | 3 | 0 | 160.000 | 160.000 | - | - | - | - | - | 12400.0 | 12400.0 | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | ec4b6dde4fd6 | 3 | 3 | 3 | 0 | 0 | 3 | 3 | 0 | 160.000 | 160.000 | 815.633 | - | - | - | - | 12400.0 | 12400.0 | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | ba875d39995e | 3 | 3 | 3 | 0 | 0 | 3 | 3 | 0 | 160.000 | 160.000 | 591.366 | - | - | - | - | 12400.0 | 12400.0 | - | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 9305c9617dbe | 3 | 3 | 3 | 0 | 0 | 3 | 3 | 3 | 160.000 | 160.000 | 1164.681 | - | - | - | - | 12400.0 | 12400.0 | -86.262 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | cd2ef624364d | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 160.000 | 160.000 | 821.355 | 6250.000 | 6250.000 | 1217.500 | 1252.694 | 12400.0 | 12400.0 | -80.520 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | e4b0c6c494ee | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 160.000 | 160.000 | 819.155 | 6250.000 | 6250.000 | 1220.771 | 1550.865 | 12400.0 | 12400.0 | -80.468 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | d1ee8f0158c3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 160.000 | 160.000 | 250.327 | 6250.000 | 6250.000 | 3994.771 | 4770.177 | 12400.0 | 12400.0 | -36.084 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | b943c38d923f | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 160.000 | 160.000 | 1193.317 | 6250.000 | 6250.000 | 838.000 | 1868.219 | 12400.0 | 12400.0 | -86.592 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | fc1728848aad | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 160.000 | 160.000 | 734.349 | 6250.000 | 6250.000 | 1361.750 | 17014.756 | 12400.0 | 12400.0 | -78.212 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | d580dff2e3e3 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 160.000 | 160.000 | 1024.350 | 6250.000 | 6250.000 | 976.229 | 1549.417 | 12400.0 | 12400.0 | -84.380 | 0.00% | 67207168 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 5b98f2570b1c | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | - | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 85cce7c2f224 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | - | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 3f9df4f0bd7a | 3 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67174400 | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | e71bfc15a184 | 3 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67174400 | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 7f007ced1a84 | 3 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67174400 | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 708be92b565a | 3 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67174400 | - |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 786eee1bfa27 | 3 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | b5c013b1b596 | 3 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | ea55b6ea98dd | 3 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 36f90ed16742 | 3 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | b1e55ed7d43f | 3 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 4981d42b1a52 | 3 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 160.000 | 160.000 | - | - | - | - | - | - | - | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 1588059f2388 | 3 | 3 | 0 | 0 | 0 | 3 | 3 | 0 | 160.000 | 160.000 | - | - | - | - | - | 11600.0 | 11600.0 | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | ec4b6dde4fd6 | 3 | 3 | 3 | 0 | 0 | 3 | 3 | 0 | 160.000 | 160.000 | 365.827 | - | - | - | - | 11600.0 | 11600.0 | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | ba875d39995e | 3 | 3 | 3 | 0 | 0 | 3 | 3 | 0 | 160.000 | 160.000 | 401.043 | - | - | - | - | 11600.0 | 11600.0 | - | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 9305c9617dbe | 3 | 3 | 3 | 0 | 0 | 3 | 3 | 3 | 160.000 | 160.000 | 752.534 | - | - | - | - | 11600.0 | 11600.0 | -78.739 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | cd2ef624364d | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 160.000 | 160.000 | 469.552 | 6250.000 | 6250.000 | 2129.688 | 2420.303 | 11600.0 | 11600.0 | -65.925 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | e4b0c6c494ee | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 160.000 | 160.000 | 320.093 | 6250.000 | 6250.000 | 3124.094 | 3144.372 | 11600.0 | 11600.0 | -50.014 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | d1ee8f0158c3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 160.000 | 160.000 | 581.047 | 6250.000 | 6250.000 | 1721.031 | 1773.766 | 11600.0 | 11600.0 | -72.463 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | b943c38d923f | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 160.000 | 160.000 | 173.654 | 6250.000 | 6250.000 | 5758.562 | 6431.931 | 11600.0 | 11600.0 | -7.863 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | fc1728848aad | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 160.000 | 160.000 | 456.321 | 6250.000 | 6250.000 | 2191.438 | 2667.791 | 11600.0 | 11600.0 | -64.937 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | d580dff2e3e3 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 160.000 | 160.000 | 369.391 | 6250.000 | 6250.000 | 2707.156 | 23130.344 | 11600.0 | 11600.0 | -56.685 | 0.00% | 67174400 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a |
| qemu_prompt_bench_summary_latest/default/-/-/- | unknown | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | - | - | - | 6250.000 | 6250.000 | 1569.411 | 1569.411 | - | - | - | - | - | - |
| qemu_prompt_bench_summary_latest/default/-/-/smoke-code | unknown | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | - | - | - | 6250.000 | 6250.000 | 976.229 | 976.229 | - | - | - | - | - | - |
| qemu_prompt_bench_summary_latest/default/-/-/smoke-short | unknown | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | - | - | - | 6250.000 | 6250.000 | 2707.156 | 2707.156 | - | - | - | - | - | - |

## Latest Summary

| Key | Records | Latest Commit | P05 tok/s | Median tok/s | Median wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Median TTFT us | P95 TTFT us | Median Host Overhead % | Max Memory Bytes |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bench_matrix_latest/ci-airgap-smoke/synthetic-smoke/Q4_0/- | 1 | f884c29d4b92 | - | - | - | 6250.000 | 6250.000 | 1400.276 | 1400.276 | - | - | - | - |
| bench_matrix_latest/ci-airgap-smoke/synthetic-smoke/Q8_0/- | 1 | f884c29d4b92 | - | - | - | 6250.000 | 6250.000 | 1817.432 | 1817.432 | - | - | - | - |
| bench_result_index_latest/ci-airgap-smoke/synthetic-smoke/Q4_0/- | 9 | 6a8bc0e18988 | - | - | - | 6250.000 | 6250.000 | 1940.141 | 14866.271 | - | - | - | - |
| bench_result_index_latest/ci-airgap-smoke/synthetic-smoke/Q8_0/- | 7 | 6a8bc0e18988 | - | - | - | 6250.000 | 6250.000 | 4158.052 | 22185.719 | - | - | - | - |
| bench_result_index_latest/synthetic-airgap-smoke/synthetic-smoke/Q4_0/- | 6 | fc1728848aad | - | - | - | 6250.000 | 6250.000 | 1858.344 | 2244.801 | - | - | - | - |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 78 | f884c29d4b92 | 160.000 | 160.000 | 708.236 | 6250.000 | 6250.000 | 1518.365 | 12534.417 | 12400.0 | 12400.0 | -77.068 | 67207168 |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 78 | f884c29d4b92 | 160.000 | 160.000 | 456.823 | 6250.000 | 6250.000 | 2189.031 | 31163.281 | 11600.0 | 11600.0 | -56.963 | 67174400 |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-code | 72 | f884c29d4b92 | 160.000 | 160.000 | 591.244 | 6250.000 | 6250.000 | 1699.417 | 4510.979 | 12400.0 | 12400.0 | -72.809 | 67207168 |
| qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q8_0/smoke-short | 72 | f884c29d4b92 | 160.000 | 160.000 | 318.924 | 6250.000 | 6250.000 | 4260.750 | 43751.344 | 11600.0 | 11600.0 | -41.392 | 67174400 |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 72 | d580dff2e3e3 | 160.000 | 160.000 | 819.155 | 6250.000 | 6250.000 | 1219.135 | 4727.099 | 12400.0 | 12400.0 | -81.613 | 67207168 |
| qemu_prompt/synthetic-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 72 | d580dff2e3e3 | 160.000 | 160.000 | 407.732 | 6250.000 | 6250.000 | 2579.875 | 23130.344 | 11600.0 | 11600.0 | -64.937 | 67174400 |
| qemu_prompt_bench_summary_latest/default/-/-/- | 1 | unknown | - | - | - | 6250.000 | 6250.000 | 1569.411 | 1569.411 | - | - | - | - |
| qemu_prompt_bench_summary_latest/default/-/-/smoke-code | 1 | unknown | - | - | - | 6250.000 | 6250.000 | 976.229 | 976.229 | - | - | - | - |
| qemu_prompt_bench_summary_latest/default/-/-/smoke-short | 1 | unknown | - | - | - | 6250.000 | 6250.000 | 2707.156 | 2707.156 | - | - | - | - |
