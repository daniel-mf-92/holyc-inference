# Benchmark Trend Export

Generated: 2026-04-29T08:59:58Z
Status: pass
Trend keys: 7
Trend points: 89

Thresholds: fail_on_empty=True, fail_on_airgap=True, fail_on_telemetry=True, min_points_per_key=2, fail_on_tok_regression_pct=5.000, fail_on_wall_tok_regression_pct=5.000, fail_on_memory_growth_pct=10.000, fail_on_window_tok_regression_pct=5.000, fail_on_window_wall_tok_regression_pct=5.000, fail_on_window_memory_growth_pct=10.000, window_points=5

| Key | Points | Latest commit | Status | Air-gap | Telemetry | Guest tok/s | Guest delta % | Wall tok/s | Wall delta % | Max memory bytes | Memory delta % | Source |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| -/-/-/68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | 4 | - | planned | pass | pass | - | - | - | - | - | - | bench/results/qemu_prompt_bench_dry_run_latest.json |
| ci-airgap-smoke/synthetic-smoke/Q4_0/68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | 25 | fbd5f28e06c8 | pass | pass | pass | 160.000 | 0.000 | 590.579 | 0.000 | 67207168 | 0.000 | bench/results/qemu_prompt_bench_latest.json |
| ci-airgap-smoke/synthetic-smoke/Q4_0/no-suite | 4 | 5bf1fd7158fd | pass | pass | pass | 160.000 | 0.000 | - | - | 67207168 | 0.000 | bench/results/bench_matrix_20260427T183959Z/ci-airgap-smoke_synthetic-smoke_Q4_0/qemu_prompt_bench_latest.json |
| ci-airgap-smoke/synthetic-smoke/Q8_0/68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | 25 | 825b6d95f7f1 | pass | pass | pass | 160.000 | 0.000 | 896.605 | 0.000 | 67207168 | 0.000 | bench/results/bench_matrix_latest.json |
| ci-airgap-smoke/synthetic-smoke/Q8_0/no-suite | 4 | 5bf1fd7158fd | pass | pass | pass | 160.000 | 0.000 | - | - | 67207168 | 0.000 | bench/results/bench_matrix_20260427T183959Z/ci-airgap-smoke_synthetic-smoke_Q8_0/qemu_prompt_bench_latest.json |
| synthetic-airgap-smoke/synthetic-smoke/Q4_0/68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | 21 | 756c0a850ab9 | pass | pass | pass | 160.000 | 0.000 | 786.126 | 25.568 | 67207168 | 0.000 | bench/results/qemu_prompt_bench_20260429T012002Z.json |
| synthetic-airgap-smoke/synthetic-smoke/Q4_0/no-suite | 6 | 708be92b565a | pass | pass | pass | 160.000 | 0.000 | - | - | 67207168 | 0.000 | bench/results/qemu_prompt_bench_20260427T192232Z.json |

## Drift
- -/-/-/68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a command_sha256: 3 value(s)
- ci-airgap-smoke/synthetic-smoke/Q4_0/68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a command_sha256: 5 value(s)
- ci-airgap-smoke/synthetic-smoke/Q8_0/68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a command_sha256: 3 value(s)
- synthetic-airgap-smoke/synthetic-smoke/Q4_0/68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a command_sha256: 3 value(s)
- -/-/-/68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a launch_plan_sha256: 3 value(s)
- ci-airgap-smoke/synthetic-smoke/Q4_0/68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a launch_plan_sha256: 4 value(s)
- -/-/-/68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a environment_sha256: 2 value(s)
- ci-airgap-smoke/synthetic-smoke/Q4_0/68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a environment_sha256: 3 value(s)
- ci-airgap-smoke/synthetic-smoke/Q8_0/68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a environment_sha256: 2 value(s)
- synthetic-airgap-smoke/synthetic-smoke/Q4_0/68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a environment_sha256: 2 value(s)
