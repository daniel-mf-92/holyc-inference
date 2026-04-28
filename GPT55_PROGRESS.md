2026-04-27T13:20:20Z | Added host-side Q4_0/Q8_0 quant audit tool with source float-leak checks, JSON results, and pytest coverage.
2026-04-27T13:33:31Z | Added host-side perf regression dashboard generator with JSON/JSONL/CSV ingestion, regression tests, and latest dashboard artifacts.
2026-04-27T13:55:17Z | Added air-gapped host-side QEMU prompt benchmark runner with smoke prompts, result normalization, docs, and tests.
2026-04-27T14:16:55Z | Added offline eval dataset packer with HellaSwag/ARC/TruthfulQA shape support, smoke binary artifact, provenance manifest, and tests.
2026-04-27T14:31:22Z | Added offline HolyC vs llama.cpp multiple-choice eval comparator with smoke predictions, result reports, docs, and tests.
2026-04-27T14:42:27Z | Added repeated QEMU prompt benchmark runs with per-prompt median/min/max summaries, Markdown reports, smoke result artifact, and tests.
2026-04-27T14:52:42Z | Added stdlib-only perf regression CI smoke gate with committed pass fixture, GitHub Actions workflow, docs, and refreshed dashboard artifact.
2026-04-27T15:19:27Z | Added host-side HCEval binary inspector with manifest/hash validation and smoke inspection results.
2026-04-27T15:27:18Z | Added host-side build benchmark comparator for QEMU prompt reports with per-prompt delta artifacts and tests.
2026-04-27T15:34:57Z | Added offline HolyC vs llama.cpp perplexity comparator with logprob/NLL input support, smoke reports, docs, and tests.
2026-04-27T15:47:56Z | Added commit-level perf regression aggregation with explicit baseline/candidate selection and refreshed dashboard artifacts.
2026-04-27T16:00:08Z | Fixed host-side perf regression memory delta reporting, added explicit memory-only regression coverage, and refreshed dashboard artifacts.
2026-04-27T16:16:27Z | Hardened QEMU prompt benchmark air-gap device rejection, added a synthetic smoke fixture, and refreshed benchmark/compare/dashboard artifacts.
2026-04-27T16:26:36Z | Added benchmark artifact air-gap audit with CI smoke coverage and latest audit result artifact.
2026-04-27T16:39:12Z | Added deterministic local eval dataset curator with provenance manifests, smoke curated artifacts, and pack/inspect validation.
2026-04-27T16:54:30Z | Extended host-side quant audit with Q4_0/Q8_0 shape checks, quant histograms, Markdown reporting, and refreshed audit artifacts.
2026-04-27T17:09:21Z | Added QEMU prompt benchmark memory telemetry normalization with refreshed smoke benchmark and dashboard artifacts.
2026-04-27T17:21:11Z | Added CSV export for QEMU prompt benchmark runs with smoke artifact refresh and coverage.
2026-04-27T17:30:28Z | Added perf regression dashboard CSV exports for commit points and regressions with refreshed artifacts and coverage.
2026-04-27T17:42:15Z | Added mixed Q4_0/Q8_0 block-file support to host-side quant audit with docs, coverage, and refreshed audit artifacts.
2026-04-27T17:53:19Z | Added QEMU prompt benchmark warmup runs recorded separately from measured throughput, with refreshed synthetic benchmark/dashboard artifacts.
2026-04-27T18:29:10Z | Added host-side QEMU benchmark matrix runner with synthetic Q4_0/Q8_0 smoke fixture and matrix result artifacts.
2026-04-27T18:41:00Z | Added CSV export for host-side benchmark matrix summaries with synthetic Q4_0/Q8_0 artifact refresh and air-gap audit.
2026-04-27T18:51:38Z | Added per-record CSV export for offline HolyC vs llama.cpp eval comparison with refreshed smoke artifact and coverage.
2026-04-27T19:02:40Z | Added deterministic answer-index balancing for local eval dataset curation with refreshed smoke curated artifacts and coverage.
2026-04-27T19:11:49Z | Added perf regression sample-coverage gating with CSV/dashboard artifacts, CI smoke checks, docs, and coverage.
2026-04-27T19:19:16Z | Added aggregate suite summaries to QEMU prompt benchmark reports with refreshed synthetic smoke artifacts and coverage.
2026-04-27T19:33:26Z | Added offline eval input audit gate for HolyC vs llama.cpp prediction coverage, metadata drift, and smoke artifacts.
2026-04-27T19:46:32Z | Added host-side benchmark prompt audit with suite hashing, length/duplicate gates, smoke artifacts, docs, and coverage.
2026-04-27T19:53:42Z | Extended air-gap artifact audit to cover benchmark matrix cell commands and refreshed passing audit output.
2026-04-27T20:05:58Z | Added prompt-suite SHA256 fingerprints to QEMU benchmark and matrix artifacts with refreshed synthetic results.
2026-04-27T20:20:32Z | Added per-record CSV export for offline perplexity comparison with refreshed smoke artifact and coverage.
2026-04-27T20:29:06Z | Added host-side benchmark result indexer with air-gap status rollups, JSON/Markdown/CSV artifacts, docs, and coverage.
2026-04-27T20:39:28Z | Added prompt-suite drift detection to the host-side benchmark result indexer with refreshed artifacts and coverage.
2026-04-27T20:47:33Z | Added offline eval dataset artifact indexer with hash/provenance validation, JSON/Markdown/CSV rollups, and refreshed smoke artifacts.
2026-04-27T20:56:32Z | Added prompt-suite drift CSV export and fail-on-drift gate to the host-side benchmark result indexer with refreshed artifacts.
2026-04-27T21:03:44Z | Added JUnit XML export for perf regression dashboards with CI smoke validation and refreshed dashboard artifacts.
2026-04-27T21:16:53Z | Added deterministic benchmark artifact manifest with SHA256/latest-key rollups, docs, and refreshed smoke artifacts.
2026-04-27T21:23:18Z | Added Markdown/CSV reporting to the benchmark air-gap audit with CI smoke coverage and refreshed audit artifacts.
2026-04-27T21:32:38Z | Added build-compare CSV export and configurable tok/s regression gate with refreshed smoke comparison artifacts.
2026-04-27T21:39:09Z | Added JUnit XML export for build benchmark comparisons with regression coverage and refreshed smoke comparison artifacts.
2026-04-27T21:45:53Z | Added macro-F1, per-answer F1, and confusion matrices to the offline HolyC vs llama.cpp eval comparator with refreshed smoke artifacts.
2026-04-27T21:53:33Z | Added fp16-to-Q16 scale statistics and max-scale gates to the host-side quant audit with docs, tests, and refreshed audit artifacts.
2026-04-27T22:02:36Z | Added build-compare memory delta reporting and optional memory-growth regression gate with refreshed comparison artifacts.
2026-04-27T22:09:50Z | Added QEMU prompt benchmark tok/s variability metrics with smoke CI coverage and refreshed synthetic artifacts.
2026-04-27T22:19:58Z | Added optional QEMU prompt benchmark tok/s CV gates with JSON/Markdown findings, smoke coverage, and refreshed benchmark indexes.
2026-04-27T22:27:49Z | qemu prompt dry-run now writes reusable air-gapped launch plan artifacts for CI review
2026-04-27T22:36:57Z | Added eval quality gates with JUnit reporting for HolyC vs llama.cpp comparisons and refreshed smoke artifacts.
2026-04-27T22:45:25Z | Added Wilson confidence intervals to offline HolyC vs llama.cpp eval comparison reports with refreshed smoke artifacts.
2026-04-27T22:55:45Z | Hardened benchmark result indexing to derive fail status from QEMU run failures and variability gate findings with refreshed index artifacts.
2026-04-27T23:02:00Z | Added JUnit XML export for benchmark result indexing with focused coverage and refreshed index artifacts.
2026-04-27T23:08:36Z | Added benchmark matrix tok/s variability gate pass-through with smoke coverage and refreshed matrix/index artifacts.
2026-04-27T23:14:13Z | Added HCEval prompt/choice/record byte stats and optional size gates to offline dataset packing and inspection with refreshed artifacts.
2026-04-27T23:19:39Z | Added JUnit XML export for QEMU prompt benchmarks with failure/variability coverage and refreshed smoke artifacts.
2026-04-27T23:25:48Z | Added offline eval dataset split-leak audit with JSON/Markdown/CSV reports, smoke artifacts, docs, and tests.
2026-04-27T23:33:06Z | Added QEMU benchmark host provenance to JSON/Markdown reports with refreshed air-gapped smoke artifacts.
2026-04-27T23:41:55Z | Added QEMU prompt byte telemetry to dry-run metadata, measured reports, CSV/Markdown outputs, and refreshed smoke artifacts.
2026-04-27T23:47:43Z | Added JUnit XML export for benchmark matrix cells with smoke coverage and refreshed matrix artifacts.
2026-04-27T23:56:01Z | Added JUnit XML export for benchmark air-gap audits with smoke coverage and refreshed audit artifacts.
2026-04-28T00:01:28Z | Added optional QEMU first-token latency telemetry with synthetic smoke coverage and refreshed benchmark artifacts.
2026-04-28T00:07:37Z | Added host wall-clock tok/s telemetry to QEMU prompt benchmarks with refreshed synthetic smoke artifacts.
2026-04-28T00:13:13Z | Hardened benchmark result indexing to audit every recorded warmup/measured QEMU command for air-gap violations with refreshed index artifacts.
2026-04-28T00:19:30Z | Added host-side Q4_0/Q8_0 quant packing distribution metrics and optional saturation/cardinality gates with refreshed audit artifacts.
2026-04-28T00:26:47Z | Added perf regression tok/s variability gating with dashboard/JUnit/CSV artifacts and refreshed CI smoke coverage.
2026-04-28T00:32:05Z | Hardened eval comparison score-vector validation for finite choice-aligned scores with refreshed smoke eval artifacts.
2026-04-28T00:38:33Z | Added score-vector calibration metrics to offline HolyC vs llama.cpp eval reports with refreshed smoke artifacts.
2026-04-28T00:45:59Z | Added JUnit XML export for offline eval dataset artifact indexing with refreshed dataset index artifacts.
2026-04-28T00:55:23Z | Added perf regression host wall-clock tok/s tracking and optional regression gate with refreshed dashboard artifacts.
2026-04-28T01:02:00Z | Added host wall-clock tok/s build comparison deltas and optional regression gate with refreshed build compare artifacts.
2026-04-28T01:09:23Z | Added perf regression commit-coverage gate with dashboard/JUnit/CSV output and CI smoke coverage.
2026-04-28T01:15:40Z | Added perf regression explicit comparison-coverage gate with dashboard/JUnit/CSV output and CI smoke coverage.
2026-04-28T01:29:28Z | Added eval input audit CSV/JUnit CI artifacts with smoke coverage and refreshed reports.
2026-04-28T01:44:14Z | Added perplexity comparison quality gates with JUnit output, smoke coverage, and refreshed reports.
2026-04-28T02:00:06Z | Added benchmark result telemetry coverage gating with refreshed index and artifact manifest reports.
2026-04-28T02:07:03Z | Added perf regression prompt-suite drift gating with CSV/JUnit/dashboard output and refreshed smoke artifacts.
2026-04-28T02:17:48Z | Added benchmark artifact manifest JUnit CI output with empty-manifest failure handling and refreshed reports.
2026-04-28T02:22:39Z | Added HCEval inspection JUnit output with refreshed smoke dataset inspection artifacts.
2026-04-28T02:36:16Z | Added benchmark artifact commit metadata and opt-in stale-commit gating with refreshed matrix/index artifacts.
2026-04-28T02:47:15Z | Added opt-in QEMU prompt benchmark telemetry gates with refreshed synthetic smoke/index artifacts.
2026-04-28T02:55:32Z | Added benchmark artifact manifest commit provenance and opt-in stale-commit gating with refreshed manifest artifacts.
2026-04-28T05:43:50Z | Added opt-in benchmark result freshness gating with age telemetry, JUnit coverage, docs, and refreshed index artifacts.
2026-04-28T03:21:49Z | Hardened benchmark result index gates for commit metadata, telemetry-only failures, and refreshed result/manifest artifacts.
2026-04-28T03:01:36Z | Added perf regression telemetry coverage gates with CSV/JUnit/dashboard output and refreshed smoke artifacts.
2026-04-28T03:11:14Z | Added perf regression first-token latency tracking with TTFT gates, CI fixture coverage, and refreshed dashboard artifacts.
2026-04-28T03:33:02Z | Added prompt audit CSV/JUnit CI artifacts with smoke coverage and refreshed reports.
2026-04-28T03:42:52Z | Added dataset leak audit JUnit CI output with refreshed smoke artifacts.
2026-04-28T04:00:04Z | Added QEMU prompt benchmark host overhead timing metrics with refreshed smoke/index artifacts.
2026-04-28T04:10:03Z | Added class-specific benchmark artifact manifest gates with smoke coverage and refreshed manifest artifacts.
2026-04-28T04:17:47Z | Added QEMU argument-file support to prompt benchmarks and matrix dry-runs with air-gap validation coverage.
2026-04-28T04:24:57Z | Added eval_compare paired correctness and exact McNemar reporting with refreshed smoke artifacts.
2026-04-28T04:31:49Z | Added stdlib dataset CI smoke gate covering curate, pack, inspect, leak audit, index, and leakage rejection.
2026-04-28T04:37:49Z | Added per-dataset eval comparison breakdowns with refreshed smoke reports and focused coverage.
2026-04-28T04:45:52Z | Added opt-in QEMU TTFT telemetry gates with refreshed synthetic benchmark, index, and air-gap audit artifacts.
2026-04-28T04:54:14Z | Added QEMU prompt benchmark per-token latency metrics with refreshed smoke, index, manifest, and air-gap artifacts.
2026-04-28T05:11:53Z | Added static QEMU source air-gap audit with JSON/Markdown/CSV/JUnit reports, docs, smoke coverage, and latest artifact.
2026-04-28T05:18:28Z | Added QEMU prompt benchmark wall-clock tok/s and memory telemetry gates with smoke coverage and refreshed artifacts.
2026-04-28T05:24:14Z | Added quant audit CSV/JUnit CI outputs with smoke coverage and refreshed latest artifacts.
2026-04-28T05:32:49Z | Added QEMU prompt benchmark host-overhead telemetry gates with smoke coverage and refreshed benchmark artifacts.
2026-04-28T05:55:23Z | Added build comparison TTFT deltas and opt-in first-token latency growth gating with refreshed reports.
2026-04-28T06:05:31Z | Added quant audit zero-scale/nonzero-payload detection with opt-in failure gate, docs, tests, and refreshed reports.
2026-04-28T06:11:51Z | Added benchmark artifact manifest freshness telemetry and opt-in stale-artifact gating with refreshed reports.
2026-04-28T06:22:13Z | Added QEMU command SHA256 fingerprints to prompt benchmark and matrix artifacts with refreshed smoke reports.
2026-04-28T06:29:39Z | Added build-compare OK-run coverage gating with CSV/JUnit reporting, docs, tests, and refreshed comparison artifacts.
2026-04-28T06:38:41Z | Added build-compare prompt-suite drift gating with CSV/JUnit reporting, docs, tests, and refreshed comparison artifacts.
2026-04-28T06:48:02Z | Added QEMU prompt benchmark low-tail tok/s percentile/spread reporting with refreshed synthetic artifacts.
2026-04-28T06:57:32Z | Added build-compare P05 tok/s deltas and opt-in low-tail regression gating with refreshed comparison artifacts.
2026-04-28T07:46:07Z | Added quant audit per-block distribution gates for low diversity and saturation with refreshed source audit artifacts.
2026-04-28T07:53:59Z | Added perf regression P05 tok/s tracking and opt-in low-tail throughput gating with refreshed dashboard artifacts.
2026-04-28T08:23:52Z | Added benchmark result command-hash drift indexing, CSV/JUnit outputs, opt-in gates, and refreshed result index artifacts.
2026-04-28T08:42:47Z | Added deterministic per-dataset and per-split caps to local eval dataset curation with refreshed smoke artifacts.
2026-04-28T08:59:27Z | Added deterministic per-dataset/split curation caps with smoke gate coverage and refreshed dataset artifacts.
2026-04-28T09:06:53Z | Made dataset index manifests portable by resolving relative artifact paths from manifest directories and refreshed dataset index artifacts.
2026-04-28T09:14:36Z | Added build-compare P05 wall-clock tok/s deltas and opt-in low-tail wall-throughput gating with refreshed comparison artifacts.
2026-04-28T09:43:31Z | Added perf regression P95 first-token latency tracking and opt-in tail-latency gating with refreshed dashboard artifacts.
2026-04-28T09:50:00Z | Added command-hash metadata propagation and opt-in gating to benchmark artifact manifests with refreshed reports.
2026-04-28T09:57:13Z | Added full-history CSV export to benchmark artifact manifests with docs, tests, and refreshed reports.
2026-04-28T10:06:08Z | Added eval_compare per-dataset/split breakdown CSV output and opt-in breakdown quality gates with refreshed smoke artifacts.
2026-04-28T10:15:29Z | Added perf regression baseline/candidate comparison CSV dashboards with docs, smoke checks, and refreshed artifacts.
2026-04-28T10:28:58Z | Added perf regression QEMU host-overhead telemetry, comparison deltas, opt-in gating, smoke checks, and refreshed dashboards.
2026-04-28T10:57:07Z | Added matrix/result-index latency and host-overhead rollups with smoke coverage and refreshed benchmark artifacts.
2026-04-28T11:10:38Z | Added eval dataset choice-count curation filters with smoke gate coverage and refreshed dataset artifacts.
2026-04-28T11:41:20Z | Added eval dataset byte-budget curation filters with smoke gate coverage and refreshed dataset artifacts.
2026-04-28T11:49:31Z | Added focused eval dataset provenance audit with JSON/Markdown/CSV/JUnit artifacts and smoke gate coverage.
2026-04-28T11:56:45Z | Added eval input prediction-distribution telemetry and opt-in majority-prediction collapse gate with refreshed smoke artifacts.
2026-04-28T12:02:39Z | Added eval input score-vector coverage telemetry and opt-in score-coverage gate with refreshed smoke artifacts.
2026-04-28T12:08:44Z | Added eval dataset provenance answer-histogram validation and opt-in majority-answer skew gate with refreshed artifacts.
2026-04-28T12:18:20Z | Added per-dataset eval provenance answer histograms and opt-in per-dataset skew gate with refreshed dataset artifacts.
2026-04-28T12:26:11Z | Added eval input gold-answer skew telemetry and opt-in gate with refreshed smoke artifacts.
2026-04-28T12:34:53Z | Added quant audit zero/subnormal fp16 scale gates with smoke coverage and refreshed quant audit artifacts.
2026-04-28T12:43:32Z | Added qemu_prompt_bench launch-budget guardrail with dry-run metadata, smoke coverage, and refreshed benchmark artifacts.
2026-04-28T12:53:37Z | Added qemu_prompt_bench host child CPU telemetry with smoke coverage and refreshed benchmark artifacts.
2026-04-28T13:02:04Z | Added eval dataset provenance/source-shard curation caps with smoke coverage and refreshed dataset artifacts.
2026-04-28T13:07:42Z | Added eval_compare confusion-matrix CSV output with docs and refreshed smoke comparison artifacts.
2026-04-28T13:18:00Z | Added eval_compare calibration-bin CSV output with smoke coverage and refreshed comparison artifacts.
2026-04-28T13:28:19Z | Added benchmark matrix/result-index host child CPU rollups with smoke coverage and refreshed benchmark artifacts.
2026-04-28T13:38:25Z | Added benchmark matrix prompt-byte rollups with smoke coverage and refreshed matrix artifacts.
2026-04-28T13:46:24Z | Added QEMU prompt benchmark aggregate summary CSV output with smoke artifact refresh and coverage.
2026-04-28T14:14:50Z | Added perf regression guest/wall us-per-token telemetry gates with smoke coverage and refreshed dashboards.
2026-04-28T14:20:41Z | Added eval_compare engine-disagreement CSV output with smoke artifact refresh and coverage.
2026-04-28T14:28:25Z | Added qemu_prompt_bench host child CPU efficiency telemetry/gate with smoke artifact refresh and coverage.
2026-04-28T14:40:50Z | Added qemu_prompt_bench host child peak RSS telemetry/gates with matrix rollups and refreshed smoke artifacts.
2026-04-28T14:50:07Z | Added benchmark matrix aggregate summary CSV output with smoke coverage and refreshed matrix artifacts.
2026-04-28T15:21:02Z | Added result-index and artifact-manifest host child CPU efficiency/RSS rollups with refreshed benchmark artifacts.
2026-04-28T15:39:29Z | Added qemu_prompt_bench failure-count rollups for failed, timeout, and nonzero-exit runs with refreshed smoke artifacts.
2026-04-28T15:56:34Z | Added benchmark environment fingerprints and env-drift gating with refreshed index/manifest artifacts.
2026-04-28T16:07:07Z | Added within-split dataset payload dedupe and answer-conflict rejection with smoke coverage and refreshed dataset artifacts.
2026-04-28T16:17:08Z | Added split-level eval dataset answer histograms and split skew gating with refreshed dataset artifacts.
2026-04-28T16:52:40Z | Added perf regression emitted-token telemetry/drop gates with smoke coverage and refreshed dashboards.
2026-04-28T17:04:58Z | Added build_compare host latency, child CPU/RSS, and CPU-efficiency deltas/gates with smoke coverage.
2026-04-28T17:31:31Z | Added benchmark result command_sha256 recomputation gate and refreshed synthetic air-gap bench artifacts
