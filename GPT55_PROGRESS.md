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
2026-04-29T00:31:55Z | Added perf regression host/QEMU environment drift reporting with opt-in CI gating and refreshed dashboard artifacts.
2026-04-28T22:12:06Z | Added dataset/split answer-skew telemetry and gates to provenance audits with smoke coverage and refreshed dataset artifacts.
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
2026-04-28T17:59:13Z | Added qemu_source_audit checks for JSON/.args QEMU argument fragments with smoke coverage and refreshed source audit artifacts.
2026-04-28T18:14:26Z | Added perf regression P05 wall-clock tok/s rollups/gate with smoke coverage and refreshed dashboards.
2026-04-28T18:27:23Z | Added eval dataset provenance contribution counts/skew gate with smoke coverage and refreshed dataset artifacts.
2026-04-28T18:33:49Z | Added eval_compare paired McNemar loss gating with smoke artifact refresh and coverage.
2026-04-28T18:50:28Z | Added eval dataset schema audit gate with smoke reports and dataset CI coverage.
2026-04-28T19:05:57Z | Added build_compare QEMU command hash drift gate with CSV/JUnit/Markdown coverage.
2026-04-28T19:12:07Z | Added eval dataset schema answer-skew telemetry/gates with smoke coverage and refreshed schema artifacts.
2026-04-28T19:18:48Z | Added HellaSwag ctx_a/ctx_b prompt normalization for offline eval dataset packing with smoke coverage.
2026-04-29T01:45:48Z | Added eval dataset choice-audit gate for prompts that leak any candidate choice text with smoke coverage.
2026-04-28T19:27:43Z | Added eval_compare score-margin telemetry, CSV output, optional HolyC margin gates, and refreshed smoke artifacts.
2026-04-28T19:36:30Z | Added expanded benchmark artifact manifest telemetry fields with refreshed manifest artifacts.
2026-04-28T19:49:43Z | Added HCEval record span telemetry to pack manifests and inspection reports with refreshed smoke dataset artifacts.
2026-04-28T20:01:33Z | Added eval_compare score-vector NLL/perplexity telemetry and gates with refreshed smoke artifacts.
2026-04-28T20:09:53Z | Added benchmark matrix/index/manifest emitted-token and elapsed-time rollups with refreshed smoke artifacts.
2026-04-28T20:19:12Z | Added perplexity_compare P95 per-record NLL delta telemetry/gate with refreshed smoke artifacts.
2026-04-28T20:28:19Z | Added perplexity_compare signed P95 record NLL tail gate with smoke coverage and refreshed reports.
2026-04-28T20:38:38Z | Added QEMU command alias coverage to air-gap audits and benchmark indexing with smoke validation.
2026-04-28T20:53:30Z | Added perf regression wall-clock tok/s variability gate with smoke coverage and refreshed dashboards.
2026-04-28T22:03:40Z | Added benchmark artifact manifest CI smoke gate for air-gapped pass/reject coverage and refreshed manifest artifacts.
2026-04-28T22:17:38Z | Added eval input top-score tie telemetry/gate with smoke audit artifacts and coverage.
2026-04-28T22:22:54Z | Added qemu_prompt_bench dry-run CSV/JUnit planning artifacts and refreshed air-gapped dry-run outputs.
2026-04-28T22:32:04Z | Added qemu_prompt_bench dry-run launch-plan hash/CSV artifacts with refreshed air-gapped smoke outputs.
2026-04-28T22:39:12Z | Added qemu_prompt_bench wall-clock P05/spread throughput telemetry with smoke coverage and refreshed artifacts.
2026-04-28T22:53:10Z | Added qemu_prompt_bench IQR tok/s variability telemetry/gates with matrix passthrough and refreshed smoke artifacts.
2026-04-28T23:03:45Z | Added dataset schema duplicate payload/conflicting-answer gates with smoke coverage and refreshed artifacts.
2026-04-28T23:15:37Z | Added qemu_prompt_bench input artifact metadata with optional image hashing and refreshed air-gapped smoke artifacts.
2026-04-28T23:22:50Z | Added qemu_source_audit coverage for standalone JSON QEMU args arrays with smoke validation and refreshed artifacts.
2026-04-28T23:28:23Z | Added dataset provenance license allow/deny gates with smoke coverage and refreshed provenance artifacts.
2026-04-28T23:38:05Z | Added perf regression host child peak RSS telemetry/gates with smoke coverage and refreshed dashboards.
2026-04-28T23:47:44Z | Added perf regression host child tok/CPU-s telemetry gates with smoke coverage and refreshed dashboards.
2026-04-28T23:54:51Z | Added perplexity_compare dataset/split breakdown reports and metadata mismatch gating with smoke coverage.
2026-04-29T00:01:22Z | Added HCEval loader-oriented record span byte telemetry with smoke validation and refreshed span artifacts.
2026-04-29T00:07:06Z | Added HCEval aggregate binary layout telemetry with manifest/inspection validation and refreshed dataset artifacts.
2026-04-29T00:16:45Z | Added HCEval choice-count telemetry to manifests/inspection with smoke validation and refreshed dataset artifacts.
2026-04-29T00:24:01Z | Added latest comparable benchmark artifact rollups to bench_result_index with smoke coverage and refreshed index artifacts.
2026-04-29T00:39:32Z | Added benchmark trend export dashboards with smoke coverage and refreshed trend artifacts.
2026-04-29T00:49:14Z | Added offline dataset choice-quality audit with CSV/Markdown/JUnit outputs, smoke gate coverage, and latest sample artifacts.
2026-04-29T00:58:32Z | Added YAML qemu_args source audit coverage for benchmark configs with smoke validation and refreshed source audit artifacts.
2026-04-29T01:10:23Z | Added quant audit tail-padding telemetry and opt-in nonzero padding gates for Q4_0/Q8_0 block streams.
2026-04-29T01:20:48Z | Added QEMU prompt benchmark OK-run percentage summaries with smoke coverage and refreshed benchmark artifacts.
2026-04-29T01:30:09Z | Added quant audit fp16 scale exponent telemetry/gates with smoke coverage and refreshed quant artifacts.
2026-04-29T01:40:10Z | Added benchmark trend min-points history gate with smoke coverage and refreshed trend dashboards.
2026-04-29T01:51:16Z | Added qemu_prompt_bench prompt input byte/s telemetry with synthetic smoke validation and refreshed benchmark artifacts.
2026-04-29T02:03:35Z | Added qemu_prompt_bench serial output byte telemetry/gates with synthetic smoke validation and refreshed benchmark artifacts.
2026-04-29T02:16:21Z | Added benchmark artifact manifest per-key history coverage gates with smoke coverage and refreshed manifest artifacts.
2026-04-29T02:24:21Z | Added perf regression minimum emitted-token drop gate with smoke coverage and refreshed dashboards.
2026-04-29T02:32:32Z | Added qemu_source_audit args-file reference following for JSON/YAML benchmark configs with smoke coverage and refreshed source audit artifacts.
2026-04-29T02:49:52Z | Added qemu_prompt_bench tokens-per-prompt-byte telemetry/gate with smoke coverage and refreshed benchmark artifacts.
2026-04-29T02:56:58Z | Added build_compare serial output byte delta telemetry and optional growth gating with refreshed comparison artifacts.
2026-04-29T03:14:00Z | Added build_compare CI smoke gate for synthetic pass, command-drift, and OK-run coverage paths with refreshed comparison artifacts.
2026-04-29T03:24:42Z | Added measured qemu_prompt_bench launch-plan hashes, launch indexes, launch CSV artifacts, smoke coverage, and refreshed benchmark outputs.
2026-04-29T03:31:04Z | Added quant audit fp16 scale sign telemetry and optional negative-scale gates with smoke coverage.
2026-04-29T03:39:32Z | Added qemu_prompt_bench dry-run launch-plan indexing with air-gap/hash telemetry and refreshed benchmark result index artifacts.
2026-04-29T04:01:25Z | Added launch-plan hash drift tracking/gating to benchmark result indexes and refreshed manifest/index artifacts.
2026-04-29T04:15:25Z | Added qemu_prompt_bench timeout budget telemetry/gate with synthetic smoke coverage and refreshed benchmark artifacts.
2026-04-29T04:23:49Z | Added benchmark result index CI smoke coverage for output artifacts, command-hash failures, air-gap rejection, stale artifacts, and environment drift.
2026-04-29T04:33:43Z | Added benchmark result index dry-run launch-plan coverage telemetry and opt-in missing-dry-run gate with smoke coverage.
2026-04-29T04:55:38Z | Added HCEval inspection record-span CSV export with dataset smoke coverage and refreshed inspection artifacts.
2026-04-29T05:05:54Z | Added eval_compare top-score tie telemetry/gating with smoke coverage and refreshed eval artifacts.
2026-04-29T05:17:55Z | Added perf regression host/QEMU environment metadata coverage gates with CSV/JUnit smoke coverage and refreshed dashboards.
2026-04-29T05:42:44Z | Added perf regression memory-per-token telemetry/gating with smoke coverage and refreshed dashboards.
2026-04-29T05:51:40Z | Added dataset choice-overlap audit gating with smoke coverage and refreshed choice audit artifacts.
2026-04-29T06:01:22Z | Added qemu_prompt_bench guest memory-per-token telemetry/gating with smoke coverage and refreshed benchmark artifacts.
2026-04-29T06:10:02Z | Added qemu_prompt_bench suite-level total-token gating with smoke coverage and README documentation.
2026-04-29T06:21:23Z | Added benchmark artifact manifest dry-run coverage gate with smoke coverage and refreshed manifest artifacts.
2026-04-29T06:32:42Z | Added qemu_prompt_bench guest prompt SHA256 match telemetry/gating with smoke coverage and refreshed benchmark artifacts.
2026-04-29T06:39:29Z | Added benchmark artifact manifest environment-drift CSV/JUnit reporting and opt-in stability gate.
2026-04-29T06:47:12Z | Added benchmark trend recent-window stats with CSV/JSON dashboard output and smoke coverage.
2026-04-29T06:54:05Z | Added qemu_prompt_bench guest prompt byte-count match telemetry/gating with smoke coverage and refreshed benchmark artifacts.
2026-04-29T07:11:07Z | Added eval input prompt/choice/input hash parity telemetry and opt-in gating with smoke coverage.
2026-04-29T07:20:48Z | Added HCEval normalized JSONL export with prompt/choice/input hashes and dataset smoke coverage.
2026-04-29T07:33:16Z | Added benchmark trend recent-window regression gates with smoke coverage and refreshed dashboard artifacts.
2026-04-29T07:39:49Z | Added offline eval dataset fingerprint manifests with prompt/choice/input hashes and smoke coverage.
2026-04-29T07:48:58Z | Added eval dataset fingerprint diff gating with smoke coverage and refreshed dataset diff artifacts.
2026-04-29T08:28:58Z | Added perf regression serial output byte/token telemetry gates with smoke coverage.
2026-04-29T08:36:41Z | Added dataset artifact-type coverage gates with smoke coverage and refreshed dataset index artifacts.
2026-04-29T08:42:28Z | Added dataset index dataset/split coverage gates with smoke coverage and refreshed dataset index artifacts.
2026-04-29T09:00:43Z | Added benchmark trend command/launch-plan/environment drift reporting and opt-in gates with smoke coverage.
2026-04-29T09:10:20Z | Added benchmark result index dry-run coverage JUnit failures with smoke coverage and refreshed index artifacts.
2026-04-29T09:17:57Z | Added benchmark trend absolute latest tok/s floor gates with smoke coverage and refreshed dashboard artifacts.
2026-04-29T09:28:06Z | Added perf regression P95 guest/wall token-latency gates with smoke coverage and refreshed dashboards.
2026-04-29T09:34:40Z | Added eval comparator CI smoke coverage for reports, paired metrics, CSV/JUnit artifacts, and gate failures.
2026-04-29T09:45:18Z | Added benchmark matrix expected-cell coverage gates with dry-run smoke coverage and refreshed matrix artifacts.
2026-04-29T10:18:03Z | Added HCEval record fingerprint manifests and inspector validation with smoke coverage and refreshed dataset artifacts.
2026-04-29T10:33:43Z | Added qemu prompt benchmark exit classification telemetry for nonzero/timeout failures with smoke coverage and refreshed artifacts.
2026-04-29T10:48:43Z | Added prompt audit pinned suite hash drift gate with smoke coverage and refreshed prompt audit artifacts.
2026-04-29T10:55:54Z | Added perplexity_compare minimum record/token coverage gates with CI smoke coverage and refreshed smoke reports.
2026-04-29T11:03:07Z | Added eval input audit choice-count gates with CI smoke coverage and refreshed smoke reports.
2026-04-29T11:38:15Z | Added qemu prompt benchmark expected-token telemetry and declared decode-length gates with smoke coverage.
2026-04-29T11:56:04Z | Added structured benchmark trend finding rows/CSV with smoke coverage and refreshed dashboard artifacts.
2026-04-29T12:52:56Z | Added dataset provenance source URL path-prefix policy gates with smoke coverage.
2026-04-29T13:00:52Z | Added build comparison prompt-key coverage gates with smoke coverage and refreshed build comparison artifacts.
2026-04-29T13:09:37Z | Propagated memory-per-token and serial-output telemetry through benchmark result indexes and artifact manifests.
2026-04-29T13:19:29Z | Added dataset curation required dataset/split coverage gates with smoke coverage and refreshed curated manifest.
2026-04-29T13:36:47Z | Added qemu prompt benchmark minimum prompt-count preflight gate with dry-run smoke artifacts.
2026-04-29T13:44:32Z | Added quant audit duplicate complete-block and identical-run gates with smoke coverage and refreshed reports.
2026-04-29T13:52:15Z | Added benchmark trend recent-window CV telemetry/gates with smoke coverage and refreshed dashboards.
2026-04-29T14:09:35Z | Added QEMU source air-gap audit CI smoke coverage and refreshed source audit artifacts.
2026-04-29T14:37:01Z | Added qemu prompt benchmark phase summaries/CSV with synthetic smoke coverage and refreshed artifacts.
2026-04-29T15:23:07Z | Added qemu prompt benchmark top-level provenance metadata and indexer fallback coverage.
2026-04-29T16:10:50Z | Added qemu prompt benchmark prompt-integrity summary telemetry with smoke coverage and refreshed artifacts.
2026-04-29T16:44:45Z | Added benchmark result-index history coverage gate with CSV/JUnit reporting and refreshed index artifacts.
2026-04-29T16:51:55Z | Added benchmark artifact manifest sample coverage gate for measured runs/tokens with smoke coverage and refreshed artifacts.
