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
2026-05-02T08:07:22Z | Added offline HolyC/llama.cpp perplexity pairing audit with metadata/token-count gates, smoke coverage, docs, and refreshed reports.
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
2026-04-30T07:28:39Z | Added QEMU benchmark latest/history retention audit with SHA256 alias checks, smoke coverage, docs, and refreshed result artifacts.
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
2026-04-30T05:58:48Z | Added bench trend export summary rollups for dashboard digests with smoke coverage and refreshed dashboard artifacts.
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
2026-05-10T15:04:30Z | Added weighted guest/wall tok/s metrics to the QEMU build throughput scorecard with smoke coverage and refreshed reports.
2026-05-10T16:12:30Z | Tightened QEMU args-file audit to reject embedded `-nic none` fragments with smoke coverage and refreshed reports.
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
2026-04-29T20:58:38Z | Added quant audit signed payload balance gate with smoke coverage and refreshed audit artifacts.
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
2026-05-02T04:43:09Z | Added direct unit coverage for eval score sparsity audits and refreshed the latest score sparsity artifact.
2026-05-02T04:32:58Z | Added QEMU prompt source audit for row-level prompt-suite hash/byte/token parity with smoke coverage and latest artifacts.
2026-04-29T06:32:42Z | Added qemu_prompt_bench guest prompt SHA256 match telemetry/gating with smoke coverage and refreshed benchmark artifacts.
2026-04-30T06:43:18Z | Added dashboard sidecar audit for perf CI artifacts with smoke coverage and refreshed dashboard outputs.
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
2026-04-29T17:02:04Z | Added qemu prompt benchmark realized launch-sequence integrity hashing with smoke coverage and refreshed artifacts.
2026-04-29T17:13:38Z | Added benchmark result-index expected-token parity telemetry with smoke coverage and refreshed index artifacts.
2026-04-29T17:30:44Z | Added dataset order-bias audit for answer-position runs/transitions with CSV/JUnit smoke coverage and refreshed artifacts.
2026-04-29T17:57:06Z | Added Q4_0 nibble-lane diversity telemetry and optional packing gate to host-side quant audit.
2026-04-29T18:12:07Z | Added signed quant-payload coverage telemetry and negative/positive entry gates to host-side quant audit.
2026-04-29T18:25:17Z | Added qemu prompt benchmark serial-output line telemetry and gating with smoke coverage and refreshed artifacts.
2026-04-29T18:38:39Z | Added build benchmark host/QEMU environment drift detection with CSV/JUnit smoke coverage.
2026-04-29T18:50:29Z | Added benchmark artifact manifest freshness-failure CSV export with stale-artifact smoke coverage and refreshed manifest artifacts.
2026-04-29T18:57:20Z | Added QEMU prompt benchmark launch-level JSONL export with smoke coverage and refreshed synthetic results.
2026-04-29T19:10:38Z | Added elapsed and wall-elapsed duration regression gates to perf dashboards with smoke coverage and refreshed artifacts.
2026-04-29T19:37:00Z | Added QEMU prompt benchmark command-level air-gap evidence across JSON/CSV/Markdown/JUnit artifacts.
2026-04-29T19:43:51Z | Added QEMU prompt benchmark wall-elapsed summary telemetry and per-run gating with smoke coverage.
2026-04-29T19:50:45Z | Added standalone dataset provenance audit CI smoke coverage and refreshed provenance reports.
2026-04-29T20:00:03Z | Added slowest-prompt ranking export to QEMU prompt benchmark artifacts with smoke coverage.
2026-04-29T20:10:46Z | Added QEMU prompt benchmark variability ranking export with smoke coverage and refreshed artifacts.
2026-04-29T20:18:04Z | Added QEMU prompt benchmark prompt-efficiency ranking export with smoke coverage and refreshed artifacts.
2026-04-29T20:24:52Z | Added QEMU prompt benchmark exit-class summary telemetry with smoke coverage and refreshed synthetic artifacts.
2026-04-29T20:32:48Z | Added QEMU prompt benchmark serial-output ranking export with smoke coverage and refreshed synthetic artifacts.
2026-04-29T20:39:22Z | Added QEMU prompt benchmark expected-token coverage gate with smoke coverage and refreshed synthetic artifacts.
2026-04-29T20:45:04Z | Added HCEval inspection fingerprint CSV export with dataset smoke coverage and refreshed binary inspection artifacts.
2026-04-29T20:52:10Z | Tightened QEMU prompt benchmark air-gap validation to reject redundant legacy -net none flags before artifact emission.
2026-04-29T21:05:43Z | Added prompt-byte efficiency telemetry to benchmark result index with smoke coverage and refreshed index artifacts.
2026-04-29T21:16:26Z | Added per-record choice telemetry CSV export to dataset choice audit with smoke coverage and refreshed artifacts.
2026-04-29T21:24:03Z | Added bench_result_index freshness-failure CSV output with smoke coverage and latest CSV artifact.
2026-04-29T22:10:52Z | Added per-record schema telemetry CSV export to dataset schema audit with smoke coverage and refreshed artifacts.
2026-04-29T22:46:18Z | Added standalone dataset leak audit CI smoke coverage and refreshed leak audit artifacts.
2026-04-29T23:02:09Z | Added QEMU prompt benchmark failure ranking export with smoke coverage and refreshed synthetic artifacts.
2026-04-29T23:19:57Z | Added quant-audit zero-payload percentage telemetry and max-zero gate with smoke coverage.
2026-04-29T23:27:10Z | Added dataset mix audit for curated eval suite dominance gates with smoke coverage and refreshed artifacts.
2026-04-29T23:34:40Z | Added benchmark artifact manifest timestamp-collision gating with CSV/JUnit smoke coverage and refreshed manifest artifacts.
2026-04-29T23:42:11Z | Added dataset contamination audit for cross-dataset prompt/payload reuse with smoke coverage and refreshed artifacts.
2026-04-29T23:50:08Z | Added dataset answer-length bias audit with smoke coverage and refreshed artifacts.
2026-04-29T23:56:46Z | Added dataset schema answer-label coverage gates with smoke coverage and refreshed schema artifacts.
2026-04-30T00:03:02Z | Added absolute perf SLO audit for benchmark artifacts with smoke coverage and refreshed dashboard outputs.
2026-04-30T00:14:34Z | Added quant-audit repeated fp16 scale telemetry and gates with smoke coverage.
2026-04-30T00:23:47Z | Added dataset provenance per-record telemetry CSV export with smoke coverage and refreshed artifacts.
2026-04-30T00:29:56Z | Added dataset/split scoped answer-length bias gates with smoke coverage and refreshed artifacts.
2026-04-30T00:34:56Z | Added dataset order per-record telemetry CSV export with smoke coverage and refreshed artifacts.
2026-04-30T00:39:16Z | Added dataset order telemetry unit coverage and refreshed per-record order audit artifacts.
2026-04-30T00:47:18Z | Added dataset mix per-record telemetry CSV export with smoke coverage and refreshed artifacts.
2026-04-30T00:53:46Z | Added dataset choice answer-byte telemetry with unit/smoke coverage and refreshed choice audit artifacts.
2026-04-30T01:07:12Z | Added dataset ID audit for explicit, bounded, unique eval record IDs with smoke coverage and refreshed artifacts.
2026-04-30T01:14:05Z | Added dataset manifest audit for curated JSONL and HCEval pack consistency with smoke coverage and refreshed artifacts.
2026-04-30T01:27:55Z | Added JSONL-to-HCEval roundtrip audit with unit/smoke coverage and refreshed artifacts.
2026-04-30T01:33:12Z | Added direct dataset mix audit unit coverage with smoke verification and refreshed artifacts.
2026-04-30T01:41:27Z | Added eval_compare score-rank CSV output with smoke coverage and refreshed comparison artifacts.
2026-04-30T01:47:13Z | Added raw Q4_0/Q8_0 quant block comparator with smoke artifacts and direct test coverage.
2026-04-30T01:53:37Z | Added dataset/split slice manifest coverage tool with smoke artifacts and direct test coverage.
2026-04-30T02:00:55Z | Added dataset text loader-safety audit with byte/control-character gates, smoke artifacts, and direct test coverage.
2026-04-30T02:07:48Z | Added dataset split-overlap audit for prompt/payload leakage across splits with smoke artifacts and direct test coverage.
2026-04-30T02:15:45Z | Added prompt expected-token metadata gates with smoke artifacts and direct test coverage.
2026-04-30T02:23:43Z | Added perplexity dataset/split coverage gates with smoke artifacts and direct test coverage.
2026-04-30T02:28:20Z | Added perplexity regression CSV output for dashboard-friendly quality gate findings with smoke artifacts and direct test coverage.
2026-04-30T02:43:40Z | Added eval input per-record telemetry CSV export with smoke artifacts and direct test coverage.
2026-04-30T02:54:47Z | Added eval input top-score margin telemetry and gate with smoke artifacts and direct test coverage.
2026-04-30T03:02:52Z | Added prompt suite parity comparer with smoke artifacts and direct stdlib test coverage.
2026-04-30T03:18:45Z | Added eval disagreement audit gates for HolyC-vs-llama reports with smoke artifacts and direct stdlib test coverage.
2026-04-30T03:31:01Z | Added dataset raw label-integrity audit with smoke artifacts and direct stdlib test coverage.
2026-04-30T03:46:12Z | Added dashboard digest gate for CI summary artifacts with smoke and direct stdlib test coverage.
2026-04-30T03:58:27Z | Added benchmark launch-plan hash validation to the result indexer with smoke coverage and refreshed index artifacts.
2026-04-30T04:07:07Z | Added eval calibration audit gates for score coverage, ECE, Brier, and HolyC-vs-llama deltas with smoke artifacts.
2026-04-30T04:14:15Z | Added direct dataset fingerprint and fingerprint-diff regression coverage with smoke verification.
2026-04-30T04:21:35Z | Added saved QEMU command air-gap audit with smoke coverage and refreshed benchmark artifact outputs.
2026-04-30T04:43:26Z | Restored shared QEMU air-gap audit helpers so source and result index smoke gates pass together.
2026-04-30T04:50:50Z | Added dataset license/source policy audit with smoke artifacts and direct stdlib test coverage.
2026-04-30T04:57:20Z | Added eval suite summary aggregator for HolyC-vs-llama reports with smoke artifacts and direct stdlib test coverage.
2026-04-30T05:10:02Z | Added dataset provenance-balance audit with smoke artifacts and direct stdlib test coverage.
2026-04-30T05:18:30Z | Added eval slice coverage audit for required dataset/split HolyC-vs-llama reports with smoke artifacts and direct stdlib test coverage.
2026-04-30T05:27:38Z | Added structured findings CSV output for dataset fingerprint diff gates with smoke and direct test coverage.
2026-04-30T05:36:25Z | Tightened QEMU source audit to reject legacy -net none fragments with smoke artifacts and direct stdlib test coverage.
2026-04-30T05:45:02Z | Added TOML qemu_args source audit coverage with args-file resolution, smoke validation, and refreshed air-gap artifacts.
2026-04-30T06:33:01Z | Added dataset prompt/choice overlap audit with smoke artifacts and direct stdlib test coverage.
2026-04-30T06:57:45Z | Added host-side benchmark matrix audit for cell coverage and air-gap-safe QEMU argument fragments with smoke coverage and result artifacts.
2026-04-30T07:06:58Z | Tightened build benchmark compare OK-run accounting to reject non-ok exit_class rows with direct and smoke coverage.
2026-04-30T07:44:12Z | Added eval outcome bucket audit for HolyC-vs-llama paired correctness gates with smoke artifacts and direct stdlib test coverage.
2026-04-30T07:56:10Z | Added eval report consistency audit for stale summary metrics with smoke artifacts and direct stdlib test coverage.
2026-04-30T08:26:46Z | Added focused dataset index CI smoke gate for artifact-type and dataset/split coverage failures.
2026-04-30T09:05:25Z | Added QEMU prompt coverage audit gates with smoke coverage and refreshed benchmark coverage artifacts.
2026-04-30T09:23:34Z | Added prompt length-bucket coverage audit with smoke artifacts and direct stdlib test coverage.
2026-04-30T09:46:58Z | Added eval suite coverage gates for required dataset, split, model, and quantization reports with smoke artifact refresh.
2026-04-30T10:10:53Z | Added QEMU prompt schema/telemetry audit with air-gap command checks, smoke coverage, and refreshed benchmark schema artifacts.
2026-04-30T10:28:31Z | Added QEMU summary consistency audit to catch stale suite/prompt aggregates with smoke artifacts and direct stdlib test coverage.
2026-04-30T10:44:15Z | Added eval/perf scorecard joining HolyC-vs-llama quality reports with air-gapped QEMU throughput artifacts.
2026-04-30T10:55:07Z | Added eval artifact drift audit for gold/prediction hash consistency with smoke artifacts and direct stdlib test coverage.
2026-04-30T11:01:50Z | Added eval quantization delta audit for Q4_0-vs-Q8_0 quality drift with smoke artifacts and direct stdlib test coverage.
2026-04-30T11:11:38Z | Added eval efficiency frontier audit for quality-vs-throughput scorecards with smoke artifacts and direct stdlib test coverage.
2026-04-30T11:23:20Z | Added dataset choice-length cue audit with smoke artifacts and direct stdlib test coverage.
2026-04-30T11:32:10Z | Added eval reproducibility metadata audit for HolyC-vs-llama seed/decoding parity with smoke artifacts and direct stdlib test coverage.
2026-04-30T11:42:31Z | Added HCEval export/repack roundtrip audit for packed dataset digest parity with smoke artifacts and direct stdlib test coverage.
2026-04-30T11:55:20Z | Added QEMU benchmark environment provenance audit with smoke coverage and refreshed result artifacts.
2026-04-30T11:58:46Z | Added QEMU resource telemetry coverage audit with smoke artifacts and direct stdlib test coverage.
2026-04-30T12:05:43Z | Added QEMU latency distribution audit with percentile gates, smoke artifacts, and direct stdlib test coverage.
2026-04-30T12:14:20Z | Added QEMU phase sequence audit for warmup/measured coverage, ordering, and measured success gates with smoke artifacts and direct stdlib test coverage.
2026-04-30T12:30:44Z | Added QEMU failure taxonomy audit for exit_class, timeout, return-code, and OK metric consistency with smoke artifacts and direct stdlib test coverage.
2026-04-30T12:46:54Z | Added QEMU args policy audit for reusable air-gapped benchmark argument fragments with smoke artifacts and direct stdlib coverage.
2026-04-30T13:03:25Z | Added QEMU timestamp chronology audit for generated_at, filename stamps, row monotonicity, and row skew with smoke artifacts and direct stdlib coverage.
2026-04-30T13:26:02Z | Added QEMU artifact budget audit for bounded benchmark files, serial output, tails, and failure reasons with smoke and direct stdlib coverage.
2026-04-30T13:45:53Z | Added QEMU throughput stability audit for per-prompt wall tok/s floors and variability with smoke artifacts and direct stdlib coverage.
2026-04-30T14:38:53Z | Added QEMU replay manifest export for air-gapped benchmark argv/provenance capture with smoke artifacts and direct stdlib coverage.
2026-04-30T14:54:39Z | Added QEMU replay manifest audit for argv sidecar parity, command hash drift, and explicit air-gap replay checks.
2026-04-30T15:09:46Z | Added curated dataset manifest row-count and dataset/split coverage consistency gates with smoke artifacts and direct stdlib coverage.
2026-04-30T15:54:56Z | Added QEMU token accounting audit for tok/s, us/token, ratio, memory/token, and expected-token consistency with smoke and direct stdlib coverage.
2026-04-30T16:46:15Z | Added QEMU CPU accounting audit for child CPU, CPU percent, and tok/CPU-second consistency with smoke artifacts and direct stdlib coverage.
2026-04-30T17:33:28Z | Added eval pairing audit for HolyC-vs-llama prediction stream parity with smoke artifacts and direct stdlib coverage.
2026-04-30T17:55:27Z | Added QEMU prompt echo audit for host/guest prompt byte and SHA parity with smoke artifacts and direct stdlib coverage.
2026-04-30T18:30:15Z | Added HCEval suite audit for packed dataset manifest and byte-budget validation with smoke artifacts and direct stdlib coverage.
2026-04-30T19:08:51Z | Added QEMU serial accounting audit for stdout/stderr-to-serial telemetry parity with smoke artifacts and direct stdlib coverage.
2026-04-30T20:12:54Z | Added QEMU timeout margin audit for wall-time timeout headroom with smoke artifacts and direct stdlib coverage.
2026-04-30T20:45:41Z | Added HCEval bundle audit for packed shard manifest parity and duplicate record/fingerprint detection with smoke artifacts.
2026-04-30T21:06:39Z | Added QEMU host overhead accounting audit for wall-vs-guest timing drift with smoke artifacts and direct stdlib coverage.
2026-04-30T21:16:13Z | Hardened QEMU args policy audit against nested response/config includes that could bypass air-gap checks.
2026-04-30T21:34:19Z | Hardened quant block compare so raw Q4_0/Q8_0 scale or payload mismatches fail by default, with explicit telemetry-only opt-out.
2026-04-30T21:53:58Z | Added QEMU launch integrity audit for launch-plan hashes, observed sequence parity, and stored integrity drift with smoke artifacts.
2026-04-30T22:05:45Z | Added QEMU result retention findings CSV export with smoke/direct coverage and refreshed retention artifacts.
2026-05-01T00:41:08Z | Added QEMU warmup isolation audit for measured/warmup row separation with smoke/direct coverage and refreshed result artifacts.
2026-05-01T01:10:43Z | Added QEMU launch order audit for contiguous launch indices, warmup-before-measured ordering, and timestamp monotonicity with smoke/direct coverage.
2026-05-01T01:21:20Z | Added QEMU command fingerprint audit for per-row command hash parity, explicit air-gap flags, and drift detection with smoke/direct coverage.
2026-05-01T01:33:41Z | Added QEMU CPU accounting budget gates for host CPU percent and tok/CPU-s with smoke coverage and refreshed audit artifacts.
2026-05-01T01:48:00Z | Added QEMU input provenance audit for prompt-suite, image, and args-file metadata drift with smoke coverage.
2026-05-01T01:58:36Z | Added QEMU timing consistency audit for derived elapsed/wall timing metrics with smoke coverage and refreshed audit artifacts.
2026-05-01T02:19:04Z | Added QEMU stdio hygiene audit for OK stderr noise, silent failures, and stdout/stderr byte-counter drift with smoke coverage.
2026-05-01T02:41:11Z | Added eval margin audit for scored HolyC-vs-llama choice margin gates with smoke/direct coverage.
2026-05-01T02:59:10Z | Added QEMU status consistency audit for stale pass/fail artifacts with smoke/direct coverage and refreshed result artifacts.
2026-05-01T03:50:57Z | Added eval score parity audit for HolyC-vs-llama paired score-vector coverage/shape gates with smoke/direct coverage.
2026-05-01T04:10:52Z | Hardened QEMU benchmark air-gap validation against socket endpoints and remote display transports with smoke/direct coverage.
2026-05-01T04:23:43Z | Added eval choice distribution audit for HolyC-vs-llama prediction collapse and gold alignment checks with smoke/direct coverage.
2026-05-01T04:35:33Z | Added build pair selector for latest comparable QEMU benchmark baseline/candidate artifacts with JSON/CSV/Markdown/JUnit sidecars.
2026-05-01T05:06:24Z | Added eval rank audit for scored top-k/MRR HolyC-vs-llama gates with smoke coverage and refreshed rank artifacts.
2026-05-01T06:09:58Z | Added QEMU NIC cardinality audit for exactly-one -nic none enforcement with smoke/direct coverage and refreshed result artifacts.
2026-05-01T06:21:58Z | Added eval score-parity top-score tie gates with smoke/direct coverage and refreshed parity artifacts.
2026-05-01T06:38:17Z | Added QEMU prompt balance audit for balanced successful samples per prompt with smoke/direct coverage and refreshed result artifacts.
2026-05-01T06:48:58Z | Added QEMU identity audit for artifact-vs-row profile/model/quantization, commit, and command-hash drift with smoke/direct coverage.
2026-05-01T07:19:57Z | Added QEMU serial BENCH_RESULT payload audit for captured metric parity with smoke/direct coverage and refreshed result artifacts.
2026-05-01T07:31:38Z | Added QEMU failure-reason audit for exit_class/returncode/timed_out/failure_reason consistency with smoke/direct coverage and refreshed result artifacts.
2026-05-01T07:43:50Z | Added eval prompt-length bucket report for HolyC-vs-llama accuracy/agreement slices with smoke artifacts and coverage.
2026-05-01T07:52:29Z | Tightened QEMU args policy audit to reject remote-display sockets with smoke coverage and refreshed result artifacts.
2026-05-01T08:01:40Z | Added QEMU stdio byte/tail budget gates with direct smoke coverage and refreshed hygiene artifacts.
2026-05-01T08:12:50Z | Tightened QEMU prompt launcher air-gap checks to reject remote-display sockets before VM launch with smoke/direct coverage.
2026-05-01T08:41:24Z | Added QEMU iteration coverage audit for prompt warmup/measured iteration gaps and duplicates with smoke/direct artifacts.
2026-05-01T08:50:47Z | Added QEMU input provenance audit direct coverage and generated latest saved-artifact provenance sidecars.
2026-05-01T08:59:12Z | Added QEMU result retention CI smoke coverage for matching latest/history, missing history, and hash mismatch cases.
2026-05-01T09:08:26Z | Added QEMU memory accounting host-RSS/guest-memory ratio gate with smoke/direct coverage and refreshed audit artifacts.
2026-05-01T09:17:56Z | Added perf CI gate for saved regression/SLO dashboard status and sidecar coverage with smoke/direct artifacts.
2026-05-01T09:42:11Z | Added QEMU prompt-length bucket audit for saved benchmark artifacts with smoke coverage and refreshed result sidecars.
2026-05-01T09:58:34Z | Tightened QEMU timeout margin audit with timeout-row underuse gates, smoke/direct coverage, and refreshed result sidecars.
2026-05-01T10:10:14Z | Added QEMU replay manifest source artifact SHA256/size provenance with audit coverage and refreshed reports.
2026-05-01T10:20:34Z | Tightened QEMU replay manifest audit to reject duplicate manifest and argv-sidecar replay keys with direct/smoke coverage.
2026-05-01T10:29:21Z | Hardened host-side QEMU air-gap checks to reject user-mode network service flags in benchmark commands and args fragments.
2026-05-01T10:38:58Z | Added dataset text audit gate for raw choice label prefixes with direct/smoke coverage and refreshed smoke artifacts.
2026-05-01T11:08:53Z | Added normalized dataset stats report with byte/choice/answer histograms, smoke/direct coverage, and refreshed smoke artifacts.
2026-05-01T11:21:10Z | Added dataset answer-position distribution audit with smoke/direct coverage and refreshed curation artifacts.
2026-05-01T11:30:28Z | Added QEMU prompt efficiency audit for prompt byte/token derived metrics with smoke/direct coverage and refreshed result sidecars.
2026-05-01T11:39:40Z | Added eval confusion-matrix audit for macro-F1/per-answer gates with smoke/direct coverage and refreshed result sidecars.
2026-05-01T11:54:02Z | Added eval top-k overlap audit for HolyC-vs-llama scored rankings with smoke/direct coverage and refreshed result sidecars.
2026-05-01T12:17:13Z | Added QEMU artifact reference audit for local/offline resource paths and command air-gap drift with smoke/direct coverage.
2026-05-01T12:44:22Z | Hardened QEMU air-gap validation against modern NIC/vsock device aliases with focused command and args-policy smoke coverage.
2026-05-01T12:55:46Z | Added eval model/tokenizer identity audit with smoke fixtures, direct tests, and refreshed comparison artifacts.
2026-05-01T13:05:15Z | Added QEMU prompt benchmark sidecar audit for JSON/CSV/Markdown/JUnit artifact completeness with smoke/direct coverage.
2026-05-01T13:13:53Z | Tightened QEMU token accounting to fail recorded expected-token mismatches with direct/smoke coverage and refreshed results.
2026-05-01T13:43:51Z | Added QEMU host-overhead median/p95 summaries with direct/smoke coverage and refreshed result sidecars.
2026-05-01T13:50:59Z | Tightened dataset manifest audit to verify selected_record_ids against curated JSONL record order with direct/smoke coverage.
2026-05-01T14:19:26Z | Added QEMU exit-rate audit for failure/timeout/nonzero/launch-error percentage gates with direct/smoke coverage and refreshed result sidecars.
2026-05-01T14:29:34Z | Added QEMU exit-class consistency audit for returncode/timed_out/failure telemetry with direct/smoke coverage and refreshed result sidecars.
2026-05-01T14:39:53Z | Added eval significance audit for paired McNemar HolyC-vs-llama losses with smoke/direct coverage and refreshed result sidecars.
2026-05-01T14:52:18Z | Hardened QEMU source air-gap fragment audits against response/config includes, sockets, remote displays, and embedded launch commands.
2026-05-01T15:32:00Z | Hardened QEMU air-gap audits and launcher validation to reject TLS credential options with focused smoke coverage.
2026-05-01T15:54:07Z | Added QEMU prompt outlier audit for repeated prompt latency/throughput drift with smoke/direct coverage and refreshed result sidecars.
2026-05-01T16:16:25Z | Tightened eval pairing audit to compare nested identity metadata fields with direct/smoke coverage.
2026-05-01T16:37:12Z | Added memory-aware eval efficiency frontier mode with direct/smoke coverage and refreshed frontier artifacts.
2026-05-01T16:57:31Z | Added QEMU token-latency consistency audit with direct/smoke coverage and refreshed result sidecars.
2026-05-01T17:17:06Z | Tightened QEMU timing consistency to reject wall-clock elapsed time below guest elapsed time with direct/smoke coverage.
2026-05-01T17:49:33Z | Added eval score-scale audit for paired HolyC-vs-llama scored predictions with smoke coverage and refreshed result sidecars.
2026-05-01T18:11:19Z | Added QEMU CSV parity audit for JSON-vs-CSV benchmark artifacts with smoke coverage and refreshed result sidecars.
2026-05-01T18:48:27Z | Added QEMU benchmark matrix planner for air-gapped build/prompt launch manifests with direct/smoke coverage and result sidecars.
2026-05-01T19:06:11Z | Added QEMU benchmark matrix artifact audit for air-gap/hash/launch-plan drift with direct/smoke coverage and result sidecars.
2026-05-01T19:34:45Z | Added QEMU matrix result coverage audit for planned-vs-observed launch completeness with direct/smoke coverage and result sidecars.
2026-05-01T19:58:57Z | Added QEMU result freshness audit for latest benchmark artifacts with direct/smoke coverage and refreshed result sidecars.
2026-05-01T20:24:07Z | Added deterministic offline dataset subset selector with balanced per-slice selection, direct/smoke coverage, and smoke result artifacts.
2026-05-01T20:38:29Z | Added QEMU environment row command provenance gating with direct/smoke coverage and refreshed result sidecars.
2026-05-01T20:51:30Z | Added host-side QEMU prompt budget audit with token/prompt ceilings, guest echo checks, CI smoke coverage, and latest result artifacts.
2026-05-01T21:03:58Z | Added direct coverage for QEMU prompt-length bucket audit and refreshed latest bucket sidecars.
2026-05-01T21:12:54Z | Verified staged host-side bench/eval/dataset audit stack with direct changed-test runners and focused QEMU smoke coverage before commit.
2026-05-01T21:22:51Z | Added eval result index for compare/suite artifacts with direct/smoke coverage and refreshed result sidecars.
2026-05-01T21:40:05Z | Tightened QEMU result freshness audit to reject future-dated artifacts with direct/smoke coverage and refreshed result sidecars.
2026-05-01T21:55:52Z | Tightened QEMU timestamp audit to flag stale row timestamps before artifact generation with direct/smoke coverage and refreshed sidecars.
2026-05-01T22:04:52Z | Tightened QEMU command fingerprint audit with an opt-in single-command-hash gate, direct/smoke coverage, and refreshed result sidecars.
2026-05-01T22:13:24Z | Added QEMU build throughput scorecard for saved prompt benchmark artifacts with direct/smoke coverage and refreshed result sidecars.
2026-05-01T22:21:52Z | Added raw eval prediction coverage audit with per-slice gates, smoke artifacts, docs, and host-side tests.
2026-05-01T22:33:09Z | Added QEMU host environment policy audit for captured proxy/URL env vars with smoke/direct coverage and latest result sidecars.
2026-05-01T22:42:54Z | Added QEMU artifact schema completeness audit for saved prompt benchmark rows with direct/smoke coverage and latest result sidecars.
2026-05-01T22:52:28Z | Added QEMU result uniqueness audit to catch duplicate benchmark row identities with direct/smoke coverage and latest result sidecars.
2026-05-01T23:04:26Z | Added QEMU prompt benchmark artifact schema version markers with schema-audit validation and refreshed synthetic result artifacts.
2026-05-01T23:14:07Z | Added eval hash audit for canonical eval_compare fingerprints with smoke coverage and refreshed result sidecars.
2026-05-01T23:22:17Z | Added HCEval binary diff tooling for offline dataset promotion review with direct/smoke coverage and latest result sidecars.
2026-05-01T23:33:30Z | Added QEMU benchmark matrix budget audit for planned launch/prompt-byte/expected-token gates with smoke coverage and result sidecars.
2026-05-01T23:53:50Z | Added build-pair manifest audit for regression pair sanity gates with direct/smoke coverage and latest result sidecars.
2026-05-02T00:14:23Z | Added eval choice-map audit for HolyC/llama prediction label-format parity with direct/smoke coverage and latest result sidecars.
2026-05-02T00:22:51Z | Tightened perplexity comparator probability telemetry validation for positive token logprobs and subunit perplexity with direct/smoke coverage.
2026-05-02T00:31:48Z | Added QEMU latest-alias audit to verify *_latest.json artifacts match newest stamped siblings, with tests, smoke gate, deterministic fixture, and bench/results outputs.
2026-05-02T00:41:36Z | Added offline dataset length-bucket coverage audit with direct/smoke coverage and refreshed dataset result sidecars.
2026-05-02T00:53:12Z | Tightened QEMU prompt benchmark air-gap policy to reject response-file and readconfig includes with smoke coverage.
2026-05-02T01:03:49Z | Added QEMU quantization coverage audit for Q4_0/Q8_0 benchmark result sets with direct/smoke coverage and latest result sidecars.
2026-05-02T01:15:18Z | Tightened QEMU quantization coverage audit with per-row air-gap command telemetry gates and refreshed matrix-backed sidecars.
2026-05-02T01:26:29Z | Added QEMU quant pairing audit for per-prompt Q4_0/Q8_0 matrix parity with smoke/direct coverage and latest result sidecars.
2026-05-02T01:36:01Z | Added offline dataset choice-similarity audit for duplicate/near-duplicate options with direct/smoke coverage and latest dataset sidecars.
2026-05-02T01:46:36Z | Added QEMU timeout recommendation reporting from saved wall-time artifacts with direct/smoke coverage and latest result sidecars.
2026-05-02T01:56:52Z | Tightened QEMU replay manifest argv sidecar hash and air-gap validation with focused coverage and refreshed replay audit artifacts.
2026-05-02T02:09:27Z | Added HolyC-vs-llama eval score-delta audit with focused tests, smoke gate, docs, and latest result sidecars.
2026-05-02T02:19:36Z | Added dashboard freshness audit for stale/future perf dashboard timestamps with direct/smoke coverage and latest dashboard sidecars.
2026-05-02T02:29:19Z | Added QEMU prompt identity audit for prompt/hash drift and collision checks with direct/smoke coverage and latest result sidecars.
2026-05-02T02:43:00Z | Added host-side HCEval binary budget audit with smoke gate, tests, docs, and latest dataset reports.
2026-05-02T02:56:47Z | Added eval top-choice tie audit for HolyC-vs-llama scored predictions with smoke/direct coverage and latest result sidecars.
2026-05-02T03:10:22Z | Tightened QEMU args policy audit to reject duplicate -nic none fragments with direct/smoke coverage and refreshed result sidecars.
2026-05-02T03:22:43Z | Added QEMU matrix plan diff gate for command/launch drift with smoke/direct coverage and latest result sidecars.
2026-05-02T03:35:18Z | Added offline eval majority/random baseline reports with smoke/direct coverage and latest result sidecars.
2026-05-02T03:46:23Z | Added QEMU serial endpoint audit for stdio-only command telemetry with smoke/direct coverage and latest result sidecars.
2026-05-02T04:02:15Z | Fixed perf CI smoke coverage for current air-gap audit sidecar outputs and verified all bench smoke gates.
2026-05-02T04:11:34Z | Verified staged host-side bench/eval/dataset infrastructure smoke gates and prepared the gpt-5.5 bench branch commit.
2026-05-02T04:20:37Z | Added eval score sparsity audit for degenerate scored prediction vectors with smoke coverage and latest result sidecars.
2026-05-02T04:53:27Z | Added eval score-order audit for prediction/top-score drift with smoke/direct coverage and latest result sidecars.
2026-05-02T05:03:06Z | Added QEMU image reference audit for artifact image.path vs recorded drive arguments with smoke coverage and latest result sidecars.
2026-05-02T05:14:03Z | Added QEMU launch profile audit for machine/CPU/accelerator/memory drift with smoke/manual coverage and latest result sidecars.
2026-05-02T05:24:58Z | Added dataset content-hash audit for row-level prompt/choices/input SHA-256 metadata with smoke/direct coverage and latest result sidecars.
2026-05-02T05:36:05Z | Added QEMU launch JSONL parity audit for warmup/measured sidecar drift with smoke coverage and latest result sidecars.
2026-05-02T05:46:56Z | Added eval entropy audit for HolyC-vs-llama scored prediction confidence collapse/drift with smoke/direct coverage and latest result sidecars.
2026-05-02T05:56:46Z | Added quant manifest audit for Q4_0/Q8_0 artifact SHA-256, byte, block, and element-count validation with smoke/direct coverage.
2026-05-02T06:07:08Z | Added QEMU output determinism audit for repeated-run generated hash/token drift with smoke/direct coverage and latest result sidecars.
2026-05-02T06:19:13Z | Added bench smoke manifest index for host-side CI smoke coverage with smoke/direct coverage and latest result sidecars.
2026-05-02T06:30:20Z | Tightened QEMU prompt benchmark launcher to reject user-supplied duplicate NIC disablement with direct/smoke coverage.
2026-05-02T06:40:06Z | Added QEMU display policy audit for headless benchmark command telemetry with smoke/direct coverage and latest result sidecars.
2026-05-02T06:50:50Z | Added offline perplexity input audit with smoke gate, docs, and latest bench/results artifacts.
2026-05-02T07:05:01Z | Tightened QEMU launch profile audit with cross-artifact profile drift gating and refreshed latest result sidecars.
2026-05-02T07:16:39Z | Tightened QEMU benchmark air-gap checks for double-dash network/display/socket options with direct/smoke coverage and refreshed result sidecars.
2026-05-02T07:27:22Z | Added offline dataset split-balance audit with CI smoke coverage, docs, and latest dataset result sidecars.
2026-05-02T07:39:47Z | Added offline dataset duplicate audit with CI smoke coverage, docs, and latest dataset result sidecars.
2026-05-02T07:53:48Z | Added eval prompt-hash audit for HolyC-vs-llama prompt/choice/input fingerprint parity with smoke coverage and latest result sidecars.
2026-05-02T08:20:59Z | Added QEMU seed metadata audit for deterministic benchmark rows with smoke/direct coverage and latest result sidecars.
2026-05-02T08:33:37Z | Added QEMU TTFT telemetry audit with smoke/direct coverage and latest result sidecars.
2026-05-02T08:48:18Z | Added QEMU executable provenance audit for direct qemu-system command/path drift with smoke/direct coverage and latest result sidecars.
2026-05-02T09:00:37Z | Added offline eval workload estimator for token/run/time projection with smoke coverage and latest result sidecars.
2026-05-02T09:12:37Z | Tightened offline eval workload estimator with per-record choice/token/launch/wall-time budget gates and refreshed latest result sidecars.
2026-05-02T09:24:17Z | Added HCEval provenance audit for packed dataset source coverage with smoke/direct coverage and latest dataset result sidecars.
2026-05-02T09:37:45Z | Tightened QEMU NIC cardinality audit to count long-form --nic none/--nic=none air-gap disablement with direct/smoke coverage.
2026-05-02T09:50:56Z | Added packed HCEval choice semantics audit for duplicate choices, answer aliases, and prompt-choice leakage with smoke artifacts.
2026-05-02T10:03:12Z | Added CI smoke gate for packed HCEval choice semantics audit clean/failing fixtures.
2026-05-02T10:07:06Z | Added packed HCEval metadata audit for canonical metadata/header consistency with smoke coverage and latest dataset sidecars.
2026-05-02T10:11:01Z | Added eval record-order audit for HolyC/llama/gold stream parity with smoke coverage and latest result sidecars.
2026-05-10T13:40:26Z | Added a global expected-token budget gate to QEMU matrix planning audits with smoke coverage and refreshed results.
2026-05-10T13:57:39Z | Added QEMU block-device policy audit for canonical raw IDE drives, remote transport rejection, smoke coverage, and refreshed reports.
2026-05-10T14:10:38Z | Added eval efficiency frontier dominance-gap telemetry and optional gates with smoke coverage and refreshed reports.
2026-05-10T14:24:18Z | Added packed HCEval record identity audit for duplicate IDs and payloads with smoke coverage and latest dataset sidecars.
2026-05-10T14:42:42Z | Added eval error-overlap audit for shared versus engine-unique HolyC/llama misses with smoke coverage and latest result sidecars.
2026-05-10T14:55:48Z | Added QEMU artifact secret-leak audit for captured commands/tails/env text with smoke coverage and latest result sidecars.
2026-05-10T15:12:52Z | Extended QEMU artifact secret-leak audit with Anthropic and HuggingFace token detectors plus smoke coverage.
2026-05-10T15:22:03Z | Added QEMU artifact host-path portability audit with CI smoke coverage and latest result sidecars.
2026-05-10T15:31:08Z | Added eval calibration engine summary CSV artifacts with smoke gate coverage and refreshed latest reports.
2026-05-10T15:40:26Z | Hardened QEMU artifact secret audit for authorization/cookie-style fields with stdlib test coverage and refreshed latest result.
2026-05-10T15:48:38Z | Added QEMU artifact network-text audit for URL/IP endpoint air-gap drift with smoke coverage and latest result sidecars.
2026-05-10T16:00:48Z | Added paired eval artifact identity audit for HolyC/llama model-tokenizer-quant metadata parity with smoke coverage and latest result sidecars.
2026-05-10T16:26:45Z | Added quant pair manifest audit for Q4_0/Q8_0 tensor coverage with smoke coverage and latest result sidecars.
2026-05-10T16:33:08Z | Hardened dataset provenance-balance CI smoke coverage with artifact assertions and failing threshold gates.
2026-05-10T16:40:44Z | Added QEMU build throughput scorecard CV stability telemetry and optional variability gates with smoke coverage and refreshed results.
2026-05-10T16:53:24Z | Fixed QEMU prompt coverage auditing to count top-level warmup rows when requested, with smoke coverage and refreshed artifacts.
2026-05-10T17:04:03Z | Extended QEMU artifact secret audit to scan text sidecars, added smoke coverage, and refreshed latest results.
2026-05-10T17:13:15Z | Made eval hash warnings visible but nonblocking by default, added warning-mode smoke coverage, and refreshed latest results.
2026-05-10T17:28:38Z | Hardened host-side QEMU air-gap audits to reject URL-backed remote disks/kernels/block devices and refreshed args-policy results.
2026-05-10T17:37:50Z | Extended QEMU artifact budget audit with stdout/stderr tail-retention consistency checks, smoke coverage, and refreshed latest results.
2026-05-10T17:44:25Z | Added QEMU host filesystem share audit for forbidden virtfs/fsdev/virtio-fs passthrough with smoke coverage and refreshed reports.
2026-05-10T17:51:55Z | Added QEMU phase summary consistency audit with smoke coverage and refreshed latest reports.
2026-05-10T17:58:27Z | Extended QEMU artifact secret audit to catch Authorization and Cookie header leaks with smoke coverage and refreshed reports.
2026-05-10T18:04:46Z | Hardened QEMU command fingerprint audit to verify stored row air-gap metadata with smoke coverage and refreshed latest reports.
2026-05-10T18:12:17Z | Added dataset_select selected-record CSV sidecars with smoke coverage, docs, and refreshed sample selection artifacts.
2026-05-10T18:20:12Z | Hardened QEMU args-fragment policy against host filesystem sharing with smoke coverage and refreshed audit results.
2026-05-10T18:27:36Z | Added quant pair size-equation audit for Q4_0/Q8_0 block/byte/element invariants with smoke coverage and refreshed reports.
2026-05-10T18:31:59Z | Added eval workload estimate CI smoke coverage for pass/fail budget paths and refreshed latest reports.
