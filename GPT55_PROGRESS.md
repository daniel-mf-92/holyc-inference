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
