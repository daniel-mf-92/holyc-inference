# HolyC Inference Engine — Ollama-Compatible LLM Runtime for TempleOS

## Mission

Build a native HolyC inference engine that can load and run GGUF-format LLM models
inside TempleOS. Ollama-compatible means: same model files, same quantization formats,
comparable output quality. The runtime is pure HolyC — no C, no C++, no Go, no FFI.

Target: a user types `Inference("What is truth?");` in TempleOS and gets a response
from a locally-loaded language model, with every token logged to the Book of Truth.

## Design Constraints

- **HolyC-only** — all inference code must be HolyC. Host-side tooling (test harnesses,
  model conversion scripts) may use Python/C for validation.
- **Integer quantized inference first** — HolyC lacks native float types. Start with
  Q4_0 and Q8_0 quantization (all integer math). Float support deferred to WS10.
- **Single model architecture first** — LLaMA family (covers LLaMA, Mistral, Qwen).
  One forward pass implementation, not 30.
- **Air-gapped** — models loaded from disk only. No downloading, no HTTP, no networking.
- **Small models first** — TinyLlama 1.1B Q4_0 (~600MB), Qwen2.5 0.5B Q4_0 (~300MB).
  Prove correctness before scaling to 7B.
- **Book of Truth integration** — every inference call, every token, every tensor op
  checkpoint is loggable by the Book of Truth ledger (WS13 in TempleOS MASTER_TASKS).
- **AVX2 inline asm for hot paths** — HolyC supports inline x86 assembly. Use it for
  quantized dot products, matmul kernels, and attention score computation.
- **No external dependencies** — no libc, no libm, no pthreads. Math functions (exp,
  log, sqrt, softmax) implemented from scratch or via integer approximations.

## North-Star Outcomes

- [ ] GGUF file parser in HolyC (read model metadata, tensor layout, quantization type)
- [ ] Q4_0 and Q8_0 dequantization kernels
- [ ] Matrix multiply (naive + AVX2 optimized)
- [ ] LLaMA-family forward pass (RMSNorm, RoPE, GQA attention, SwiGLU FFN)
- [ ] BPE tokenizer in HolyC (encode/decode, vocab from GGUF)
- [ ] Top-k / top-p / temperature sampling
- [ ] Interactive `Inference("prompt");` HolyC command
- [ ] Benchmark: tokens/sec on TinyLlama 1.1B Q4_0 (target: >2 tok/s single-core)
- [ ] Validation: output matches llama.cpp reference for same prompt+seed+model

## Workstream Map

### WS0 — Project Setup & Reference Material
- [ ] WS0-01 Create project structure (src/, tests/, models/, docs/, automation/)
- [ ] WS0-02 Document GGUF format spec (from llama.cpp/gguf-py reference)
- [ ] WS0-03 Document Q4_0 and Q8_0 quantization math (integer-only formulas)
- [ ] WS0-04 Document LLaMA architecture (layer structure, tensor names, dimensions)
- [ ] WS0-05 Create reference output corpus (llama.cpp outputs for known prompts+seeds)

### WS1 — Integer Math Foundations
- [ ] WS1-01 Implement fixed-point arithmetic helpers (mul, div, shift, round)
- [ ] WS1-02 Implement integer exp/log/sqrt approximations
- [ ] WS1-03 Implement integer softmax approximation
- [ ] WS1-04 Implement RMSNorm (integer path)
- [ ] WS1-05 Add test harness for math functions (compare against reference)

### WS2 — GGUF Parser
- [ ] WS2-01 Implement GGUF header parser (magic, version, tensor count, metadata count)
- [ ] WS2-02 Implement GGUF metadata reader (key-value pairs, string/int/float/array types)
- [ ] WS2-03 Implement GGUF tensor info reader (name, dims, type, offset)
- [ ] WS2-04 Implement tensor data loader (read quantized blocks from file into memory)
- [ ] WS2-05 Add GGUF validation tool (print model info, tensor list, total size)

### WS3 — Quantization Kernels
- [ ] WS3-01 Implement Q4_0 block structure (32 values per block, 1 scale float16)
- [ ] WS3-02 Implement Q4_0 dequantize (block -> integer array)
- [ ] WS3-03 Implement Q4_0 dot product (quantized vec * quantized vec -> scalar)
- [ ] WS3-04 Implement Q8_0 block structure and dequantize
- [ ] WS3-05 Implement Q8_0 dot product
- [ ] WS3-06 AVX2 optimized Q4_0 dot product (inline asm)
- [ ] WS3-07 AVX2 optimized Q8_0 dot product (inline asm)
- [ ] WS3-08 Benchmark: Q4_0 dot product ops/sec (naive vs AVX2)

### WS4 — Matrix Operations
- [ ] WS4-01 Implement naive matmul (quantized input, integer accumulator)
- [ ] WS4-02 Implement row-major / col-major layout handling
- [ ] WS4-03 AVX2 tiled matmul (cache-friendly blocking)
- [ ] WS4-04 Implement element-wise operations (add, mul, scale)
- [ ] WS4-05 Benchmark: matmul GOPS for typical LLM dimensions (4096x4096)

### WS5 — LLaMA Forward Pass
- [ ] WS5-01 Implement RoPE (Rotary Position Embedding) in integer math
- [ ] WS5-02 Implement GQA (Grouped Query Attention) — Q/K/V projection, score, softmax, output
- [ ] WS5-03 Implement SwiGLU FFN (gate * up projection, silu activation, down projection)
- [ ] WS5-04 Implement transformer block (attention + FFN + residual + RMSNorm)
- [ ] WS5-05 Implement full forward pass (embed -> N blocks -> final norm -> logits)
- [ ] WS5-06 Implement KV cache for autoregressive generation
- [ ] WS5-07 Validation: compare logits against llama.cpp for first token of known prompt

### WS6 — Tokenizer
- [ ] WS6-01 Implement BPE vocabulary loader from GGUF metadata
- [ ] WS6-02 Implement BPE encode (text -> token IDs)
- [ ] WS6-03 Implement BPE decode (token IDs -> text)
- [ ] WS6-04 Handle special tokens (BOS, EOS, padding)
- [ ] WS6-05 Validation: tokenize/detokenize round-trip matches llama.cpp

### WS7 — Sampling & Generation
- [ ] WS7-01 Implement temperature scaling on logits
- [ ] WS7-02 Implement top-k sampling
- [ ] WS7-03 Implement top-p (nucleus) sampling
- [ ] WS7-04 Implement repetition penalty
- [ ] WS7-05 Implement autoregressive generation loop (prompt -> tokens one at a time)
- [ ] WS7-06 Implement `Inference(prompt, max_tokens, temp, top_k, top_p);` HolyC API

### WS8 — Integration & Polish
- [ ] WS8-01 End-to-end test: load TinyLlama Q4_0, generate 50 tokens, compare quality
- [ ] WS8-02 Memory usage profiling and optimization
- [ ] WS8-03 Book of Truth integration hooks (log model load, each token, anomalies)
- [ ] WS8-04 Interactive mode with streaming token output to TempleOS console
- [ ] WS8-05 Performance tuning: identify and optimize top-5 hottest functions
- [ ] WS8-06 Documentation: user guide for loading models and running inference

### WS9 — GPU Compute Partition (IOMMU-Safe, Book-of-Truth-Audited)
- Philosophy: GPU acceleration is allowed ONLY under strict isolation. The GPU is a
  powerful but untrustworthy co-processor — it has DMA, opaque firmware, and its own
  execution engine. It must be caged, monitored, and logged.
- [ ] WS9-01 GPU threat model: DMA risks, firmware opacity, MMIO attack surface, IOMMU bypass vectors
- [ ] WS9-02 IOMMU (VT-d/AMD-Vi) initialization and enforcement — mandatory, no GPU without it
- [ ] WS9-03 GPU memory partition design — dedicated physical region, never overlaps kernel/log pages
- [ ] WS9-04 Explicit copy-in/copy-out protocol — CPU copies tensors to GPU region, triggers compute, copies results
- [ ] WS9-05 PCIe device enumeration in HolyC — discover GPU via config space, map BARs
- [ ] WS9-06 GPU command submission interface — minimal ring buffer, no proprietary driver
- [ ] WS9-07 MMIO register map for target GPU compute dispatch (start with virtio-gpu or simple compute)
- [ ] WS9-08 Book of Truth hooks: log every DMA mapping, every GPU command, every MMIO write
- [ ] WS9-09 GPU compute shader/kernel format — simple dispatachable compute tasks
- [ ] WS9-10 Matmul offload: move quantized matrix multiply to GPU, keep results in GPU partition
- [ ] WS9-11 Attention score computation offload to GPU
- [ ] WS9-12 GPU error handling: timeout, hang detection, forced reset, logged to Book of Truth
- [ ] WS9-13 Benchmark: GPU vs CPU inference speed for TinyLlama 1.1B Q4_0
- [ ] WS9-14 Fallback: if no IOMMU available, GPU disabled entirely — CPU-only mode, no exceptions

### WS10 — Multi-Architecture Model Support
- [ ] WS10-01 Abstract forward pass interface — model-agnostic layer dispatch
- [ ] WS10-02 Mistral architecture support (sliding window attention, different GQA config)
- [ ] WS10-03 Qwen2 architecture support (different FFN layout, tied embeddings)
- [ ] WS10-04 Phi-3 architecture support (different attention pattern, block structure)
- [ ] WS10-05 Architecture auto-detection from GGUF metadata keys
- [ ] WS10-06 Model registry: map GGUF architecture string to forward pass implementation

### WS11 — Extended Quantization Support
- [ ] WS11-01 Q5_0 and Q5_1 quantization (5-bit, wider dynamic range)
- [ ] WS11-02 Q2_K, Q3_K, Q4_K, Q5_K, Q6_K (k-quant family, mixed precision)
- [ ] WS11-03 IQ quantization formats (importance-based, variable bit-width per weight)
- [ ] WS11-04 Quantization format auto-selection from GGUF tensor type field
- [ ] WS11-05 Performance comparison matrix: tok/s vs quantization level vs model size

### WS12 — Model Conversion & Preparation Tooling (Host-Side)
- Host-side Python/C tools — NOT part of the HolyC runtime. Used to prepare models
  for TempleOS consumption from standard Hugging Face / Ollama sources.
- [ ] WS12-01 GGUF model validator — verify file integrity, tensor checksums, format version
- [ ] WS12-02 Model preparation guide: download from Hugging Face → quantize → copy to TempleOS disk
- [ ] WS12-03 Ollama model extraction: pull Ollama blob → extract GGUF → validate
- [ ] WS12-04 Model size calculator: predict RAM/disk requirements before loading
- [ ] WS12-05 Reference output generator: run llama.cpp on known prompts, save expected outputs

### WS14 — safetensors Format Support (Hugging Face Native)
- [ ] WS14-01 Document safetensors binary format (header JSON + raw tensor data, no pickle)
- [ ] WS14-02 Implement safetensors header parser in HolyC (JSON key-value, tensor offsets)
- [ ] WS14-03 Implement safetensors tensor loader (mmap-friendly, direct read to memory)
- [ ] WS14-04 Model format auto-detection: GGUF vs safetensors from magic bytes
- [ ] WS14-05 Conversion tool (host-side): safetensors → quantized GGUF for TempleOS

### WS15 — Local LLM Ecosystem Compatibility
- Compatibility with other local LLM systems — same models, interoperable formats.
- [ ] WS15-01 Ollama blob extraction: parse Ollama manifests, extract GGUF from blobs
- [ ] WS15-02 llama.cpp CLI parity: match command-line args for model/prompt/params
- [ ] WS15-03 LM Studio model directory layout compatibility (scan and list available models)
- [ ] WS15-04 OpenAI-compatible local API (CLI-based, serial-port accessible, no HTTP)
- [ ] WS15-05 Model card parser: extract metadata from Hugging Face model cards for display
- [ ] WS15-06 Benchmark suite: compare tok/s against llama.cpp, Ollama, LM Studio for same model

### WS13 — Advanced Inference Features
- [ ] WS13-01 Multi-turn conversation context management
- [ ] WS13-02 System prompt / instruction template support (ChatML, Alpaca, Llama-style)
- [ ] WS13-03 Batch inference: process multiple prompts efficiently
- [ ] WS13-04 Speculative decoding: use small model to draft, large model to verify
- [ ] WS13-05 Persistent KV cache across sessions (save/load to disk)
- [ ] WS13-06 Token streaming callback for real-time display

## Queue Semantics

- Rolling work queue, keep at least 15 unchecked IQ items at all times.
- One IQ item per iteration. Each must be small, implementable, and testable.
- IQ = Inference Queue (to distinguish from CQ in the TempleOS modernization loop).

## Inference Queue (Rolling, unbounded)

- [x] IQ-001 Create project directory structure: `src/`, `tests/`, `docs/`, `automation/`, `models/` (WS0-01)
- [x] IQ-002 Write GGUF format reference doc at `docs/GGUF_FORMAT.md` from llama.cpp spec (WS0-02)
- [x] IQ-003 Write Q4_0/Q8_0 quantization reference doc at `docs/QUANTIZATION.md` (WS0-03)
- [x] IQ-004 Write LLaMA architecture reference doc at `docs/LLAMA_ARCH.md` (WS0-04)
- [x] IQ-005 Implement fixed-point multiply/divide helpers in `src/math/fixedpoint.HC` (WS1-01)
- [x] IQ-006 Implement integer exp approximation in `src/math/intexp.HC` (WS1-02)
- [x] IQ-007 Implement integer sqrt approximation in `src/math/intsqrt.HC` (WS1-02)
- [x] IQ-008 Implement integer softmax in `src/math/softmax.HC` (WS1-03)
- [x] IQ-009 Implement RMSNorm (integer path) in `src/math/rmsnorm.HC` (WS1-04)
- [x] IQ-010 Implement GGUF magic/version/header parser in `src/gguf/header.HC` (WS2-01)
- [x] IQ-011 Implement GGUF metadata KV reader in `src/gguf/metadata.HC` (WS2-02)
- [x] IQ-012 Implement GGUF tensor info reader in `src/gguf/tensorinfo.HC` (WS2-03)
- [x] IQ-013 Implement Q4_0 block struct and dequantize in `src/quant/q4_0.HC` (WS3-01, WS3-02)
- [x] IQ-014 Implement Q4_0 dot product (naive) in `src/quant/q4_0_dot.HC` (WS3-03)
- [x] IQ-015 Implement Q8_0 block struct and dequantize in `src/quant/q8_0.HC` (WS3-04)
- [ ] IQ-016 Draft `docs/GGUF_FORMAT.md` skeleton with sections for header, metadata, tensor info, and alignment rules (WS0-02)
- [ ] IQ-017 Write host-side GGUF header/tensor dump fixture plan in `tests/README.md` for llama.cpp parity checks (WS0-05, WS2-05)
- [ ] IQ-018 Create `tests/README.md` skeleton for llama.cpp parity fixtures (header, metadata, tensor offsets) (WS0-05, WS2-05)
- [ ] IQ-019 Create `src/math/fixedpoint.HC` skeleton with Q16 constants, core type aliases, and TODO stubs for mul/div helpers (WS1-01)
- [ ] IQ-020 Implement Q16 integer exponent range clamp and base constants in `src/math/intexp.HC` (WS1-02)
- [ ] IQ-021 Add host-side Q16 exp parity harness in `tests/test_intexp_q16.py` against `math.exp` samples (WS1-05)
- [ ] IQ-022 Implement GGUF header constants and struct layout notes in `src/gguf/header.HC` before parser logic (WS2-01)
- [ ] IQ-023 Create `src/math/rmsnorm.HC` skeleton with Q16 constants, tensor shape assumptions, and TODO stubs for scale/variance accumulation (WS1-04)
- [ ] IQ-024 Create `src/gguf/header.HC` skeleton with `GGUFHeader` struct and endian-safe integer read helper stubs (WS2-01)
- [ ] IQ-025 Add host-side GGUF header parser parity fixture in `tests/test_gguf_header_parse.py` covering valid/magic/version/truncation cases (WS2-01, WS2-05)
- [x] IQ-026 Implement metadata key lookup helpers (`GGUFMetaFindByKey`, scalar extractors) in `src/gguf/metadata.HC` (WS2-02)
- [ ] IQ-027 Add host-side metadata parser parity fixture for scalar/string/array/nested-array cases in `tests/test_gguf_metadata_parse.py` (WS2-02, WS2-05)
- [x] IQ-028 Implement GGUF tensor data base alignment helper in `src/gguf/tensor_data_base.HC` (WS2-04)
- [x] IQ-029 Implement Q8_0 dot product (naive, integer-only accumulator) in `src/quant/q8_0_dot.HC` (WS3-05)
- [x] IQ-030 Implement mixed Q4_0 x Q8_0 dot product kernel in `src/quant/q4_0_q8_0_dot.HC` (WS3-05)
- [x] IQ-031 Implement Q8_0 blockwise dot-to-Q16 accumulation helper in `src/quant/q8_0_dot.HC` for matmul callers (WS4-01)
- [ ] IQ-032 Add mixed Q4_0 x Q8_0 dot parity harness in `tests/test_q4_0_q8_0_dot.py` with GGML-math bounds (WS3-05)
- [x] IQ-033 Implement Q4_0 row-dot helper (`Q4_0DotRowBlocksQ16`) in `src/quant/q4_0_dot.HC` for quant matmul row kernels (WS4-01)
- [x] IQ-034 Implement Q4_0 x Q8_0 blockwise Q16 accumulation helper in `src/quant/q4_0_q8_0_dot.HC` for mixed matmul callers (WS4-01)
- [ ] IQ-035 Add Q8_0 Q16-accumulator parity harness in `tests/test_q8_0_dot_accum_q16.py` with seeded blockwise rounding checks (WS4-01)
- [x] IQ-036 Implement GGUF tensor data offset resolver (`GGUFTensorDataBaseOffset`) in `src/gguf/tensor_data_base.HC` with alignment validation (WS2-04)
- [x] IQ-037 Implement GGML tensor block-size/byte-size helpers in `src/gguf/tensor_data_base.HC` for F32/F16/Q4_0/Q8_0 sizing math (WS2-04)
- [x] IQ-038 Implement GGUF tensor absolute range validator (`GGUFTensorResolveRange`) in `src/gguf/tensor_data_base.HC` with overflow-safe `abs+size` checks (WS2-04)
- [ ] IQ-039 Implement GGUF tensor payload byte-size helper (`GGUFTensorBytesForType`) in `src/gguf/tensor_data_base.HC` for F32/F16/Q4_0/Q8_0 (WS2-04)
- [x] IQ-040 Implement GGUF tensor-table range validator (`GGUFValidateTensorRanges`) in `src/gguf/tensor_data_base.HC` to enforce non-overlap and EOF bounds (WS2-04)
- [x] IQ-041 Implement `GGUFValidateTensorRangesSorted` in `src/gguf/tensor_data_base.HC` using in-place sort + single-pass overlap scan for large tensor tables (WS2-04)
- [x] IQ-042 Implement `GGUFTensorInfoResolveByteSpans` in `src/gguf/tensor_data_base.HC` to derive per-tensor payload bytes from dims+type and feed sorted range validation (WS2-04)
- [x] IQ-043 Implement `GGUFTensorInfoResolveAbsRanges` in `src/gguf/tensor_data_base.HC` to output per-tensor absolute `[start,end)` spans after byte-size derivation and range validation (WS2-04)
- [ ] IQ-044 Implement `GGUFTensorRangeFindByAbsOffset` in `src/gguf/tensor_data_base.HC` (binary search over sorted `[start,end)` spans) for O(log n) tensor payload lookup (WS2-04)

## Progress Ledger

| Date | Iteration | Task | Result | Notes |
|---|---|---|---|---|
| 2026-04-12 | bootstrap | MASTER_TASKS init | done | Initial roadmap + queue created |
| 2026-04-12 | loop-001 | IQ-001 directory structure | done | Verified `src/`, `tests/`, `docs/`, `automation/`, `models/` exist locally |
| 2026-04-12 | loop-002 | IQ-002 GGUF format reference doc | done | Added `docs/GGUF_FORMAT.md` with header/metadata/tensor/alignment rules and HolyC parser safety checks |
| 2026-04-12 | loop-003 | IQ-003 quantization reference doc | done | Added `docs/QUANTIZATION.md` with integer-only Q4_0/Q8_0 decode, fixed-point scaling, and dot-product formulas |
| 2026-04-12 | loop-004 | IQ-004 LLaMA architecture reference doc | done | Added `docs/LLAMA_ARCH.md` covering decoder flow, RMSNorm, RoPE, GQA, SwiGLU, KV cache, and required GGUF fields |
| 2026-04-12 | loop-005 | IQ-005 fixed-point mul/div helpers | done | Added `src/math/fixedpoint.HC` with Q16 helpers (`FPQ16Mul`, `FPQ16Div`, conversions); validated host-side with `python3` arithmetic parity checks |
| 2026-04-12 | loop-006 | IQ-006 integer exp approximation | done | Added `src/math/intexp.HC` with Q16 range reduction + 4th-order polynomial; validated host-side over x∈[-8,8] (`max_rel_err=4.48%`) |
| 2026-04-12 | loop-007 | IQ-007 integer sqrt approximation | done | Added `src/math/intsqrt.HC` with bitwise `IntSqrtU64` + `FPQ16Sqrt`; validated via host parity checks (`intsqrt_exact_checks=ok`, `max_rel_err=0.007690%`) |
| 2026-04-12 | loop-008 | IQ-008 integer softmax | done | Added `src/math/softmax.HC` with stable max-shifted Q16 softmax and sum-to-one correction; validated host-side parity (`softmax_q16_checks=ok`, `max_abs_err=0.003334`) |
| 2026-04-12 | loop-009 | IQ-009 integer RMSNorm | done | Added `src/math/rmsnorm.HC` with Q16 RMSNorm (`mean(x^2)+eps`, rsqrt via `FPQ16Sqrt`, per-channel scale); validated host-side parity (`rmsnorm_q16_checks=ok`, `max_abs_err=0.000075`) |
| 2026-04-12 | loop-010 | IQ-010 GGUF header parser | done | Added `src/gguf/header.HC` with little-endian U32/U64 readers and `GGUFParseHeader` validation (magic/version/count/truncation); validated via focused host parity check (`gguf_header_reference_checks=ok`) |
| 2026-04-12 | loop-011 | IQ-011 GGUF metadata KV reader | done | Added `src/gguf/metadata.HC` with full key/value parsing (scalar, string, array), strict bounds/type checks, and metadata table lifetime helpers; validated with `python3 tests/test_gguf_metadata_parse.py` (`gguf_metadata_reference_checks=ok`) |
| 2026-04-12 | loop-012 | IQ-012 GGUF tensor info reader | done | Added `src/gguf/tensorinfo.HC` with tensor-name parsing, dim/product overflow guards, GGML type validation, and table lifetime helpers; validated with `python3 tests/test_gguf_tensorinfo_parse.py` + `python3 tests/test_gguf_metadata_parse.py` (`gguf_tensorinfo_reference_checks=ok`, `gguf_metadata_reference_checks=ok`) |
| 2026-04-12 | loop-013 | IQ-013 Q4_0 block struct + dequant | done | Added `src/quant/q4_0.HC` with Q4_0 block layout, fp16->Q16 integer scale conversion, nibble unpack (`q-8`), single/multi-block dequant helpers; validated with `python3 tests/test_q4_0_dequant.py` + regression parsers (`q4_0_dequant_reference_checks=ok`) |
| 2026-04-12 | loop-014 | IQ-014 Q4_0 dot product (naive) | done | Added `src/quant/q4_0_dot.HC` with per-block and multi-block integer dot kernels (`Q4_0DotProductBlockQ32`, `Q4_0DotProductBlocksQ32`, `Q4_0DotQ32ToQ16`); validated with `python3 tests/test_q4_0_dot.py` + `python3 tests/test_q4_0_dequant.py` (`q4_0_dot_reference_checks=ok`, `q4_0_dequant_reference_checks=ok`) |
| 2026-04-12 | loop-015 | IQ-015 Q8_0 block struct + dequant | done | Added `src/quant/q8_0.HC` with Q8_0 block layout, fp16->Q16 integer scale conversion, signed-byte unpack, single/multi-block dequant helpers; added `tests/test_q8_0_dequant.py`; validated with `python3 tests/test_q8_0_dequant.py && python3 tests/test_q4_0_dequant.py && python3 tests/test_q4_0_dot.py` (`q8_0_dequant_reference_checks=ok`, `q4_0_dequant_reference_checks=ok`, `q4_0_dot_reference_checks=ok`) |
| 2026-04-12 | loop-016 | IQ-029 Q8_0 dot product (naive) | done | Added `src/quant/q8_0_dot.HC` integer block/multi-block dot kernels + `tests/test_q8_0_dot.py`; validated with `python3 tests/test_q8_0_dot.py && python3 tests/test_q8_0_dequant.py` (`q8_0_dot_reference_checks=ok`, `q8_0_dequant_reference_checks=ok`) |
| 2026-04-12 | loop-017 | IQ-030 Q4_0 x Q8_0 mixed dot | done | Added `src/quant/q4_0_q8_0_dot.HC` mixed integer dot kernels; validated with `python3 tests/test_q4_0_q8_0_dot_kernel.py && python3 tests/test_q4_0_dot.py && python3 tests/test_q8_0_dot.py` (`q4_0_q8_0_dot_kernel_reference_checks=ok`, `q4_0_dot_reference_checks=ok`, `q8_0_dot_reference_checks=ok`) |
| 2026-04-12 | loop-018 | IQ-031 Q8_0 Q16 accumulator | done | Added Q16 blockwise accumulator helper + parity checks; `python3 tests/test_q8_0_dot.py && python3 tests/test_q8_0_dequant.py` passed |
| 2026-04-12 | loop-019 | IQ-033 Q4_0 row-dot helper | done | Added `Q4_0DotRowBlocksQ16` + parity rounding tests; `python3 tests/test_q4_0_dot.py && python3 tests/test_q4_0_dequant.py && python3 tests/test_q4_0_q8_0_dot_kernel.py` passed |
| 2026-04-12 | loop-020 | IQ-034 Q4_0 x Q8_0 Q16 accumulator | done | Added mixed Q16 blockwise accumulator helper + parity checks; `python3 tests/test_q4_0_q8_0_dot_kernel.py && python3 tests/test_q4_0_dot.py && python3 tests/test_q8_0_dot.py` passed |
| 2026-04-12 | loop-021 | IQ-026 metadata key lookup helpers | done | Added `GGUFMetaFindByKey` + typed scalar extractors; `python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-12 | loop-022 | IQ-028 tensor data base alignment helper | done | Added `src/gguf/tensor_data_base.HC` alignment/overflow checks + `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_metadata_parse.py && python3 tests/test_gguf_tensorinfo_parse.py` passed |
| 2026-04-12 | loop-023 | IQ-036 tensor data offset resolver | done | Added `GGUFTensorDataBaseOffset`/default in `src/gguf/tensor_data_base.HC` with base/relative alignment and overflow guards; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-12 | loop-024 | IQ-037 tensor block-size/byte-size helpers | done | Added `GGUFTensorTypeBlockSize`/`GGUFTensorTypeBlockBytes` in `src/gguf/tensor_data_base.HC`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-12 | loop-025 | IQ-038 tensor absolute range validator | done | Added `GGUFTensorResolveRange`/default in `src/gguf/tensor_data_base.HC` with overflow + file-bound checks and expanded parity harness; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-12 | loop-026 | IQ-040 tensor-table range validator | done | Added `GGUFValidateTensorRanges` in `src/gguf/tensor_data_base.HC` with per-range validation + non-overlap enforcement; validated with `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` |
| 2026-04-12 | loop-027 | IQ-041 sorted tensor-table range validator | done | Added `GGUFValidateTensorRangesSorted` + in-place heapsort tie-broken by `(offset,nbytes)` in `src/gguf/tensor_data_base.HC`; validated with `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` |
| 2026-04-12 | loop-028 | IQ-042 tensor info byte-span resolver | done | Added `GGUFTensorInfoResolveByteSpans` + parity tests in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-12 | loop-029 | IQ-043 tensor info absolute range resolver | done | Added `GGUFTensorInfoResolveAbsRanges` in `src/gguf/tensor_data_base.HC` + parity tests for sorted absolute `[start,end)` output and failure propagation; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |

## Blockers & Decisions

- HolyC float support: deferred. Use integer-only quantized inference (Q4_0, Q8_0).
- Model architecture: LLaMA family only. Single forward pass implementation.
- Target model: TinyLlama 1.1B Q4_0 (~600MB). Proves correctness before scaling up.
- All .HC files are HolyC source intended for TempleOS compilation. Host-side test
  harnesses may use Python/C to validate outputs against llama.cpp reference.
- Air-gap mandate: networking stack work is out-of-scope for TempleOS guest; any VM run
  command must explicitly disable NICs (`-nic none`, legacy fallback `-net none`).
