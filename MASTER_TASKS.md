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
- [x] IQ-032 Add mixed Q4_0 x Q8_0 dot parity harness in `tests/test_q4_0_q8_0_dot.py` with GGML-math bounds (WS3-05)
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
- [x] IQ-044 Implement `GGUFTensorRangeFindByAbsOffset` in `src/gguf/tensor_data_base.HC` (binary search over sorted `[start,end)` spans) for O(log n) tensor payload lookup (WS2-04)
- [x] IQ-045 Implement `GGUFTensorRangeFindByRelOffset` in `src/gguf/tensor_data_base.HC` by translating relative offset to absolute and delegating to `GGUFTensorRangeFindByAbsOffset` (WS2-04)
- [x] IQ-046 Implement `GGUFTensorRangeFindByRelOffsetDefault` in `src/gguf/tensor_data_base.HC` using `GGUF_DEFAULT_ALIGNMENT` and delegate to `GGUFTensorRangeFindByRelOffset` (WS2-04)
- [x] IQ-047 Implement `GGUFTensorInfoBuildOffsetIndex` in `src/gguf/tensor_data_base.HC` to build sorted absolute ranges + original tensor indices for stable lookup (WS2-04)
- [x] IQ-048 Add indexed tensor lookup helpers (`GGUFTensorRangeFindIndexByAbsOffset`, `GGUFTensorRangeFindIndexByRelOffset`) in `src/gguf/tensor_data_base.HC` returning original tensor index (WS2-04)
- [x] IQ-049 Add default-alignment indexed lookup helper `GGUFTensorRangeFindIndexByRelOffsetDefault` in `src/gguf/tensor_data_base.HC` and parity tests in `tests/test_gguf_tensor_data_base.py` (WS2-04)
- [x] IQ-050 Add offset-index integrity validator `GGUFValidateSortedTensorIndex` (permutation+range checks) in `src/gguf/tensor_data_base.HC` with adversarial host parity tests (WS2-04)
- [x] IQ-051 Add sorted-index inverse map builder `GGUFBuildTensorSortedPositionMap` in `src/gguf/tensor_data_base.HC` (original index -> sorted position) with parity checks in `tests/test_gguf_tensor_data_base.py` (WS2-04)
- [x] IQ-052 Implement `GGUFValidateSortedPositionMap` in `src/gguf/tensor_data_base.HC` to verify original->sorted inverse map bounds/permutation consistency against `sorted_tensor_indices` (WS2-04)
- [x] IQ-053 Implement `GGUFTensorIndexToAbsRange` in `src/gguf/tensor_data_base.HC` using inverse map + sorted span arrays for O(1)+lookup tensor range retrieval by original index (WS2-04)
- [x] IQ-054 Add host-side parity harnesses for inverse-map/range-index helpers in `tests/test_gguf_tensor_data_base.py` with randomized permutation adversarial cases (WS2-04)
- [x] IQ-055 Implement HolyC helper `GGUFTensorRangeFindByIndex` in `src/gguf/tensor_data_base.HC` to resolve original tensor index directly to `[abs_start,abs_end)` via sorted-position map (WS2-04)
- [x] IQ-056 Implement HolyC helper `GGUFTensorLookupByRelOffset` in `src/gguf/tensor_data_base.HC` returning original tensor index plus `[abs_start,abs_end)` for a relative tensor offset via existing offset-index maps (WS2-04)
- [x] IQ-057 Implement HolyC helper `GGUFTensorLookupByAbsOffset` in `src/gguf/tensor_data_base.HC` returning original tensor index plus `[abs_start,abs_end)` for an absolute payload offset via sorted offset-index maps (WS2-04)

- [x] IQ-058 Implement HolyC helper `GGUFTensorLookupByAbsOffsetDefault` in `src/gguf/tensor_data_base.HC` as default-alignment wrapper over absolute lookup for metadata-absent callers (WS2-04)
- [x] IQ-059 Implement HolyC helper `GGUFTensorLookupByAbsOffsetWithInnerOffset` in `src/gguf/tensor_data_base.HC` to return original tensor index, `[abs_start,abs_end)`, and `offset_in_tensor` for direct payload-addressed reads (WS2-04)
- [x] IQ-060 Implement HolyC helper `GGUFTensorLookupByRelOffsetWithInnerOffset` in `src/gguf/tensor_data_base.HC` to resolve relative payload offsets into original tensor index, absolute span, and inner offset bytes (WS2-04)
- [x] IQ-061 Implement HolyC helper `GGUFTensorLookupByRelOffsetWithInnerOffsetDefault` in `src/gguf/tensor_data_base.HC` as a `GGUF_DEFAULT_ALIGNMENT` wrapper for metadata-absent callers (WS2-04)
- [x] IQ-062 Implement HolyC lookup-table validator `GGUFValidateTensorLookupTables` in `src/gguf/tensor_data_base.HC` to check sorted ranges + sorted-index permutation + inverse-map consistency in one pass (WS2-04)
- [x] IQ-063 Implement HolyC helper `GGUFTensorLookupByIndexWithInnerOffset` in `src/gguf/tensor_data_base.HC` to resolve original tensor index + caller absolute offset into validated `[abs_start,abs_end)` plus `offset_in_tensor` (WS2-04)
- [x] IQ-064 Implement HolyC helper `GGUFTensorLookupByIndexWithRelOffset` in `src/gguf/tensor_data_base.HC` to resolve original tensor index + caller relative offset into validated `[abs_start,abs_end)` plus `offset_in_tensor` with alignment checks (WS2-04)
- [x] IQ-065 Implement HolyC helper `GGUFTensorInfoBuildLookupTables` in `src/gguf/tensor_data_base.HC` to compose offset index build + inverse-map build + lookup-table validation in one checked setup path (WS2-04)
- [x] IQ-066 Add host-side parity/adversarial harness for `GGUFTensorInfoBuildLookupTables` in `tests/test_gguf_tensor_data_base.py` with malformed permutation and overlap cases (WS2-04)
- [x] IQ-067 Implement HolyC helper `GGUFTensorLookupSpanByAbsRange` in `src/gguf/tensor_data_base.HC` to resolve `[abs_start,abs_end)` windows into containing tensor index plus validated in-tensor span offsets (WS2-04)
- [x] IQ-068 Implement HolyC helper `GGUFTensorLookupSpanByRelRange` in `src/gguf/tensor_data_base.HC` to resolve relative `[start,end)` windows via tensor-data base offset and return tensor index + in-tensor span offsets (WS2-04)
- [x] IQ-069 Implement HolyC helper `GGUFTensorLookupSpanByIndexAndAbsRange` in `src/gguf/tensor_data_base.HC` to validate caller-selected tensor index against an absolute byte window and return checked in-tensor span offsets (WS2-04)
- [x] IQ-070 Implement HolyC helper `GGUFTensorLookupSpanByIndexAndRelRange` in `src/gguf/tensor_data_base.HC` to validate caller-selected tensor index against a relative byte window (via tensor-data base + alignment) and return checked in-tensor span offsets (WS2-04)
- [x] IQ-071 Implement HolyC helper `GGUFTensorLookupSpanByIndexAndRelRangeDefault` in `src/gguf/tensor_data_base.HC` as a `GGUF_DEFAULT_ALIGNMENT` wrapper over index+relative range span lookup, with host parity tests in `tests/test_gguf_tensor_data_base.py` (WS2-04)
- [x] IQ-072 Implement HolyC helper `GGUFTensorInfoLoadPayloadWindowByIndexRelDefault` in `src/gguf/tensor_data_base.HC` to resolve index+relative window (default alignment) into checked `[abs_start,abs_end)` plus in-tensor span for parser-side chunked reads (WS2-04)
- [x] IQ-073 Implement HolyC helper `GGUFTensorInfoLoadPayloadWindowByIndexAbs` in `src/gguf/tensor_data_base.HC` to validate caller absolute payload windows against selected tensor index and return checked in-tensor span for direct file-pointer reads (WS2-04)
- [x] IQ-074 Implement HolyC helper `GGUFTensorInfoLoadPayloadWindowByIndexRel` in `src/gguf/tensor_data_base.HC` with caller-specified alignment (non-default) and shared absolute-window validation for parser paths that honor `general.alignment` (WS2-04)
- [x] IQ-075 Implement HolyC helper `GGUFTensorInfoLoadPayloadWindowByRelSpan` in `src/gguf/tensor_data_base.HC` to resolve relative payload windows (caller alignment) to owning original tensor index plus validated absolute/in-tensor spans (WS2-04)
- [x] IQ-076 Implement HolyC helper `GGUFTensorInfoLoadPayloadWindowByAbsSpan` in `src/gguf/tensor_data_base.HC` to resolve absolute payload windows to owning original tensor index plus validated in-tensor spans for direct stream readers (WS2-04)
- [x] IQ-077 Implement HolyC integer natural-log approximation core in `src/math/intlog.HC` (Q16 input/output, overflow-safe domain clamps, monotonicity-preserving piecewise polynomial) with parity-target hooks for WS1-02/WS1-05 harnesses (WS1-02)
- [x] IQ-078 Implement HolyC helper `FPQ16Log2` in `src/math/intlog.HC` via `FPQ16Ln` and fixed-point `1/ln(2)` scaling with overflow-safe rounding (WS1-02)
- [x] IQ-079 Implement HolyC helper `FPQ16LnRatio` in `src/math/intlog.HC` for `ln(num/den)` with positive-domain validation and shared core range reduction (WS1-02)
- [x] IQ-080 Implement HolyC helper `FPQ16EntropyFromProbs` in `src/math/softmax.HC` using `FPQ16Ln` to compute Shannon entropy diagnostics from Q16 probabilities (WS1-03)
- [x] IQ-081 Add host-side parity harness `tests/test_intlog_ratio_q16.py` covering `FPQ16Log2` + `FPQ16LnRatio` against `math.log2`/`math.log` bounds (WS1-05)
- [x] IQ-082 Add host-side parity harness `tests/test_softmax_entropy_q16.py` validating Q16 entropy helper against float reference vectors and edge-case clamps (WS1-05)
- [x] IQ-083 Implement HolyC `Q8_0DotProductBlocksQ32` in `src/quant/q8_0_dot.HC` with widened I64 accumulator and Q16 output rounding for WS4 matmul callers (WS4-01)
- [x] IQ-084 Implement HolyC `Q4_0DotProductBlocksQ32` in `src/quant/q4_0_dot.HC` with widened I64 accumulator and Q16 output rounding for WS4 matmul callers (WS4-01)
- [x] IQ-085 Implement HolyC `Q4_0Q8_0DotProductBlocksQ32` in `src/quant/q4_0_q8_0_dot.HC` with widened I64 accumulator and Q16 output rounding for mixed-kernel WS4 matmul callers (WS4-01)
- [x] IQ-086 Implement HolyC `Q4_0Q8_0DotProductBlocksQ32ToQ16` row-helper integration tests in `src/quant/q4_0_q8_0_dot.HC` + `tests/test_q4_0_q8_0_dot_kernel.py` for single-rounding-vs-blockwise invariants (WS4-01)
- [x] IQ-087 Implement HolyC `Q8_0DotRowBlocksQ16` in `src/quant/q8_0_dot.HC` with parity/semantic tests in `tests/test_q8_0_dot.py` (blockwise rounding lane contract for WS4 matmul row kernels) (WS4-01)
- [x] IQ-088 Implement HolyC `Q8_0DotRowsQ16MatrixVector` in `src/quant/q8_0_dot.HC` for contiguous row-block matrix×vector kernels with explicit `row_stride_blocks` (WS4-02)
- [x] IQ-089 Add host-side parity harness `tests/test_q8_0_dot_matrix_vector.py` for `Q8_0DotRowsQ16MatrixVector` covering stride, sign, and per-row rounding semantics (WS4-02)
- [x] IQ-090 Implement HolyC `Q8_0DotProductBlocksQ32Checked` in `src/quant/q8_0_dot.HC` with overflow-safe accumulation and parity/adversarial tests for hostile block streams (WS4-01)
- [x] IQ-091 Implement HolyC Q16 fixed-point multiply/divide/round helpers (`FPQ16Mul`, `FPQ16Div`, `FPQ16MulSat`) in `src/math/fixedpoint.HC` with overflow-checked I64 intermediates for WS1 arithmetic callers (WS1-01)
- [x] IQ-092 Implement HolyC `Q4_0DotRowsQ16MatrixVector` in `src/quant/q4_0_dot.HC` for contiguous row-block matrix×vector kernels with explicit `row_stride_blocks`, plus parity coverage in `tests/test_q4_0_dot.py` (WS4-02)
- [x] IQ-093 Implement HolyC `Q4_0Q8_0DotRowsQ16MatrixVector` in `src/quant/q4_0_q8_0_dot.HC` for mixed quantized row-block matrix×vector kernels with explicit `row_stride_blocks`, plus parity coverage in `tests/test_q4_0_q8_0_dot_kernel.py` (WS4-02)
- [x] IQ-094 Implement HolyC `Q4_0Q8_0DotRowsQ16MatrixVectorChecked` in `src/quant/q4_0_q8_0_dot.HC` with explicit matrix/vector extent and accumulator-overflow guards, plus adversarial parity coverage in `tests/test_q4_0_q8_0_dot_kernel.py` (WS4-02)
- [x] IQ-095 Implement HolyC `Q8_0DotRowsQ16MatrixVectorChecked` in `src/quant/q8_0_dot.HC` with explicit matrix/vector extent + row accumulator overflow guards and adversarial parity coverage in `tests/test_q8_0_dot.py` (WS4-02)
- [x] IQ-096 Implement HolyC `Q8_0DotRowBlocksQ16Checked` alias helper in `src/quant/q8_0_dot.HC` for explicit checked row-dot call-sites reusing `Q8_0DotProductBlocksQ16AccumulateChecked` semantics (WS4-02)
- [x] IQ-097 Implement HolyC compatibility wrapper `Q8_0_DotRowsQ16MatrixVectorChecked` in `src/quant/q8_0_dot.HC` for mixed kernel naming parity with checked extent/overflow path (WS4-02)
- [x] IQ-098 Add host-side parity harness `tests/test_q8_0_dot_matrix_vector_checked.py` for `Q8_0DotRowsQ16MatrixVectorChecked` covering capacity bounds, row-base overflow guards, and adversarial accumulator overflow vectors (WS4-02)
- [x] IQ-099 Implement HolyC checked Q8_0 tiled matmul kernel `Q8_0MatMulQ16TiledChecked` in `src/matmul/q8_0_matmul.HC` (explicit tile bounds, row/col stride, and I64 overflow guards) (WS4-03)
- [x] IQ-100 Implement HolyC checked mixed tiled matmul kernel `Q4_0Q8_0MatMulQ16TiledChecked` in `src/matmul/q4_0_q8_0_matmul.HC` reusing mixed dot kernels with explicit K-block contracts (WS4-03)
- [x] IQ-101 Implement HolyC AVX2 lane-pack helper `Q8_0Pack32ToI16LanesAVX2` in `src/quant/q8_0_avx2.HC` as a preparatory primitive for WS4 tiled AVX2 matmul inner loops (WS4-03)
- [x] IQ-102 Implement HolyC AVX2 pairwise multiply helper `Q8_0MulI16LanesToI32PairsAVX2` in `src/quant/q8_0_avx2.HC` (32 widened lanes -> 16 pairwise I32 products) for WS4 tiled AVX2 dot/matmul inner loops (WS4-03)
- [x] IQ-103 Implement HolyC AVX2 horizontal-sum helper `Q8_0HSumI32PairsAVX2` in `src/quant/q8_0_avx2.HC` (16 pairwise I32 products -> scalar I64) with explicit signed overflow checks for WS4 dot/matmul reductions (WS4-03)
- [x] IQ-104 Implement HolyC AVX2 lane-dot helper `Q8_0DotI16LanesAVX2` in `src/quant/q8_0_avx2.HC` by composing pairwise multiply + checked horizontal reduction for fixed 32-lane inner loops (WS4-03)
- [x] IQ-105 Add host-side parity/adversarial harness `tests/test_q8_0_avx2_dot_lanes.py` for `Q8_0MulI16LanesToI32PairsAVX2` + planned lane-dot reduction helpers (signed-edge vectors, lane-order invariants, overflow probes) (WS4-03)
- [x] IQ-106 Implement HolyC `Q8_0DotBlocksAVX2Q32Checked` in `src/quant/q8_0_avx2.HC` to compose lane-pack + lane-dot helpers across N Q8_0 blocks with explicit accumulator overflow guards (WS4-03)
- [x] IQ-107 Add host-side parity harness `tests/test_q8_0_avx2_blocks_q32.py` for multi-block AVX2 Q32 dot composition (scale conversion, lane-dot parity, checked accumulation) (WS4-03)
- [x] IQ-108 Implement HolyC `Q8_0DotRowsAVX2Q32Checked` in `src/quant/q8_0_avx2.HC` for row-wise matrix×vector AVX2 block-dot accumulation with stride/capacity guards (WS4-03)
- [x] IQ-109 Implement HolyC `Q8_0MatMulQ32TiledAVX2Checked` in `src/matmul/q8_0_matmul.HC` using AVX2 block-dot inner loops and checked tile bounds (WS4-03)
- [x] IQ-110 Add host-side parity harness `tests/test_q8_0_matmul_tiled_avx2_q32.py` for AVX2 tiled Q32 matmul against scalar checked reference (WS4-03)
- [x] IQ-111 Implement HolyC `Q8_0DotBlocksAVX2Q32ToQ16Checked` in `src/quant/q8_0_avx2.HC` for single-rounding Q32->Q16 handoff used by Q16 row kernels (WS4-03)

- [x] IQ-112 Implement HolyC `Q8_0DotRowsAVX2Q32ToQ16Checked` in `src/quant/q8_0_avx2.HC` to convert row-wise AVX2 Q32 outputs into Q16 with one rounded downshift per row (WS4-03)
- [x] IQ-113 Implement HolyC `Q8_0MatMulQ16TiledAVX2Checked` in `src/matmul/q8_0_matmul.HC` by composing `Q8_0DotBlocksAVX2Q32ToQ16Checked` with checked tile bounds and Q16 output stride contracts (WS4-03)
- [x] IQ-114 Add host-side parity harness `tests/test_q8_0_matmul_tiled_avx2_q16.py` for `Q8_0MatMulQ16TiledAVX2Checked` against scalar checked Q16 reference (WS4-03)
- [x] IQ-115 Implement HolyC `Q4_0Q8_0DotBlocksAVX2Q32Checked` in `src/quant/q4_0_q8_0_dot.HC` by extending AVX2 lane-pack/multiply/reduce flow to mixed-nibble Q4_0 and signed Q8_0 blocks with checked I64 accumulation (WS4-03)
- [x] IQ-116 Implement HolyC `Q4_0Q8_0MatMulQ32TiledAVX2Checked` in `src/matmul/q4_0_q8_0_matmul.HC` using mixed AVX2 block dots with explicit tile/stride/capacity overflow guards (WS4-03)
- [x] IQ-117 Add host-side parity harness `tests/test_q4_0_q8_0_matmul_tiled_avx2_q32.py` for mixed AVX2 tiled Q32 matmul against scalar checked reference outputs (WS4-03)
- [x] IQ-118 Implement HolyC `Q8_0MatMulTiledValidateArgsChecked` helper in `src/matmul/q8_0_matmul.HC` and route both AVX2 tiled entrypoints through the shared checked preflight contract (WS4-03)
- [x] IQ-119 Add host-side parity harness `tests/test_q8_0_matmul_tiled_avx2_preflight.py` to assert Q32/Q16 tiled entrypoints return identical BAD_LEN/OVERFLOW surfaces for shared preflight invariants (WS4-03)
- [x] IQ-120 Implement HolyC helper `Q8_0MatMulTiledComputeRequiredCapacitiesChecked` in `src/matmul/q8_0_matmul.HC` to return checked `{lhs_required_blocks,rhs_required_blocks,out_required_cells}` for shared preflight/math callers (WS4-03)
- [x] IQ-121 Implement HolyC helper `Q8_0MatMulTiledValidateStridesAndKChecked` in `src/matmul/q8_0_matmul.HC` to centralize `{k_block_count<=strides, out_stride>=cols}` invariants for both scalar and AVX2 tiled matmul entrypoints (WS4-03)
- [x] IQ-122 Implement HolyC helper `Q8_0MatMulTiledValidateTileShapeChecked` in `src/matmul/q8_0_matmul.HC` to centralize `{tile_rows>0, tile_cols>0}` invariants for scalar and AVX2 tiled entrypoints (WS4-03)
- [x] IQ-123 Implement HolyC helper `Q8_0MatMulTiledComputeTileEndChecked` in `src/matmul/q8_0_matmul.HC` to centralize checked `tile_end=min(tile_start+tile_span,axis_len)` math for row/col traversal loops across scalar and AVX2 tiled entrypoints (WS4-03)
- [x] IQ-124 Implement HolyC helper `Q8_0MatMulTiledComputeRowBaseChecked` in `src/matmul/q8_0_matmul.HC` to centralize checked `row_index*{lhs_row_stride_blocks,out_row_stride_cells}` base math reused by scalar and AVX2 tiled loops (WS4-03)
- [x] IQ-125 Implement HolyC helper `Q8_0MatMulTiledComputeOutIndexChecked` in `src/matmul/q8_0_matmul.HC` to centralize checked `out_row_base+col_index` index derivation and overflow handling across scalar/AVX2 tiled paths (WS4-03)

- [x] IQ-126 Implement HolyC helper `Q8_0MatMulTiledComputeRhsColBaseChecked` in `src/matmul/q8_0_matmul.HC` to centralize checked `col_index*rhs_col_stride_blocks` math across scalar/AVX2 tiled paths (WS4-03)
- [x] IQ-127 Implement HolyC helper `Q8_0MatMulTiledComputeSpanBoundsChecked` in `src/matmul/q8_0_matmul.HC` to unify tile span validation and `min(start+span,len)` semantics for row/col loops (WS4-03)
- [x] IQ-128 Add host-side parity harness `tests/test_q8_0_matmul_tiled_rhs_base_preflight.py` asserting scalar/AVX2 parity for checked RHS column-base overflow surfaces (WS4-03)
- [x] IQ-129 Add host-side parity harness `tests/test_q8_0_matmul_tiled_row_base_preflight.py` asserting scalar/AVX2 parity for checked row-base overflow surfaces (`row_index*row_stride`) across tiled Q16/Q32 entrypoints (WS4-03)
- [x] IQ-130 Implement HolyC helper `Q8_0MatMulTiledValidateRowSliceChecked` in `src/matmul/q8_0_matmul.HC` to centralize checked `{row_base + k_block_count <= lhs_block_capacity}` math and unify BAD_LEN/OVERFLOW contracts across scalar and AVX2 tiled entrypoints (WS4-03)
- [x] IQ-131 Implement HolyC helper `Q8_0MatMulTiledValidateOutIndexCapacityChecked` in `src/matmul/q8_0_matmul.HC` to centralize checked `{out_row_base + col_index < out_cell_capacity}` invariants and unify BAD_LEN/OVERFLOW behavior across scalar and AVX2 tiled writers (WS4-03)
- [x] IQ-132 Add host-side parity harness `tests/test_q8_0_matmul_tiled_row_slice_preflight.py` to assert scalar/AVX2 tiled entrypoints surface identical BAD_LEN/OVERFLOW contracts for checked row-slice bounds (`row_base + k_block_count`) (WS4-03)
- [x] IQ-133 Implement HolyC `RoPEQ16AngleStepChecked` in `src/model/rope.HC` to compute per-head rotary angle increments with checked Q16 fixed-point multiply/divide and explicit overflow domain guards before WS5 attention integration (WS5-01)
- [x] IQ-134 Implement HolyC `RoPEQ16AngleForPositionChecked` in `src/model/rope.HC` to compute checked `pos * angle_step` radians in Q16 with explicit overflow/domain guards for WS5 attention rotation kernels (WS5-01)
- [x] IQ-135 Implement HolyC `RoPEQ16RotatePairChecked` in `src/model/rope.HC` to rotate `(x_q16,y_q16)` by `angle_q16` using integer sin/cos Q16 helpers with checked mul/add overflow guards for WS5 rotary application (WS5-01)
- [x] IQ-136 Implement HolyC `RoPEQ16RotatePairByPositionChecked` in `src/model/rope.HC` that composes `RoPEQ16AngleStepChecked` + `RoPEQ16AngleForPositionChecked` + `RoPEQ16RotatePairChecked` for one-step checked rotary application at `(head_dim,pair_index,position)` (WS5-01)
- [x] IQ-137 Implement HolyC `RoPEQ16RotateHeadByPositionChecked` in `src/model/rope.HC` to apply checked rotary updates across all `(x,y)` pairs in one head slice with stride/capacity guards for WS5 attention kernels (WS5-01)
- [x] IQ-138 Implement HolyC `RoPEQ16RotateHeadRangeByPositionChecked` in `src/model/rope.HC` to apply checked rotary updates across contiguous head slices (`head_count`, `head_stride_cells`) with per-head base overflow/capacity guards for WS5 attention kernels (WS5-01)
- [ ] IQ-139 Add host-side parity harness `tests/test_rope_q16_rotate_head_range_position.py` to validate range-wise head rotation contracts (base/stride/capacity/overflow) against per-head composition references (WS5-01)
- [x] IQ-140 Implement HolyC helper `RoPEQ16ComputeHeadBaseIndexChecked` in `src/model/rope.HC` for checked `head_base + head_index*head_stride_cells` derivation reused by range rotation kernels (WS5-01)
- [x] IQ-141 Implement HolyC helper `RoPEQ16ValidateHeadRangeCapacityChecked` in `src/model/rope.HC` to centralize head-range bounds/overflow checks before multi-head in-place rotation (WS5-01)
- [x] IQ-142 Implement HolyC helper `RoPEQ16ValidateHeadSpanForDimChecked` in `src/model/rope.HC` to centralize checked `{head_base + ((head_dim/2)-1)*pair_stride + 1 < head_cell_capacity}` bounds validation for per-head rotate kernels before pair iteration (WS5-01)
- [x] IQ-143 Implement HolyC helper `RoPEQ16ValidateHeadRangeSpanForDimChecked` in `src/model/rope.HC` to preflight per-head last-pair span bounds (`head_base + ((head_dim/2)-1)*pair_stride + 1`) across `head_count` before multi-head rotation loops (WS5-01)
- [x] IQ-144 Add host-side parity harness `tests/test_rope_q16_validate_head_range_span_for_dim.py` for range-span preflight helper with overflow and capacity adversarial cases (WS5-01)
- [x] IQ-145 Implement HolyC helper `RoPEQ16RotateHeadRangeByPositionPreflightedChecked` in `src/model/rope.HC` that composes range-capacity + range-span preflight helpers then applies per-head rotation for one position (WS5-01)
- [x] IQ-146 Implement HolyC helper `RoPEQ16RotateHeadRangeByTokenWindowChecked` in `src/model/rope.HC` to apply checked head-range rotation across `token_count` positions with checked `position_start + token_index*position_step` math and fail-fast preflight reuse (WS5-01)
- [ ] IQ-147 Add host-side parity harness `tests/test_rope_q16_rotate_head_range_token_window.py` for token-window range rotation contract and overflow/capacity adversarial cases against per-position composition references (WS5-01)

## Progress Ledger

| Date | Iteration | Task | Result | Notes |
|---|---|---|---|---|
| 2026-04-17 | loop-124 | IQ-144 RoPE head-range span parity harness | done | Added `tests/test_rope_q16_validate_head_range_span_for_dim.py`; `python3 tests/test_rope_q16_validate_head_range_span_for_dim.py && python3 tests/test_rope_q16_rotate_head_range_preflighted_position.py && python3 tests/test_rope_q16_rotate_head_range_position.py && python3 tests/test_rope_q16_validate_head_span_for_dim.py && python3 tests/test_rope_q16_validate_head_range_capacity.py` passed |
| 2026-04-17 | loop-125 | IQ-146 RoPE token-window range rotate helper | done | Added `RoPEQ16RotateHeadRangeByTokenWindowChecked` in `src/model/rope.HC`; `python3 tests/test_rope_q16_rotate_head_range_preflighted_position.py && python3 tests/test_rope_q16_rotate_head_range_position.py && python3 - <<'PY' ... rope_q16_rotate_head_range_token_window_reference_checks=ok ... PY` passed |
| 2026-04-17 | loop-123 | IQ-145 RoPE preflighted head-range rotate helper | done | Added `RoPEQ16RotateHeadRangeByPositionPreflightedChecked` in `src/model/rope.HC` and `tests/test_rope_q16_rotate_head_range_preflighted_position.py`; `python3 tests/test_rope_q16_rotate_head_range_preflighted_position.py && python3 tests/test_rope_q16_rotate_head_range_position.py && python3 tests/test_rope_q16_validate_head_span_for_dim.py && python3 tests/test_rope_q16_validate_head_range_capacity.py && python3 tests/test_rope_q16_rotate_head_position.py` passed |
| 2026-04-17 | loop-122 | IQ-143 RoPE head-range span validator helper | done | Added `RoPEQ16ValidateHeadRangeSpanForDimChecked` in `src/model/rope.HC`; `python3 tests/test_rope_q16_validate_head_span_for_dim.py && python3 tests/test_rope_q16_validate_head_range_capacity.py && python3 tests/test_rope_q16_rotate_head_range_position.py` passed |
| 2026-04-17 | loop-121 | IQ-142 RoPE head-span validator helper | done | Added `RoPEQ16ValidateHeadSpanForDimChecked` in `src/model/rope.HC` and parity harness `tests/test_rope_q16_validate_head_span_for_dim.py`; `python3 tests/test_rope_q16_validate_head_span_for_dim.py && python3 tests/test_rope_q16_rotate_head_position.py && python3 tests/test_rope_q16_rotate_head_range_position.py && python3 tests/test_rope_q16_validate_head_range_capacity.py` passed |
| 2026-04-16 | loop-120 | IQ-141 RoPE head-range capacity helper | done | Added `RoPEQ16ValidateHeadRangeCapacityChecked` in `src/model/rope.HC` + parity harness `tests/test_rope_q16_validate_head_range_capacity.py`; `python3 tests/test_rope_q16_validate_head_range_capacity.py && python3 tests/test_rope_q16_rotate_head_range_position.py && python3 tests/test_rope_q16_compute_head_base_index.py` passed |
| 2026-04-16 | loop-119 | IQ-140 RoPE Q16 head-base index helper | done | Added `RoPEQ16ComputeHeadBaseIndexChecked` in `src/model/rope.HC` + parity harness `tests/test_rope_q16_compute_head_base_index.py`; `python3 tests/test_rope_q16_compute_head_base_index.py && python3 tests/test_rope_q16_rotate_head_range_position.py && python3 tests/test_rope_q16_rotate_head_position.py && python3 tests/test_rope_q16_rotate_pair_position.py && python3 tests/test_rope_q16_angle_position.py && python3 tests/test_rope_q16_angle_step.py` passed |
| 2026-04-16 | loop-118 | IQ-138 RoPE Q16 rotate-head-range helper | done | Added `RoPEQ16RotateHeadRangeByPositionChecked` in `src/model/rope.HC` + `tests/test_rope_q16_rotate_head_range_position.py`; `python3 tests/test_rope_q16_rotate_head_range_position.py && python3 tests/test_rope_q16_rotate_head_position.py && python3 tests/test_rope_q16_rotate_pair_position.py && python3 tests/test_rope_q16_angle_position.py && python3 tests/test_rope_q16_angle_step.py` passed |
| 2026-04-16 | loop-117 | IQ-134 RoPE Q16 angle-for-position helper | done | Updated `RoPEQ16AngleForPositionChecked` in `src/model/rope.HC` to checked integer `step*position` (no lossy intermediate Q16 upcast) and expanded `tests/test_rope_q16_angle_position.py`; `python3 tests/test_rope_q16_angle_position.py && python3 tests/test_rope_q16_rotate_pair_position.py && python3 tests/test_rope_q16_rotate_head_position.py && python3 tests/test_rope_q16_angle_step.py` passed |
| 2026-04-16 | loop-116 | IQ-136 RoPE Q16 rotate-by-position helper | done | Added `RoPEQ16RotatePairByPositionChecked` in `src/model/rope.HC` plus parity harness `tests/test_rope_q16_rotate_pair_position.py`; `python3 tests/test_rope_q16_rotate_pair_position.py && python3 tests/test_rope_q16_rotate_pair.py && python3 tests/test_rope_q16_angle_position.py && python3 tests/test_rope_q16_angle_step.py` passed |
| 2026-04-16 | loop-114 | IQ-032 mixed Q4_0×Q8_0 parity harness | done | Added `tests/test_q4_0_q8_0_dot.py` with GGML-bounded single/multiblock Q32 parity and checked Q16-accumulator bounds; `python3 tests/test_q4_0_q8_0_dot.py && python3 tests/test_q4_0_q8_0_dot_kernel.py` passed |
| 2026-04-16 | loop-115 | IQ-135 RoPE Q16 rotate-pair helper | done | Added integer-only sin/cos approximation + checked rotate pair helper in `src/model/rope.HC` and parity harness `tests/test_rope_q16_rotate_pair.py`; `python3 tests/test_rope_q16_rotate_pair.py && python3 tests/test_rope_q16_angle_position.py && python3 tests/test_rope_q16_angle_step.py` passed |
| 2026-04-16 | loop-113 | IQ-133 RoPE Q16 angle-step helper | done | Added `src/model/rope.HC` (`RoPEQ16AngleStepChecked` + checked Q16 mul/div/add primitives) and `tests/test_rope_q16_angle_step.py`; `python3 tests/test_rope_q16_angle_step.py && python3 tests/test_intlog_q16.py` passed |
| 2026-04-16 | loop-112 | IQ-132 row-slice preflight parity harness | done | Added `tests/test_q8_0_matmul_tiled_row_slice_preflight.py` with targeted/randomized helper parity and scalar/AVX2 tiled BAD_LEN/OVERFLOW surface checks; `python3 tests/test_q8_0_matmul_tiled_row_slice_preflight.py && python3 tests/test_q8_0_matmul_tiled_row_base_preflight.py && python3 tests/test_q8_0_matmul_tiled_rhs_base_preflight.py && python3 tests/test_q8_0_matmul_tiled_out_capacity_preflight.py` passed |
| 2026-04-16 | loop-111 | IQ-131 tiled out-index capacity validator helper | done | Added `Q8_0MatMulTiledValidateOutIndexCapacityChecked` in `src/matmul/q8_0_matmul.HC`, routed scalar+AVX2 tiled writers through it, and added `tests/test_q8_0_matmul_tiled_out_capacity_preflight.py`; `python3 tests/test_q8_0_matmul_tiled_out_capacity_preflight.py && python3 tests/test_q8_0_matmul_tiled_out_index_preflight.py && python3 tests/test_q8_0_matmul_tiled_avx2_preflight.py && python3 tests/test_q8_0_matmul_tiled_avx2_q32.py && python3 tests/test_q8_0_matmul_tiled_avx2_q16.py` passed |
| 2026-04-16 | loop-110 | IQ-130 tiled row-slice validator helper | done | Added `Q8_0MatMulTiledValidateRowSliceChecked` in `src/matmul/q8_0_matmul.HC`, routed scalar/AVX2 tiled loops through it, and mirrored row-slice contract checks in `tests/test_q8_0_matmul_tiled_checked.py`; `python3 - <<'PY' ... q8_0_matmul_row_slice_helper_checks=ok ... PY && python3 tests/test_q8_0_avx2_blocks_q32.py && python3 tests/test_q8_0_avx2_blocks_q32_to_q16.py` passed |
| 2026-04-16 | loop-109 | IQ-129 row-base preflight parity harness | done | Added `tests/test_q8_0_matmul_tiled_row_base_preflight.py`; `python3 tests/test_q8_0_matmul_tiled_row_base_preflight.py && python3 tests/test_q8_0_matmul_tiled_avx2_preflight.py && python3 tests/test_q8_0_matmul_tiled_rhs_base_preflight.py` passed |
| 2026-04-16 | loop-108 | IQ-127 tiled span-bounds helper | done | Added `Q8_0MatMulTiledComputeSpanBoundsChecked` in `src/matmul/q8_0_matmul.HC`, routed scalar+AVX2 tiled row/col spans through it, and updated AVX2 parity helper naming; `python3 tests/test_q8_0_matmul_tiled_avx2_q32.py && python3 tests/test_q8_0_matmul_tiled_avx2_q16.py && python3 tests/test_q8_0_matmul_tiled_avx2_preflight.py` passed |
| 2026-04-16 | loop-107 | IQ-126 tiled RHS column-base helper | done | Added `Q8_0MatMulTiledComputeRhsColBaseChecked` in `src/matmul/q8_0_matmul.HC` and routed scalar+AVX2 tiled kernels through it; `python3 tests/test_q8_0_matmul_tiled_avx2_preflight.py && python3 tests/test_q8_0_matmul_tiled_avx2_q32.py && python3 tests/test_q8_0_matmul_tiled_avx2_q16.py` passed |
| 2026-04-16 | loop-106 | IQ-125 tiled out-index helper | done | Added `Q8_0MatMulTiledComputeOutIndexChecked` in `src/matmul/q8_0_matmul.HC`, routed scalar+AVX2 tiled kernels through it, and added `tests/test_q8_0_matmul_tiled_out_index_preflight.py`; `python3 tests/test_q8_0_matmul_tiled_out_index_preflight.py && python3 tests/test_q8_0_matmul_tiled_avx2_q32.py && python3 tests/test_q8_0_matmul_tiled_avx2_q16.py && python3 tests/test_q8_0_matmul_tiled_avx2_preflight.py && python3 tests/test_q8_0_matmul_tiled_stride_k_preflight.py && python3 tests/test_q8_0_matmul_tiled_tile_shape_preflight.py` passed |
| 2026-04-16 | loop-105 | IQ-124 tiled row-base helper | done | Added `Q8_0MatMulTiledComputeRowBaseChecked` in `src/matmul/q8_0_matmul.HC` and routed scalar+AVX2 tiled kernels through the shared checked row-base contract; `python3 tests/test_q8_0_matmul_tiled_avx2_q32.py && python3 tests/test_q8_0_matmul_tiled_avx2_q16.py && python3 tests/test_q8_0_matmul_tiled_avx2_preflight.py && python3 tests/test_q8_0_matmul_tiled_stride_k_preflight.py && python3 tests/test_q8_0_matmul_tiled_tile_shape_preflight.py` passed |
| 2026-04-16 | loop-104 | IQ-123 tiled tile-end helper | done | Added `Q8_0MatMulTiledComputeTileEndChecked` in `src/matmul/q8_0_matmul.HC`, routed scalar+AVX2 tiled row/col traversal through the helper, and extended AVX2 parity harnesses with helper-contract checks; `python3 tests/test_q8_0_matmul_tiled_avx2_q32.py && python3 tests/test_q8_0_matmul_tiled_avx2_q16.py && python3 tests/test_q8_0_matmul_tiled_avx2_preflight.py && python3 tests/test_q8_0_matmul_tiled_stride_k_preflight.py && python3 tests/test_q8_0_matmul_tiled_tile_shape_preflight.py` passed |
| 2026-04-16 | loop-103 | IQ-122 tiled tile-shape preflight helper | done | Added `Q8_0MatMulTiledValidateTileShapeChecked` in `src/matmul/q8_0_matmul.HC`, routed shared preflight through it, and added `tests/test_q8_0_matmul_tiled_tile_shape_preflight.py`; `python3 tests/test_q8_0_matmul_tiled_tile_shape_preflight.py && python3 tests/test_q8_0_matmul_tiled_stride_k_preflight.py && python3 tests/test_q8_0_matmul_tiled_avx2_preflight.py && python3 tests/test_q8_0_matmul_tiled_avx2_q16.py && python3 tests/test_q8_0_matmul_tiled_avx2_q32.py` passed |
| 2026-04-16 | loop-102 | IQ-121 tiled stride/K preflight helper | done | Added `Q8_0MatMulTiledValidateStridesAndKChecked` in `src/matmul/q8_0_matmul.HC`, routed scalar+AVX2 preflight through it, and added `tests/test_q8_0_matmul_tiled_stride_k_preflight.py`; `python3 tests/test_q8_0_matmul_tiled_stride_k_preflight.py && python3 tests/test_q8_0_matmul_tiled_avx2_preflight.py && python3 tests/test_q8_0_matmul_tiled_avx2_q16.py && python3 tests/test_q8_0_matmul_tiled_avx2_q32.py` passed |
| 2026-04-16 | loop-101 | IQ-120 tiled capacity helper | done | Added `Q8_0MatMulTiledComputeRequiredCapacitiesChecked` and routed checked preflight/math callers through it; `python3 tests/test_q8_0_matmul_tiled_avx2_preflight.py && python3 tests/test_q8_0_matmul_tiled_avx2_q32.py && python3 tests/test_q8_0_matmul_tiled_avx2_q16.py` passed |
| 2026-04-16 | loop-100 | IQ-119 AVX2 tiled preflight parity harness | done | Added `tests/test_q8_0_matmul_tiled_avx2_preflight.py` with targeted + randomized BAD_LEN/OVERFLOW parity checks across Q32/Q16 tiled entrypoints; `python3 tests/test_q8_0_matmul_tiled_avx2_preflight.py && python3 tests/test_q8_0_matmul_tiled_avx2_q32.py && python3 tests/test_q8_0_matmul_tiled_avx2_q16.py` passed |
| 2026-04-16 | loop-098 | IQ-113 AVX2 tiled Q16 matmul preflight hardening | done | Shared Q32/Q16 tiled argument validator in `src/matmul/q8_0_matmul.HC` + parity-surface check in `tests/test_q8_0_matmul_tiled_avx2_q16.py`; `python3 tests/test_q8_0_matmul_tiled_avx2_q16.py && python3 tests/test_q8_0_matmul_tiled_avx2_q32.py && python3 tests/test_q8_0_avx2_rows_q32_to_q16.py` passed |
| 2026-04-16 | loop-097 | IQ-117 mixed AVX2 tiled Q32 matmul parity harness | done | Added `tests/test_q4_0_q8_0_matmul_tiled_avx2_q32.py` covering known-vector parity, randomized tile/stride shapes, BAD_LEN/NULL checks, and overflow propagation from the mixed AVX2 dot kernel; `python3 tests/test_q4_0_q8_0_matmul_tiled_avx2_q32.py && python3 tests/test_q4_0_q8_0_avx2_blocks_q32.py && python3 tests/test_q4_0_q8_0_matmul_tiled_checked.py` passed |
| 2026-04-16 | loop-096 | IQ-116 mixed AVX2 tiled Q32 matmul kernel | done | Added `Q4_0Q8_0MatMulQ32TiledAVX2Checked` in `src/matmul/q4_0_q8_0_matmul.HC` with checked tile/stride/capacity overflow guards over mixed AVX2 block dots; `python3 tests/test_q4_0_q8_0_dot_kernel.py && python3 tests/test_q4_0_q8_0_matmul_tiled_checked.py && python3 - <<'PY' ... PY` (`q4_0_q8_0_matmul_tiled_avx2_q32_reference_checks=ok`) passed |
| 2026-04-16 | loop-095 | IQ-115 mixed AVX2 Q32 dot kernel | done | Added `Q4_0Q8_0DotBlocksAVX2Q32Checked` plus mixed AVX2 lane-pack/pairwise/hsum helpers in `src/quant/q4_0_q8_0_dot.HC` and parity harness `tests/test_q4_0_q8_0_avx2_blocks_q32.py`; `python3 tests/test_q4_0_q8_0_avx2_blocks_q32.py && python3 tests/test_q4_0_q8_0_dot_kernel.py && python3 tests/test_q8_0_avx2_blocks_q32.py` passed |
| 2026-04-16 | loop-094 | IQ-113 AVX2 tiled Q16 matmul kernel | done | Verified `Q8_0MatMulQ16TiledAVX2Checked` in `src/matmul/q8_0_matmul.HC` matches single-rounding Q32->Q16 contract with focused parity suite; `python3 tests/test_q8_0_matmul_tiled_avx2_q16.py && python3 tests/test_q8_0_matmul_tiled_avx2_q32.py && python3 tests/test_q8_0_avx2_rows_q32_to_q16.py` passed |
| 2026-04-16 | loop-093 | IQ-114 AVX2 tiled Q16 matmul parity harness | done | Added `tests/test_q8_0_matmul_tiled_avx2_q16.py` covering known vectors, randomized parity, tile/stride invariants, and overflow/null error paths; `python3 tests/test_q8_0_matmul_tiled_avx2_q16.py && python3 tests/test_q8_0_matmul_tiled_avx2_q32.py && python3 tests/test_q8_0_avx2_blocks_q32_to_q16.py` passed |
| 2026-04-16 | loop-092 | IQ-112 AVX2 row-wise Q32->Q16 kernel | done | Added `Q8_0DotRowsAVX2Q32ToQ16Checked` in `src/quant/q8_0_avx2.HC` with single-rounding row semantics + parity harness `tests/test_q8_0_avx2_rows_q32_to_q16.py`; `python3 tests/test_q8_0_avx2_rows_q32_to_q16.py && python3 tests/test_q8_0_avx2_rows_q32.py && python3 tests/test_q8_0_avx2_blocks_q32_to_q16.py && python3 tests/test_q8_0_avx2_blocks_q32.py` passed |
| 2026-04-16 | loop-091 | IQ-107 AVX2 multi-block Q32 parity harness | done | Hardened `tests/test_q8_0_avx2_blocks_q32.py` with fp16->Q16 spot checks, zero-block semantics, sign/lane invariants, and overflow probes; `python3 tests/test_q8_0_avx2_blocks_q32.py && python3 tests/test_q8_0_avx2_dot_lanes.py && python3 tests/test_q8_0_avx2_blocks_q32_to_q16.py` passed |
| 2026-04-16 | loop-090 | IQ-110 AVX2 tiled Q32 matmul parity harness | done | Added `tests/test_q8_0_matmul_tiled_avx2_q32.py` plus `Q8_0MatMulTiledAVX2Q32Checked` in `src/quant/q8_0_avx2.HC`; `python3 tests/test_q8_0_matmul_tiled_avx2_q32.py && python3 tests/test_q8_0_avx2_blocks_q32.py && python3 tests/test_q8_0_avx2_blocks_q32_to_q16.py` passed |
| 2026-04-16 | loop-089 | IQ-109 AVX2 tiled Q32 matmul kernel | done | Added `Q8_0MatMulQ32TiledAVX2Checked` in `src/matmul/q8_0_matmul.HC` with checked tile/stride/capacity guards and expanded parity coverage in `tests/test_q8_0_matmul_tiled_checked.py`; `python3 tests/test_q8_0_avx2_blocks_q32.py && python3 tests/test_q8_0_matmul_tiled_checked.py` passed |
| 2026-04-16 | loop-088 | IQ-111 AVX2 Q32->Q16 handoff helper | done | Added `Q8_0DotBlocksAVX2Q32ToQ16Checked` (+ signed round-shift helper) in `src/quant/q8_0_avx2.HC` and parity harness `tests/test_q8_0_avx2_blocks_q32_to_q16.py`; `python3 tests/test_q8_0_avx2_blocks_q32.py && python3 tests/test_q8_0_avx2_rows_q32.py && python3 tests/test_q8_0_avx2_blocks_q32_to_q16.py` passed |
| 2026-04-16 | loop-087 | IQ-108 AVX2 row-wise Q32 dot kernel | done | Added `Q8_0DotRowsAVX2Q32Checked` in `src/quant/q8_0_avx2.HC` with stride/capacity and overflow guards; added parity harness `tests/test_q8_0_avx2_rows_q32.py`; `python3 tests/test_q8_0_avx2_rows_q32.py && python3 tests/test_q8_0_avx2_blocks_q32.py && python3 tests/test_q8_0_avx2_dot_lanes.py && python3 tests/test_q8_0_dot.py` passed |
| 2026-04-16 | loop-086 | IQ-106 AVX2 multi-block Q32 dot helper | done | Added `Q8_0DotBlocksAVX2Q32Checked` in `src/quant/q8_0_avx2.HC` with checked scale+block accumulation and added parity harness `tests/test_q8_0_avx2_blocks_q32.py`; `python3 tests/test_q8_0_avx2_blocks_q32.py && python3 tests/test_q8_0_avx2_dot_lanes.py && python3 tests/test_q8_0_dot.py` passed |
| 2026-04-16 | loop-085 | IQ-105 AVX2 lane-dot adversarial parity harness | done | Expanded `tests/test_q8_0_avx2_dot_lanes.py` with pair-grouping composition checks (`mul_pairs`→`hsum` parity), lane-order mismatch sensitivity, and explicit reduction overflow probes; `python3 tests/test_q8_0_avx2_dot_lanes.py && python3 tests/test_q8_0_avx2_mul_pairs.py && python3 tests/test_q8_0_avx2_hsum_pairs.py` passed |
| 2026-04-16 | loop-084 | IQ-104 AVX2 lane-dot helper | done | Added `Q8_0DotI16LanesAVX2` in `src/quant/q8_0_avx2.HC` with composed pairwise multiply + checked horizontal reduction; added parity harness `tests/test_q8_0_avx2_dot_lanes.py`; `python3 tests/test_q8_0_avx2_dot_lanes.py && python3 tests/test_q8_0_avx2_mul_pairs.py && python3 tests/test_q8_0_avx2_hsum_pairs.py && python3 tests/test_q8_0_dot.py` passed |
| 2026-04-16 | loop-083 | IQ-103 AVX2 horizontal-sum helper | done | Added `Q8_0HSumI32PairsAVX2` in `src/quant/q8_0_avx2.HC` with signed I64 overflow checks and parity harness `tests/test_q8_0_avx2_hsum_pairs.py`; `python3 tests/test_q8_0_avx2_mul_pairs.py && python3 tests/test_q8_0_avx2_hsum_pairs.py` passed |
| 2026-04-16 | loop-082 | IQ-102 AVX2 pairwise multiply helper | done | Added `Q8_0MulI16LanesToI32PairsAVX2` in `src/quant/q8_0_avx2.HC` + parity harness `tests/test_q8_0_avx2_mul_pairs.py`; `python3 tests/test_q8_0_avx2_pack.py && python3 tests/test_q8_0_avx2_mul_pairs.py && python3 tests/test_q8_0_dot.py && python3 tests/test_q8_0_matmul_tiled_checked.py` passed |
| 2026-04-16 | loop-081 | IQ-101 Q8_0 AVX2 lane-pack helper | done | Added `src/quant/q8_0_avx2.HC` with `Q8_0Pack32ToI16LanesAVX2` + `Q8_0PackBlockToI16LanesAVX2`; added parity harness `tests/test_q8_0_avx2_pack.py`; `python3 tests/test_q8_0_avx2_pack.py && python3 tests/test_q8_0_dot.py && python3 tests/test_q8_0_matmul_tiled_checked.py` passed |
| 2026-04-16 | loop-080 | IQ-100 checked mixed tiled matmul kernel | done | Added `Q4_0Q8_0MatMulQ16TiledChecked` in `src/matmul/q4_0_q8_0_matmul.HC` + parity harness `tests/test_q4_0_q8_0_matmul_tiled_checked.py`; `python3 tests/test_q4_0_q8_0_matmul_tiled_checked.py && python3 tests/test_q4_0_q8_0_dot_kernel.py` passed |
| 2026-04-16 | loop-079 | IQ-099 checked Q8_0 tiled matmul kernel | done | Added `Q8_0MatMulQ16TiledChecked` in `src/matmul/q8_0_matmul.HC` + parity harness `tests/test_q8_0_matmul_tiled_checked.py`; `python3 tests/test_q8_0_matmul_tiled_checked.py && python3 tests/test_q8_0_dot.py && python3 tests/test_q8_0_dot_matrix_vector_checked.py` passed |
| 2026-04-16 | loop-078 | IQ-098 checked Q8_0 matrix×vector parity harness | done | Added `tests/test_q8_0_dot_matrix_vector_checked.py`; `python3 tests/test_q8_0_dot_matrix_vector_checked.py && python3 tests/test_q8_0_dot.py` passed |
| 2026-04-16 | loop-076 | IQ-096 checked Q8_0 row-dot alias helper | done | Added `Q8_0DotRowBlocksQ16Checked` in `src/quant/q8_0_dot.HC` and checked-row parity/overflow coverage in `tests/test_q8_0_dot.py`; `python3 tests/test_q8_0_dot.py` passed |
| 2026-04-16 | loop-075 | IQ-095 Q8_0 checked matrix×vector kernel | done | Added `Q8_0DotProductBlocksQ16AccumulateChecked` + `Q8_0DotRowsQ16MatrixVectorChecked` in `src/quant/q8_0_dot.HC` with explicit capacity/overflow guards and adversarial parity coverage in `tests/test_q8_0_dot.py`; `python3 tests/test_q8_0_dot.py` passed |
| 2026-04-16 | loop-074 | IQ-094 mixed Q4_0xQ8_0 checked matrix×vector kernel | done | Added `Q4_0Q8_0DotRowsQ16MatrixVectorChecked` + checked Q16 accumulator helper in `src/quant/q4_0_q8_0_dot.HC` with extent/overflow guards and adversarial parity coverage in `tests/test_q4_0_q8_0_dot_kernel.py`; `python3 tests/test_q4_0_q8_0_dot_kernel.py` passed |
| 2026-04-16 | loop-073 | IQ-093 mixed Q4_0xQ8_0 matrix×vector helper | done | Added `Q4_0Q8_0DotRowsQ16MatrixVector` in `src/quant/q4_0_q8_0_dot.HC` with stride validation and expanded `tests/test_q4_0_q8_0_dot_kernel.py`; `python3 tests/test_q4_0_q8_0_dot_kernel.py && python3 tests/test_q4_0_dot.py && python3 tests/test_q8_0_dot_matrix_vector.py` passed |
| 2026-04-16 | loop-072 | IQ-091 fixedpoint Q16 checked arithmetic | done | Implemented checked/saturating `FPQ16Mul`/`FPQ16Div`/`FPQ16MulSat` plus rounding helpers in `src/math/fixedpoint.HC` and added `tests/test_fixedpoint_q16.py`; `python3 tests/test_fixedpoint_q16.py && python3 tests/test_intlog_q16.py && python3 tests/test_intlog_ratio_q16.py && python3 tests/test_softmax_entropy_q16.py` passed |
| 2026-04-16 | loop-071 | IQ-089 Q8_0 matrix×vector parity harness | done | Added `tests/test_q8_0_dot_matrix_vector.py` with stride/sign/per-row semantic coverage for `Q8_0DotRowsQ16MatrixVector`; `python3 tests/test_q8_0_dot_matrix_vector.py && python3 tests/test_q8_0_dot.py && python3 tests/test_q8_0_dequant.py` passed |
| 2026-04-15 | loop-070 | IQ-090 checked Q8_0 Q32 accumulator | done | Added `Q8_0DotProductBlocksQ32Checked` + overflow guard helper in `src/quant/q8_0_dot.HC` and adversarial overflow parity coverage in `tests/test_q8_0_dot.py`; `python3 tests/test_q8_0_dot.py && python3 tests/test_q8_0_dequant.py` passed |
| 2026-04-15 | loop-069 | IQ-088 Q8_0 matrix×vector row kernel | done | Added `Q8_0DotRowsQ16MatrixVector` in `src/quant/q8_0_dot.HC` with explicit `row_stride_blocks` validation and row-wise Q16 outputs; `python3 tests/test_q8_0_dot.py` passed |
| 2026-04-15 | loop-066 | IQ-085 mixed Q4_0xQ8_0 Q32->Q16 matmul helper | done | Added canonical `Q4_0Q8_0DotProductBlocksQ32` + `Q4_0Q8_0DotProductBlocksQ32ToQ16` in `src/quant/q4_0_q8_0_dot.HC` (I64 checked accumulation) and parity coverage in `tests/test_q4_0_q8_0_dot_kernel.py`; `python3 tests/test_q4_0_q8_0_dot_kernel.py && python3 tests/test_q4_0_dot.py && python3 tests/test_q8_0_dot.py` passed |
| 2026-04-15 | loop-067 | IQ-086 mixed Q4_0xQ8_0 row-helper integration | done | Added `Q4_0Q8_0DotRowBlocksQ16`/`Q4_0_Q8_0DotRowBlocksQ16` in `src/quant/q4_0_q8_0_dot.HC` and new row semantic parity checks in `tests/test_q4_0_q8_0_dot_kernel.py`; `python3 tests/test_q4_0_q8_0_dot_kernel.py && python3 tests/test_q4_0_dot.py && python3 tests/test_q8_0_dot.py` passed |
| 2026-04-15 | loop-065 | IQ-084 Q4_0 Q32->Q16 matmul helper | done | Added `Q4_0DotProductBlocksQ32ToQ16` in `src/quant/q4_0_dot.HC` + parity cases in `tests/test_q4_0_dot.py`; `python3 tests/test_q4_0_dot.py && python3 tests/test_q4_0_q8_0_dot_kernel.py && python3 tests/test_q8_0_dot.py` passed |
| 2026-04-15 | loop-064 | IQ-081 intlog ratio/log2 parity harness | done | Added `tests/test_intlog_ratio_q16.py` covering FPQ16Log2/FPQ16LnRatio domain floors, powers-of-two identities, reciprocal symmetry, ln/log2-delta consistency, and random float parity bounds; `python3 tests/test_intlog_ratio_q16.py && python3 tests/test_intlog_q16.py` passed |
| 2026-04-15 | loop-063 | IQ-082 softmax entropy parity harness | done | Added `tests/test_softmax_entropy_q16.py` with Q16 entropy clamp/known-vector/random-simplex parity checks against float references; `python3 tests/test_softmax_entropy_q16.py` passed |
| 2026-04-15 | loop-062 | IQ-079 FPQ16LnRatio helper | done | Added `FPQ16LnRatio` in `src/math/intlog.HC` with positive-domain checks + shared range-reduction polynomial path and parity coverage in `tests/test_intlog_q16.py`; `python3 tests/test_intlog_q16.py` passed |
| 2026-04-15 | loop-061 | IQ-077 intlog Q16 core | done | Added `src/math/intlog.HC` (`FPQ16LnReduce`, `FPQ16Ln1pPoly`, `FPQ16Ln`) with integer-only domain clamps/range reduction/piecewise polynomial and added `tests/test_intlog_q16.py`; `python3 tests/test_intlog_q16.py` passed |
| 2026-04-15 | loop-060 | IQ-076 abs-span payload-window loader | done | Added `GGUFTensorInfoLoadPayloadWindowByAbsSpan` in `src/gguf/tensor_data_base.HC` with parity/adversarial coverage in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-15 | loop-058 | IQ-074 index+rel payload-window loader (custom alignment) | done | Added `GGUFTensorInfoLoadPayloadWindowByIndexRel` + parity/adversarial coverage in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-15 | loop-059 | IQ-075 rel-span payload-window loader | done | Added `GGUFTensorInfoLoadPayloadWindowByRelSpan` + parity/adversarial coverage in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-15 | loop-057 | IQ-073 index+abs payload-window loader | done | Added `GGUFTensorInfoLoadPayloadWindowByIndexAbs` in `src/gguf/tensor_data_base.HC`, tightened overflow parity in `tests/test_gguf_tensor_data_base.py`, and added abs-window parity/adversarial coverage; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-15 | loop-056 | IQ-072 index+rel payload-window loader | done | Added `GGUFTensorInfoLoadPayloadWindowByIndexRelDefault` in `src/gguf/tensor_data_base.HC` with parity/adversarial coverage in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-15 | loop-055 | IQ-071 index+rel-range default span wrapper | done | Added `GGUFTensorLookupSpanByIndexAndRelRangeDefault` in `src/gguf/tensor_data_base.HC` with parity coverage in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py` passed |
| 2026-04-15 | loop-054 | IQ-070 index+rel-range span lookup helper | done | Added `GGUFTensorLookupSpanByIndexAndRelRange` in `src/gguf/tensor_data_base.HC` + parity/adversarial coverage in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-15 | loop-053 | IQ-069 index+abs-range span lookup helper | done | Added `GGUFTensorLookupSpanByIndexAndAbsRange` in `src/gguf/tensor_data_base.HC` + parity/adversarial coverage in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-15 | loop-052 | IQ-068 rel-range span lookup helper | done | Added `GGUFTensorLookupSpanByRelRange` + parity coverage; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-15 | loop-046 | IQ-067 abs-range span lookup helper | done | Added `GGUFTensorLookupSpanByAbsRange` in `src/gguf/tensor_data_base.HC` + parity coverage in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
|---|---|---|---|---|
| 2026-04-16 | loop-117 | IQ-134 RoPE Q16 angle-for-position helper | done | Updated `RoPEQ16AngleForPositionChecked` in `src/model/rope.HC` to checked integer `step*position` (no lossy intermediate Q16 upcast) and expanded `tests/test_rope_q16_angle_position.py`; `python3 tests/test_rope_q16_angle_position.py && python3 tests/test_rope_q16_rotate_pair_position.py && python3 tests/test_rope_q16_rotate_head_position.py && python3 tests/test_rope_q16_angle_step.py` passed |
| 2026-04-15 | loop-051 | IQ-066 build-lookup-table adversarial harness | done | Added staged malformed-permutation + randomized overlap-matrix adversarial coverage for `tensor_info_build_lookup_tables` in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py` passed |
| 2026-04-15 | loop-050 | IQ-065 lookup-table setup helper | done | Added `GGUFTensorInfoBuildLookupTables` in `src/gguf/tensor_data_base.HC` with composed checked build+validate path; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-15 | loop-049 | IQ-064 index+rel-offset lookup helper | done | Added `GGUFTensorLookupByIndexWithRelOffset` in `src/gguf/tensor_data_base.HC` + parity coverage in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-15 | loop-048 | IQ-063 index+inner-offset lookup helper | done | Added `GGUFTensorLookupByIndexWithInnerOffset` in `src/gguf/tensor_data_base.HC` + parity/adversarial coverage in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-15 | loop-047 | IQ-061 rel-offset default inner-offset wrapper | done | Added `GGUFTensorLookupByRelOffsetWithInnerOffsetDefault` in `src/gguf/tensor_data_base.HC` + default-alignment parity coverage in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-15 | loop-046 | IQ-062 lookup-table consistency validator | done | Added `GGUFValidateTensorLookupTables` in `src/gguf/tensor_data_base.HC` + parity/adversarial coverage in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-15 | loop-045 | IQ-060 rel-offset lookup with inner offset | done | Added `GGUFTensorLookupByRelOffsetWithInnerOffset` + parity coverage in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-15 | loop-044 | IQ-059 abs-offset lookup with inner offset | done | Added `GGUFTensorLookupByAbsOffsetWithInnerOffset` + parity coverage in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-15 | loop-043 | IQ-058 abs-offset default lookup wrapper | done | Added `GGUFTensorLookupByAbsOffsetDefault` + parity coverage in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-15 | loop-042 | IQ-057 abs-offset tensor lookup helper | done | Added `GGUFTensorLookupByAbsOffset` + parity coverage in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-15 | loop-041 | IQ-056 rel-offset tensor lookup helper | done | Added `GGUFTensorLookupByRelOffset` (+ default) in `src/gguf/tensor_data_base.HC` and parity coverage in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-15 | loop-040 | IQ-055 tensor index direct range helper | done | Added `GGUFTensorRangeFindByIndex` + parity coverage in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py` passed |
| 2026-04-12 | loop-039 | IQ-054 inverse-map/range-index parity harnesses | done | Added randomized permutation + adversarial parity coverage in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py` passed |
| 2026-04-12 | loop-038 | IQ-053 tensor index→abs range lookup | done | Added `GGUFTensorIndexToAbsRange` + focused parity checks; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-12 | loop-037 | IQ-052 sorted position-map validator | done | Added `GGUFValidateSortedPositionMap` in `src/gguf/tensor_data_base.HC` + adversarial parity coverage in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-12 | loop-036 | IQ-050 sorted index integrity validator | done | Added `GGUFValidateSortedTensorIndex` in `src/gguf/tensor_data_base.HC` plus adversarial integrity tests in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py` passed |
| 2026-04-12 | loop-035 | IQ-049 default-alignment indexed lookup | done | Added `GGUFTensorRangeFindIndexByRelOffsetDefault` + default-alignment parity tests; `python3 tests/test_gguf_tensor_data_base.py` passed |
| 2026-04-12 | loop-034 | IQ-048 indexed tensor lookup helpers | done | Added indexed abs/rel lookup in `src/gguf/tensor_data_base.HC` + parity coverage in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-12 | loop-032 | IQ-046 tensor range default relative-offset lookup | done | Added `GGUFTensorRangeFindByRelOffsetDefault`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-12 | loop-033 | IQ-047 tensor offset index builder | done | Added `GGUFTensorInfoBuildOffsetIndex` + offset-index parity tests; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
| 2026-04-12 | loop-031 | IQ-045 tensor range relative-offset lookup | done | Added `GGUFTensorRangeFindByRelOffset` + parity cases; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |
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
| 2026-04-12 | loop-030 | IQ-044 tensor range binary lookup | done | Added `GGUFTensorRangeFindByAbsOffset` in `src/gguf/tensor_data_base.HC` + binary-search parity tests; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |

| 2026-04-12 | loop-031 | IQ-051 sorted-index inverse map builder | done | Added `GGUFBuildTensorSortedPositionMap` in `src/gguf/tensor_data_base.HC` + parity tests in `tests/test_gguf_tensor_data_base.py`; `python3 tests/test_gguf_tensor_data_base.py && python3 tests/test_gguf_tensorinfo_parse.py && python3 tests/test_gguf_metadata_parse.py` passed |

| 2026-04-15 | loop-032 | IQ-078 FPQ16Log2 helper | done | Added `FPQ16Log2` in `src/math/intlog.HC` with overflow-safe Q16 scaling via `1/ln(2)`; expanded `tests/test_intlog_q16.py`; `python3 tests/test_intlog_q16.py` passed |
| 2026-04-15 | loop-033 | IQ-080 FPQ16 entropy helper | done | Added `FPQ16EntropyFromProbs` in `src/math/softmax.HC`; validated with `python3 - <<'PY' ... PY` (`softmax_entropy_q16_reference_checks=ok`) |

| 2026-04-16 | loop-100 | IQ-128 RHS column-base preflight parity | done | Added `tests/test_q8_0_matmul_tiled_rhs_base_preflight.py` + scalar helper reuse in `tests/test_q8_0_matmul_tiled_checked.py`; `python3 tests/test_q8_0_matmul_tiled_rhs_base_preflight.py && python3 tests/test_q8_0_matmul_tiled_out_index_preflight.py && python3 tests/test_q8_0_matmul_tiled_avx2_preflight.py` passed |
## Blockers & Decisions

- HolyC float support: deferred. Use integer-only quantized inference (Q4_0, Q8_0).
- Model architecture: LLaMA family only. Single forward pass implementation.
- Target model: TinyLlama 1.1B Q4_0 (~600MB). Proves correctness before scaling up.
- All .HC files are HolyC source intended for TempleOS compilation. Host-side test
  harnesses may use Python/C to validate outputs against llama.cpp reference.
- Air-gap mandate: networking stack work is out-of-scope for TempleOS guest; any VM run
  command must explicitly disable NICs (`-nic none`, legacy fallback `-net none`).

| 2026-04-15 | loop-035 | IQ-083 Q8_0 Q32->Q16 matmul helper | done | Added `Q8_0DotProductBlocksQ32ToQ16` in `src/quant/q8_0_dot.HC` + parity cases in `tests/test_q8_0_dot.py`; `python3 tests/test_q8_0_dot.py && python3 tests/test_q8_0_dequant.py` passed |
| 2026-04-15 | loop-068 | IQ-087 Q8_0 row-dot helper | done | Added `Q8_0DotRowBlocksQ16` in `src/quant/q8_0_dot.HC` + row-rounding parity tests in `tests/test_q8_0_dot.py`; `python3 tests/test_q8_0_dot.py` passed |

| 2026-04-16 | loop-069 | IQ-092 Q4_0 row-matrix×vector helper | done | Added Q4_0DotRowsQ16MatrixVector in src/quant/q4_0_dot.HC + stride/length parity coverage in tests/test_q4_0_dot.py; python3 tests/test_q4_0_dot.py && python3 tests/test_q4_0_dequant.py && python3 tests/test_q4_0_q8_0_dot_kernel.py passed |
| 2026-04-16 | loop-077 | IQ-097 Q8_0 checked matrix wrapper compat | done | Added `Q8_0_DotRowsQ16MatrixVectorChecked` alias in `src/quant/q8_0_dot.HC` with parity test in `tests/test_q8_0_dot.py`; `python3 tests/test_q8_0_dot.py && python3 tests/test_q8_0_dot_matrix_vector.py` passed |
| 2026-04-16 | loop-099 | IQ-118 AVX2 tiled preflight unification | done | Unified AVX2 tiled Q32/Q16 preflight through `Q8_0MatMulTiledValidateArgsChecked` in `src/matmul/q8_0_matmul.HC`; `python3 tests/test_q8_0_matmul_tiled_avx2_q32.py && python3 tests/test_q8_0_matmul_tiled_avx2_q16.py` passed |
| 2026-04-16 | loop-117 | IQ-137 RoPE head-slice rotate kernel | done | Added `RoPEQ16RotateHeadByPositionChecked` (+ checked `RoPETryMulI64Checked`) in `src/model/rope.HC` and parity harness `tests/test_rope_q16_rotate_head_position.py`; `python3 tests/test_rope_q16_angle_step.py && python3 tests/test_rope_q16_angle_position.py && python3 tests/test_rope_q16_rotate_pair.py && python3 tests/test_rope_q16_rotate_pair_position.py && python3 tests/test_rope_q16_rotate_head_position.py` passed |
