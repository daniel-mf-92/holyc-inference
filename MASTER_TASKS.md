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
- [ ] IQ-009 Implement RMSNorm (integer path) in `src/math/rmsnorm.HC` (WS1-04)
- [ ] IQ-010 Implement GGUF magic/version/header parser in `src/gguf/header.HC` (WS2-01)
- [ ] IQ-011 Implement GGUF metadata KV reader in `src/gguf/metadata.HC` (WS2-02)
- [ ] IQ-012 Implement GGUF tensor info reader in `src/gguf/tensorinfo.HC` (WS2-03)
- [ ] IQ-013 Implement Q4_0 block struct and dequantize in `src/quant/q4_0.HC` (WS3-01, WS3-02)
- [ ] IQ-014 Implement Q4_0 dot product (naive) in `src/quant/q4_0_dot.HC` (WS3-03)
- [ ] IQ-015 Implement Q8_0 block struct and dequantize in `src/quant/q8_0.HC` (WS3-04)
- [ ] IQ-016 Draft `docs/GGUF_FORMAT.md` skeleton with sections for header, metadata, tensor info, and alignment rules (WS0-02)
- [ ] IQ-017 Write host-side GGUF header/tensor dump fixture plan in `tests/README.md` for llama.cpp parity checks (WS0-05, WS2-05)
- [ ] IQ-018 Create `tests/README.md` skeleton for llama.cpp parity fixtures (header, metadata, tensor offsets) (WS0-05, WS2-05)
- [ ] IQ-019 Create `src/math/fixedpoint.HC` skeleton with Q16 constants, core type aliases, and TODO stubs for mul/div helpers (WS1-01)
- [ ] IQ-020 Implement Q16 integer exponent range clamp and base constants in `src/math/intexp.HC` (WS1-02)
- [ ] IQ-021 Add host-side Q16 exp parity harness in `tests/test_intexp_q16.py` against `math.exp` samples (WS1-05)
- [ ] IQ-022 Implement GGUF header constants and struct layout notes in `src/gguf/header.HC` before parser logic (WS2-01)
- [ ] IQ-023 Create `src/math/rmsnorm.HC` skeleton with Q16 constants, tensor shape assumptions, and TODO stubs for scale/variance accumulation (WS1-04)

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

## Blockers & Decisions

- HolyC float support: deferred. Use integer-only quantized inference (Q4_0, Q8_0).
- Model architecture: LLaMA family only. Single forward pass implementation.
- Target model: TinyLlama 1.1B Q4_0 (~600MB). Proves correctness before scaling up.
- All .HC files are HolyC source intended for TempleOS compilation. Host-side test
  harnesses may use Python/C to validate outputs against llama.cpp reference.
- Air-gap mandate: networking stack work is out-of-scope for TempleOS guest; any VM run
  command must explicitly disable NICs (`-nic none`, legacy fallback `-net none`).
