# LLaMA Architecture Reference (HolyC Integer Inference)

This document describes the LLaMA-family forward pass target for the TempleOS
HolyC runtime. It is an implementation guide for direct, auditable inference
code in `src/*.HC`, using integer-only math paths.

Scope for first architecture target:
- Decoder-only transformer (no encoder stack)
- RMSNorm pre-norm blocks
- RoPE position encoding on Q/K
- Grouped Query Attention (GQA)
- SwiGLU feed-forward network
- Final RMSNorm + logits projection

## 1) Model Family Scope

Supported by first-pass implementation:
- LLaMA-family tensor layout (LLaMA, Mistral, Qwen-style decoder variants)
- Causal, autoregressive next-token inference

Out of scope for initial pass:
- Encoder-decoder architectures
- Mixture-of-Experts routing
- Cross-attention

## 2) Canonical Forward Pass

Given token IDs for context `t0..tn`:
1. Lookup token embeddings -> hidden state `x`
2. For each transformer block `l = 0..N-1`:
   - `x = x + Attention(RMSNorm(x), kv_cache_l, pos)`
   - `x = x + FFN(RMSNorm(x))`
3. `x = RMSNorm(x)`
4. `logits = x * W_out` (or tied token embedding matrix)
5. Sample next token from logits

All math is implemented with quantized integer kernels and fixed-point scaling.

## 3) Tensor Shapes (Typical LLaMA Family)

Common symbols:
- `B`: batch size (initially 1)
- `T`: sequence length (prompt + generated)
- `D`: model dimension (`n_embd`)
- `Hq`: query head count (`n_head`)
- `Hkv`: key/value head count (`n_head_kv`)
- `Dh`: head dimension (`D / Hq`)
- `F`: FFN hidden dimension (`n_ff`)

Important shape rules:
- Queries: `[B, T, Hq, Dh]`
- Keys/Values: `[B, T, Hkv, Dh]`
- GQA replication/grouping maps `Hq` query heads onto `Hkv` KV heads

## 4) RMSNorm (Pre-Norm)

Per token vector `x` with dimension `D`:
- Compute mean square of elements
- Multiply by reciprocal-root normalization factor
- Apply learned weight vector

Integer path note:
- Use fixed-point accumulator for sum of squares
- Reciprocal sqrt uses integer approximation (WS1)
- Apply scale/shift policy consistently at writeback boundaries

## 5) RoPE (Rotary Position Embedding)

RoPE is applied to Q and K channels before attention score computation.
For each head vector, channel pairs are rotated by position-dependent angles.

Implementation constraints:
- Integer approximation path for sin/cos or equivalent recurrence
- Deterministic scaling/rounding so parity checks are reproducible
- Apply RoPE only to configured rotary dimensions

## 6) Grouped Query Attention (GQA)

Attention sub-steps per block:
1. Project normalized `x` into Q, K, V
2. Apply RoPE to Q and K
3. Append K/V to layer KV cache at current time index
4. For each query head, select mapped KV head-group
5. Compute causal attention scores
6. Softmax scores (integer approximation)
7. Weighted sum over V cache
8. Output projection back to `D`

Causal masking rule:
- Position `t` can attend only to `<= t`.

## 7) SwiGLU Feed-Forward Network

FFN path in each block:
1. `u = x * W_up`
2. `g = x * W_gate`
3. `a = SiLU(g)` (integer approximation)
4. `m = a * u` (element-wise)
5. `y = m * W_down`

Residual update:
- `x = x + y`

## 8) KV Cache Layout

Per layer, persist K and V across generation steps:
- `K_cache[layer][time][Hkv][Dh]`
- `V_cache[layer][time][Hkv][Dh]`

Requirements:
- Pre-allocate for `max_context`
- O(1) append per new token
- Bounds checks on every write/read index
- Deterministic memory layout (no hidden indirection)

## 9) GGUF Metadata/Tensor Names Needed

Runtime must read these model properties from GGUF metadata (key names vary by
model family/version but semantic fields are required):
- Architecture identifier
- Layer count
- Embedding size
- FFN size
- Attention head counts (`n_head`, `n_head_kv`)
- Context length
- RoPE parameters
- Vocabulary size

Core tensor groups needed for decoder inference:
- Token embedding
- Per-layer RMSNorm weights
- Per-layer attention projections (Q/K/V/O)
- Per-layer FFN projections (gate/up/down)
- Final RMSNorm
- Output projection / LM head

## 10) Determinism and Auditing Rules

- No hidden state outside explicit tensor buffers and KV cache
- Every projection and residual add is explicit in code
- Integer rounding policy is centralized and stable
- Optional Book of Truth checkpoints can log:
  - layer start/end
  - attention score stats
  - sampled token and score

## 11) First Validation Targets

Host-side validation in `tests/` should compare HolyC math decisions against
llama.cpp reference outputs for same model/prompt/seed:
- First-token logits top-k ordering
- Attention masking behavior
- KV cache growth per token
- Deterministic repeatability with fixed seed

These validation scripts are not part of TempleOS runtime.
