# holyc-inference — NORTH STAR

This is the **single concrete deliverable** that defines "done" for the inference loop.

## North Star

**Run one forward pass of a small GPT-2 model in pure HolyC inside the TempleOS guest, output a token id over serial.**

Concretely, the following script must exit 0:

```bash
bash automation/north-star-e2e.sh
```

Which means:
1. A Q4_0-quantized GPT-2 124M weight blob lives on `shared.img`
2. A HolyC program loads weights, runs ONE forward pass on a fixed prompt (token id sequence `[15496, 11, 995]` = "Hello, world"), outputs the next-token id over serial
3. Output matches the reference produced by `tests/reference_q4_gpt2.py` (using llama.cpp / GGML for ground truth) — bit-exact
4. Wall time on QEMU instance < 30s
5. Memory peak < 256 MB

## Why this North Star

It forces the whole stack to work end-to-end: weight format, dequantization, matmul kernel, attention, sampling. Every IQ item must move toward this. Optimizations are evaluated against reference accuracy + wall time, not synthetic micro-benchmarks. **If an IQ item does not advance the forward-pass pipeline, it does not belong in MASTER_TASKS.md.**

## Status

RED. Will stay RED until the forward pass works bit-exactly. Every iteration runs the E2E test and reports the result.
