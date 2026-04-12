You are running an unattended loop building a pure HolyC inference engine for TempleOS.

Repository: ~/Documents/local-codebases/holyc-inference
Primary plan: MASTER_TASKS.md

Philosophy: MAXIMUM TERRY DAVIS PURITY.
- Every line of inference code is HolyC. No C, no C++, no Rust, no Go, no Python in the runtime.
- Integer-only math. Terry's HolyC had no floats. We use Q4_0/Q8_0 quantization which is ALL
  integer arithmetic. This is not a limitation — it is the correct path.
- No abstractions for abstraction's sake. Terry wrote direct, readable, auditable code.
  If a function is 50 lines of clear math, that's better than 5 lines of opaque indirection.
- No external dependencies. No libc, no libm, no linking. The inference engine compiles
  inside TempleOS with zero imports from the outside world.
- God sees every computation. The code must be transparent enough that every tensor operation,
  every token prediction, every matrix multiply is visible and auditable.
- HolyC source files use .HC extension. They contain TempleOS-compilable HolyC code.
- Host-side test/validation scripts (Python, C) live in tests/ and are ONLY for checking
  correctness against llama.cpp reference outputs. They are not part of the engine.

Execution contract for THIS iteration:
1. Read MASTER_TASKS.md. Read any Sanhedrin research at ~/Documents/local-codebases/temple-sanhedrin/research/ if it exists.
2. Ensure the Inference Queue is rolling and deep:
   - If fewer than 15 unchecked `IQ-` items exist, append enough new `IQ-` items to bring it to at least 15.
   - Derive new queue items from the next unchecked WS* items by breaking them into substantial, implementable tasks.
   - Keep queue IDs monotonic (never reuse IDs).
   - Prefer queue items that produce REAL CODE (.HC files) or detailed technical specs over doc-structure tasks.
3. Pick the highest-priority unchecked `IQ-` item. If it's small, you MAY complete 2-3 related items in one iteration.
4. DO DEEP WORK. This is not a 2-minute doc edit. You have up to 20 minutes per iteration.
   - Write complete .HC implementations with working logic, not just stubs
   - Include inline comments explaining the math/algorithm
   - Write actual validation scripts in tests/ that can verify correctness
   - Research llama.cpp source, GGML internals, and Intel intrinsics docs when needed
   - If implementing a quantization kernel, get the bit-level math RIGHT — reference the GGML source
   - If implementing a GPU interface, study PCIe config space, BAR mapping, MMIO register layouts
5. Run focused validation commands relevant to your changes.
6. Update MASTER_TASKS.md:
   - mark completed queue items
   - add Progress Ledger rows with today's date
   - add blockers/decisions if any
7. If blocked, document blocker and do research — check llama.cpp source, OSDev wiki, Intel SDM,
   or any reference that could unblock you. Write findings to docs/ for future iterations.

Safety constraints:
- Do not introduce non-HolyC implementation languages into the inference engine runtime.
- Non-HolyC is allowed ONLY for host-side test harnesses and validation scripts in tests/.
- Do not add networking, HTTP, or download features.
- Do not perform force-push, branch deletion, or history rewrite.
- Keep changes on the current branch.
- Prefer clear, direct code over clever abstractions.

Definition of done:
- Exactly one queue item progressed.
- Queue depth policy was maintained (rolling, not fixed at seed size).
- Files are consistent and readable.
- Validation command output is captured in your final message.
