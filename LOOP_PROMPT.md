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
1. Read MASTER_TASKS.md.
2. Ensure the Inference Queue is rolling and deep:
   - If fewer than 15 unchecked `IQ-` items exist, append enough new `IQ-` items to bring it to at least 15.
   - Derive new queue items from the next unchecked WS* items by breaking them into small, implementable tasks.
   - Keep queue IDs monotonic (never reuse IDs).
3. Pick exactly ONE highest-priority unchecked `IQ-` item.
4. Implement it with minimal, reviewable changes.
5. Run a focused validation command relevant to your change.
6. Update MASTER_TASKS.md:
   - mark the completed queue item
   - add one Progress Ledger row with today's date
   - add blockers/decisions if any
7. If blocked, document blocker in MASTER_TASKS and complete a smaller prerequisite task instead.

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
