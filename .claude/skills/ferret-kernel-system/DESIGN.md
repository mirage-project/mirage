# Ferret invocation refactor — frozen-gate, dispatcher-constrained, in-session, Codex-reviewed

## Why (the failure this eliminates)
The attention mega-kernel came out WRONG (theta-10000 rope not YaRN, 1/sqrt(576) scale not mscale, head-sum o_proj skipping W_UV, no kv_a_layernorm) yet self-reported cosine 1.0 — because **ferret wrote its OWN test against its OWN simplified reference**. Marking its own homework. The fix is structural: **take the judging + the constraints OUT of ferret's hands.**

## /cd mechanics (claude-code-guide verified, v2.1.172+)
- `/cd` is **session-level** (moves the whole session to a sibling repo; NOT a per-subagent cwd). Subagents inherit the parent cwd AT SPAWN.
- **Nesting ≤ 5 levels**; a subagent gets the `Agent` tool unless at depth 5 / omitted from `tools`.
- Nested subagents **inherit MCP** access; add explicitly via `mcpServers: [codex]` frontmatter (string ref to a configured server). Plugin agents can't — must live in `.claude/agents/`.
- **NO per-subagent standing-goal Stop hook.** The autonomous loop must be **encoded in the agent's prompt** (or the parent re-spawns it each round).
- → DESIGN CHOICE: the ferret subagent runs IN-SESSION, drives its OWN prompt-encoded bounded loop, `cd ~/ferret/workspace<N>` via Bash (cwd persists across its Bash calls), Reads `~/ferret/CLAUDE.md` for its rules, calls `mcp__codex__codex`. No `claude -p`, no detached driver.

## The 3 levels
### L1 — DISPATCHER (main agent / `ferret-kernel-agent`) — PINS EVERY CONSTRAINT
Authors the COMPLETE requirement; THIS is where nothing-gets-missed + no-simplification is enforced. Must pin, explicitly:
- TARGET op + what it replaces; exact shapes (derive from builder/weights, don't guess).
- **The REAL math contract — NO simplification allowed** (e.g. attention: YaRN rope w/ mscale, real softmax scale, W_UV per-head BMM + o_proj_original, kv_a_layernorm — enumerate every step the kernel MUST do).
- **The production compile flags** — `-rdc=true` / `MPK_FORCE_RDC_TRUE=1`, arch sm_100a, single-stream/no-CUDA-graph/no-cta_group::2.
- The consumer ABI (`__device__ task_impl` signature, NS/NE).
- The gate spec (what correctness reference + what perf harness the test-writer must build).
Then orchestrates L2a → (Codex review) → L2b → (Codex review per round).

### L2a — TEST-WRITER (`ferret-test-writer`, NEW) — FREEZES THE GATE, runs BEFORE ferret
Writes, into the workspace, a FROZEN gate ferret may READ but NOT modify:
1. **Correctness harness** = the kernel's output vs a **REAL reference** (PyTorch DSv3 attention with the real math; OR the in-MPK faithful test). cosine vs the TRUE output — never a simplified ref.
2. **Perf harness** = built with the PRODUCTION flags (`-rdc=true` via `MPK_FORCE_RDC_TRUE=1`); the faithful in-MPK per-task measure where one exists, else a prod-flag standalone with the documented launch-overhead caveat.
3. A `gate.md` stating: the exact reference, the cos floor, the perf metric + flags, and "FERRET MUST NOT EDIT THESE FILES."
→ Codex MCP **Integrity Review** (does the gate encode every L1 constraint — real math, prod flags, shapes?) + **Plan Review** (is the reference itself correct?). Iterate the gate until Codex passes BOTH.

### L2b — FERRET (`ferret-kernel-agent` body, refactored) — OPTIMIZES vs the FROZEN gate
In-session nested subagent. Given: the frozen gate (read-only) + the requirement. Runs bounded optimization rounds (prompt-encoded loop, anti-early-stop): edit kernel → compile with the PROD flags → run the FROZEN correctness + perf gate → record. Validates ONLY against the frozen gate.
→ Codex MCP per round: **Integrity Review** (is the candidate still doing the REAL math + prod ABI, or did it drift/simplify to pass perf?) + **Plan Review** (is the next lever sound?).

## The two Codex reviews (active, every stage)
- (a) **Integrity Review** = ALIGNMENT: candidate/gate matches the L1 requirement (real math, prod flags, exact shapes, ABI). This is the anti-"simplified-attention" guard.
- (b) **Plan Review** = the approach/lever itself is sound (the optimization plan, the reference correctness).

## Files
- `.claude/agents/ferret-test-writer.md` (NEW) — L2a gate-freezer.
- `.claude/agents/ferret-kernel-agent.md` (REFACTOR) — L1 dispatcher + L2b in-session ferret loop + Codex reviews; drop the `claude -p`/`cc-run.sh` detached-episode model.

## Codex Plan-Review hardening (incorporated — the gate is now the load-bearing risk)
Codex's adversarial verdict: the refactor fixes "ferret marks its own homework", but the guarantee now rides on GATE FIDELITY. A wrong-but-consistent gate still blesses a wrong kernel. So:
1. **The reference must be CANONICAL/already-trusted, never a fresh re-derivation by the test-writer** (else the test-writer just moves the simplification one level up). For the attention: the reference IS the existing MPK attention **task-chain output** (the trusted, in-tree-correct DSv3 attention) — the gate compares the fused candidate vs the CHAIN, in-MPK. (Equivalently the official DSv3 PyTorch modeling code, if used, must be cross-checked against the chain.)
2. **Check INTERMEDIATES, not just final cosine** — golden vectors for kv_a_layernorm out, YaRN rope (positions/scales), softmax mscale, W_UV per-head un-absorption, o_proj_original — emitted by the canonical reference. A final-only cosine is too weak (a sign error deep in one head can wash out).
3. **MULTIPLE metrics** (max-abs err, relative err, per-head + per-token cosine, final cosine) and **EDGE cases** (varied positions incl. long-context YaRN-sensitive, KV-cache-state boundaries) — not one happy-path vector.
4. **Mechanical immutability**: after Codex passes the gate, the L1 dispatcher **hashes the gate files (sha256) and re-verifies the hash before EVERY ferret round.** Frozen-by-permission, not frozen-by-instruction. Any hash mismatch = abort + flag (ferret tampered or drifted the gate).
5. **FINAL ACCEPTANCE = in-MPK faithful build** (`MPK_FORCE_RDC_TRUE=1`, the candidate compiled INTO the real megakernel). A standalone `-rdc=true` compile is **diagnostic only** — it cannot reproduce the whole-megakernel register-pressure/spill. No kernel ships on a standalone number.
6. **Early stop = round INCOMPLETE, never success.** The L1 parent treats a ferret subagent that stops without a frozen-gate PASS as an unfinished round → respawn / run the next round itself. A self-reported "done" is not acceptance; only a frozen-gate PASS + Codex Integrity-clear is.

→ These make L2a's job heavier (validated canonical reference + golden intermediates + multi-metric + edge cases + hash-lock) and L1's job include hash-verification each round + in-MPK-faithful final acceptance. Implement accordingly.

## First test of the new system
Fully-fused REAL-DSv3 attention mega-kernel (the one that came out simplified). L1 pins the real-math contract + RDC=true; L2a freezes the gate vs the real DSv3 attention reference; L2b ferret optimizes vs it. Success = a fused attention that passes the FROZEN real-DSv3 correctness gate (not a self-reported cosine).
