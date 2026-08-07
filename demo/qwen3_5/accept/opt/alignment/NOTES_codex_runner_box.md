# Running ferret episodes on codex instead of Claude Code

Done 2026-07-31 because the Claude budget is scarce and the user has abundant codex tokens (plus
spares). The port is live and validated; the claude path still works unchanged.

## First, two corrections to things I asserted

1. **codex IS installed on the B200 box** — `/home/muhengl/.nvm/versions/node/v25.9.0/bin/codex`,
   codex-cli 0.145.0, with `~/.codex/auth.json` present and `models_cache.json` refreshed the same
   day. I twice concluded it was absent because `ssh host 'command -v codex'` came back empty: nvm
   is not on the login PATH, and codex's shebang is `#!/usr/bin/env node` so even the absolute path
   fails without node. **This is the THIRD time on this project that a remote-PATH assumption
   produced a wrong conclusion** (cf. `remote_run.sh` needing `bash -ls` for nvcc). RULE: to decide
   whether a tool exists on a remote host, SEARCH THE FILESYSTEM (`find`/known install dirs), never
   trust `command -v` over ssh.
2. **codex has a full subagent mechanism** — I said it had none. It exposes
   `collaboration.spawn_agent / wait_agent / list_agents / send_message / followup_task /
   interrupt_agent`, gated by the `multi_agent` feature flag (stable, enabled). `spawn_agent` takes
   `{task_name, message, agent_type?, fork_turns?, model?, reasoning_effort?}` — strictly richer
   than Claude Code's Task tool, since model and reasoning effort are per-spawn and an existing
   agent can be continued instead of re-spawned.

## The mapping

| Claude Code | codex |
|---|---|
| `Task(subagent_type=X, prompt=P)` | `collaboration.spawn_agent(agent_type=X, task_name=…, message=P)` + `wait_agent` |
| `.claude/agents/X.md` (YAML frontmatter + body) | `~/.codex/agents/X.toml` (`name` / `description` / `developer_instructions`) |
| frontmatter `model:` | per-spawn `model` / `reasoning_effort` |
| `--append-system-prompt` | no flag; goal goes in the prompt (AGENTS.md carries the protocol) |
| `CLAUDE.md` | `AGENTS.md` (symlinked to CLAUDE.md — one source of truth) |
| `/goal` slash command | `goals` feature exists; prompt preamble is sufficient |

Ferret's 8 agents are installed as **`ferret-*`** (`ferret-planner`, `ferret-iterator`,
`ferret-reviewer`, `ferret-kernel-extractor`, `ferret-memory-keeper`, `ferret-profiler`,
`ferret-mpk-validator`, `ferret-codex-dispatcher`) via `scratchpad/port_agents.py`. The prefix is
LOAD-BEARING: `~/.codex/agents/` already held OUR framework's general templates under the bare names
`iterator`, `planner`, `memory-keeper`, so an unprefixed install would have silently dispatched the
wrong agent. `collaboration.list_agents` confirms all 19 + built-ins `default`/`explorer`/`worker`.

## Operational details that cost time to find

- `--skip-git-repo-check` is REQUIRED outside a trusted dir, else it aborts with "Not inside a
  trusted directory".
- nvm's bin must be prepended to PATH (`FERRET_CODEX_NODE_BIN` in the runner).
- `-C <dir>` sets the working root; `-o <file>` captures the final message cleanly; `--json` emits
  JSONL events (much easier to observe than parsing a transcript); `exec resume --last` continues a
  session — a better fit for multi-round loops than re-seeding from scratch.
- `~/.codex/config.toml` already pins `model = "gpt-5.6-sol"`, `model_reasoning_effort = "max"`.

## How it is wired

`FERRET_RUNNER=codex|claude` branches ONLY the final exec inside `scripts/cc-run.sh`
(backup: `cc-run.sh.bak-preCodex`). Everything before that point — workspace init, dev-memory
seeding, `pick_gpu.sh` pinning, `FERRET_*` exports — is runner-agnostic and reused, so
`chain_longhaul.sh` and `.run_episode.sh` need no change and **the default stays claude**. A parallel
`codex-run.sh` was rejected: it would have duplicated ~100 lines of setup and drifted.
The prompt carries a translation preamble (Task→spawn_agent, the `ferret-` prefix, the
only-the-mainthread-spawns rule, and the unchanged measurement/honesty rules).

## Validated live, first episode

All four shakedown checks passed: read CLAUDE.md and restated the stage machine; ran the state CLI
(`REPRODUCE / 0.570 / worst bs1`); **spawned `ferret-iterator` and quoted its return**; reported no
tool/permission/path failures. It then began a normal iteration with pre-registered kill criteria.
It also found a real doc bug: CLAUDE.md §6.5 contradicts itself on whether the mainthread or the
reviewer triggers final extraction.

OBSERVED FRICTION TO FIX: the episode read ~1 MB of context including tangential skill docs before
starting work. Narrow the preamble to point at CLAUDE.md + task.yaml + progress.md and explicitly
tell it not to sweep the skills corpus at shakedown.
