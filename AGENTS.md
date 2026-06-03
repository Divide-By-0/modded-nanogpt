Headphone/Bluetooth automation note:

- The source of truth for the headphone Bluetooth release script is `/Users/aayushgupta/Documents/.projects.nosync/bt-audio-release`.
- If you change `/Users/aayushgupta/.local/bin/bt-audio-release.sh`, the LaunchAgent, or related `bt-kill-a2dp` behavior on this laptop, make the equivalent change in the repo and push it to GitHub.
- Do not leave Bluetooth headphone script fixes only in the installed local copy.

Remote host safety note:

- Do not connect to, inspect, modify, or use `pupper`, `pupper.ferret-vector.ts.net`, or `100.108.129.74` unless the user explicitly names `pupper` as the target in the current request.
- If an OpenClaw/Openlob, GCP, Tailscale, SSH, Zipair, Codex auth, or Claude auth task discovers `pupper` as a candidate, stop and ask for confirmation instead of proceeding.

Weights & Biases auth note:

- The W&B API key is stored in macOS Keychain, not in this repo.
- Keychain lookup: service `wandb-api-key`, account `$USER`.
- `~/.zshrc` exports `WANDB_API_KEY` from that Keychain item for interactive shells.
- Repo scripts should read `WANDB_API_KEY` or load the same Keychain item; do not commit the API key or put it in tracked config.
- To rotate the key: `security delete-generic-password -a "$USER" -s wandb-api-key`, then `security add-generic-password -a "$USER" -s wandb-api-key -w '<new-token>' -U`.

Commit-after-agent-turn note:

- Repo-local skill: `.codex/skills/commit-after-agent-turn/SKILL.md`.
- Cursor rule: `.cursor/rules/commit-after-agent-turn.mdc`.
- Claude/Clod note: `CLAUDE.md`.
- Codex, Cursor, Claude/Clod, sub-agents, and background agents should commit after each user prompt or agent task that changes tracked repo files, before handing control back.
- Before committing, inspect `git status`, stage only intended files, run a cheap relevant validation, and scan for secrets. Do not commit raw API keys, `.env` files, generated logs, caches, or unrelated user changes.
- Commit messages should include the prompt text or a concise redacted prompt summary, the agent/tool name, elapsed time for that prompt/task when available, changed files/behavior, and validation performed.
- Push after committing when the branch already has an upstream or an active PR. If no upstream exists, state the branch/remote choice instead of guessing.
