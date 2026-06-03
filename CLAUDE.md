# Claude/Clod Repo Notes

## Commit After Agent Turn

Claude/Clod agents, sub-agents, and background agents should commit after each user prompt or completed agent task that changes tracked repo files, before handing control back.

Shared workflow: `.codex/skills/commit-after-agent-turn/SKILL.md`.

Before committing, inspect status and diff, stage only intended files, run a cheap relevant validation, and scan staged content for secrets. Do not commit raw API keys, `.env` files, generated logs, caches, model weights, downloaded datasets, or unrelated user changes.

Commit messages should include a redacted prompt summary, the agent/tool name, elapsed time when available, the changed files/behavior, and validation performed. Push when the branch already has an upstream or active PR.
