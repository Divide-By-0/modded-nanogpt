---
name: commit-after-agent-turn
description: Use before handing control back after any Codex, Cursor, Claude/Clod, sub-agent, or background-agent prompt/task that changes this repo, so work is committed, pushed when appropriate, and annotated with prompt and timing provenance.
---

# Commit After Agent Turn

Use this skill at the end of every user prompt, sub-agent run, background-agent run, or agent handoff that changed tracked repo files.

Do not make an empty commit when there are no repo changes.

## Workflow

1. Inspect the worktree:
   - `git status -sb --untracked-files=all`
   - `git diff --stat`
   - `git diff --check`
2. Separate intended changes from unrelated user edits. Stage explicit files only.
3. Run the cheapest relevant validation for the files changed, such as `bash -n` for shell scripts, `python3 -m py_compile` for Python, or the focused project test command.
4. Scan staged content for secrets before committing. At minimum, look for raw service-token prefixes, private keys, `.env` content, and pasted API keys. Redact any secret in terminal output or commit text.
5. Commit with a detailed message:
   - Subject: imperative summary of the change.
   - Body: include the redacted prompt text or concise prompt summary, agent/tool name, elapsed time for that prompt/task if available, files/behavior changed, and validation performed.
6. Push after the commit when the branch has an upstream or an active PR. If no upstream exists, pick the existing repo remote/branch convention when obvious; otherwise report the blocker.
7. Report the commit SHA, branch, PR URL if known, and validation result.

## Guardrails

- Never commit raw API keys, `.env` files, credentials, generated training logs, caches, model weights, downloaded datasets, or local machine state.
- Never rewrite public history, amend someone else's commit, or use destructive git commands unless explicitly requested.
- Preserve user changes you did not make. If the worktree is mixed, commit only the requested scope.
- Do not connect to remote hosts as part of this skill unless the prompt explicitly asks for remote work.
- If elapsed time is available from agent logs, include it. If not available, write `elapsed time: not recorded`.
