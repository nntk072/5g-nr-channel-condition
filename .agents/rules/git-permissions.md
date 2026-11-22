---
trigger: model_decision
description: Use for commits, pushes, merges, rebases, resets, cleaning, and workflow mutation.
---

# Git permissions

Inspect branch, source revision, and dirty paths before source mutation. Preserve unrelated work. Do not reset, clean, stash, overwrite, rebase, merge, commit, push, or rerun remote workflows unless the user or active task explicitly authorizes that action.

Stage only files owned by the current task. Never use Git operations to erase evidence or bypass a failed verification gate.
