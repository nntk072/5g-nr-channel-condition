---
trigger: always_on
description: Use when waiting for background jobs, gates or long commands.
---

# Waiting on work

Do not block the session to wait for a background job. Do not run `sleep` in the foreground, and do not chain short sleeps to get around a block.

To wait for a job, use a background command (`run_in_background`) or a Monitor with an event filter that reports every terminal state. Then read the logs or summary file without sleeping, and report what finished.
