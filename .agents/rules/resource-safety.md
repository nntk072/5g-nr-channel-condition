---
trigger: model_decision
description: Use before native, GPU, scheduler, production D1, or resource-sensitive work.
---

# Resource Safety

Use the existing scientific resource-budget workflow as the source of measured resource estimates. Engineering resource authorization is a separate permission receipt and never scientific evidence.

CPU-only checks may run under `cpu_test`. `native_smoke`, `scientific_native`, and `production_d1` require an exact, unexpired resource authorization bound to task card, source revision, host, canonical device (`cpu` or `cuda:<index>`), and budget digest. `scientific_native` additionally requires scientific gate admission. Production D1 remains denied while the existing D1 production gate has no trusted signature keyring; an AgentOps authorization flag cannot substitute for a gate-issued production authorization.

Do not cancel jobs, delete outputs, repeat expensive runs, allocate a GPU, invoke a scheduler, or run native Sionna from a generic doctor or unit test.
