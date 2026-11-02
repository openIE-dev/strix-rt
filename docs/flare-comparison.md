# NVIDIA Flare vs strix-rt (Nov 2, 2026)

**Source caveat:** this is based on a video summary of NVIDIA Flare, not its code or paper. Its speed claims vs llama.cpp are **unverified**; nothing here was measured on our box.

## What Flare does (as described)
- **Ignition (Python, build step):** records CUDA graphs for a batch-size ladder (1, 2, 4, 8, ...), so every steady-state shape is captured ahead of time.
- **Helios (Rust, runtime):** replays the recorded graphs. There is no allocation and no compilation in steady state.
- **KV admission:** KV-cache pages are reserved *before* a request is admitted, so a running batch can't OOM partway through.
- **Prefill/decode interleave:** at most one bounded prefill chunk runs between decode batches, which bounds decode latency jitter.
- **Packaging:** one content-hashed package holds the weights, graphs and kernels, so a deployment is identified by a single hash.
- **Early bug (instructive):** one recorded graph needed about 56 KiB of shared memory, roughly 1/6 over the limit configured at rebuild. It only crashed at prepare time. Recorded artifacts encode device assumptions that must be checked against the real device.

## Mapping to strix-rt
| Flare | strix-rt today |
|---|---|
| CUDA graph capture per batch size | Vulkan command buffers recorded once and resubmitted (`vkrt` "whole-token graph"; Beam/decision kernels prerecorded). Pipelines are specialized with spec constants at load. |
| Rust replay, no alloc/compile | Buffers and descriptor sets are created at load; steady-state decode resubmits. Recordings mostly cover batch = 1. |
| KV pages reserved at admission | KV is a fixed `max_ctx` buffer per model; there's no per-request page accounting in the `/v1/decisions` server. |
| Bounded prefill chunks | Prefill and decode are not interleaved across requests; a long prefill blocks decodes. |
| Content-hashed package | Weights, SPIR-V and configs are deployed separately to each of the 3 Frameworks; nothing hashes them as a unit. |
| Device-limit validation | Shared-memory, subgroup-size and push-constant limits are trusted at pipeline creation; a mismatch surfaces at the first dispatch. |

## Next steps for strix-rt
1. **Per-batch-size recordings for `/v1/decisions`:** prerecord command buffers for a ladder B = 1, 2, 4, 8 (pad up to the next rung) and replay the matching one, so batched decisions get the no-alloc, no-record steady state that B = 1 already has.
2. **Admission-time KV reservation + bounded prefill chunks:** reserve KV pages when a request is accepted (reject or queue if none are free), and schedule at most one fixed-size prefill chunk between decode steps, to cap decode p99.
3. **Content-hashed package for the 3 Frameworks:** a manifest hashing the weights, SPIR-V, spec-constant sets, recorded-buffer descriptors and config, with one package id that the servers report and refuse to mix.
4. **Load-time validation of recordings vs real device limits:** when a package loads, check every recorded pipeline's shared-memory, workgroup, subgroup and push-constant needs against `vkGetPhysicalDeviceProperties` limits and fail loudly with the offending kernel, rather than crashing at first dispatch (Flare's 56 KiB bug).
