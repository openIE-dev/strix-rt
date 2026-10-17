# The lab notebook

These files were written while the work happened, by the people and AI coding agents working on the box. We publish them
mostly unedited, because the dead ends and wrong turns are part of what we want to teach. We only replaced machine-specific paths
(`$MODELS` is where model files live, `~` is a home directory), took out a few private notes, and renamed `competitors.md` to
`comparisons.md`. Several agents often shared the machine at the same time. When you read "another worker", that is what it means,
and it is why so many entries say whether the box was quiet.

| File | What's in it |
|---|---|
| [benchmarking.md](benchmarking.md) | How measurements are serialized (GPU lock, quiet-CPU check), a GPU hang incident, the fan curve |
| [comparisons.md](comparisons.md) | Every model next to llama.cpp / vLLM / FastFlowLM: speed, concurrency, energy, parity |
| [prefill.md](prefill.md) | Batched prompt processing: coop-matrix GEMMs, grouped MoE experts, probes, failed tile/queue experiments |
| [glimmer.md](glimmer.md) | Muse Glimmer 30B port, decode energy investigation, integer-dot and fused-FFN experiments |
| [new-models.md](new-models.md) | Gemma 4, Nemotron 3 / 3.5 Super VL, Cosmos 3, Glimmer scoping; Super VL real-image parity and the stb JPEG port |
| [memory-tier.md](memory-tier.md) | Zero-copy GGUF import (userptr, dma-buf kernel module), NVMe as a memory tier |
| [npu.md](npu.md) | XDNA2 NPU from Rust via XRT, NPU+GPU split prefill, energy |
| [ternary.md](ternary.md) | 1.58-bit (ternary) GEMV/GEMM kernels and the Walsh-Hadamard transform |
| [minicpmv.md](minicpmv.md) | MiniCPM-V 4.7 text + vision path |
| [qimage.md](qimage.md) | Qwen-Image-2.1 text-to-image |
| [cosmos3.md](cosmos3.md) | Cosmos3-Nano text-to-video |
| [parakeet.md](parakeet.md) | Parakeet Redux speech-to-text |
| [decision_models.md](decision_models.md) | "System One" decision models (Kev, Laya) and energy per decision |
| [../PROGRESS.md](../PROGRESS.md) | Day-by-day progress log from the first GEMV to video generation |

Logs and session scripts named in these files are under `../logs/`. A few large logs, binary dumps, audio, video and images were
removed from the public copy to keep it small. The staging report lists them.
