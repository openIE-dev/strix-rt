# Models

**No model weights are included in this repository.** Download them from the sources below. Each model's license and terms of
use are set by its upstream publisher (see the model card at the source); this repo's MIT / Apache-2.0 license does not cover them.

The scripts expect files under `$MODELS` (we used `export MODELS=~/models`) at the paths shown. Where our notes don't record the
exact repository, search Hugging Face for the file name.

| Model (as used here) | File(s) under `$MODELS` | Where to get it | strix-rt crate |
|---|---|---|---|
| Muse Glimmer 30B Q4_K_M + mmproj | `new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`, `mmproj-Muse-Glimmer-30B-Q4_K_M.gguf` | huggingface.co/meta-models/Muse-Glimmer-30B-GGUF (script `logs/glimmer/dl.sh`) | `glimmer` |
| Gemma 4 26B-A4B-it UD-Q4_K_M | `new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` | unsloth/gemma-4-26B-A4B-it-GGUF (base model google/gemma-4-26B-A4B-it) | `gemma4` |
| Gemma 4 E4B-it Q8_0 + mmproj | `new/gemma4/e4b/gemma-4-E4B-it-Q8_0.gguf`, `mmproj-gemma-4-E4B-it-Q8_0.gguf` | google/gemma-4-E4B-it (GGUF) | `gemma4` |
| Nemotron 3 Nano 30B-A3B Q4_K_M | `new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf` | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B (GGUF) | `nemotron`, bench bin `nh` |
| Nemotron 3 Nano 4B | `new/nemotron3/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf` | nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF | `nemotron` |
| Nemotron 3.5 Super VL 120B-A12B Q4_0 + mmproj BF16 + MTP head | `new/nemotron35vl/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf`, `mmproj-...-BF16.gguf`, `mtp-...-Q4_0.gguf`; `mmproj_st/` is made by `crates/mcv/ref/mmproj2st.py` | ggml-org GGUF on Hugging Face | `nemotron35` |
| Cosmos3-Nano (bf16 safetensors) | `new/cosmos3/Cosmos3-Nano` | nvidia/Cosmos3-Nano (Diffusers) | `cosmos3` |
| Qwen3.8-27B UD-Q4_K_M | `Qwen3.8-27B-UD-Q4_K_M.gguf` | unsloth on Hugging Face | `qwen38` |
| Bonsai 2 27B PQ2_0 / PTQ1_0 | `bonsai2/Ternary-Bonsai-2-27B-{PQ2_0,PTQ1_0}.gguf` | PrismML on Hugging Face | `qwen38`, `ternary` |
| LFM2.5-24B-A2B Q4_K_M | `LFM2.5-24B-A2B-Q4_K_M.gguf` | Hugging Face | `lfm2` |
| MiniCPM-V 4.7 35B-A3B Q4_K_M + mmproj f16 | `minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf`, `mmproj-model-f16.gguf` | openbmb/MiniCPM-V-4.7-35B-A3B-gguf (script `crates/mcv/ref/dl.sh`) | `mcv` |
| Qwen3.6-35B-A3B Q8_0 | `bench/Qwen3.6-35B-A3B-Q8_0.gguf` | Hugging Face | `mcv` |
| gpt-oss-20b MXFP4 | `bench/gpt-oss-20b-MXFP4.gguf` | Hugging Face | `gptoss` |
| Qwen2-architecture 7B Q4_K_M (first milestones) | `q7.gguf` | any Qwen2 7B Q4_K_M GGUF | `qwen2`, bench bin `strix` |
| Kitten TTS 2 (speech LM TQ2_1 + exported decoder) | `kitten2/hf/cpp/model-tq2_1.gguf`, `kitten2/rt/decoder` (made by `crates/kitten/ktexport.py`) | the kitten-tts-2-cpp project | `kitten` |
| Parakeet Redux (thrush-ternary-v2) | `parakeet-redux` | Hugging Face | `parakeet` |
| Qwen-Image-2.1 (BF16) | `qwen-image-2.1` | Qwen/Qwen-Image-2.1 | `qimage` |
| Kandinsky 6.0 Pro-distill (30B DiT) | `k6-pro-distill` | Hugging Face (Kandinsky 6.0) | `k6` |
| YuE2-3B (music) | (path in docs) | m-a-p on Hugging Face | `yue2` |
| Kev 0.8B / 4B / 9B / 27B | `decision/kev-*-bf16.gguf` + head (built by `crates/kev/ref/build_model.sh`) | jaredpalmer/kev-* on Hugging Face | `kev` |
| Laya (421M) | `decision/laya` | convaiinnovations/laya on Hugging Face | `laya` |

Benchmark-only weights for other engines (vLLM AWQ/FP8 checkpoints under `$MODELS/hf/`, an EAGLE3 drafter for llama.cpp) are
named in [docs/comparisons.md](docs/comparisons.md); their licenses are also the upstream publishers'.
