# Parakeet Redux STT on strix-rt (crate `parakeet`)

- Weights are in `$MODELS/parakeet-redux`, packed as thrush-ternary-v2. Each byte holds 5 trits (base 3, least significant first), with one f16 scale per 128 weights.
  - At load time the weights are repacked to PQ2_0 rows, which takes about 0.2 s on all threads.
  - The 264 ternary linears (FF, attention, relative position, conv pointwise) then go through prefill's coop-matrix GEMM (`Pf::gemm_tile`).
  - The dense f16 matrices (subsampling pointwise, subsampling linear, encoder projector) use the F16 GEMM path.
- Encoder (FastConformer, 24 layers, d=1024, Vulkan):
  - `pk_sub`: dw-striding subsampling convolutions, valid frames only.
  - `pk_ln`, `pk_ew`: LayerNorm, SiLU, residuals and biases.
  - `pk_dwc`: GLU, depthwise k9 conv, BatchNorm and SiLU, fused.
  - Relative-position attention: `pk_aprep` produces per-head f16 operands. Then QK^T, (Q+v)P^T and P.V run as GEMMs, with `pk_smax` (shifted softmax) in between. The older scalar kernel `pk_attn` is still available with `PK_ATTN_GEMM=0`.
- CPU parts:
  - Log-mel front end: own FFT and Slaney filterbank, identical to HF to about 2e-6 relative.
  - TDT greedy decoder: 2-layer LSTM, projector and joint, run on a 16-thread spin pool with AVX2/FMA GEMVs.
- GEMM tile policy: 32x128 tiles up to 256 frames, 128x128 above that (`PK_M32`, `PK_TILE`).
- Reference: `crates/parakeet/ref`.
  - `ref.py`: HF transformers ParakeetForTDT running the unpacked ternary weights.
  - `photon.py`: Moondream Photon.
  - `rtf_ref.py`: CPU timings.
  - `wer.py`: the comparison table.
  - Clips: LibriSpeech dummy ls0–7 and two kitten TTS clips (resampled to 16 kHz in `~/pk_ref/clips16`).

## Results

Runs `logs/pk_r2_table.txt` and `logs/pk_r3.log`, on an idle GPU with perf level high, median of 7. The CPU references are the median of 3 with 16 threads.

- Parity:
  - All 10 transcripts are byte-identical to both HF and Photon (WER 0 against each).
  - WER against LibriSpeech ground truth is 4.39% for all three implementations.
  - The encoder hidden state differs from HF fp32 by about 5e-4 relative (f16 activations into the GEMMs).
- RTF over 96.8 s of audio, end to end (mel, encoder and decode):

| implementation | RTF |
|---|---|
| strix-rt | 0.0026–0.0028 (about 370× real time) |
| Photon CPU | 0.0093 |
| HF CPU | 0.068 |

- Breakdown for ls4 (29.4 s of audio, 368 encoder frames): mel 4.4 ms, encoder 34 ms on the GPU, decode 15–19 ms.
- Short clips (5 s) take about 12 ms of encoder time, dominated by per-dispatch overhead across about 900 dispatches.
