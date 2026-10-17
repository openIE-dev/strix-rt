# Benchmarking and heavy GPU jobs on the Framework (Strix Halo)

Several workers share one GPU (and the NPU). Unsynchronized jobs give wrong numbers and, worse, hang the GPU.

## Rule: hold the exclusive GPU lock

Every benchmark, parity/reference run, or other heavy GPU job runs under `logs/gpu_lock.sh`. That includes
Vulkan binaries (`target*/release/*`), NPU+GPU runs, llama.cpp, and **ROCm/HIP/PyTorch Python** (e.g. `.venv_rocm`
reference scripts).

```sh
logs/gpu_lock.sh ./target/release/pf_q38 model.gguf 512          # waits for the lock, runs, releases
GPU_LOCK_TAG="qi ref1024" logs/gpu_lock.sh .venv_rocm/bin/python ref.py golden1024
GPU_LOCK_WAIT=600 logs/gpu_lock.sh ...                            # give up after 600 s (exit 75)
logs/gpu_lock.sh --status                                          # "free" or "held: pid .. user .. since ..: tag"
logs/gpu_lock.sh sh my_session.sh                                  # hold it for a whole session (perf level, A/B, ...)
```

- The lock is `flock` on `/tmp/strix-gpu.lock` (mode 666; the holder is in `/tmp/strix-gpu.lock.owner`). The kernel
  releases it when the holder and all its children exit, including on crash or `kill -9`. Never delete the file.
- It is re-entrant: anything started by a holder sees `STRIX_GPU_LOCK_HELD=1` and runs directly, so wrappers can nest.
- `logs/gpu_idle_run.sh LOG ENV... CMD ARGS...` now runs under the lock (it used to poll pgrep).
- Scripts that source `logs/pf_busy.sh` re-execute themselves under the lock, so the whole script is exclusive; its
  `busy`/`waitidle` remain as no-ops for compatibility. `logs/kt/kt_e2e.sh` and `logs/kt/ab.sh` do the same.
- Do the perf level inside the lock: `echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level`
  at the start, `echo auto` before you exit (use `trap ... EXIT`).
- Keep each hold short (one measurement session, minutes). Builds, CPU work and editing don't need the lock.
- Short smoke tests (< a few seconds of GPU) may skip it, but anything that can run long or allocate tens of GB
  must take it.

Why the pgrep idle checks were replaced: they only matched known binary names (ROCm Python was invisible),
matched the waiting wrapper itself (self-deadlock, round 11), and were racy (two waiters start together).

## Rule: check for CPU-heavy jobs first

The GPU shares memory bandwidth (and power/thermal budget) with the CPU, so a CPU-only job skews GPU numbers even though
it never touches the GPU. Example 2026-10-15: a CPU llama-server (LFM2.5-24B, -t 16, ~900% CPU) running during qimage
profiling raised the 1024 step from 3.92 s to 4.6 s and SwiGLU from 44 to 55 ms, and the 1024 wall time from 179-188 s to 194.8 s.

- Before measuring, run `ps -eo pid,etime,pcpu,comm --sort=-pcpu | head`. If anything uses heavy CPU (more than ~100%), wait in
  short polls for it to finish. Do not kill other workers' jobs. If you have to measure anyway, say so in the log.
- Ideally, CPU-heavy baselines (llama-server, llama-bench -ngl 0, CPU references, big builds) also run under `logs/gpu_lock.sh`,
  so they serialize with GPU benchmarks instead of colliding with them.

## GPU hang incident 2026-10-14 (UTC)

`journalctl -k`: 41 `ring gfx_0.0.0 timeout` + ring resets since boot (all "recovered through reset", no reboot needed).
- 21:20-21:32: `pf_gemm` (a `[gfxhub] page fault` at 21:31:58 first, an out-of-bounds shader access) and other single
  long/hung dispatches.
- 22:00-22:24: ROCm PyTorch reference jobs (`qi_ref` te_ref/ref512/ref1024, `yue_ref`) ran through KFD at the same
  time as Vulkan benchmarks. The kernel logged `svm_range_restore_work` / `amdgpu_amdkfd_restore_userptr_worker`
  hogging the CPU (KFD memory eviction/restore) right before the timeouts at 22:02, 22:04 (x2) and 22:24. The
  Vulkan job running at that moment (`pfq`, NPU_SPLIT and plain prefill alike) was blamed and lost its device.
- 22:27-22:32: `yue2` (new crate, `yue2 parity`/`yue2 dbg`) hung the gfx ring by itself four times (22:27:14,
  22:30:22, 22:32:31; the pfq timeout at 22:27:28 is collateral right after a yue2 reset). The devcoredump
  (`/sys/class/drm/card1/device/devcoredump/data`) names `yue2`. That is a yue2 bug (likely an over-long or
  out-of-bounds dispatch) and needs a fix in the crate, not the lock.
- 22:37:57 and 22:40:00: `qi --size 1024` (target-qi, logs/qi1024_bench.sh) lost the device each time ~120 s in,
  right after "prefix gpu", i.e. in the first denoise step at 4096 image tokens. The ring timeout names `qi`. Likely
  one command buffer per denoise step (or the 4096-token flash attention) running longer than the amdgpu gfx lockup
  timeout (10 s), so the kernel treats it as a hang. Fix in qimage: split each step into several submits (e.g. per
  block, aim for < 1-2 s per submit), and time the attention kernel alone at 4096 tokens.
- Possible fault source, flag for the kev/bf16 owner: commit 280d9e2 ("bf16 prefill GEMM: padded-row BF16 ...", 22:32,
  kev work) changed `crates/prefill/shaders/pf_deq.glsl` BF16/F16 `deq32` to `uvec4` loads with `b4 = (rb + it*64) >> 4`,
  assuming 16-byte aligned rows. Weights straight from safetensors/GGUF are only guaranteed 8-byte (GGUF: 32-byte) aligned.
  A misaligned row base is silently floored to 16 B, so it reads the wrong weights (wrong output, not a fault) and can
  read up to 8 bytes past the end of the last row. Any F16/BF16 user of `deq32` (qimage, yue2, decision models)
  needs either a 16-byte-aligned upload (the RT_PAD16 / qtype 13 path) or the old 4 x u16 loads when misaligned.
  qimage and yue2 started failing right after that commit (yue2 hangs 22:27-22:32 started before it, so it doesn't
  explain those).
- Temps at the time: GPU edge 92 C, k10temp Tctl 91 C, ~107 W at sclk 2900 MHz. Hot but under the limit; no thermal
  messages in the log. Not the trigger.
- Mitigation: run ROCm and Vulkan jobs serially (this lock), and debug hanging kernels in isolation. Each gfx timeout
  costs the 10 s lockup timeout plus a ring reset and kills the innocent Vulkan contexts that were running.

### pf_deq alignment fix verification (7845320)
BF16/F16 `deq32` now uses 16-byte loads only for 16-byte-aligned blocks (2-byte loads otherwise). Verified under the lock:
kev 0.8b and 9B bf16 parity unchanged (9B state 2048 tok 1.52 s), qimage 512 PSNR 22.71 dB (at fbedf9e + fix and at
HEAD target-qi), yue2 parity (ar logits cos 0.999984, codec argmax 30/31). Note: `qi` defaults to 1024; a 512 golden
check needs `--size 512` (without it, the run compares against the wrong golden and the 1024 run can hit the gfx timeout).

## Fan curve (2026-10-15)
- Stock EC curve held the fan at ~926 rpm target with CPU ~92 C; GPU hit 95-100 C and sclk sagged 2890 -> ~2000 MHz.
- Now: `fw-fancurve.service` (/usr/local/bin/fw-fancurve, ectool built from ~/src/ectool) drives duty from
  max(k10temp Tctl, amdgpu edge): <=40 C 20%, 55 C 35%, 70 C 60%, >=85 C 100% (linear between).
- 60 s back-to-back llama-bench pp600 (Qwen3.8-27B Q4_K_M, Vulkan): sclk avg 2665 / steady 2830-2880 MHz, GPU edge max 85 C,
  Tctl max 83 C, ~135-145 W, fan up to ~2230 rpm, pp600 ~397 t/s. No throttle sag.
- Only EC fan index 1 (hwmon fan2) spins (max ~2300 rpm at 100%); fan index 0 (hwmon fan1) reads 0 even at 100% duty.
- Restore stock EC control: `~/setup/fan-backup/restore.sh` (old state saved in ~/setup/fan-backup/state-20261015.txt).
