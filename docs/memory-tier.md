# strix-rt memory tier: UMA + page cache + NVMe

Design rule (David): max out UMA and treat the NVMe SSD as a memory tier, so the runtime does not need RAM >= model size.
Weights should come from mmap'd GGUF pages without copies where that pays off; page cache = weight cache; explicit prefetch
for MoE experts / layers that do not fit. If stock drivers cannot do it, we build it.

All numbers below were measured on the Framework Desktop (Ryzen AI Max+ 395, 128 GB, Radeon 8060S, RADV Mesa 26.0.8,
kernel 7.0.0-38, Crucial T710 4 TB) on 2026-10-14. Logs are in `logs/`.

## 1. Findings

### 1.1 Does RADV accept host pointers from a file-backed mmap?  **Not on stock Mesa/libdrm. The kernel can.**
- `VK_EXT_external_memory_host` is exposed (`minImportedHostPointerAlignment` = 4096) and is now enabled in `vkrt` (`Ctx::import_host`).
- Importing a page-aligned range of a `MAP_SHARED, PROT_READ` (or `MAP_PRIVATE` rw) mmap of q7.gguf: `vkAllocateMemory` -> `VK_ERROR_UNKNOWN`
  (`logs/hostimport_warm.log`). Anonymous memory imports fine.
- Root cause (`tools/userptr_probe.c`, `logs/userptr_probe.log`): raw `DRM_IOCTL_AMDGPU_GEM_USERPTR` on the same file pages:

  | mapping | ANONONLY\|VALIDATE\|REGISTER (0xe, what RADV/libdrm sends) | VALIDATE\|REGISTER (0xc) | READONLY\|VALIDATE\|REGISTER (0xd) | READONLY\|REGISTER | READONLY |
  |---|---|---|---|---|---|
  | anon rw | OK | OK | OK | OK | OK |
  | file MAP_SHARED ro | EPERM | EPERM | **OK** | OK | OK |
  | file MAP_PRIVATE rw | EPERM | EAGAIN | **OK** | OK | OK |

  libdrm's `amdgpu_create_bo_from_user_mem` hard-codes `AMDGPU_GEM_USERPTR_ANONONLY`; amdgpu rejects file-backed VMAs with that flag.
  With `READONLY` and without `ANONONLY` the kernel accepts page-cache pages.
- Proof via `tools/libuserptr_shim.so` (LD_PRELOAD; rewrites the userptr ioctl flags 0xe -> 0xd): RADV imports the file-backed mmap,
  GEMV results are **bit-identical** to copied buffers (max |diff| = 0), and the full q7 decode with `ZEROCOPY=1` produces the same
  text as the copied path (`logs/gen_zerocopy_warm.log`). Imported memory reports `HOST_VISIBLE|HOST_COHERENT|HOST_CACHED`.
- DMA-BUF route: exporting a file-backed userptr BO via `DRM_IOCTL_PRIME_HANDLE_TO_FD` -> EPERM (`logs/prime_probe.log`).
  udmabuf needs memfd/shmem, not ext4 page cache. So stock dma-buf paths cannot carry page-cache pages either.

### 1.2 GEMV on imported pages vs copied DL|HV buffers (Q4_K 18944x3584, cycling 28 layers = 1020 MiB)
| run | imported (mmap, shim) | copied (DL\|HV) |
|---|---|---|
| warm, logs/hostimport_shim_warm.log (shared_ro) | 193.7 GB/s | 208.1 GB/s |
| warm, same log (private_rw) | 194.0 GB/s | 200.8 GB/s |
| after cold import, logs/cold_import.log | 170.1 GB/s | 203.1 GB/s |
| warm, logs/cold_import.log | 169.1 GB/s | 197.0 GB/s |
| anon import control, logs/hostimport_warm.log | 191.5 GB/s | 199.3 GB/s |

Imported memory is 3-15% slower across runs (snooped, cacheable system memory vs the write-combined DL|HV type).

### 1.3 Hidden cost: per-submit userptr tax
- Empty command-buffer submit+wait with 1020 MiB of userptr imports alive: **1094-1364 us**; after freeing the imports: **23 us** (`logs/cold_import.log`).
- Full q7 decode (128 tok): copied 44.6 tok/s wall / 47.0 GPU; zero-copy (29 per-layer imports) **35.1 wall / 43.7 GPU**;
  3 big overlapping 3 GiB windows 33.5 / 43.1; 5 windows of 2 GiB 32.5 / 42.2 (`logs/gen_zerocopy_*.log`).
  The wall-vs-GPU gap grows from ~1.2 ms to ~5-7 ms per token and does not shrink with fewer imports, so it scales with imported bytes,
  not BO count. Hypothesis (not yet verified with ftrace): amdgpu revalidates every userptr BO on the CS BO list (HMM range walk)
  on each submission, roughly 1.1 ms per GiB here.

### 1.4 NVMe tier (logs/ssd_bench.log, page cache dropped before each test)
- Link: drive is PCIe 5.0 capable (32 GT/s) but the slot trains at **16 GT/s x4** (Gen4) -> every test saturates at ~7.2 GB/s.
- Cold sequential buffered `dd` bs=16M: 7.0 GB/s (q7.gguf, 4.68 GB), 6.9 GB/s (16 GiB file).
- fio io_uring O_DIRECT: seq 1M qd32: 7221 MB/s; seq 4M qd16: 7209; seq 1M qd32 x4 jobs: 7225; rand 2M qd16: 7194;
  rand 2M qd32 x4: 7204; rand 128k qd64 x4: 7205 MB/s (55.0k IOPS); rand 4k qd128 x4: 7022 MB/s (1714k IOPS).
- fio io_uring buffered seq 1M qd32: 7209 MB/s.
- Cold zero-copy import of the whole q7 (shim; page faults pull from SSD): 0.62 s; cold import of 1020 MiB of scattered ffn_gate tensors: 0.199 s (5.38 GB/s).
- O_DIRECT into Vulkan-mapped memory (DL|HV or HOST_CACHED): **EFAULT** (GPU mappings are PFN/IO maps, GUP fails) (`logs/odirect.log`).
- Cold buffered pread straight into a mapped DL|HV buffer (qd1, 8 MiB): 6.60 GB/s. O_DIRECT into anonymous memory: 6.83 GB/s;
  importing that 4 GiB anon buffer on stock RADV: 0.400 s.

Takeaway: UMA streams ~200 GB/s, the SSD ~7.2 GB/s (link-limited), about 28x less. Anything read per token must be UMA-resident.
The SSD tier only works for sparse/rarely touched weights (MoE experts, cold layers at load time) and for capacity, not for dense streaming.
Example: dense 40 GB model fully on SSD = ~5.6 s/token; MoE with 3 GB active/token at a 90% expert hit rate = 0.3 GB misses ≈ 42 ms of I/O per token.

## 2. Design: SSD <-> UMA streaming tier

### 2.1 Tiers
| tier | backing | GPU access | measured | role |
|---|---|---|---|---|
| T0 slab pool | DL\|HV vkAllocateMemory, persistently mapped | full speed, no submit tax | ~200 GB/s GEMV | hot weights, attention/shared/dense layers, resident experts |
| T1 page-cache import | file mmap imported as userptr (READONLY) | ~3-15% slower + ~1.1 ms/GiB/submit (as measured) | 169-194 GB/s | zero-copy for models that fit, while submits are few (batched tokens) |
| T2 NVMe | GGUF file | via T0 fill or T1 fault | 7.2 GB/s | capacity: experts and layers that do not fit |

### 2.2 Default data path (works on stock drivers today)
1. Parse GGUF from mmap (no copies for metadata). Mark tensors resident vs streamable through a placement policy.
2. Dense/always-used tensors -> T0 slabs filled by buffered `pread`/io_uring reads directly into the mapped slab
   (6.6 GB/s cold measured). Then `posix_fadvise(DONTNEED)` / uncached buffered I/O (`RWF_DONTCACHE`, to test on 7.0) so the page cache
   does not hold a second copy. Net RAM = one copy of the resident set.
3. MoE experts -> fixed-size expert slots in a T0 slab pool (slot = one expert's gate/up/down, all quant formats aligned to 4 KiB).
   An expert table maps (layer, expert) -> slot or MISS. Kernels take slot indices through a small indirection buffer, so the recorded
   command buffer never changes; only the table contents change.
4. Miss handling: the router output for layer L is read back (tiny), missing experts are queued on io_uring (2 MiB-class reads saturate
   the drive: rand 2M qd16 = 7.19 GB/s), and compute for resident experts proceeds. Decode overlaps per-layer: issue reads for layer L+1..L+k
   from predictions while layer L computes.
5. Eviction: LRU with frequency (expert usage is heavy-tailed); pin shared experts and dense layers; never evict anything referenced
   by an in-flight submit (per-slot fence epoch).
6. Prefetch from router predictions: (a) same-layer temporal locality (last token's experts), (b) cross-layer predictor: router logits of
   layer L applied to the layer-L hidden state with layer L+1's router (cheap GEMV, approximates the next layer's choice), (c) prompt
   prefill statistics as a warm-up set. Measure hit rate per policy; target miss bytes/token << 7.2 GB/s x token budget.
7. Load time: `madvise(MADV_WILLNEED)` / io_uring readahead on the resident set; cold full-model fill is drive-bound (~0.65 s per 4.7 GB measured).

### 2.3 Zero-copy path (T1) and what we have to build
- Prototype now: `tools/libuserptr_shim.so` (drops ANONONLY, sets READONLY). Production options, in order of effort:
  1. **Patch libdrm/RADV**: when `VkImportMemoryHostPointerInfoEXT` points at a read-only/file mapping, pass `READONLY` without `ANONONLY`
     (small upstreamable change; Vulkan semantics: imported memory is then GPU read-only, fine for weights).
  2. **Kill the per-submit tax**: the hypothesis above has to be confirmed (ftrace `amdgpu_cs_ioctl`, `hmm_range_fault`). Mitigations: submit
     many tokens per submission (GPU-side sampling + embedding lookup -> one submit per N tokens), or keep userptr BOs off the per-CS list
     (RADV uses a global BO list; a per-submit list with only the tensors that submit reads).
  3. **Custom kernel module `strix_pagecache`**: given (fd, offset, len), pin page-cache folios (`pin_user_pages` FOLL_LONGTERM is allowed for
     read-only file pins) or `filemap_get_folios`, build an sg_table and export a **dma-buf**; import with `VK_EXT_external_memory_dma_buf`.
     This gives zero-copy page-cache weights with no per-submit HMM walk, and lets the module also do eviction/refill
     (unpin + `invalidate_mapping_pages`) for the expert tier. Risks: pinned page cache cannot be reclaimed (the module must enforce a
     budget), and the dma-buf importer must map system pages (amdgpu does for GTT).
  4. Patched amdgpu: allow exporting READONLY file userptr BOs as dma-buf (today EPERM).
- Until 2/3 land, T1 is only a win for capacity (no second copy) and not for speed: on q7 zero-copy decode is ~20% slower wall-clock.

### 2.4 Pluggable weight formats (targets: modern models < 50 GB, possibly ternary, e.g. PrismML Bonsai 2)
- `gguf` crate stays format-agnostic: (type id -> block geometry, CPU reference dequant). New formats (ternary TQ1_0/TQ2_0-style or
  model-specific layouts) register a `WeightFormat` entry: block bytes/elements, CPU dequant (reference + tests), GLSL `item()` body for the GEMV
  template (`shaders/gemv.comp` already builds per-format variants from one template via -D), and later a coop-matrix GEMM tile loader.
- The memory tier only sees byte ranges + alignment (4 KiB), never formats; slots are sized per model at load time.
- Ternary implications: ~1.6-2 bits/weight halves the bytes vs Q4_K, so decode becomes more compute-bound per byte; unpack cost in the
  GEMV `item()` matters more, and a 50 GB budget fits far larger models resident in 128 GB UMA.

## 3. Next experiments
1. ftrace the submit path to confirm/refute the per-GiB userptr revalidation hypothesis; measure tax vs imported size (0.25-8 GiB).
2. GPU-side argmax + embedding, N tokens per submit, then re-measure zero-copy wall tok/s.
3. Prototype `strix_pagecache` dma-buf module (DKMS, out of tree) and compare GEMV GB/s + submit cost against T0 copies.
4. Expert slot pool + io_uring fill on a real MoE GGUF (< 50 GB), with hit rate / miss bytes / tok/s vs RAM budget (e.g. cap UMA at 16/32/64 GB).
5. `RWF_DONTCACHE` buffered reads to fill T0 without page-cache duplication; measure fill GB/s.


## 4. Zero-copy at copied-path speed (2026-10-14, round 2)

### 4.1 Per-submit cost: cause confirmed
bpftrace (`tools/trace_cs.bt`, `logs/trace_cs_hostimport.log`) on `hostimport` with 28 userptr imports (1020 MiB) alive:
- 410 `amdgpu_cs_ioctl` calls. The 200 empty submits with imports alive land in the 1-2 ms bucket (210 calls), the others in 4-8 us.
- `amdgpu_ttm_tt_get_user_pages` and `hmm_range_fault` each ran 5908 times (28 BOs x 211 submits); `hmm_range_fault` total 379.7 ms,
  `amdgpu_cs_parser_bos` 226.5 ms, `amdgpu_cs_ioctl` total 231.0 ms.
- So amdgpu re-runs `hmm_range_fault` over every userptr BO on the CS BO list on every submission. The hypothesis from section 1.3 is confirmed.
  It is inherent to userptr (HMM, unpinned pages), so no userptr flag or libdrm patch removes it.

### 4.2 Why imported pages were slower: page/fragment alignment, not caching
`memtype` (`logs/memtype*.log`), same Q4_K GEMV, 28 x ffn_gate:
| backing | GB/s |
|---|---|
| copied DL\|HV | 208.5-208.6 |
| copied HOST_CACHED (GTT, snooped) | 204.9-205.1 |
| anon import, 4K pages | 162.6-164.9 |
| anon import, THP 2 MiB | 194.0-196.5 |
| file import, 4 KiB-aligned ranges | 158.7-174.1 |
| file import, **2 MiB-aligned ranges** | 192.4-196.7 |
- Snooped/cached memory costs ~2% (205 vs 208.5). The loss came from GPU page size: 4 KiB mappings vs 2 MiB fragments.
- The page cache is already almost ideal after a sequential read: `tools/folio_stats.py` found 99.6% of the 2 MiB-aligned chunks of q7.gguf
  physically contiguous (`logs/folio_stats.log`). Folio contiguity barely changed with read_ahead_kb 512 KB to 64 MB (154.8-175.2 GB/s with 4 KiB-aligned imports, `logs/ra_folio.log`).
  The import has to start at a 2 MiB-aligned file offset so GPU VA, file offset and physical folios agree mod 2 MiB.
- MADV_COLLAPSE on the file mapping returns EINVAL (`CONFIG_READ_ONLY_THP_FOR_FS` is not set), and it isn't needed.
- In a clean process (`zctest`, `logs/zctest1.log`): copy 208.4 GB/s; 2 MiB-aligned userptr 206.0 GB/s, but an empty submit costs 448.5 us;
  strixpc dma-buf 187.9 GB/s with empty submit 44.1 us. RADV placed every dma-buf BO at VA = 1 MiB mod 2 MiB.
  The fix is `qwen2::import_pagecache_aligned`: start the export earlier by the VA misalignment so VA + offset is 2 MiB aligned.

### 4.3 The fix: `kmod/strixpc` (out-of-tree dma-buf exporter) + aligned dma-buf import
- `/dev/strixpc` pins the page-cache folios behind the read-only GGUF mmap (`pin_user_pages_fast(FOLL_LONGTERM)`, read-only, allowed for file pages)
  and exports them as a dma-buf. vkrt imports it with `VK_EXT_external_memory_dma_buf` (`Ctx::import_dmabuf`, `vkrt::pagecache::export`).
  The result is an ordinary sg BO: no HMM walk per submit, no libdrm/RADV/amdgpu changes, no shim. Rollback is `rmmod strixpc`.
- Cheaper options were considered and ruled out: patching libdrm/RADV flags (the tax is in amdgpu, not the flags), userptr flag variants
  (all use HMM), and multiple tokens per submit (only amortizes a cost that scales with model bytes: ~1.1 ms/GiB, so ~55 ms per submit at 50 GB).
- Loader (`ZEROCOPY=dmabuf`): one export per layer group with 2 MiB-aligned ranges, then VA alignment. Cold pages are populated first with
  `MADV_HUGEPAGE` + `MADV_SEQUENTIAL` + `MADV_POPULATE_READ` (0.651 s for the whole q7 from a dropped cache), which leaves 99.6% of 2 MiB chunks contiguous.
  Letting GUP fault pages in instead (`ZC_WARM=0`) left 96.6% contiguous.

### 4.4 Results (q7 decode, 12-token prompt + 128 tokens; `logs/gen_zc2.log`, `logs/zc_dmabuf_cold_kbench.log`)
| path | wall tok/s | GPU ms/tok (tok/s) | weight setup |
|---|---|---|---|
| copied (3 runs) | 44.51 / 44.59 / 44.59 | 21.27-21.28 (47.00-47.02) | 0.546 s copy |
| **zero-copy dma-buf (strixpc)**, warm (3 runs) | **45.06 / 44.81 / 45.03** | **21.25 (47.05-47.06)** | 0.021 s |
| zero-copy userptr + shim, 2 MiB aligned | 38.53 | 22.23 (44.98) | 0.051 s |
| cold start, dma-buf + warm | 45.68 | 21.30 (46.96) | 0.661 s (process 3.85 s) |
| cold start, dma-buf, ZC_WARM=0 | 43.94 | 21.35 (46.84) | 0.610 s |
| cold start, copied | 44.20 | 21.37 (46.80) | 0.942 s (process 4.27 s) |
- The 64 greedy token ids are identical for copied and dma-buf.
- Per-kernel kbench with ZEROCOPY=dmabuf: attn_q 176.2, attn_output 178.4, ffn_gate 180.9, gate+up 226.6, ffn_down Q4K 184.9 / Q6K 218.0, output 231.3 GB/s.
  Several of these are below the copied numbers (192-198), even though the whole-token GPU time is the same. That isn't explained yet; the next step is per-tensor alignment/contiguity checks.
- Memory: weights exist once, in the page cache, pinned while imported. The copied path keeps both the page cache and the arena copies unless the cache is dropped.

### 4.5 Modality-agnostic core (LLM, VLM, image/video/audio generation)
- The memory tier deals in **byte ranges of backing files** (GGUF today; safetensors or any container tomorrow): `export(range) -> dma-buf -> Buffer + offset`.
  It never sees tensors, layers or tokens. Placement policies (resident, streamable, slot pool) are keyed by opaque tensor ids plus a usage
  schedule supplied by the model graph: per token for LLMs, per denoising step / per block for diffusion and video DiTs, per frame for audio codecs.
- Expert/slot pools generalize to any "conditional weight set": MoE experts, LoRA/adapters, ControlNet branches, per-resolution VAE tiles,
  vision encoders that run once per image.
- The kernel layer is a set of format-specialized GEMV/GEMM/conv/attention kernels behind (format, op, shape) dispatch. Decode GEMV is just
  the first entry; diffusion/video are GEMM- and attention-bound (coop matrix), audio adds conv1d/STFT. Activations, KV caches, latents and
  image tensors all use the same Buffer/arena API.
- Recorded command buffers with parameters in buffers (pos today; timestep, frame index, guidance scale tomorrow) are the graph-replay unit
  for any iterative generator.

### 4.6 Next
1. Find out why some kbench kernels lose 5-10% on dma-buf memory while the whole-token time is equal (per-arena VA/phys alignment audit).
2. Budget and eviction for pinned page cache (strixpc `max_pinned_mb`, release/re-export per slot) for the MoE expert tier.
3. Sign the module / DKMS packaging; upstream discussion: let amdgpu export READONLY userptr BOs, or a generic page-cache dma-buf exporter.
