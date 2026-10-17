# vLLM num_warps patch (not redistributed)

The experiment in `docs/comparisons.md` ("vLLM gpt-oss 4 -> 5 client step: root cause") bind-mounted a patched copy of
`vllm/third_party/triton_kernels/matmul_details/opt_flags.py` from the vLLM container (`kyuz0 vllm-therock-gfx1151`, vLLM 0.31.0+strix).
That file is third-party code, so this public copy does not include it. The patch is a one-line change on the AMD path:

```diff
-    num_warps = 2 if (m is not None and m <= 16) else 8
+    num_warps = 8  # PATCH: was 2 if m <= 16
```

To reproduce: copy `opt_flags.py` out of the container into this directory as `opt_flags_nw8.py`, apply the line above,
then run `tools/comp/vllm_nw8.sh`.
