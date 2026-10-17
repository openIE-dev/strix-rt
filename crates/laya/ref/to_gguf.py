"""Laya model.safetensors (F16) -> GGUF for strix-rt: 2-D weights stay F16 (GEMM arenas), 1-D tensors become F32 (f32b).
Names are kept as in the checkpoint. Head attention in_proj / out_proj keep their PyTorch [out, in] layout."""
import sys, json, numpy as np
sys.path.insert(0, "~/llama.cpp/gguf-py")
import gguf
from safetensors.numpy import load_file
src, out = sys.argv[1], sys.argv[2]
W = load_file(f"{src}/model.safetensors"); cfg = json.load(open(f"{src}/encoder/config.json")); rl = json.load(open(f"{src}/rl_agent_config.json"))
w = gguf.GGUFWriter(out, "laya")
for k, v in [("hidden", cfg["hidden_size"]), ("layers", cfg["num_hidden_layers"]), ("heads", cfg["num_attention_heads"]), ("ffn", cfg["intermediate_size"]),
             ("window", cfg["local_attention"]), ("global_every", cfg["global_attn_every_n_layers"]), ("head_layers", rl["head_layers"])]: w.add_uint32(f"laya.{k}", k and v)
w.add_float32("laya.theta_global", cfg["rope_parameters"]["full_attention"]["rope_theta"]); w.add_float32("laya.theta_local", cfg["rope_parameters"]["sliding_attention"]["rope_theta"])
w.add_float32("laya.norm_eps", cfg["norm_eps"])
for k in sorted(W):
    a = W[k]
    if a.ndim >= 2 and a.shape[-1] % 64 == 0 and "type_emb" not in k: w.add_tensor(k, a.astype(np.float16))
    else: w.add_tensor(k, a.astype(np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close(); print("wrote", out, len(W))
