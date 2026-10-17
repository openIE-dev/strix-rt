# Cosmos3 understanding reference (CPU, bf16): Qwen3-VL (= vllm-cosmos3 Cosmos3Reasoner) built from the und-tower +
# vision_encoder weights with the transformers_cosmos3 KEY_MAPPING. Dumps processor inputs, ViT outputs (merger +
# 3 deepstack), first-step logits and a greedy answer into OUT/.  Usage: vit_ref.py IMAGE "question" OUT [max_pixels]
import sys, os, json, re, time, numpy as np, torch
from safetensors import safe_open
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3VLConfig
M = os.path.expanduser("$MODELS/new/cosmos3/Cosmos3-Nano")
img, q, out = sys.argv[1], sys.argv[2], sys.argv[3]; maxpx = int(sys.argv[4]) if len(sys.argv) > 4 else 448 * 448
os.makedirs(out, exist_ok=True); torch.set_num_threads(int(os.environ.get("THREADS", "8")))
c = json.load(open(f"{M}/config.json"))
cfg = Qwen3VLConfig(text_config=c["text_config"], vision_config=c["vision_config"], image_token_id=c["image_token_id"], video_token_id=c["video_token_id"],
                    vision_start_token_id=c["vision_start_token_id"], vision_end_token_id=c["vision_end_token_id"], tie_word_embeddings=c.get("tie_word_embeddings", False))
DROP = re.compile(r"_moe_gen|^proj_(in|out)\.|^time_embedder\.|\.self_attn\.add_[qkv]_proj\.|\.self_attn\.to_add_out\.|\.self_attn\.norm_added_[qk]\.|^(audio|action)_|^(sound|action|audio)|^llm2|^vae2llm|latent_pos_embed")
MAP = [(r"\.self_attn\.to_q\.", ".self_attn.q_proj."), (r"\.self_attn\.to_k\.", ".self_attn.k_proj."), (r"\.self_attn\.to_v\.", ".self_attn.v_proj."),
       (r"\.self_attn\.to_out\.", ".self_attn.o_proj."), (r"\.self_attn\.norm_q\.", ".self_attn.q_norm."), (r"\.self_attn\.norm_k\.", ".self_attn.k_norm."),
       (r"^(embed_tokens\.|layers\.|norm\.)(.*)$", r"model.language_model.\1\2"), (r"^(blocks\.|merger\.|patch_embed\.|pos_embed\.|deepstack_merger_list\.)(.*)$", r"model.visual.\1\2")]
t0 = time.time()
with torch.device("meta"): model = Qwen3VLForConditionalGeneration(cfg)
sd = {}
idx = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]
for f in sorted(set(idx.values())):
    with safe_open(f"{M}/{f}", "pt") as h:
        for k in h.keys():
            if DROP.search(k): continue
            n = k
            for a, b in MAP: n = re.sub(a, b, n)
            sd[n] = h.get_tensor(k)
r = model.load_state_dict(sd, strict=False, assign=True); print("missing", r.missing_keys[:8], "unexpected", r.unexpected_keys[:8], flush=True)
lm, vis_ = model.model.language_model, model.model.visual  # non-persistent rotary buffers stayed on meta: rebuild on CPU
lm.rotary_emb = type(lm.rotary_emb)(config=cfg.text_config); vis_.rotary_pos_emb = type(vis_.rotary_pos_emb)(cfg.vision_config)
left = [n for n, b in model.named_buffers() if b.is_meta]; print("meta buffers", left, flush=True)
model = model.to(torch.bfloat16).eval(); print(f"loaded {time.time()-t0:.1f}s", flush=True)
proc = AutoProcessor.from_pretrained(M); proc.image_processor.size = {"shortest_edge": 65536, "longest_edge": maxpx}
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
text = proc.apply_chat_template(msgs, add_generation_prompt=True)
inp = proc(text=[text], images=[Image.open(img).convert("RGB")], return_tensors="pt")
open(f"{out}/prompt.txt", "w").write(text)
np.save(f"{out}/input_ids.npy", inp["input_ids"][0].numpy().astype(np.int64)); np.save(f"{out}/pixel_values.npy", inp["pixel_values"].float().numpy())
np.save(f"{out}/grid_thw.npy", inp["image_grid_thw"].numpy().astype(np.int64)); print("grid", inp["image_grid_thw"].tolist(), "ids", inp["input_ids"].shape, flush=True)
with torch.no_grad():
    vis = model.model.visual
    o = vis(inp["pixel_values"].to(torch.bfloat16), grid_thw=inp["image_grid_thw"])
    emb, ds = (o.pooler_output, o.deepstack_features) if hasattr(o, "pooler_output") else o
    emb = emb[0] if isinstance(emb, (list, tuple)) else emb
    np.save(f"{out}/vit_out.npy", emb.float().numpy())
    for i, d in enumerate(ds): np.save(f"{out}/vit_ds{i}.npy", d.float().numpy())
    print(f"vit {tuple(emb.shape)} {time.time()-t0:.1f}s", flush=True)
    pos, _ = model.model.get_rope_index(inp["input_ids"], (inp["input_ids"] == cfg.image_token_id).int(), inp["image_grid_thw"])
    np.save(f"{out}/position_ids.npy", pos[:, 0].numpy().astype(np.int64))
    lg = model(**inp).logits[0, -1].float().numpy(); np.save(f"{out}/logits0.npy", lg); print("top5", np.argsort(-lg)[:5].tolist(), flush=True)
    g = model.generate(**inp, max_new_tokens=int(os.environ.get("NEW", "32")), do_sample=False)
    a = g[0, inp["input_ids"].shape[1]:]; np.save(f"{out}/answer_ids.npy", a.numpy().astype(np.int64))
    ans = proc.decode(a, skip_special_tokens=True); open(f"{out}/answer.txt", "w").write(ans); print("answer:", ans, f"{time.time()-t0:.1f}s", flush=True)
