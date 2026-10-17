import sys
sys.path.insert(0, '~/llama.cpp/gguf-py')
import gguf
r = gguf.GGUFReader(sys.argv[1])
for f in r.fields.values():
    if f.name.startswith(('tokenizer.ggml.tokens','tokenizer.ggml.merges','tokenizer.ggml.token_type','tokenizer.ggml.scores')):
        continue
    try:
        v = f.contents()
    except Exception:
        v = '?'
    s = str(v)
    print('KV', f.name, s[:160])
seen = set()
for t in r.tensors:
    import re
    k = re.sub(r'blk\.\d+\.', 'blk.N.', t.name)
    if k in seen: continue
    seen.add(k)
    print('T', t.name, t.tensor_type.name, list(t.shape))
