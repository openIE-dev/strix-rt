# Export the KittenTTS TorchScript decoder graph to a flat IR (JSON) + f32 tensor blob for strix-rt's Rust interpreter.
import torch, json, sys, struct, numpy as np
src, out = sys.argv[1], sys.argv[2]
m = torch.jit.load(src); g = m.graph
torch._C._jit_pass_inline(g)
ids = {}; nodes = []; blob = bytearray(); tensors = []
def vid(v):
    if v.debugName() not in ids: ids[v.debugName()] = len(ids)
    return ids[v.debugName()]
for i, v in enumerate(g.inputs()):
    vid(v)
for n in g.nodes():
    k = n.kind(); outs = [vid(o) for o in n.outputs()]
    if k == 'prim::Constant':
        o = n.output(); t = o.type()
        if n.hasAttribute('value'):
            kind = n.kindOf('value')
            if kind == 't':
                tv = n.t('value'); a = tv.detach().contiguous()
                dt = str(a.dtype).replace('torch.', '')
                arr = a.to(torch.float32).numpy() if a.is_floating_point() else a.to(torch.int64).numpy().astype(np.float32)
                off = len(blob) // 4; blob += arr.astype('<f4').tobytes()
                c = {'tensor': [off, list(a.shape), dt]}
            elif kind == 'i': c = {'int': n.i('value')}
            elif kind == 'f': c = {'float': n.f('value')}
            elif kind == 's': c = {'str': n.s('value')}
            elif kind == 'ival':
                iv = n.ival('value'); c = {'ival': repr(iv)}
            else: c = {'other': kind}
            if str(t) == 'bool' and 'int' in c: c = {'bool': bool(c['int'])}
            if str(t) == 'Device': c = {'str': str(n.output().toIValue())}
        else: c = {'none': True}
        nodes.append({'op': 'const', 'out': outs, 'c': c, 'ty': str(t)})
    else:
        nodes.append({'op': k, 'in': [vid(i) for i in n.inputs()], 'out': outs})
ret = [vid(v) for v in g.outputs()]
json.dump({'n_values': len(ids), 'inputs': [ids[v.debugName()] for v in g.inputs()], 'outputs': ret, 'nodes': nodes}, open(out + '.json', 'w'))
open(out + '.bin', 'wb').write(blob)
from collections import Counter
print(len(nodes), 'nodes', len(blob) / 4e6, 'M floats', Counter(json.dumps(x['c'])[:8] for x in nodes if x['op'] == 'const'))
print('unhandled consts', [x for x in nodes if x['op'] == 'const' and ('other' in x['c'] or 'ival' in x['c'])][:10])
