# WER / parity table: python wer.py ours.jsonl ref_times.json refs.tsv
import sys, json, jiwer, re
ours = {j["name"]: j for j in map(json.loads, open(sys.argv[1]))}
ref = json.load(open(sys.argv[2]))
gt = dict(l.rstrip("\n").split("\t", 1) for l in open(sys.argv[3]) if "\t" in l)
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", s.lower())).strip()
rows, O, Hh, P, G = [], [], [], [], []
print(f"{'clip':8} {'sec':>6} {'ours ms':>8} {'RTF':>7} {'HF RTF':>7} {'Photon RTF':>10} {'=HF':>4} {'=Photon':>7} {'WER(ours,HF)':>12}")
for n, o in ours.items():
    r = ref[n]; o_t, h_t, p_t = o["text"], r["hf_text"], r["photon_text"]
    print(f"{n:8} {o['sec']:6.2f} {o['total_s']*1e3:8.1f} {o['rtf']:7.4f} {r['hf_s']/r['sec']:7.4f} {r['photon_s']/r['sec']:10.4f} {str(o_t == h_t):>4} {str(o_t == p_t):>7} {jiwer.wer(norm(h_t), norm(o_t)):12.4f}")
    O.append(norm(o_t)); Hh.append(norm(h_t)); P.append(norm(p_t)); G.append(norm(gt.get(n, h_t)))
sec = sum(o["sec"] for o in ours.values())
print(f"total {sec:.1f}s audio: ours {sum(o['total_s'] for o in ours.values()):.3f}s (RTF {sum(o['total_s'] for o in ours.values())/sec:.4f}), "
      f"HF {sum(r['hf_s'] for r in ref.values()):.3f}s (RTF {sum(r['hf_s'] for r in ref.values())/sec:.4f}), Photon {sum(r['photon_s'] for r in ref.values()):.3f}s (RTF {sum(r['photon_s'] for r in ref.values())/sec:.4f})")
print(f"corpus WER vs ground truth: ours {jiwer.wer(G, O):.4f}  HF {jiwer.wer(G, Hh):.4f}  Photon {jiwer.wer(G, P):.4f};  WER(ours vs HF) {jiwer.wer(Hh, O):.4f}  WER(ours vs Photon) {jiwer.wer(P, O):.4f}")
