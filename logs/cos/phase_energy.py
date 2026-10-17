# phase_energy.py CSV T0_EPOCH LOG: integrate the energy recorder power columns over the "PHASE name start end" windows (epoch s) in LOG.
# t_s in the CSV counts from the recorder start (T0_EPOCH, recorded by the wrapper just before it).
import sys, csv
rows = list(csv.DictReader(open(sys.argv[1]))); t0 = float(sys.argv[2])
ph = [l.split() for l in open(sys.argv[3]) if l.startswith("PHASE ")]
def integ(a, b, col):
    e = 0.0
    for r0, r1 in zip(rows, rows[1:]):
        x0, x1 = float(r0["t_s"]), float(r1["t_s"]); lo, hi = max(x0, a), min(x1, b)
        if hi > lo: e += float(r1[col]) * (hi - lo)
    return e
tot = {}
for _, name, s, e in ph:
    a, b = float(s) - t0, float(e) - t0
    j = {c: integ(a, b, c) for c in ("socket_w", "gfx_w", "all_core_w", "ppt_w")}
    print(f"{name:10} {b - a:8.2f} s  socket {j['socket_w']:9.1f} J ({j['socket_w'] / max(b - a, 1e-9):6.1f} W)  gfx {j['gfx_w']:8.1f} J  cpu {j['all_core_w']:7.1f} J")
    for k, v in j.items(): tot[k] = tot.get(k, 0) + v
    tot["s"] = tot.get("s", 0) + b - a
print(f"{'total':10} {tot.get('s', 0):8.2f} s  socket {tot.get('socket_w', 0):9.1f} J")
