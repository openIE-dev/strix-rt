#!/usr/bin/env python3
"""Stand-alone socket-energy recorder for AMD APUs on Linux, using only standard sysfs files.

  tools/energy/record.py record [--hz N] [--idle S] [--json F] [--csv F] -- CMD [ARGS...]
      Run CMD, sample power at N Hz (default 20) while it runs, integrate to joules (trapezoid rule), write JSON/CSV.
      --idle S sleeps S seconds before starting CMD (and samples during it), to settle the readings.
  tools/energy/record.py snapshot
      Print one JSON line with the current readings.

Channels (each one is only reported if its sysfs file exists and can be read):
  socket       amdgpu gpu_metrics v3.x `average_socket_power` (mW, SMU firmware average), from
               /sys/class/drm/cardN/device/gpu_metrics.  This is the APU *socket* power: the CPU cores, GPU, memory
               controller and fabric of the SoC. It is NOT wall power: DRAM devices, VRM losses, fans, SSD and PSU are not in it.
  gfx          gpu_metrics v3.x `average_gfx_power` (mW): the GPU core rail.
  all_core     gpu_metrics v3.x `average_all_core_power` (mW): all CPU cores.
  hwmon_ppt    /sys/class/drm/cardN/device/hwmon/hwmonM/power1_average (uW, label "PPT"): another view of package power.
  rapl_package /sys/class/powercap/intel-rapl:0/energy_uj (uJ counter; on AMD Zen this is the core package, not the GPU).
               Often root-only; read directly as an energy counter instead of integrating power.
The gpu_metrics byte offsets below follow `struct gpu_metrics_v3_0` in the Linux kernel
(drivers/gpu/drm/amd/include/kgd_pp_interface.h). Check them against your kernel if the header is not v3.

CSV columns (--csv): t_s,socket_w,gfx_w,all_core_w,ppt_w (t_s counts from the recorder start; ppt_w is hwmon PPT).
JSON output: {"duration_s", "hz", "samples", "breakdown": {"channels": [{"fabric", "joules", "avg_w", "source"}, ...]}}.
tools/comp/energy.sh reads the "socket" and "gfx" joules from it.

Method used throughout this repo: run the same workload with a short and a long token count, take
(J_long - J_short) / (N_long - N_short). Model load, prefill and warm-up are in both runs and cancel.
"""
import glob, json, os, struct, subprocess, sys, threading, time

def card_dir():
    env = os.environ.get("AMDGPU_CARD")
    if env: return f"/sys/class/drm/{env}/device"
    for d in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
        if os.path.exists(d + "/gpu_metrics"): return d
    return None

DEV = card_dir()
HWMON = (glob.glob(DEV + "/hwmon/hwmon*/power1_average") or [None])[0] if DEV else None
RAPL = "/sys/class/powercap/intel-rapl:0/energy_uj"

def read_gm():
    """socket and gfx power in W from gpu_metrics v3.x, or {}"""
    try:
        b = open(DEV + "/gpu_metrics", "rb").read()
    except Exception:
        return {}
    if len(b) < 128 or b[2] != 3: return {}
    sock, gfx = struct.unpack_from("<I", b, 112)[0], struct.unpack_from("<I", b, 124)[0]
    r = {}
    if sock != 0xFFFFFFFF: r["socket"] = sock / 1000.0
    if gfx != 0xFFFFFFFF: r["gfx"] = gfx / 1000.0
    if len(b) >= 136:
        core = struct.unpack_from("<I", b, 132)[0]
        if core != 0xFFFFFFFF: r["all_core"] = core / 1000.0
    return r

def read_hwmon():
    try: return {"hwmon_ppt": int(open(HWMON).read()) / 1e6}
    except Exception: return {}

def read_rapl_uj():
    try: return int(open(RAPL).read())
    except Exception: return None

def dpm_cur(name):
    try:
        for l in open(f"{DEV}/{name}"):
            if l.strip().endswith("*"): return float(l.split(":")[1].strip().rstrip("*").strip().lower().replace("mhz", ""))
    except Exception: pass
    return None

def sample():
    s = {"t": time.monotonic()}; s.update(read_gm()); s.update(read_hwmon()); return s

GM_CLOCKS = (("gfxclk_mhz", 174), ("fclk_mhz", 182), ("uclk_mhz", 186))  # gpu_metrics_v3_0 average_*_frequency (u16 MHz)

def snapshot():
    """one JSON line {"gm": {socket_w, gfx_w, all_core_w, gfxclk_mhz, fclk_mhz, uclk_mhz, gfx_activity_pct, hwmon_ppt_w}}"""
    p = sample(); gm = {}
    for k in ("socket", "gfx", "all_core"):
        if k in p: gm[k + "_w"] = p[k]
    try:
        b = open(DEV + "/gpu_metrics", "rb").read()
        if b[2] == 3 and len(b) >= 188:
            for k, off in GM_CLOCKS:
                v = struct.unpack_from("<H", b, off)[0]
                if v != 0xFFFF: gm[k] = float(v)
            act = struct.unpack_from("<H", b, 42)[0]  # average_gfx_activity, percent
            if act != 0xFFFF: gm["gfx_activity_pct"] = float(act)
    except Exception: pass
    for k, f in (("gfxclk_mhz", "pp_dpm_sclk"), ("fclk_mhz", "pp_dpm_fclk"), ("uclk_mhz", "pp_dpm_mclk")):
        if k not in gm:
            v = dpm_cur(f)
            if v is not None: gm[k] = v
    if "gfx_activity_pct" not in gm:
        try: gm["gfx_activity_pct"] = float(open(DEV + "/gpu_busy_percent").read())
        except Exception: pass
    if "hwmon_ppt" in p: gm["hwmon_ppt_w"] = p["hwmon_ppt"]
    print(json.dumps({"gm": gm}))

def record(argv):
    hz, idle, jf, cf = 20.0, 0.0, None, None
    while argv and argv[0] != "--":
        a = argv.pop(0)
        if a == "--hz": hz = float(argv.pop(0))
        elif a == "--idle": idle = float(argv.pop(0))
        elif a == "--json": jf = argv.pop(0)
        elif a == "--csv": cf = argv.pop(0)
        else: sys.exit(f"unknown option {a}")
    if not argv: sys.exit("usage: record.py record [--hz N] [--idle S] [--json F] [--csv F] -- CMD...")
    cmd = argv[1:]
    samples, stop = [], threading.Event()
    def loop():
        dt = 1.0 / hz
        while not stop.is_set():
            samples.append(sample()); stop.wait(dt)
    r0 = read_rapl_uj(); t0 = time.monotonic()
    th = threading.Thread(target=loop, daemon=True); th.start()
    if idle > 0: time.sleep(idle)
    rc = subprocess.call(cmd)
    stop.set(); th.join(); samples.append(sample())
    t1 = time.monotonic(); r1 = read_rapl_uj()
    chans = []
    for k in ("socket", "gfx", "all_core", "hwmon_ppt"):
        pts = [(s["t"], s[k]) for s in samples if k in s]
        if len(pts) < 2: continue
        j = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(pts, pts[1:]))
        src = {"socket": "gpu_metrics average_socket_power", "gfx": "gpu_metrics average_gfx_power",
               "all_core": "gpu_metrics average_all_core_power", "hwmon_ppt": HWMON}[k]
        chans.append({"fabric": k, "joules": j, "avg_w": j / max(pts[-1][0] - pts[0][0], 1e-9), "source": src})
    if r0 is not None and r1 is not None and r1 >= r0:
        chans.append({"fabric": "rapl_package", "joules": (r1 - r0) / 1e6, "avg_w": (r1 - r0) / 1e6 / (t1 - t0), "source": RAPL})
    out = {"duration_s": t1 - t0, "hz": hz, "samples": len(samples), "rc": rc, "breakdown": {"channels": chans}}
    if jf:
        with open(jf, "w") as f: json.dump(out, f, indent=1)
    if cf:
        with open(cf, "w") as f:
            f.write("t_s,socket_w,gfx_w,all_core_w,ppt_w\n")
            for s in samples:
                f.write(f"{s['t'] - t0:.4f}," + ",".join(str(s.get(k, "")) for k in ("socket", "gfx", "all_core", "hwmon_ppt")) + "\n")
    print("energy: " + ", ".join(f"{c['fabric']} {c['joules']:.1f} J" for c in chans) + f" over {t1 - t0:.2f} s", file=sys.stderr)
    return rc

if __name__ == "__main__":
    if DEV is None: sys.exit("no amdgpu device with gpu_metrics found (set AMDGPU_CARD=cardN)")
    a = sys.argv[1:]
    if a[:1] == ["snapshot"]: snapshot()
    elif a[:1] == ["record"]: sys.exit(record(a[1:]))
    else: sys.exit(__doc__)
