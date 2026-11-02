#!/usr/bin/env python3
# cpubusy.py gate MAXCORES [MAXWAIT_S] : block until system-wide busy cores (excl. iowait) < MAXCORES for 2 consecutive 4 s samples (exit 1 on timeout)
# cpubusy.py snap : print "busy_jiffies total_jiffies"
import sys, time
def snap():
    f = open("/proc/stat").readline().split()[1:]; v = list(map(int, f)); idle = v[3] + v[4]; tot = sum(v[:8]); return tot - idle, tot
def cores(a, b): return 32.0 * (b[0] - a[0]) / max(b[1] - a[1], 1)
if sys.argv[1] == "snap": print(*snap()); sys.exit()
mx = float(sys.argv[2]); wait = float(sys.argv[3]) if len(sys.argv) > 3 else 3600; t0 = time.time(); ok = 0
while time.time() - t0 < wait:
    a = snap(); time.sleep(4); c = cores(a, snap())
    ok = ok + 1 if c < mx else 0
    if ok >= 2: print("%s quiet (%.2f busy cores)" % (time.strftime("%T"), c), flush=True); sys.exit(0)
    print("%s busy %.1f cores" % (time.strftime("%T"), c), flush=True)
sys.exit(1)
