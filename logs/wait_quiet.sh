#!/bin/bash
# logs/wait_quiet.sh CMD..: poll (30 s) until no process uses >100% CPU (ps pcpu is lifetime-average; uses top 2 samples), then run CMD
busy() { top -bn2 -d2 -w 200 | awk "/^top -/{n++} n==2 && \$1 ~ /^[0-9]+$/ && \$9+0 > 100 {print \$1, \$9, \$12}"; }
while b=$(busy); [ -n "$b" ]; do echo "$(date +%T) busy: $b"; sleep 30; done
echo "$(date +%T) quiet"; exec "$@"
