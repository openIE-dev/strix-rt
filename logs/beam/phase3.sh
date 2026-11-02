#!/bin/bash
cd ~/strix-rt
while ! grep -q "EXIT" logs/beam/rq/q3.log 2>/dev/null; do sleep 30; done
bash logs/beam/evalq.sh
bash logs/beam/bench.sh
