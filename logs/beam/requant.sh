#!/bin/bash
cd ~/strix-rt; mkdir -p logs/beam/rq
for b in 2 3; do ~/venvs/ml/bin/python tools/beam_requant.py $b $MODELS/new/beam/q$b --device cuda > logs/beam/rq/q$b.log 2>&1; echo "EXIT $?" >> logs/beam/rq/q$b.log; done
