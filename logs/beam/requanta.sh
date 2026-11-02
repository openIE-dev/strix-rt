#!/bin/bash
cd ~/strix-rt; while [ ! -e logs/beam/act/done ]; do sleep 20; done
mkdir -p logs/beam/rq; ~/venvs/ml/bin/python tools/beam_requant.py 2 $MODELS/new/beam/act/q2 --act logs/beam/act --device cuda > logs/beam/rq/act_q2.log 2>&1; echo "EXIT $?" >> logs/beam/rq/act_q2.log
