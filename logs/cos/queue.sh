#!/bin/bash
# cosmos3 GPU queue after reboot: one lock session per job, each under timeout < 300 s
cd ~/strix-rt; L=logs/cos
run() { tag=$1; log=$2; shift 2; echo "$(date +%T) start $tag" >> $L/queue.log
  GPU_LOCK_TAG="cosmos3 $tag" logs/wait_quiet.sh logs/gpu_lock.sh "$@" > $log 2>&1; echo "$(date +%T) end $tag rc=$?" >> $L/queue.log; }
run gen_dog $L/gen_dog.log timeout 280 target-cos/release/cos gen --prompt "A golden retriever runs across a sunny beach, waves rolling in behind it" --size 256x448 --frames 17 --steps 20 --out $L/gen_dog
run vae_gpu_ref $L/vae_gpu_ref.out timeout 290 $L/vae_gpu_ref.sh
run vattn_t $L/vattn_t.log $L/vattn_t.sh
run ask_ref $L/ask_ref.log $L/ask_ref.sh
run step4_ab $L/step4_ab.log timeout 290 $L/step4_ab.sh
run gen720 $L/gen720.log $L/gen720.sh
