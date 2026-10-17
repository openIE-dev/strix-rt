cd ~/strix-rt
for m in kev08 laya kev4b kev9q8 kev9bf kev27; do GPU_LOCK_TAG="dm $m" logs/gpu_lock.sh tools/comp/perf_high.sh bash logs/dm_session.sh $m > logs/dm/s_$m.log 2>&1; done
echo done > logs/dm/all.done
