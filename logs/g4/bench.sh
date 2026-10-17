# logs/g4/bench.sh <tag> <model> <ours|llama|energy>: one GPU session (run via wait_quiet + gpu_lock + perf_high)
cd ~/strix-rt; T=$1; M=$2; B=~/llama-latest/build-vulkan/bin; P=@logs/g4/e4b_prompt_ids.txt; O=logs/g4/bench; mkdir -p $O
case $3 in
ours)
  for L in 512 2048; do for r in 1 2 3; do G4_PF=1 MAX_CTX=4096 timeout 60 ./target/release/g4_run $M rand:$L 2 | grep prompt | sed "s/^/pp$L /"; done; done
  G4_PF=1 RUNS=3 MAX_CTX=4096 timeout 60 ./target/release/g4_run $M $P 129 | grep decode | sed "s/^/tg128 /" ;;
llama)
  timeout 280 $B/llama-bench -m $M -ngl 999 -fa on -b 2048 -ub 512 -p 512,2048 -n 128 -r 3 -o json > $O/${T}_llama_vk.json 2> $O/${T}_llama_vk.err; echo "rc $?"
  python3 -c "import json,sys;[print(r['n_prompt'],r['n_gen'],round(r['avg_ts'],2),round(r['stddev_ts'],2)) for r in json.load(open(sys.argv[1]))]" $O/${T}_llama_vk.json ;;
energy)
  tools/comp/energy.sh g4_${T}_ours "G4_PF=1 MAX_CTX=1024 ./target/release/g4_run $M $P @N@"
  tools/comp/energy.sh g4_${T}_llama_vk "$B/llama-bench -m $M -ngl 999 -fa on -p 0 -n @N@ -r 1" ;;
esac
