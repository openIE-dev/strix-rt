# logs/g4/par.sh <tag> <model> [n]: greedy parity + teacher-forced vs llama.cpp Vulkan reference (run under gpu_lock)
cd ~/strix-rt; T=$1; M=$2; N=${3:-64}
P=@logs/g4/${T}_prompt_ids.txt; R=@logs/g4/${T}_${RC:-cont}_ids.txt
timeout 120 ./target/release/g4_run $M $P $N $R
FORCED=1 timeout 120 ./target/release/g4_run $M $P $N $R
