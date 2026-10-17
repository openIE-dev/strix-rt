# CPU (and ROCm) llama.cpp references: logs/g4/ref2.sh <tag> <model> [n]  (run under logs/gpu_lock.sh)
cd ~/strix-rt; B=~/llama-latest/build-vulkan/bin; H=~/llama-latest/build-rocm713/bin; T=$1; M=$2; N=${3:-64}
PR="The history of the Roman Empire can be summarized as follows:"
GGML_VK_VISIBLE_DEVICES=none timeout 200 $B/llama-simple -m $M -ngl 0 -n $N "$PR" > logs/g4/${T}_ref_cpu.txt 2> logs/g4/${T}_ref_cpu.err
$B/llama-tokenize -m $M -f logs/g4/${T}_ref_cpu.txt --ids --log-disable 2>/dev/null | tail -1 > logs/g4/${T}_ref_cpu_ids.txt
python3 -c "p=eval(open('logs/g4/${T}_prompt_ids.txt').read());r=eval(open('logs/g4/${T}_ref_cpu_ids.txt').read());assert r[:len(p)]==p;open('logs/g4/${T}_cpu_cont_ids.txt','w').write(str(r[len(p):]))"
cat logs/g4/${T}_ref_cpu.txt; echo
