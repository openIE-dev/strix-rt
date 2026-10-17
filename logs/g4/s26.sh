# 26B-A4B session: llama.cpp reference, then our greedy + teacher-forced parity (stepwise)
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
bash logs/g4/ref.sh 26b $M 64
python3 -c "p=eval(open('logs/g4/26b_prompt_ids.txt').read());r=eval(open('logs/g4/26b_ref_vk_ids.txt').read());assert r[:len(p)]==p;open('logs/g4/26b_cont_ids.txt','w').write(str(r[len(p):]))"
bash logs/g4/par.sh 26b $M 64
