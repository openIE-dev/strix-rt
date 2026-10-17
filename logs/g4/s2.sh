cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf; E=$MODELS/new/gemma4/e4b/gemma-4-E4B-it-Q8_0.gguf
bash logs/g4/ref2.sh 26b $M 64
RC=cpu_cont bash logs/g4/par.sh 26b $M 64
bash logs/g4/pfpar.sh e4b $E
