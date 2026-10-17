# logs/g4/pfpar.sh <tag> <model>: batched-prefill parity (G4_PF=1, small chunk to cross chunk boundaries) + pp smoke
cd ~/strix-rt; T=$1; M=$2
P=@logs/g4/${T}_prompt_ids.txt; R=@logs/g4/${T}_${RC:-cont}_ids.txt
G4_PF=1 G4_PF_CHUNK=4 timeout 60 ./target/release/g4_run $M $P 64 $R
G4_PF=1 FORCED=1 timeout 60 ./target/release/g4_run $M $P 64 $R
G4_PF=1 timeout 60 ./target/release/g4_run $M rand:512 8
G4_PF=0 timeout 60 ./target/release/g4_run $M rand:512 8
