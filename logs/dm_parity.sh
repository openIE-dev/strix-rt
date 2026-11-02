cd ~/strix-rt; M=$MODELS/decision; O=logs/dm/parity
r(){ n=$1; shift; GPU_LOCK_TAG="dm parity $n" logs/gpu_lock.sh "$@" > $O/$n.log 2>&1; }
r k08 ./target/release/kev_parity $M/kev-0.8b-bf16.gguf $M/kev-0.8b-head.bin crates/kev/ref/golden.bin 248061
r k4b ./target/release/kev_parity $M/kev-4b-bf16.gguf $M/kev-4b-head.bin crates/kev/ref/4b/golden.bin 248061
r k9q8 ./target/release/kev_parity $M/kev-9b-q8_0.gguf $M/kev-9b-head.bin crates/kev/ref/9b/golden.bin 248061
r k9bf ./target/release/kev_parity $M/kev-9b-bf16.gguf $M/kev-9b-head.bin crates/kev/ref/9b/golden.bin 248061
r k27 ./target/release/kev_parity $M/kev-27b-q8_0.gguf $M/kev-27b-head.bin crates/kev/ref/27b/golden.bin 248061
r laya ./target/release/laya_check $M/laya-f16.gguf $M/laya/tokenizer/tokenizer.json $M/laya/rl_agent_config.json crates/laya/ref/golden.json
r layap ./target/release/laya_parity $MODELS/decision/laya-f16.gguf crates/laya/ref/golden.bin
echo done > $O/done
