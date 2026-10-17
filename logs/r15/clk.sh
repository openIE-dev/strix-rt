#!/bin/bash
# round 15: is the NPU slowdown under GPU load fabric/power? fuse512 + off512 at different clock settings (one lock hold)
cd ~/strix-rt; B=logs/r14/pfq; M=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; D=logs/r15; G=/sys/class/drm/card1/device
trap "echo auto | sudo tee $G/power_dpm_force_performance_level >/dev/null" EXIT
S="NPU_SPLIT=1 NPU_STATS=1 NPU_SPLIT_OPS=ffn_gate,ffn_up,ffn_down NPU_NN=5120x17408:4096,17408x5120:1536 PF_M=512 NOREF=1 REPS=5"
for cfg in high fclkmax sclk1100; do
  case $cfg in high) echo high | sudo tee $G/power_dpm_force_performance_level;;
   fclkmax) echo manual | sudo tee $G/power_dpm_force_performance_level; echo 2 | sudo tee $G/pp_dpm_sclk; echo 7 | sudo tee $G/pp_dpm_fclk; echo 2 | sudo tee $G/pp_dpm_mclk;;
   sclk1100) echo manual | sudo tee $G/power_dpm_force_performance_level; echo 1 | sudo tee $G/pp_dpm_sclk; echo 7 | sudo tee $G/pp_dpm_fclk;; esac
  echo "== $cfg: $(grep "\*" $G/pp_dpm_sclk) fclk $(grep "\*" $G/pp_dpm_fclk)" > $D/$cfg.log
  env $S $B $M 512 1 >> $D/$cfg.log 2>&1; env PF_M=512 NOREF=1 REPS=5 $B $M 512 1 2>&1 | grep prefill >> $D/$cfg.log
done
