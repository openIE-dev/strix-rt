#!/bin/sh
# Step 4b: per-layer times zero-copy (mmap'd safetensors pages) vs QI_ZC=0 (copied into driver allocations); + ask parity rerun
cd ~/strix-rt; B="target-cos/release/cos --golden $HOME/cosmos_ref/g256 --stage tower"
cat /sys/class/drm/card*/device/mem_info_vram_total /sys/class/drm/card*/device/mem_info_gtt_total 2>/dev/null
for v in "QI_ZC=0 COS_LTIME=1" "QI_ZC=0" "X=1"; do echo "== $v"; env $v timeout 70 $B 2>&1 | grep -E "loaded|step gpu|layer ms" | head -4; done
