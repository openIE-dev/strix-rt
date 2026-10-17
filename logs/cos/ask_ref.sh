#!/bin/bash
# Step 6 parity: ViT + understanding vs logs/cos/vit_ref.py dump (~/cosmos_ref/vit1, CPU bf16 HF Qwen3-VL with the und weights).
cd ~/strix-rt
timeout 280 target-cos/release/cos ask --ref ~/cosmos_ref/vit1 --question "Describe this image in one sentence." --new 32
