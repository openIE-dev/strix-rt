#!/bin/bash
# llama.cpp HIP build against ROCm 7.13 (hipBLASLt via ROCBLAS_USE_HIPBLASLT=1)
source ~/strix-rt/tools/comp/rocm713.env; exec ~/strix-rt/tools/comp/lb.sh "$@"
