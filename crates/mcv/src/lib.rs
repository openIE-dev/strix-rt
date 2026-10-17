//! MiniCPM-V-4.7 (35B-A3B, GGUF Q4_K_M + mmproj) on strix-rt: SigLIP vision tower with the 4.7 window-attention
//! merger and DownsampleMLP (vision.rs), qwen35moe text tower (llm.rs).
pub mod llm; pub mod prefill;
pub mod vision;
