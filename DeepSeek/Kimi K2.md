# Kimi K2

Model Architecture
- Based on DeepSeek V3
- 1T parameters 
- 256 -> 384 experts, but # of activated experts are the same as DeepSeek
  - SwiGLU FFN: matrix parameters are separated to 3 weights (W1, W2 & V)
  - #parameters in FFN: FFN Dense = FFN MoE = 396M
- 128 -> 64 attention heads
  - LoRA’s Kv_latent_dim  = 512, very small, *32 num_head = 16384
  - Scaling Law: sparsity scaling law, # of activated experts 8, * sparsity 48. Attention heads = 64

Training Recipe
- MuonClip optimizer = Muon optimizer + QK-Clip (to scale-lower the weights, due to exploding attention logits during training caused by Muon optimizer)
- WSD (warm-up, stable, decay) learning rate scheduler: no need to converge too much, which is not good for model training afterwards
- Context window (YARN): batch size keep const at 67M tokens (memory constraint), learning rate decayed from 2e-5 to 7e-6 (stable training), gradually increase context window to 128k with YARN.

Algorithm
- Pretraining: knowledge data rephrasing for synthetic data generation, which is a rewrite model (LLM for data processing) in an auto-regressive way.
- SFT: data synthesis pipeline for agentic intelligence
- RL: with verified reward, self-critique rubric reward, with rubric anchors (AReal) / rubric as rewards.

System Design
- Training pipeline: based on Megatron, selective recomputation (LayerNorm, SwiGLU), FP8 storage for intensive application, activation CPU offload
- RL infra: parameter synchronization (train -> checkpoint -> inference, separate architecture)

