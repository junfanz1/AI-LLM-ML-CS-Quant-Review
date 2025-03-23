# Contents

[Generative AI System Design Interview](https://www.amazon.com/Generative-AI-System-Design-Interview/dp/1736049143)

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [Introduction](#introduction)
   * [Transformer's Self-attention Architecture](#transformers-self-attention-architecture)
   * [Model Training Techniques for Large-scale models](#model-training-techniques-for-large-scale-models)
      + [Pipeline Parallelism (PP), inter-layer](#pipeline-parallelism-pp-inter-layer)
      + [Tensor Parallelsim (TP), intra-layer](#tensor-parallelsim-tp-intra-layer)
      + [Hybrid Parallelism](#hybrid-parallelism)
   * [Model Sampling Methods](#model-sampling-methods)
   * [Evaluation](#evaluation)
      + [Offline Evaluation](#offline-evaluation)
      + [Online Evaluation](#online-evaluation)
- [Gmail Smart Compose](#gmail-smart-compose)

<!-- TOC end -->

<!-- TOC --><a name="introduction"></a>
# Introduction

<!-- TOC --><a name="transformers-self-attention-architecture"></a>
## Transformer's Self-attention Architecture

- Self-attention: each element in the input sequence can focus on every other element, by converting inpupt embeddings for each token into 3 vectors: query Q, key K, value V.
- Attention score has scaling factor to prevent dot-product being too large (causing very small gradients during backpropagation).
- Softmax function ensures attention scores are normalized, summing to 1. Producing weighted sum of the value vectors V, where weights are determined by relevance of each input token indicated by attention scores.
- Multi-head attention: Instead of computing single set of Q, K, V, input is projected into multiple heads, each with its own learnable weight matrices: MultiHead(Q, K, V) = Concat(head1, head2, ...)W_O.
- Results of different heads are concatenated and then linearly transformed using output weight matrix W_O. Allowing model to jointly attend to info from different representation subspaces and capture richer dependencies. 

<!-- TOC --><a name="model-training-techniques-for-large-scale-models"></a>
## Model Training Techniques for Large-scale models

- Gradient checkpointing: reduce memory usage during model training by saving only a selected subset of activations. During the backward pass, missing activations are recomputed. This reduces memory usage.
- Automatic mixed precision (AMP) training: automatically handles transition between half and single precision, optimizing where to use each precision type and applying scaling techniques to maintain numerical stability during training.
- Distributed training: Model(Tensor+Pipeline)/Data/Hybrid Parallelism

<!-- TOC --><a name="pipeline-parallelism-pp-inter-layer"></a>
### Pipeline Parallelism (PP), inter-layer
- Model layers are split across multiple devices, computations in pipeline.
- Forward pass: each device forwards intermediate activation to next device in pipeline; Backward pass: reverse. Good for 
- Good for deep models, as it allows multiple devices to work concurrently, reducing idle time and improving training efficiency.

<!-- TOC --><a name="tensor-parallelsim-tp-intra-layer"></a>
### Tensor Parallelsim (TP), intra-layer
- Each device handles a portion of computaions for that layer, and outputs are combined before moving to next layer. Different part of matrix processed in parallel across multiple devices.
- Good when single layer is too large to fit in memory.

<!-- TOC --><a name="hybrid-parallelism"></a>
### Hybrid Parallelism
- ZeRO (Zero Redundancy Optimizer)
- FSDP (Fully Sharded Data Parallel)

<!-- TOC --><a name="model-sampling-methods"></a>
## Model Sampling Methods

Text generation: greedy search, beam search (produce coherent and relevant text but with limited diversity), top-k sampling.

<!-- TOC --><a name="evaluation"></a>
## Evaluation

<!-- TOC --><a name="offline-evaluation"></a>
### Offline Evaluation

Evalute using pre-collected data without deploying to real-time environment. 

- Discriminative Tasks Metrics
  - Classification: Precision, Recall, F1, Accuracy, Confusion matrix
  - Regression: MSE, MAE, RMSE
  - Ranking: Precision@k, Recall@k, MRR, mAP, nDCG
- Generative Tasks Metrics
  - Text Generation: Perplexity, BLEU, METEOR, ROUGE, CIDEr
  - Image Generation: FID, IS, KID, SWD, PPL, LPIPS
  - Text-to-Video: FVD, CLIPScore, FID, LPIPS, KID

<!-- TOC --><a name="online-evaluation"></a>
### Online Evaluation

How models perform after deployment to production.

- Click Through Rate (CTR)
- Conversion Rate
- Latency (Inference time)
- Engagement Rate
- Revenue Per User
- Churn Rate
- User Retention/Satisfaction
- Completion Rate 

<!-- TOC --><a name="gmail-smart-compose"></a>
# Gmail Smart Compose

























