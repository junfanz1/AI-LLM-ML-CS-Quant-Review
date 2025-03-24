# Contents

[Generative AI System Design Interview](https://www.amazon.com/Generative-AI-System-Design-Interview/dp/1736049143)

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [1. Introduction](#1-introduction)
   * [1.1 Transformer's Self-attention Architecture](#11-transformers-self-attention-architecture)
   * [1.2 Model Training Techniques for Large-scale models](#12-model-training-techniques-for-large-scale-models)
      + [1.2.1 Pipeline Parallelism (PP), inter-layer](#121-pipeline-parallelism-pp-inter-layer)
      + [1.2.2 Tensor Parallelsim (TP), intra-layer](#122-tensor-parallelsim-tp-intra-layer)
      + [1.2.3 Hybrid Parallelism](#123-hybrid-parallelism)
   * [1.3 Model Sampling Methods](#13-model-sampling-methods)
   * [1.4 Evaluation](#14-evaluation)
      + [1.4.1 Offline Evaluation](#141-offline-evaluation)
      + [1.4.2 Online Evaluation](#142-online-evaluation)
- [2. Gmail Smart Compose](#2-gmail-smart-compose)
   * [2.1 Positional encoding](#21-positional-encoding)
   * [2.2 Transformer Architecture](#22-transformer-architecture)
   * [2.3 Evaluation Metrics](#23-evaluation-metrics)
- [3. Google Translate](#3-google-translate)

<!-- TOC end -->

<!-- TOC --><a name="1-introduction"></a>
# 1. Introduction

<!-- TOC --><a name="11-transformers-self-attention-architecture"></a>
## 1.1 Transformer's Self-attention Architecture

- Self-attention: each element in the input sequence can focus on every other element, by converting inpupt embeddings for each token into 3 vectors: query Q, key K, value V.
- Attention score has scaling factor to prevent dot-product being too large (causing very small gradients during backpropagation).
- Softmax function ensures attention scores are normalized, summing to 1. Producing weighted sum of the value vectors V, where weights are determined by relevance of each input token indicated by attention scores.
- Multi-head attention: Instead of computing single set of Q, K, V, input is projected into multiple heads, each with its own learnable weight matrices: MultiHead(Q, K, V) = Concat(head1, head2, ...)W_O.
- Results of different heads are concatenated and then linearly transformed using output weight matrix W_O. Allowing model to jointly attend to info from different representation subspaces and capture richer dependencies.
- While Transformers are parallelizable due to lack of strict sequential dependencies, their self-attention has O(n^2) complexity, as self-attention requires calculation of attention scores between every pair of tokens in the sequence. So we have Group Attention and Flash Attention.

<!-- TOC --><a name="12-model-training-techniques-for-large-scale-models"></a>
## 1.2 Model Training Techniques for Large-scale models

- Gradient checkpointing: reduce memory usage during model training by saving only a selected subset of activations. During the backward pass, missing activations are recomputed. This reduces memory usage.
- Automatic mixed precision (AMP) training: automatically handles transition between half and single precision, optimizing where to use each precision type and applying scaling techniques to maintain numerical stability during training.
- Distributed training: Model(Tensor+Pipeline)/Data/Hybrid Parallelism

<!-- TOC --><a name="121-pipeline-parallelism-pp-inter-layer"></a>
### 1.2.1 Pipeline Parallelism (PP), inter-layer
- Model layers are split across multiple devices, computations in pipeline.
- Forward pass: each device forwards intermediate activation to next device in pipeline; Backward pass: reverse. Good for 
- Good for deep models, as it allows multiple devices to work concurrently, reducing idle time and improving training efficiency.

<!-- TOC --><a name="122-tensor-parallelsim-tp-intra-layer"></a>
### 1.2.2 Tensor Parallelsim (TP), intra-layer
- Each device handles a portion of computaions for that layer, and outputs are combined before moving to next layer. Different part of matrix processed in parallel across multiple devices.
- Good when single layer is too large to fit in memory.

<!-- TOC --><a name="123-hybrid-parallelism"></a>
### 1.2.3 Hybrid Parallelism
- ZeRO (Zero Redundancy Optimizer)
- FSDP (Fully Sharded Data Parallel)

<!-- TOC --><a name="13-model-sampling-methods"></a>
## 1.3 Model Sampling Methods

Text generation: greedy search, beam search (produce coherent and relevant text but with limited diversity, track multiple potential sequences of tokens simultaneously, each step select top-k most probable sequences), top-k sampling.

<!-- TOC --><a name="14-evaluation"></a>
## 1.4 Evaluation

<!-- TOC --><a name="141-offline-evaluation"></a>
### 1.4.1 Offline Evaluation

Evalute using pre-collected data without deploying to real-time environment. 

- Discriminative Tasks Metrics
  - Classification: Precision, Recall, F1, Accuracy, Confusion matrix
  - Regression: MSE, MAE, RMSE
  - Ranking: Precision@k, Recall@k, MRR, mAP, nDCG
- Generative Tasks Metrics
  - Text Generation: Perplexity, BLEU, METEOR, ROUGE, CIDEr
  - Image Generation: FID, IS, KID, SWD, PPL, LPIPS
  - Text-to-Video: FVD, CLIPScore, FID, LPIPS, KID

<!-- TOC --><a name="142-online-evaluation"></a>
### 1.4.2 Online Evaluation

How models perform after deployment to production.

- Click Through Rate (CTR)
- Conversion Rate
- Latency (Inference time)
- Engagement Rate
- Revenue Per User
- Churn Rate
- User Retention/Satisfaction
- Completion Rate 

<!-- TOC --><a name="2-gmail-smart-compose"></a>
# 2. Gmail Smart Compose

Input -> Triggering Service -> Phrase Generator (Beam Search, Long/Low-confidence Filtering) -> Post-processing -> Output.

<!-- TOC --><a name="21-positional-encoding"></a>
## 2.1 Positional encoding

Each token's position is encoded, so the model can understand coherent semantic meanings.

- Sin-cosine positional encoding
  - Pros: Fixed encoding don't add extra trainable parameters to the model, computationally efficient. Support for long sequences, as fixed methods can map any position into a representation, such flexibility can handle longer sequences beyond model's training data.
  - Cons: Predefined limits to their applicability to sequences below that maximum. Suboptimal performance, as fixed encodings may not capture positional relationships effectively.
- Learned positional encoding: Positional representations are learned during training process.
  - Pros: Optimal performance
  - Cons: Inefficiency, as it requires additional parameters to be learned during training. Lack of generalization, may overfit.
  
<!-- TOC --><a name="22-transformer-architecture"></a>
## 2.2 Transformer Architecture

Transformer architecture consists of a stack of blocks. Each block contains:
- Multi-head/Self attention: updates each embedding by using the attention mechanism, capturing relationships in sequence by allowing each embedding to attend to its preceding embeddings.
- Feed forward: 2 linear transformations, with ReLU activation in between, to each embedding in sequence independently.

Pretraining: Cross-entropy loss as loss function for next-token prediction, it measures difference between predicted probabilities and the correct token.

<!-- TOC --><a name="23-evaluation-metrics"></a>
## 2.3 Evaluation Metrics

Offline
- Perplexity: how accurately the model predicts exact sequence of tokens in data, exp(avg(negative log-likelihood of Prob(predicted | previous tokens in a sequence)). The lower the better.
- ExactMatch@N

Online
- Based on specific requirements: user engagement metrics, effectiveness metrics, latency, quality.

<!-- TOC --><a name="3-google-translate"></a>
# 3. Google Translate

## 3.1 Architecture

- Encoder: Input Sequence -> Text Embedding -> Positional Encoding -> Transformer ([Self Attention (MHA), Normalization, Feed Forward, Normalization] * N) -> Output Sequence
- Decoder: Previously generated tokens -> Positional Encoding -> Transformer ([Self Attention (MHA), Normalization, Cross Attention (MHA), Feed Forward, Normalization] * N) -> Prediction head (linear layer + softmax to convert Transformer's output to probabilities over vocabulary) -> Predicted next token 

Difference: Encoder, Decoder 
- Cross-attention layer: Each token in decoder can attend to all embeddings in encoder, can integrate info from input sequence during output.
- Self-attention: Encoder, each token attends to all other tokens, to understand entire sequence. Decoder, each token is restricted to only tokens come before.

## 3.2 Training 

Next-token prediction is not ideal for encoder-decoder pretraining, because it's unsupervised training and decoder prediction will cause cheating. So we use masked language modeling (MLM).
- Randomly select a subset of tokens in input, and mask them.
- Feed masked sequence to encoder to understand context
- Feed decoder with the same input, but none of tokens are mased and sequence has been shift one position to the right by insertion of a start token.
- Decoder predicts next token for each position in sequence. Each prediction uses all previous input tokens from encoder.
- Calculate cross-entropy loss over predicted probabilities.

Fine-tuning stage is supervised.

Sampling with beam search for accuracy and consistency.

## 3.3 Evaluation

Offline evaluation metrics
- BLEU (BiLingual Evaluation Understudy): count the ratio of matches, with brevity penalty, n-grams precision, weight for different n-gram precisions
- ROUGE (Recall-Oriented Understudy for Gisting Evaluation): recall = # matching n-grams / total # n-grams in reference. Lack of contextual understanding.
- METEOR (Metric for Evaluation of Translation with Explicit ORdering): combines precision, recall using weighted harmonic mean.
  - Pros: Semantic understanding, balanced evaluation, correlation with human judgements
  - Cons: Computational complexity, resource dependence.

Online evaluation metrics
- User feedback/engagements

# 4. ChatGPT: Personal Assistant Chatbot 















