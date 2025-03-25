[Book: Generative AI System Design Interview](https://www.amazon.com/Generative-AI-System-Design-Interview/dp/1736049143)


# Contents

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [1. Introduction](#1-introduction)
   * [1.1 Transformer](#11-transformer)
   * [1.2 Training](#12-training)
   * [1.3 Parallelism](#13-parallelism)
   * [1.4 Sampling](#14-sampling)
   * [1.5 Evaluation](#15-evaluation)
- [2. Gmail Smart Compose](#2-gmail-smart-compose)
   * [2.1 Positional encoding](#21-positional-encoding)
   * [2.2 Transformer Architecture](#22-transformer-architecture)
   * [2.3 Evaluation Metrics](#23-evaluation-metrics)
- [3. Google Translate](#3-google-translate)
   * [3.1 Architecture](#31-architecture)
   * [3.2 Training ](#32-training)
   * [3.3 Evaluation](#33-evaluation)
- [4. ChatGPT: Personal Assistant Chatbot ](#4-chatgpt-personal-assistant-chatbot)
   * [4.1 Positional Encoding](#41-positional-encoding)
   * [4.2 Training](#42-training)
   * [4.3 Sampling ](#43-sampling)
   * [4.4 ML System Design Pipeline ](#44-ml-system-design-pipeline)
- [5. Image Captioning (Image2Text)](#5-image-captioning-image2text)
   * [5.1 Image Encoder ](#51-image-encoder)
   * [5.2 Pipeline](#52-pipeline)
- [6. RAG ](#6-rag)
   * [6.1 Model Architecture](#61-model-architecture)
   * [6.2 Sampling](#62-sampling)
   * [6.3 Evaluation](#63-evaluation)
- [7. Realistic Face Generation](#7-realistic-face-generation)
   * [7.1 Choose ML](#71-choose-ml)
      + [7.1.1 VAE](#711-vae)
      + [7.1.2 GAN ](#712-gan)
      + [7.1.3 Autoregressive (DALL-E)](#713-autoregressive-dall-e)
      + [7.1.4 Diffusion](#714-diffusion)
   * [7.2 GAN Architecture ](#72-gan-architecture)
      + [7.2.1 Generator](#721-generator)
      + [7.2.2 Discriminator ](#722-discriminator)
   * [7.3 Adversarial Training ](#73-adversarial-training)
      + [7.3.1 Training Challenges](#731-training-challenges)
   * [7.4 Sampling](#74-sampling)
   * [7.5 Evaluation](#75-evaluation)

<!-- TOC end -->

<!-- TOC --><a name="1-introduction"></a>
# 1. Introduction

<!-- TOC --><a name="11-transformer"></a>
## 1.1 Transformer

Transformer's Self-attention Architecture
- Self-attention: each element in the input sequence can focus on every other element, by converting inpupt embeddings for each token into 3 vectors: query Q, key K, value V.
- Attention score has scaling factor to prevent dot-product being too large (causing very small gradients during backpropagation).
- Softmax function ensures attention scores are normalized, summing to 1. Producing weighted sum of the value vectors V, where weights are determined by relevance of each input token indicated by attention scores.
- Multi-head attention: Instead of computing single set of Q, K, V, input is projected into multiple heads, each with its own learnable weight matrices: MultiHead(Q, K, V) = Concat(head1, head2, ...)W_O.
- Results of different heads are concatenated and then linearly transformed using output weight matrix W_O. Allowing model to jointly attend to info from different representation subspaces and capture richer dependencies.
- While Transformers are parallelizable due to lack of strict sequential dependencies, their self-attention has O(n^2) complexity, as self-attention requires calculation of attention scores between every pair of tokens in the sequence. So we have Group Attention and Flash Attention.

<!-- TOC --><a name="12-training"></a>
## 1.2 Training

Model Training Techniques for Large-scale models
- Gradient checkpointing: reduce memory usage during model training by saving only a selected subset of activations. During the backward pass, missing activations are recomputed. This reduces memory usage.
- Automatic mixed precision (AMP) training: automatically handles transition between half and single precision, optimizing where to use each precision type and applying scaling techniques to maintain numerical stability during training.
- Distributed training: Model(Tensor+Pipeline)/Data/Hybrid Parallelism

<!-- TOC --><a name="13-parallelism"></a>
## 1.3 Parallelism

Pipeline Parallelism (PP), inter-layer
- Model layers are split across multiple devices, computations in pipeline.
- Forward pass: each device forwards intermediate activation to next device in pipeline; Backward pass: reverse. Good for 
- Good for deep models, as it allows multiple devices to work concurrently, reducing idle time and improving training efficiency.

Tensor Parallelsim (TP), intra-layer
- Each device handles a portion of computaions for that layer, and outputs are combined before moving to next layer. Different part of matrix processed in parallel across multiple devices.
- Good when single layer is too large to fit in memory.

Hybrid Parallelism
- ZeRO (Zero Redundancy Optimizer)
- FSDP (Fully Sharded Data Parallel)

<!-- TOC --><a name="14-sampling"></a>
## 1.4 Sampling

Deterministic
- greedy search
- beam search
  - produce coherent and relevant text but with limited diversity (not open-ended)
  - improves greedy search by considering multiple sequences simultaneously, each step tracking top-k most probable sequences

Stochastic
- Top-k sampling: balance coherence and diversity by picking top-k tokens, but predicted token prob can be sharply or evenly distribued.
- Top-p (nucleus) sampling: dynamically adjust number of tokens considered based on combined probabilities, choose smallest possible set of tokens whose cumulative prob > probability p. More adaptive and flexible than top-k sampling (selecting fixed number of tokens).

<!-- TOC --><a name="15-evaluation"></a>
## 1.5 Evaluation

Offline Evaluation: Evalute using pre-collected data without deploying to real-time environment. 
- Discriminative Tasks Metrics
  - Classification: Precision, Recall, F1, Accuracy, Confusion matrix
  - Regression: MSE, MAE, RMSE
  - Ranking: Precision@k, Recall@k, MRR, mAP, nDCG
- Generative Tasks Metrics
  - Text Generation: Perplexity, BLEU, METEOR, ROUGE, CIDEr
  - Image Generation: FID, IS, KID, SWD, PPL, LPIPS
  - Text-to-Video: FVD, CLIPScore, FID, LPIPS, KID

Online Evaluation: How models perform after deployment to production.
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

<!-- TOC --><a name="31-architecture"></a>
## 3.1 Architecture

- Encoder: Input Sequence -> Text Embedding -> Positional Encoding -> Transformer ([Self Attention (MHA), Normalization, Feed Forward, Normalization] * N) -> Output Sequence
- Decoder: Previously generated tokens -> Positional Encoding -> Transformer ([Self Attention (MHA), Normalization, Cross Attention (MHA), Feed Forward, Normalization] * N) -> Prediction head (linear layer + softmax to convert Transformer's output to probabilities over vocabulary) -> Predicted next token 

Difference: Encoder, Decoder 
- Cross-attention layer: Each token in decoder can attend to all embeddings in encoder, can integrate info from input sequence during output.
- Self-attention: Encoder, each token attends to all other tokens, to understand entire sequence. Decoder, each token is restricted to only tokens come before.

<!-- TOC --><a name="32-training"></a>
## 3.2 Training 

Next-token prediction is not ideal for encoder-decoder pretraining, because it's unsupervised training and decoder prediction will cause cheating. So we use masked language modeling (MLM).
- Randomly select a subset of tokens in input, and mask them.
- Feed masked sequence to encoder to understand context
- Feed decoder with the same input, but none of tokens are mased and sequence has been shift one position to the right by insertion of a start token.
- Decoder predicts next token for each position in sequence. Each prediction uses all previous input tokens from encoder.
- Calculate cross-entropy loss over predicted probabilities.

Fine-tuning stage is supervised.

Sampling with beam search for accuracy and consistency.

<!-- TOC --><a name="33-evaluation"></a>
## 3.3 Evaluation

Offline evaluation metrics
- BLEU (BiLingual Evaluation Understudy): count the ratio of matches, with brevity penalty, n-grams precision, weight for different n-gram precisions
- ROUGE (Recall-Oriented Understudy for Gisting Evaluation): recall = # matching n-grams / total # n-grams in reference. Lack of contextual understanding.
- METEOR (Metric for Evaluation of Translation with Explicit ORdering): combines precision, recall using weighted harmonic mean.
  - Pros: Semantic understanding, balanced evaluation, correlation with human judgements
  - Cons: Computational complexity, resource dependence.

Online evaluation metrics
- User feedback/engagements

<!-- TOC --><a name="4-chatgpt-personal-assistant-chatbot"></a>
# 4. ChatGPT: Personal Assistant Chatbot 

<!-- TOC --><a name="41-positional-encoding"></a>
## 4.1 Positional Encoding

- Relative positional encoding: encode differences in two tokens' positions
- Rotary positional encoding (RoPE): represent positional info as rotation matrix applied to token embeddings.
  - Translational invariance: encodes positional info that remains consistent even when positions of tokens shift, can handle changes in position.
  - Relative position representation
  - Generalization to unseen positions across varying sequence lengths 

<!-- TOC --><a name="42-training"></a>
## 4.2 Training

3 stage training: pretraining (on large corpus), SFT (finetunes model to adapt output to prompt-response format), RLHF (further refines model response with human alignment).

RLHF: Alignment stage, final stage in training process.
- Training reward model
  - loss function to penalize the model when difference between winning and losinng scores is too small, with hyperparameter defining the margin (min desired diff between scores of winning and losing responses).
  - Output: predicts relevance scores for (prompt, response) pairs, reflects human judgements.
- Optimizing SFT model with RL
  - Proximal Policy Optimization (PPO), to max scores predicted by reward model. Update model weights to max the expected reward that scores the responses.
  - Direct Policy Optimization (DPO)

<!-- TOC --><a name="43-sampling"></a>
## 4.3 Sampling 

How we select tokens from model's predicted probability distribution to generate response.

- temperature: control randomness by scaling logits (raw scores) of model's output before applying softmax to generate prob.
- repetition penalty 

<!-- TOC --><a name="44-ml-system-design-pipeline"></a>
## 4.4 ML System Design Pipeline 

- Training pipeline: pretraining, SFT, RLHF
- Inference pipeline: safety filterinng, prompt enhancer, response generator (choose one from multiple responses), response safety evaluator (detect harmful content), rejection response generator (generate a proper response when input prompt is unsafe or generated response is unsuitable, explain why request can't be fulfilled), session management 

<!-- TOC --><a name="5-image-captioning-image2text"></a>
# 5. Image Captioning (Image2Text)

<!-- TOC --><a name="51-image-encoder"></a>
## 5.1 Image Encoder 

Attention mechanism works best with sequence inputs, as it enables decoder to focus dynamically on different regions of image during caption generation. This selectively attending to various parts of image leads to more accurate captions.

CNN-based
- Process input image, output a grid of feature vectors.
- CNN produces 3 x 3 x c output. Transformer in the text decoder needs a sequence of features (9 x c) by flattening/reshaping operation that reorganizes features from each of 9 positions in 3 x 3 grid into a sequential format.
- Good to capture local patterns in images, but bad at long-range dependencies between distant regions of image.

Transformer-based 
- Patchify: Divide image to fixed-size patches, flatten each patch, linearly project each patch.
- Positional-encoding: Assign position info to each patch
  - 2D positional-encoding: maps 2 integers (row, column), preserving spatial structure; 1D: maps integer to c-dimensional vector, e.g., ViT.
  - learnable: learns positional encoding during training; fixed: positional encoding determined by sine-cosine fixed functions.
- Can capture both local and global relationships with self-attention, context aware.

<!-- TOC --><a name="52-pipeline"></a>
## 5.2 Pipeline

- Training: Supervised finetuning on 400 million image-caption pairs with cross entropy loss on next-token prediction.
- Sampling: beam search for coherence
- Offline evaluation metric: CIDEr, use consensus to evaluate similarity of generated caption to a set of reference captions (robust to different caption variations).
  - Represent captions using TF-IDF (good that sensitive to important words, but bad that lack of semantic understanding), calculate cosine similarities, aggregate similarity scores
- System Design: Image preprocessing -> caption generator (beam search for trained model) -> post-processing (fairness, inclusivity)

<!-- TOC --><a name="6-rag"></a>
# 6. RAG 

Company-wide StackOverflow ChatPDF system: Indexing process (data chunks embedded using CLIP text encoder) -> Safety input filtering -> Query expansion (expand user's query to have a better flow, broaden scope of search) -> Retrieval (user query embedded, then use ANN to retrieve similar data chunks in index table) -> Generation (CoT prompt engineering, LLM with top-p sampling)

- PDF document parsing: OCR (Optical Character Recognition) to identify texts, tables, diagrams from docs.
- Doc chunking (index them into searchable database): length-based (LangChain RecursiveCharacterTextSplitter), regex-based (semantic logical breaks), html markdown code (LangChain PythonCodeTextSplitter)
- Indexing (organize chunked data into structure for retrieval): knowledge-graph based (relationship), vector-based (good for semantic understanding, scalability, efficiency)

<!-- TOC --><a name="61-model-architecture"></a>
## 6.1 Model Architecture

- Indexing: text encoder + image encoder (CLIP provides pretrained encoders with a shared embedding space, enabling cross-modal retrieval where text and data shares embedding space; or image captioning to generate textual description of image)
- Retrieval: convert user query into the same embedding space as indexed data
- Generation: by LLM 

RAFT (Retrieval-Augmented Finetuning)
- Document labeling: relevant doc or not
- Joint training: LLM trained on relevant docs, minimizing influence of irrelevant docs, by penalizing irrelevant docs in loss function 

<!-- TOC --><a name="62-sampling"></a>
## 6.2 Sampling

RAG system, multiple components work together to produce response.

- Retrieval stage: compute query embedding (capture semantic meaning of query), use FAISS/Elasticsearch frameworks for ANN (approximate nearest neighbor) to find data similar to query, retrieve close-enough neighbors without searchign entire datasets.
  - ANN Tree-based (data space into multiple partitions), locality-sensitive hashing (LSH, points close in space are hased into same bucket)
  - ANN clustering-based (clusters using distance metrics like cosine similarity). Two step process (inter-cluster and intra-cluster search): narrowing down the search to a cluster, then finer search within that cluster.
  - ANN graph-based (Hierarchical nagivable small world, HNSW): begin with higher-level coarse graph, then gradually move down to finer levels. Search refined at each level, exploring only nearby nodes, thus reducing search space.
- Generation stage:
  - prompt engineering (CoT, few-shot providing some examples, role-specific with necessary style, user-context)

<!-- TOC --><a name="63-evaluation"></a>
## 6.3 Evaluation

- context relevance: hit rate, mean reciprocal rank (MRR), normalized discounted cumulative gain (NDCG), Precision@k
- faithfulness: human evaluation, automated fact-checking tools, consistency checks
- answer relevance/correctness: BLUE, ROGUE, METEOR to measure how closely the answer matches correct reference answer

<!-- TOC --><a name="7-realistic-face-generation"></a>
# 7. Realistic Face Generation

Face generator (StyleGAN), Training service, Evaluation service, Deployment service

<!-- TOC --><a name="71-choose-ml"></a>
## 7.1 Choose ML

<!-- TOC --><a name="711-vae"></a>
### 7.1.1 VAE
VAE can generate new images by sampling points from learned distribution and use decoder to map these points into image.
- Encoder: NN that maps input image into lower-dimensiona space (latent space, as an output).
- Decoder: Another NN that maps encoded representation into an image.
- Pros: simple architecture, fast image generation, stable training, compression capability
- Cons: less realistic images, blurriness, limited novelty, limited control in generation

<!-- TOC --><a name="712-gan"></a>
### 7.1.2 GAN 

- Generator: NN that converts random noise into image. Learn to make realistic images.
- Discriminator: Another NN that determines whether a given image is real or human-generated. Distinguish real from generated ones.
- Pros: high-quality, fast generation, attribute control (e.g. age, expression in face)
- Cons: training instability (mode collapse, non-convergence), limited control, limited novelty

<!-- TOC --><a name="713-autoregressive-dall-e"></a>
### 7.1.3 Autoregressive (DALL-E)

Each part of image is generated sequentially using Transformer.
- Autoregressive training: Image -> convert to sequence -> Transformer
- Autoregressive inference: Transformer (random seed to start, ..., end token) -> convert back to image -> generated image 
- Pros: high detail and realism, stable training, control over generation using additional inputs (text prompts), support multimodal conditioning (audio, etc), novelty
- Cons: slow, resource-intensive, limited image manipulation (no strucuted latent space like VAE or GAN)

<!-- TOC --><a name="714-diffusion"></a>
### 7.1.4 Diffusion

Formulate image generation as iterative process. Noise is gradually added to images, NN is trained to predict this noise, beginning with random noise and iteratively denoise the image.
- Pros: high detail and realism, stable training, control over generation, novelty, robustness to noisy images
- Cons: slow, resource-intensive, limited image manipulation

<!-- TOC --><a name="72-gan-architecture"></a>
## 7.2 GAN Architecture 

<!-- TOC --><a name="721-generator"></a>
### 7.2.1 Generator

Transform low-dimensional noise vector into 2D image, generator = N * upsampling blocks [= ConvTranspose2D + BatchNorm2D + ReLU], where final block uses Tanh instead of ReLU (to ensure output range [-1, 1], matching range of image pixels).
- Transposed/Upsampling convolution (Deconvolution): to increase spatial resolution of feature maps. For image generation, semantic segmentaion, super-resolution.
  -  Insert zeroes between pixels of input feature map, expanded input is convolved with a filter, where stride (controls how much filter moves across the input during convolution, larger strides skip more pixels) and paddling (adds extra borders around input to control output size during convolution) are adjusted to achieve desired output size.
  -  PyTorch layer name `ConvTranspose2d`
- Normalization layer: scale input data to have consistent distribution to improve training stability, as GAN is unstable due to generator-discriminator compete against each other, causing mode collapse, oscillations. Normalization stabilize training by scaling activations at each layer, reducing risk of vanishing/exploding gradients, balance competition between generator-discriminator. We can use high learning rate, speed up training, reduce time for convergence. 
  - Batch Normalization: normalize inputs of a layer across the batch dimension by calculating mean and variance for each feature, normalized data then scaled and shifted using learnable parameters.
    - Good: allow for higher learning rates, speed up training. Regularizer, reduce overfitting.
    - Used in deep NN like CNN, GAN.
  - Layer Normalization: normalize inputs across features of each individual sample, rather than across batch dimension, by calculating mean and variance for each feature across entire feature vector of each sample.
    - Good: effective in settings where batch sizes are small or variable, like RNN, Transformer.
    - Used in sequence models and consistent-behavior across samples.
  - Instance Normalization: normalize across each feature map individually for each sample
    - Good: for appearance of individual samples varies widely, allowing network to focus on content rather than style
    - Used in style transfer, image generation
  - Group Normalization: divide features into groups and normalize within each group, a balance between Batch Normalization and Layer Normalization.
    - Good: for small batch sizes, where batch normalization is not effective
    - Used when Batch Normalization fails due to small batch size, or layer behavior consistency is needed across groups of features
- Nonlinear activation (ReLU) 

<!-- TOC --><a name="722-discriminator"></a>
### 7.2.2 Discriminator 

Binary classifier, take in image, output prob that image is real discriminator = N * downsampling blocks [= Conv2D + BatchNorm2D + ReLU] + classification head = [Fully Connected + Sigmoid activation function] (ensure prob output in [0, 1]).
- Downsampling blocks: reduce spatial dimensions of input image while extracing features, with convolution. PyTorch layer name `Conv2D` with stride = 2 to have spatial dimensions.
- Classification head: given extracted features, predict prob that it's real. 

<!-- TOC --><a name="73-adversarial-training"></a>
## 7.3 Adversarial Training 

Generator-discriminator trained simultaneously, avoid one dominates the other, ensure both improve together. Alternative between 2 steps:
- Train discriminator for a few iterations, freeze generator.
- Train generator for a few iterations, freeze discriminator.

Loss function
- Discriminator: min binary cross-entropy, loss contribution from real images + loss contribution from fake images
- Generator: max for all fake images
- GAN's minimax loss: unify generator's and discriminator's losses into a single function: discriminator max the loss and generator min the loss.

<!-- TOC --><a name="731-training-challenges"></a>
### 7.3.1 Training Challenges

- Vanishing gradients: when discriminator too good at distinguishing fake and reak, it provides small gradient for generator. Solution:
  - Modify minimax loss (generator to max prob of fake images being identified as real, rather than min the prob of fake images being identified as fake)
  - Wasserstein GAN. Discriminator for WGAN (critic) outputs a score representing realness of image, instead of classifying image as real/fake, objective is to max the critic loss. WGAN generator to max prob of fake images being identified as eral.
- Mode collapse: generator produce limited variety of images to trick the system by keep producing same image to fool discriminator, and never learns to generate other images. Solution: WGAN, Unrolled GAN.
- Not converge: as generator improves, discriminator declines, because of difficulty, and it can't converge as feedback becomes less useful to generator but generator keep training on useless feedback. Solution:
  - batch normalization to stablize training by ensuring consistent distributions across layers.
  - different learning rates to balance their progress and avoid instability
  - regularization (weight decay) prevents overfitting
  - add noise to discriminator inputs to prevent it being too powerful early on, balance competition

<!-- TOC --><a name="74-sampling"></a>
## 7.4 Sampling

Sampling process to generate new images from trained GAN.

Sample a latent vector from a learned latent space. 
  - random sampling (ensure diversity by exploring entire latent space), by Gaussian distribution to draw latent vectors from latent space. 
  - truncated sampling (focus on high-prob region to enhance realism), to restrict latent vectors to samller, high-prob region of latent space. Can reduce generating outliers, high-quality images, good for realism.

<!-- TOC --><a name="75-evaluation"></a>
## 7.5 Evaluation

Inception score: evaluate quality of generated images in GAN, in diversity (check if generated images has a nearly uniform marginal distribution across classes, that images are spread evenly across different classes), and quality (high quality images have sharp, peaked prob distribution that's clearly recognized as belonging to a particular class, compares this distribution with marginal distribution to assess quality).
- generating images
- compute class probability distribution: high quality image has a distribution with a peak (high prob for one class) that the model recognizes it as clear instance of a class.
- calculate marginal distribution: average of predicted class probabilities across all images, to understand overall distribution of classes. If images are diverse, marginal distribution is flat and spread across many classes.
- compute KL divergence: how different the predicted class distribution for each image is from the marginal distribution. High quality image has distribution different from marginal distribution, because it has a peak in distribution, not uniform, if image is diverse.
- calculate inception score: exponentiated average of KL divergence across all images. High inception score = individual images are confidently classified into various classes, and generated images are diverse and high quality.

Frechet inception distance (FID): how similar the distribution of generated images is to the distribution of real images. Unlike Inception score using class probabilities, FID considers the statistics of features extracted by pretrained model (Inception v3, trained on large diverse dataset ImageNet and can extract meaningful features of content/style of images). FID measures diversity (covaraince of features reflect spread and variation in image) and quality (FID ensures generated images are high-quality by comparing their feature distribution to real images).
- generating images
- extracting features: pass (generated and real) images through Inception v3 and extract features (activations) from specific layer (one near the end of network). Features from this deep layer capture high-level info (shapes, textures, objects) to assess realism of images.
- calculate mean and covariance: summarize distribution of features for both sets of images
- compute Frechet distance between mean and covariance of generated and real images, that how close two distributions are, lower FID means more similar.  

# 8. High-Resolution Image Synthesis 

Generation service, decoding service, super-resolution service

As resolution increases, need decoder with high capacity to capture details, but decoder can be powerful to ignore input from latent space, so latent variables contribute little to generation process, reducing diversity of images. 

Autoregressive models are slow due to sequential nature, where each pixel depends on the ones generated before it with O(N^2) complexity. We can use Transformer-based autoregressive model to fastly generate image chunk by chunk instead of pixel by pixel. Diffusion model increase complexity super-linearly with image size with O(TN^2) complexity, N = pixels, T = denoising steps.

## 8.1 Architecture

Image tokenizer
- encode image into sequence of discrete tokens
- decode sequence of discrete tokens back to image

Vector-Quantized VAE (VQ-VAE)
- encoder (deep CNN): image -> N * [Conv2D + ReLU] -> Conv2D -> Encoded representation in lower-dimensional latent space
- quantizer: convert continuous latent vectors to discrete tokens and output a collection of token IDs, it's an embedding table with sole parameter being a codebook, which is learned during training.
  - avoid posterior collapse: decoder generate accurate outputs without using latent space, quantization force model to use discrete latent variables during reconstruction so decoder doesn't overpower the latent space, and keep latent variables actively involved in shaping outputs.
  - reduce learning space: continuous vectors are difficult to predict sequentially as they have endless possibilities and small differences, quantizer allow Transformer to focus on fewer options.
- decoder (deep CNN) : codebook -> embedding lookup -> N * [Transposed conv `ConvTranspose2d` + ReLU] -> image 

Image generator, with decoder-only Transformer
- embedding lookup: replace each discrete token with its embedding from codebook
- projection: project each token embedding into dimensionality 
- positionnal encoding: adds positional encodings to sequence to add spatial info
- transformer = N * [Multi-head Attention + Normalization + Feed Forward + Normalization]: process input sequence and outputs updated sequence of vectors
- prediction head: use updated embeddings to predict next token

## 8.2 Training 

Image tokenizer

- encoder process input image and convert to continuous representation
- quantizer replace continuous representation with discrete tokens using internal cookbook
- decoder use discrete tokens to reconstruct original image

loss function = weighted sum of below 4
- reconstruction loss: difference between original and reconstructed images
- quantization loss: distance between encoder's outputs and nearest embeddings in codebook, encourage encoder to produce outputs closer to codebook embeddings
- For high resolutions
  - perceptual loss: difference between features of original and reconstructed images extracted from a specific layer of pretrained model (VGG)
  - adversarial loss: how well image reconstructed by image tokenizer can fool discriminator

Image generator: cross-entropy loss function to measure how accurate predicted prob are compared to correct visual tokens

## 8.3 Sampling

- generate sequence of discrete tokens: randomly select token from codebook as initial token, seed for the rest of generation. Then augoregressively generate tokens one by one (predict prob distribution over codebook, top-p sampling for next token).
- decode discrete tokens into image

# 9. Text2Image

## 






