# Contents

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [1. DeepSeek‑V3](#1-deepseekv3)
   * [1.1 Mixture of Experts (MoE)](#11-mixture-of-experts-moe)
   * [1.2 DeepSeek Experts](#12-deepseek-experts)
   * [1.3 How are DeepSeek’s gating mechanism and routing different?](#13-how-are-deepseeks-gating-mechanism-and-routing-different)
   * [1.4 Multi-Latent Head Attention](#14-multi-latent-head-attention)
   * [1.5 Multi-Token prediction](#15-multi-token-prediction)
   * [1.6 FP8 mixed precision training](#16-fp8-mixed-precision-training)
   * [1.7 Parallelism](#17-parallelism)
      + [1.7.1 DualPipe pipeline parallelism: Reducing communication overhead](#171-dualpipe-pipeline-parallelism-reducing-communication-overhead)
      + [1.7.2 Expert parallelism + ZeRO data parallelism: Handling MoE efficiently](#172-expert-parallelism-zero-data-parallelism-handling-moe-efficiently)
- [2. DeepSeek‑R1](#2-deepseekr1)
   * [2.1 R1-Zero: Group Relative Policy Optimization (GRPO)](#21-r1-zero-group-relative-policy-optimization-grpo)
   * [2.2 R1 vs. R1-Zero](#22-r1-vs-r1-zero)
      + [2.2.1 How does the cold start data improve readability?](#221-how-does-the-cold-start-data-improve-readability)
      + [2.2.2 How is the reasoning-oriented RL conducted?](#222-how-is-the-reasoning-oriented-rl-conducted)
      + [2.2.3 What’s rejection sampling and fine-tuning?](#223-whats-rejection-sampling-and-fine-tuning)
      + [2.2.4 Why another RL pass for all scenarios?](#224-why-another-rl-pass-for-all-scenarios)
- [3. Making Distilled Models from DeepSeek-R1](#3-making-distilled-models-from-deepseek-r1)
- [Acknowledgements](#acknowledgements)

<!-- TOC end -->

---

<!-- TOC --><a name="1-deepseekv3"></a>
# 1. DeepSeek‑V3
As the robust, fully open‑source base model, DeepSeek‑V3 leverages a Mixture‑of‑Experts architecture, incorporating innovations like Multi‑Head Latent Attention (MLA) and advanced load balancing. This design ensures high performance even on modest hardware setups, offering speed and cost efficiency.

Instead of scaling up and running into massive computational costs, memory demands, and inefficiencies, DeepSeek employs optimization techniques to tackle these challenges. It leverages the Mixture-of-Experts (MoE) framework, which selectively activates only the most relevant parts of the model for each task, significantly reducing memory usage and computational overhead while maintaining high performance. Also, further enhancements such as Multi‑Head Latent Attention and Multi‑Token Prediction further boost efficiency, allowing the model to handle long-context tasks and diverse data with ease.
<!-- TOC --><a name="11-mixture-of-experts-moe"></a>
## 1.1 Mixture of Experts (MoE)


DeepSeek models are built on the transformer architecture, which can be thought of as a bustling factory where every worker (or neuron) processes every task. Transformers use self‑attention and feed‑forward networks to understand context and generate language. What sets DeepSeek apart is the integration of the Mixture‑of‑Experts (MoE) approach—a strategy that brings in specialized teams of submodels only when needed, rather than having every part of the network work on every task.

Imagine you’re at a busy gourmet coffee shop that offers a wide range of specialty drinks, each crafted by a barista with unique expertise. Instead of having every barista make every drink, a smart ordering system directs your order to the barista best suited for that particular beverage. This is the essence of MoE: rather than processing each input through the entire network, a gating function acts like that smart ordering system—selecting only a small subset of experts (specialized submodels) for each input token.

In traditional MoE, there are two common approaches:
- Dense MoE: All experts are activated for every input. While this can improve accuracy, it is computationally heavy.
- Sparse MoE: Only the top‑k experts are activated for each input, dramatically reducing the computation required. Most large‑scale models, including those in DeepSeek’s family, use this sparse method.

How do these sparse MoE systems work? Instead of having every expert work on every task, a smart system called the gating function decides which few experts are best suited for each specific input—much like a manager who assigns tasks only to the most appropriate team members. However, if the manager always picks the same few experts, those specialists can become overworked while others sit idle, which isn’t efficient. Older models dealt with this by adding extra rules to force a fair distribution of tasks among all experts.

DeepSeek‑V3 takes a smarter approach: It automatically adjusts how tasks are assigned so that every expert is used evenly without needing any extra balancing rules. It also breaks each expert into smaller parts, allowing the system to handle even more detailed or nuanced information within each expert’s knowledge.


- Experts: Specialized submodels that process different aspects of the input. They form the building blocks of the MoE by partitioning the FFN layers.
- Gating Mechanism: A dynamic “smart switch” that selects the most relevant experts for each input token, ensuring that only a subset of the model is engaged.
- Load Balancing and Routing: Strategies that distribute tasks evenly among experts, preventing some from being overused while others remain idle.





<!-- TOC --><a name="12-deepseek-experts"></a>
## 1.2 DeepSeek Experts


DeepSeek‑V3 distinguishes between two types of experts. It divides the experts into:
- Shared experts: These experts are always available, providing a stable processing backbone for every token.
- Routed experts: These experts are conditionally activated based on the input token, allowing the model to leverage specialized, context‑dependent expertise.

This dual structure improves efficiency and enhances the model’s ability to handle diverse tasks. Secondly, DeepSeek‑V3 applies fine‑grained segmentation to its routed experts. Rather than treating each routed expert as a single monolithic block, DeepSeek‑V3 partitions them into multiple smaller segments. This fine‑grained expert segmentation enables the model to capture subtle nuances in the data, as each segment can focus on different aspects of the input. 


<!-- TOC --><a name="13-how-are-deepseeks-gating-mechanism-and-routing-different"></a>
## 1.3 How are DeepSeek’s gating mechanism and routing different?
In a basic MoE, imagine each input token is like an order at a restaurant. A simple system (the gating mechanism) gives each chef (expert) a score using a linear projection and then applies a softmax function—this is like ranking the chefs based on their expertise for that specific order. The highest-ranked chefs (top‑k) are chosen to prepare the order. However, if one chef keeps getting the highest score, they can get overwhelmed while others hardly get any work. DeepSeek‑V3 refines this process in two key ways:
- Instead of softmax, DeepSeek‑V3 uses a sigmoid function to calculate affinity scores. Think of this as giving each chef a rating that isn’t forced into a strict ranking but instead reflects a more balanced “likability” for handling the order. This helps prevent one chef from always dominating the orders.
- Before deciding which chefs get the order, DeepSeek‑V3 adds a dynamic bias to each chef’s score. Imagine that if a chef is already very busy, the system slightly lowers their score—like a traffic regulator ensuring no single chef is overloaded. Conversely, if a chef is underutilized, their score gets slightly boosted. After this adjustment, the system picks the top chefs based on these modified scores. Importantly, while these bias adjustments influence who gets chosen, the actual contribution from each chef (how much they help with the order) is determined by the original, unbiased score.

This two-step approach—balancing the initial ratings with a dynamic bias adjustment—ensures that orders are distributed more evenly among all chefs, leading to a more efficient and stable system overall.

| Feature | Mixture of Experts (MoE) | Standard Transformer |
|---|---|---|
| Computation per Token | Activates only a subset of experts for each token, significantly reducing computation overhead. | Activates all parameters for every token, leading to higher computational cost. |
| Parameter Utilization | Sparse activation enables the selective use of a vast parameter pool, maximizing efficiency. | Dense activation utilizes the entire parameter set for every token. |
| Scalability | Can scale to trillions of parameters while keeping inference costs low, thanks to conditional computation. | Scaling increases computation linearly, making it more resource‑intensive at large scales. |
| Training Complexity | Requires advanced routing and load balancing techniques, but yields high efficiency at scale. | Simpler training procedures but with higher computational demands during scaling. |

<!-- TOC --><a name="14-multi-latent-head-attention"></a>
## 1.4 Multi-Latent Head Attention


DeepSeek-V3 addresses the computational challenges of self-attention using Multi-Head Latent Attention (MHLA), which is an optimized mechanism that reduces memory overhead while preserving contextual understanding.


DeepSeek-V3 introduces latent compression, KV caching, and efficient positional encoding. Instead of computing full-sized query, key, and value matrices for every token, MHLA compresses them into latent representations, reducing redundant calculations. The model first derives a latent query representation from the input hidden state, which is then used to generate content-based queries and residual queries. These queries are concatenated and passed into multi-head attention. Similarly, keys and values undergo a latent compression mechanism, reducing the number of key-value pairs needed for attention computations. The keys are further divided into content keys, which are dynamically computed for each token, and residual keys, which are cached and reused during inference. 

Beyond latent compression and caching, MHLA also improves efficiency by refining how positional information is handled. Instead of adding absolute position embeddings to token representations, DeepSeek-V3 employs Rotary Positional Embeddings (RoPE), which are applied only to the residual queries and residual keys before attention computations. This separation ensures that the model can efficiently retain word order information without increasing memory requirements.

By modifying the traditional attention equation and leveraging latent compression, KV caching, and RoPE-based positional encoding, DeepSeek-V3’s MHLA mechanism significantly reduces computational overhead while maintaining high performance for long-context processing and large-scale inference. This enhanced efficiency is essential for enabling subsequent innovations such as Multi-Token Prediction (MTP) and optimized training techniques, positioning DeepSeek-V3 as a highly scalable and powerful AI model.
<!-- TOC --><a name="15-multi-token-prediction"></a>
## 1.5 Multi-Token prediction
Another key innovation that boosts DeepSeek‑V3’s efficiency is its Multi‑Token Prediction (MTP) strategy. In a traditional transformer, the model is trained to predict the next token in a sequence—one word at a time. Imagine you’re reading a sentence and trying to guess the next word based solely on what you’ve seen so far. This one-word-at-a-time approach works, but it can sometimes miss out on longer-term patterns in the text. DeepSeek‑V3 takes a step further with its Multi‑Token Prediction (MTP) strategy. Instead of predicting just the very next token, MTP trains the model to forecast several tokens into the future. Think of it like planning an entire sentence rather than a single word. This allows the model to pre-plan its responses and capture more context, ultimately leading to richer and more coherent outputs.


<!-- TOC --><a name="16-fp8-mixed-precision-training"></a>
## 1.6 FP8 mixed precision training


DeepSeek-V3 employs FP8 (8-bit Floating Point) mixed precision training to significantly reduce memory usage and accelerate computation. This approach allows faster training speeds, larger batch sizes, and improved efficiency while maintaining numerical stability. The mixed precision framework strategically switches between FP8, BF16 (16-bit Brain Floating Point), and FP32 (32-bit Floating Point) for different computations to balance efficiency and accuracy. The illustration below shows how the linear operator computations handle different precision formats.


<!-- TOC --><a name="17-parallelism"></a>
## 1.7 Parallelism


<!-- TOC --><a name="171-dualpipe-pipeline-parallelism-reducing-communication-overhead"></a>
### 1.7.1 DualPipe pipeline parallelism: Reducing communication overhead


DeepSeek-V3 implements DualPipe, an advanced pipeline parallelism algorithm that overlaps forward and backward computations with inter-node communication, reducing idle time. Each micro-batch is split into functional components such as:
- Attention computation:rocesses token relationships using multi-head self-attention, enabling the model to capture long-range dependencies efficiently.
- All-to-all dispatch (for MoE layers):istributes input tokens across multiple expert networks, ensuring that each token is processed by the most relevant subset of parameters.
- MLP (Feedforward networks): pplies nonlinear transformations using fully connected layers to refine token representations after attention processing.
- All-to-all combine:athers outputs from distributed expert networks, merging them back into a unified representation for the next model stage.

By minimizing pipeline bubbles and synchronization delays, DualPipe significantly reduces communication overhead, making scalability more efficient across multiple GPUs.
<!-- TOC --><a name="172-expert-parallelism-zero-data-parallelism-handling-moe-efficiently"></a>
### 1.7.2 Expert parallelism + ZeRO data parallelism: Handling MoE efficiently



<!-- TOC --><a name="2-deepseekr1"></a>
# 2. DeepSeek‑R1
Building on the V3 foundation, DeepSeek‑R1 is tailored for advanced reasoning. Unlike many models focusing solely on text generation, DeepSeek‑R1 is fine‑tuned through reinforcement learning to excel at logical problem‑solving and decision‑making. It doesn’t just predict the next word—it thoughtfully navigates complex challenges.

<!-- TOC --><a name="21-r1-zero-group-relative-policy-optimization-grpo"></a>
## 2.1 R1-Zero: Group Relative Policy Optimization (GRPO)
R1‑Zero’s training is driven by a specialized RL algorithm called Group Relative Policy Optimization (GRPO).

- Generate a batch of solutions: The model’s older checkpoint (or old policy) comes up with multiple possible solutions to a problem.
- Score each solution: Each solution is given a reward or penalty, depending on correctness (for instance, math solutions that match the answer key or code solutions that pass tests) and formatting (like having chain-of-thought between `<think>` tags).
- Compare against the average: We compute the group’s average reward. Any solution that’s above this average gets a positive push (the model learns from it), while solutions below average get pushed away.
- Update the model: Over many iterations, the model competes with its own previous snapshots, continually nudging itself toward better, more reasoned answers.

Unlike some other RL methods (e.g., those needing a large critic model), GRPO estimates a baseline reward by looking at that group’s mean performance. This approach reduces computational overhead and still provides a clear incentive for solutions that outscore the average. Over time, the model self-evolves, discovering that writing out thorough steps can lead to higher rewards—and, thus, better solutions.

However, there are some drawbacks as well. R1-Zero’s outputs were sometimes messy. Without curated chain-of-thought examples, it sometimes spews stream-of-consciousness text, mixes multiple languages, or wanders off-topic. This is intriguing from a research angle but not ideal for user-facing tasks. Hence, DeepSeek tried to do something better, which resulted in DeepSeek-R1.
<!-- TOC --><a name="22-r1-vs-r1-zero"></a>
## 2.2 R1 vs. R1-Zero
While R1-Zero proved that purely RL-trained models can discover thorough reasoning patterns, it sometimes produced messy, stream-of-consciousness answers. So, the DeepSeek team designed a four-stage pipeline for DeepSeek-R1 that preserves the benefits of R1-Zero’s think-out-loud capabilities but removes its worst quirks.

They start with a cold-start dataset of neatly formatted chain-of-thought examples, do reasoning-oriented RL to encourage correctness and structure, filter the best outputs to build an expanded supervised dataset, and finally apply RL for all scenarios to produce a well-rounded, alignment-friendly model.
<!-- TOC --><a name="221-how-does-the-cold-start-data-improve-readability"></a>
### 2.2.1 How does the cold start data improve readability?
The goal is to give the model a gentler introduction to writing chain-of-thought (CoT) so it doesn’t wander or produce unreadable tangents. The team manually curated a few thousand exemplars where a person (or a high-quality AI) solved a problem step-by-step—e.g., a math question with each step clearly explained, or a coding challenge broken down line by line. These CoT exemplars were designed to be human-friendly, so there was no random mixing of languages or half-finished statements.

Instead of letting RL fling the model directly into trial-and-error (like R1-Zero), the base model (DeepSeek‑V3’s core) is briefly fine-tuned on these curated CoT examples. This gives it a sense of how people typically structure reasoning (e.g., bullet-point logic, consistent variable naming, final summary, etc.). Technically, this stage is still a supervised fine-tuning pass, but it uses only a small dataset—often described as a seed or cold start set.

Thanks to these curated patterns, the model is less prone to spewing random text when asked to think aloud. Even if it hasn’t yet been trained via RL, it already knows a more coherent style for chain-of-thought.

<!-- TOC --><a name="222-how-is-the-reasoning-oriented-rl-conducted"></a>
### 2.2.2 How is the reasoning-oriented RL conducted?
After that mini-supervised stage, they run a large-scale RL loop (similar to R1‑Zero’s approach). The goal is to improve the model’s step-by-step logic and problem-solving skills (math, code, logic puzzles) while maintaining a well-structured format. How do they do it? Once again, R1 uses the GRPO algorithm. Solutions better than the average get nudged up; worse ones get nudged down. This works because it doesn’t require a large critical network that matches the size of the model, which saves training resources.

This RL process once again runs for thousands of updates, each time sampling a batch (group) of solutions from the model, rating them with the rule-based or code-based reward system, and adjusting the model’s parameters. Since the model was already cold-started with coherent CoT, it doesn’t revert to the messy stream of consciousness that plagued R1-Zero. A model that thinks out loud in a more stable, structured way, and consistently produces correct solutions for math/coding tasks. The official paper highlights significant gains on numeric benchmarks (e.g., AIME) purely from letting the model be rewarded for each detailed reasoning step.


<!-- TOC --><a name="223-whats-rejection-sampling-and-fine-tuning"></a>
### 2.2.3 What’s rejection sampling and fine-tuning?

The goal of this step is to filter out nonsense or erroneous solutions from the newly RL-trained checkpoint and broaden the model’s skill set with an even larger supervised dataset. After the RL converges, they generate a huge batch of solutions from the model. Some are correct and nicely formatted; others are half-baked or obviously wrong. The team systematically rejects the worst ones—messy, inaccurate, or completely off-topic. That filtering is done either with automated checks (like math correctness, coding test results, or a quality reward model) or manual spot checks.

Only the high-quality solutions remain, forming a large dataset of chain-of-thought exemplars. The paper mentions gathering ~600k–800k data points at this stage—far more than the initial cold-start dataset. They also add tasks from creative writing, open Q&A, alignment data (for safety/policy compliance), so the model can handle general user requests beyond just math/coding.

The base model is then retrained (fine-tuned) on this curated super-dataset, effectively absorbing the best RL outputs plus additional tasks. This ensures not only advanced reasoning but also a more polished style, with fewer tangents or off-topic rambles.

A strong SFT model draws on the best solutions from the RL checkpoint, covering code, math, creative Q&A, etc. The chain-of-thought remains fairly neat since the messy ones were mostly filtered out.


<!-- TOC --><a name="224-why-another-rl-pass-for-all-scenarios"></a>
### 2.2.4 Why another RL pass for all scenarios?
Finally, the model does one more RL stage, mixing advanced reasoning tasks (math, code, logic) with general helpfulness or alignment tasks. The goal is to round out the model’s alignment, correctness, and helpfulness across multiple domains, not just math/coding. This last RL phase includes a wider variety of prompts—code tasks, math tasks, role-play, knowledge queries, and alignment-critical prompts (e.g., refusal of disallowed content). By blending different prompt types, the model learns to respond helpfully to general queries while still writing the correct chain-of-thought for math or coding.

The final solution’s accuracy is still rewarded in math and coding tasks. The reward checks whether the model remains polite, aligns with policy, or avoids disallowed content. Typically, this is done via a preference model or a pairwise comparison approach—similar to how alignment training is done in other large LLMs.

After enough RL updates, the model is good at reasoning and everyday tasks, with a user-friendly style. This final checkpoint is DeepSeek‑R1.
<!-- TOC --><a name="3-making-distilled-models-from-deepseek-r1"></a>
# 3. Making Distilled Models from DeepSeek-R1

DeepSeek employed a sophisticated model distillation process to create efficient versions of its DeepSeek-R1 model. Their approach shares similarities with traditional knowledge distillation but also introduces unique methodologies.
- Teacher model training: DeepSeek developed a powerful teacher model, DeepSeek-R1, comprising 671 billion parameters. This model was trained using reinforcement learning (RL) techniques, specifically Group Relative Policy Optimization (GRPO), to enhance reasoning capabilities.
- Synthetic data generation: DeepSeek-R1 generated synthetic datasets instead of relying solely on real-world data. Approximately 600,000 reasoning data points were created through rejection sampling, ensuring only accurate outputs were retained. An additional 200,000 non-reasoning data points were synthesized for tasks like writing and translation.
- Student model training: Smaller models, referred to as DeepSeek-R1-Distill, were initialized from existing open-weight architectures such as Llama and Qwen. These student models were fine-tuned using the synthetic data produced by DeepSeek-R1, enabling them to learn the reasoning patterns and knowledge of the teacher model.

| Aspect | DeepSeek’s Approach | Traditional Distillation |
|---|---|---|
| Synthetic Data Utilization | Extensive use of synthetic data generated by the teacher model to train the student models. Reduces reliance on large labeled datasets. | Primarily relies on real-world labeled datasets for training. |
| Reinforcement Learning (RL) Integration | Incorporates Group Relative Policy Optimization (GRPO) to enhance reasoning capabilities in the teacher model. | RL is not commonly used in conventional knowledge distillation processes. |
| Core Distillation Principles | Follows traditional distillation principles: knowledge transfer from a larger teacher model to a smaller, efficient student model. | Also follows the same principle, aiming for efficient model compression. |

<!-- TOC --><a name="acknowledgements"></a>
# Acknowledgements

[Educative: Everything You Need to Know About DeepSeek](https://www.educative.io/verify-certificate/GZjlABCqZ1G2n7mWjuroy1MXK2GBIm)

<img src="https://github.com/user-attachments/assets/0cec5763-b0fd-4278-a7fa-a289fffab5be" width="50%" height="50%">
