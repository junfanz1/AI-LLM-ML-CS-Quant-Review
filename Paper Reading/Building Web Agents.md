数据集：SimpleQA, HotpotQA (multi-hop reasoning跨文档多跳推理), GAIA, WebArena, WebVoyager多模态，WebWalkerQA (web traversal search)，BrowseComp（网络难以查找的信息）。
- WebDancer：browsing data construction，目前数据集小，用crawlQA（浏览网页根目录，给GPT-4O synthesize）和e2hQA（从简单问题出发，在搜索中给名词加很多定语模糊化，让最终问题没有直接简单答案）
- WebSailer：用knowledge graph构造问题做path construction，构造multi-hop问题
- NNetNav (Chris Manning)：zero shot Llama 70B 在网页乱逛exploration，每4步summarize trajectory changes，用trajectory labeler标记retroactive labeling+剪枝+语言模型奖励，检查outcome reward model的instruction-trajectory是否匹配，post-hoc action regeneration因为原始动作可能非最优所以让LLM review一遍输出更好动作、减少noise。

Methodology: Decouple Clip & Dynamic Sampling Policy Optimization (DAPO), DUPO
- PPO：importance sampling, CLIP, surrogate objective, GAE，reward有两种：可以是rule based或NN训练的
- GRPO：去掉critic，advantage function变成组内优势，
- DAPO：在GRPO基础上4个改进：
  - 1、Clip-Higher：增加CLIP上界epsilon，增加低概率事件explore次数，避免PPO上限太紧导致entropy collapse探索能力坍缩；
  - 2、Dynamic Sampling：因为GRPO组内所有问题都对或都错是没有advantage的就导致0 policy gradient，随着模型越来越好全组答对情况更多，所以定义rule-based reward让组内不全对或全错（否则继续sample）；
  - 3、Token-level Policy Gradient Loss：GRPO是sample-level loss，这里重组求和，考虑token-level loss，给又臭又长回答penalty（长样本总体影响更大但按token一视同仁，长且high quality sample是好好利用的，而GRPO里的优质长sample没有得到足够重视）；
  - 4、Overlong Reward Shaping：过长sample不计入gradient update、临界长度sample用soft overlong punishment。

WebDancer
- Data construction构造数据集 + SFT用一部分数据集（SFT对cold start必要，学会复杂tool use） + DAPO（用剩下数据集train）
- LLM + LargeReasoningModel（长CoT sample）
- Takeaway
  - 高质量数据很重要，SFT for cold start很重要让agent学会多步多tool use，LLM for short CoT and LRM for long CoT，DAPO让推理更长。
  - RL目前还没有GAN的思想，自己跟自己对抗然后产生好数据？

WebSailer区别：主要还是在构建数据集的区别，用knowledge graph得到长path trajectory，像BrowseComp dataset需要跳步+加密才能得到答案。
- WebDancer = SFT + DAPO
- WebSailer = RFT (SFT一种，rejection-sampling Fine-Tuning，先用reasoning模型给prompt生成多个候选，用LLM judge只保留好样本做SFT，就是Best-of-N蒸馏，SFT很必要因为首先要教模型怎样使用tool）+DUPO（非DAPO，如果一组全对全错就再抽样，让这组变成有对有错，不去sample新样本而是在组里复制有对有错的样本，来填满batch从而节省时间）

<img width="1488" height="992" alt="image" src="https://github.com/user-attachments/assets/d9b05651-e4d4-4354-a8d6-b83bccc24a53" />


# Building Web Agents

How to push the boundaries of web agents that can truly research + tool use, not just retrieve information?

Goal: multi-hop reasoning across documents, dynamic web traversal, and robust handling of information that’s hard to find.

## Datasets & Benchmarks
We trained and evaluated on several challenging datasets:

- SimpleQA, HotpotQA – multi-hop reasoning across documents
- GAIA, WebArena, WebVoyager – multimodal browsing environments
- WebWalkerQA – web traversal search tasks
- BrowseComp – questions that require non-trivial, hard-to-find information

Some recent work in the industry that are worth looking into:

## WebDancer = Data Construction + SFT + DAPO Pipeline
WebDancer is a two-stage pipeline designed to maximize reasoning ability:

- Data Construction: crawlQA: Crawl root web pages, then use GPT-4o to synthesize challenging Q&A pairs. e2hQA: Start from simple questions, iteratively obfuscate nouns and attributes so that no single page gives a direct answer — forcing multi-hop reasoning.
- Training: SFT (Supervised Fine-Tuning): Train on a curated subset of the data to give the agent a cold start and teach it basic multi-step reasoning and tool use. DAPO (Decoupled Clip & Dynamic Sampling Policy Optimization): Train on the remaining data to extend reasoning depth and improve exploration.
- Model Mix: LLM handles short CoT (Chain-of-Thought) reasoning. Large Reasoning Model (LRM) handles long CoT trajectories, ensuring robust reasoning over extended browsing sessions.

DAPO: RL optimization with

- Clip-Higher for better exploration
- Dynamic Sampling to avoid zero-gradient groups
- Token-Level Loss to penalize bad long answers
- Soft Overlong Penalty to keep reasoning concise

Takeaway:
- High-quality, diverse data is the single most important factor.
- SFT is crucial to bootstrap multi-step planning and tool use.
- DAPO helps prevent exploration collapse and encourages longer, more meaningful reasoning chains.

A still-open question:

- Could we incorporate GAN-like adversarial self-play, where agents generate harder queries for themselves and iteratively improve via competition?

## WebSailer: Data Construction with Knowledge Graphs
WebSailer differs primarily in how the data is constructed:

- Uses knowledge graphs to build long path trajectories — ideal for tasks like BrowseComp where answers are hidden several hops away and require information “encryption” or indirection to solve.

Training recipe:

- RFT (Rejection-Sampling Fine-Tuning): Generate multiple candidate trajectories with a reasoning model. Use an LLM judge to keep only the best samples — essentially Best-of-N distillation. This ensures the agent learns not just to act, but to act optimally.
- DUPO (Dynamic Unbiased Policy Optimization): Similar to DAPO but more efficient: when a training group is all-correct or all-wrong, don’t resample new data — instead, duplicate within-group samples to create a balanced batch. Saves compute while maintaining gradient signal.

This approach excels at long-path, knowledge-graph-based reasoning — perfect for datasets like BrowseComp where multiple hops and obfuscation are required.

## NNetNav: Inspiration from Exploration
Chris Manning’s NNetNav experimented with zero-shot Llama 70B exploration:

- Summarize the trajectory every 4 steps
- Apply retroactive labeling and pruning
- Use an outcome reward model to check instruction-trajectory alignment
- Regenerate actions post-hoc to denoise and improve trajectory quality

## Key Takeaways
- WebDancer = SFT + DAPO. Best for teaching tool use, step-by-step reasoning, and pushing agents toward longer reasoning chains.
- WebSailer = RFT + DUPO. Best for building highly structured datasets via knowledge-graph paths and training agents to execute precise, long-horizon plans.
- LLM + LRM: Use LLM for short-chain reasoning and LRM for complex, long-chain reasoning.
- Future Directions: RL for web agents might benefit from a GAN-style setup, where agents iteratively create harder tasks for themselves — pushing both data quality and reasoning skill forward.

| Component | WebDancer | WebSailer |
|--------------|--------------|--------------|
| Data Construction | crawlQA, e2hQA | Knowledge-graph path construction |
| Cold Start | SFT (tool use, planning) | RFT (Best-of-N distillation) |
| RL Optimization | DAPO (Clip-Higher, token-level loss) | DUPO (batch balancing) |
| Focus | Learn to explore & extend reasoning | Execute long, precise trajectories |

## References 

- [LinkedIn: Building Web Agents](https://www.linkedin.com/pulse/building-web-agents-jf-ai-khjic/)
