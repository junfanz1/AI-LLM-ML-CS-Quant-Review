LLM Cheat Sheet
---

[Contents](https://bitdowntoc.derlin.ch/)

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [1. Udemy LLM Project Notes](#1-udemy-llm-project-notes)
   * [LangChain](#langchain)
   * [LangGraph](#langgraph)
      + [LangGraph Researcher Agent](#langgraph-researcher-agent)
      + [RAG Self-Reflection Workflow ](#rag-self-reflection-workflow)
   * [Agent Framework](#agent-framework)
      + [Project](#project)
      + [RAG](#rag)
      + [LoRA (Low Rank Adaptation)](#lora-low-rank-adaptation)
      + [Training](#training)
- [2. DeepSeek MoE](#2-deepseek-moe)
- [3. DeepSeek-V3/R1](#3-deepseek-v3r1)
   * [DeepSeek-V3](#deepseek-v3)
   * [DeepSeek-R1](#deepseek-r1)
   * [DeepSeek Janus](#deepseek-janus)
   * [Kimi-K1.5](#kimi-k15)
- [101. Transformer如何设定learning rate?](#101-transformerlearning-rate)
- [102. Transformer: Why Positional Encoding?](#102-transformer-why-positional-encoding)
- [103. Deploy ML Applications?](#103-deploy-ml-applications)
- [104. MLOps：Model health in Prod?](#104-mlopsmodel-health-in-prod)
- [105. 优化RAG？](#105-rag)
- [106. LLM微调与优化？](#106-llm)

<!-- TOC end -->



<!-- TOC --><a name="1-udemy-llm-project-notes"></a>
# 1. Udemy LLM Project Notes


<!-- TOC --><a name="langchain"></a>
## LangChain

[Eden Marco: LangChain- Develop LLM powered applications with LangChain](https://www.udemy.com/course/langchain/?srsltid=AfmBOooPg0Xkc19q5W1430Dzq6MHGKWqHtq5a1WY4uUl9sQkrh_b_pej&couponCode=ST4MT240225B)

<img src="https://github.com/user-attachments/assets/545885af-9c0b-431c-b8d4-cc28a0b7d64f" width="50%" height="50%">

Projects
- https://github.com/junfanz1/Code-Interpreter-ReAct-LangChain-Agent
- https://github.com/junfanz1/LLM-Documentation-Chatbot

LangChain = LLM + Retriever (Chroma, vector storer) + Memory (list of dicts, history chats)
- LLM applications = RAG + Agents
- Simplifies creation of applications using LLMs (AI assistants, RAG, summarization), fast time to market
- Wrapper code around LLMs makes it easy to swap models 
- As APIs for LLMs have matured, converged and simplified, need for unifying framework like LangChain has decreased 

ReAct (Reason-Act)
- Paradigm that integrates language models with reasoning and acting capabilities, allowing for dynamic reasoning and interaction with external environments to accomplish complex tasks.
- Simplest agent is for-loop (ReAct), ReAct agents are flexible and any state is possible, but have poor reliability (eg. invoking the same tool always and stuck, due to hallucination, tool-misuse, task ambiguity, LLM non-determinism).

Autonomy in LLM applications (5 levels): 
- human code
- LLM call: one-step only
- Chain: multiple steps, but one-directional
- Router (LangChain): decide output of steps and steps to take (but no cycles), still human-driven (not agent executed)
- State Machine (LangGraph): Agent executed, where agent is a control flow controlled by LLM, use LLM to reason where to go in this flow and tools-calling to execute steps, agent can have cycles.


<!-- TOC --><a name="langgraph"></a>
## LangGraph

[Eden Marco: LangGraph-Develop LLM powered AI agents with LangGraph](https://www.udemy.com/course/langgraph)

<img src="https://github.com/user-attachments/assets/0511a3b1-a5d4-4255-8916-fc9cb2d08e99" width="50%" height="50%">

Projects:
- https://github.com/junfanz1/Cognito-LangGraph-RAG
- https://github.com/junfanz1/LangGraph-Reflection-Researcher


LangGraph
- LangGraph is both reliable (like Chain, that architects our state machine) and flexible (like ReAct).
- Flow Engineering (planning+testing), can build highly customized agents. (AutoGPT can do long-term planning, but we want to define the flow.) 
- Controllability (we define control flow, LLM make decisions inside flow) + Persistence + Human-in-the-loop + Streaming. 
- LangChain Agent. Memory (shared state across the graph), tools (nodes can call tools and modify state), planning (edges can route control flow based on LLM decisions).

<!-- TOC --><a name="langgraph-researcher-agent"></a>
### LangGraph Researcher Agent
https://github.com/assafelovic/gpt-researcher
- Implementing agent production-ready. There’re nodes and edges, but no cycles. We can integrate GPT Researcher (as a node under LangGraph graph) within Multi-Agent Architecture. (https://github.com/assafelovic/gpt-researcher/tree/master/multi_agents)
- Every agent in a multi-agent system can be a researcher, as part of workflow. e.g., `Technology` agent is talor-made for technological subjects, and is dynamically created/chosen
- Research automation needs to make a decision for a few deeper levels and iterate again again again until the optimal answer. Key difference here is not only width (in parallel then aggregation) but also depth

Reason for LangGraph in Multi-Agent Architecture
- LangGraph (Flow Engineering techniques addresses the tradeoff between agent freedom and our control) is more flexible in production than CrewAI (doesn’t have as much control of the flow)
- breaks down the problem into specific actions, like microservices, (1) with specialized tasks, we can control quality of nodes, (2) can scale up the application as it grows
- Customizability, creative framework
- Contextual compression is the best method for retrieving in RAG workflow
- Allow both web and local data indexing, with LangChain easy integration can embed anything
- Human-in-the-loop, let user decide how much feedback autonomy to interact with, especially useful when finding two knowledge sources that conflict or contradict each other. When this happens, AI needs human assistance.


<!-- TOC --><a name="rag-self-reflection-workflow"></a>
### RAG Self-Reflection Workflow 

LangGraph Components
- Nodes (Python functions)
- Edges (connect nodes)
- Conditional Edges (make dynamic decisions to go to node A or B)

State Management: dictionary to track the graph’s execution result, chat history, etc.

Reflection Agents: prompt to improve quality and success rate of agents/AI systems.

Self-reflects on 
- Document we retrieve
- Curate documents and add new info
- Answers if grounded in documents

We also implement a routing element, routing our request to correct datastore with info of the answer.

RAG Idea Foundations
- Self-RAG: reflect on the answer the model generated, check if answer is grounded in the docs.
- Adaptive RAG: (1) taking the route to search on a website, then continuing downstream on the same logic (2) use RAG from the vector store. Use conditional entry points for routing.
- Corrective RAG: Take query, vector search + semantic search, retrieve all docs, start to self-reflect and critique the docs, determine whether they’re relevant or not. If relevant, send to LLM, if not relevant, filter out and perform external Internet search to get more info, to augment our prompt with real-time online info, then augment the prompt and send to LLM.

Further Improvements

- LangGraph has a persistence layer via checkpoint object (save the state after each node execution, in persistence storage, e.g. SQLite). Can interrupt the graph and checkpoint the state of the graph and stop to get human feedback, and resume graph execution from the stop point.
- Create conditional branches for parallel node execution
- Use Docker to deploy to LangGraph Cloud, or use LangGraph Studio, LangGraph API to build LLM applications without frontend


<!-- TOC --><a name="agent-framework"></a>
## Agent Framework

[Ed Donnoer: LLM Engineering: Master AI, Large Language Models & Agents](https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models/learn/lecture/)

<img src="https://github.com/user-attachments/assets/e5bb6fb6-9c70-42e6-9d4a-7603d9646b26" width="50%" height="50%">

Building AI UIs with Gradio from HuggingFace using LLMs behind its scenes, implementing streaming responses

DALL-E-3, image generation model behind GPT-4o

Agent Framework, build multimodal (image, audio) AI assistant

Use HuggingFace pipelines, tokenizers and models, libraries: hub, datasets, transformers, peft (parameter efficient fine tuning), trl, accelerate

Use Frontier models/open source models to convert audio to text

Benchmarks comparing LLMs - HuggingFace Open LLM Leaderboard 

- ELO, evaluating Chats, results from head-to-head face-offs with other LLMs, as with ELO in Chess
- HumanEval, evaluating Python coding, 164 problems writing code based on docstrings
- MultiPL-E, evaluating broader coding, translation of HumanEval to 18 programming languages

Metrics to train LLM
- Cross-entropy loss: -log(predicted probability of the thing that turned out to be actual next token)
- Perplexity: e^{Cross-entropy loss}, if = 1 then model is 100% correct, if = 2 then model is 50% correct, if = 4 then model is 25% correct. Higher perplexity: how many tokens would need to be to predict next token

<!-- TOC --><a name="project"></a>
### Project

Autonomous Agentic AI framework (watches for deals published online, estimate price of products, send push notifications when it’s opportunity)

modal.com to deploy LLM to production, serverless platform for AI teams 

Agent Architecture/Workflows: 7 Agents work together (GPT-4o model identify deals from RSS feed, frontier-busting fine-tuned model estimate prices, use Frontier model with massive RAG Chroma datastore)
- UI, with Gradio
- Agent Framework: with memory and logging
- Planning Agent: coordinate activities
- Scanner Agent: identify promising deals
- Ensemble Agent: estimate prices using multiple models, and collaborate with other 3 agents
- Messaging Agent: send push notifications
- Frontier Agent: RAG pricer (based on inventory of lots of products, good use case for RAG)
- Specialist Agent: estimate prices 
- Random Forest Agent: estimate prices (transformer architecture)

SentenceTransformer from HuggingFace maps sentences to 384 dimensional dense vector space and is ideal for semantic search. 


<!-- TOC --><a name="rag"></a>
### RAG

RAG (Retrieval Augmented Generation) uses vector embeddings and vector databases to add contexts to prompts, define LangChain and read/split documents. 

Convert chunks of text into vectors using OpenAI Embeddings, store vectors in Chroma (open source AI vector datastores) or FAISS, and visualize the vector store in 3D, and reduce the dimension of vectors to 2D using t-SNE.
- Autoregressive LLM: predict future token from the past
- Autoencoding LLM: produce output based on full input. Good at sentiment analysis, classification. (BERT, OpenAI Embeddings)

LangChain’s decorative language LCEL

Professional private knowledge base can be vectorized in Chroma, vector datastore, and build conversational AI. Use libraries to connect to email history, Microsoft Office files, Google Drive (can map to Google Colab to vectorize all documents) and Slack texts in Chroma. Use RAG to get the 25 closest documents in the vector database. Use open source model BERT to do vectorization by myself. Use Llama.CPP library to vectorize all documents without the need to go to cloud. 

Use Transfer learning to train LLMs, take pretrained models as base, use additional training data to fine-tune for our task. 

Generate text and code with Frontier models including AI assistants with Tools and with open source models with HuggingFace transformers. Create advanced RAG solutions with LangChain. Make baseline model with traditional ML and making Frontier solution, and fine-tuning Frontier models.

Fine-tuning open source model (smaller than Frontier model)

Llama 3.1 architecture
- 8B parameters, 32G memory, too large and costly to train.
- 32 groups of layers, each group = llama decoder layer

<!-- TOC --><a name="lora-low-rank-adaptation"></a>
### LoRA (Low Rank Adaptation)

Freeze main model, come up with a bunch of smaller matrices with fewer dimensions, they’ll get trained and be applied using simple formulas to target modules. So we can make a base model that gets better as it learns because of the application of LoRA matrices.
- Freeze weights, we don’t optimize 8B weights (too many gradients), but we pick a few layers (target modules) that we think are key things we want to train. We create new matrices (Low Rank Adaptor) with fewer dimensions, and apply these matrices into target modules. So fewer weights are applied to target modules.
- Quantization (Q in QLoRA): Keep the number of weights but reduce precision of each weight. Model performance is worse, but impact is small. 

3 Hyperparameters for LoRA fine-tuning
- r, rank, how many dimensions in low-rank matrices. Start with 8, 16, 32 until diminishing returns 
- Alpha, scaling factor that multiplies the lower rank matrices. Alpha = 2 * r, the bigger the more effective.
- Target modules, which layers of NN are adapted. Target the attention head layers.

fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL), after quantized to 8 bit or 4 bit, model size reduced to 5000MB, after fine-tuned LoRA matrices applying to big model, size of weights reduced to 100MB.

5 Hyperparameters for QLoRA fine-tuning
- Target modules
- r, how many dimensions
- alpha, scaling factor to multiply up the importance of adaptor when applying to target modules, by default = 2 * r
- Quantization
- Dropout, regularization technique, to prevent overfitting

<!-- TOC --><a name="training"></a>
### Training
- Epochs, how many times we go through the entire dataset when training. At the end of each epoch, we save the model and the model gets better in each epoch before overfitting then gets worse; we pick the best model and that’s the result of training.
- Batch size, take a bunch of data together rather than step by step, it’s faster and better performance, because for multiple epochs, in each epoch the batch is different. 
- Learning rate, = 0.0001, predict the next token vs. the actual next token should be -> loss, how poorly it predicts the actual, use loss to do back propagation to figure out how to adjust weight to do better, the amount that it shifts the weights is learning rate. During the epochs we can gradually lower the learning rate, to make tiny adjustments as the model gets trained.
- Gradient accumulation. Improve speed of going through training. We can do forward pass and get the gradient, and don’t take a step, just do a second forward pass and add up the gradients and keep accumulating gradients and then take a step and optimize the network. Steps less frequently = faster. 
- Optimizer. Algorithm that updates NN to shift everything a bit to increase the prediction accuracy of the next token.

4 Steps in Training
- Forward pass, predict next token in training data
- Loss calculation, how different was it to the true token
- Backpropagation, how much (sensitivity) should we tweak parameters to do better next time (gradients)
- Optimization, update parameters a tiny step to do better next time

Loss function: cross-entropy loss = -log Prob(next true token), = 0 : 100% confident of right answer, higher number = lower confidence.

Carry out end-to-end process for selecting and training open source models to build proprietary verticalized LLM (deploy multiple models to production, including LLM on Modal, RAG workflow with Frontier model) to solve business problems that can outperform the Frontier model. 

Run inference on a QLoRA fine-tuned model.

## Cursor

> [Eden Marco: Cursor Course: FullStack development with Cursor AI Copilot](https://www.udemy.com/course/cursor-ai-ide/)

<img src="https://github.com/user-attachments/assets/71c2bd39-a1a1-410c-a541-0615e4608995" width="50%" height="50%">

<!-- TOC --><a name="2-deepseek-moe"></a>
# 2. DeepSeek MoE
https://space.bilibili.com/517221395/upload/video

https://github.com/chenzomi12/AIFoundation/

DeepSeek V-1 MoE (2024.01)
- 专家共享机制，部分专家在不同Token或层间共享参数，减少冗余。
- 内存优化，用多头潜在注意力机制MLA+键值缓存优化，减少延迟。

Mixture of Experts (MoE) 混合专家模型。参数小、多专家，监督学习+分而治之，模块化神经网络的基础，像集成学习。根据scaling law大模型性能好，而推理时只执行部分参数，故DeepSeek成本低。

MoE模型架构
- 稀疏MoE层，代替了Transformer FFN层（节省算力），包含很多专家，每个专家是一个神经网络。稀疏性让部分专家被激活，非所有参数参与计算。高效推理同时，扩展到超大规模，提升模型表征能力。
- 专家模块化，不同专家学习不同特征，处理大数据。门控网络Gating Network或路由=可学习的门控网络+专家间负载均衡，动态协调哪些token激活哪些专家参与计算，与专家一起学习。稀疏门控激活部分专家，稠密门控激活所有专家，软门控合并专家与token并可微。

训练效率提升
- 训练。专家并行EP使用All2All通讯（带宽少），每个专家处理一部分batch（增加吞吐）。
- 推理。只激活少量专家（低延迟），增加专家数量（推理成本不变）。

MoE优化
- 专家并行计算
- 提高容量因子Capacity Factor和显存带宽
- MoE模型蒸馏回对应的稠密小模型
- 任务级别路由+专家聚合，简化模型减少专家

DeepSeek GRPO与LLM主流RLHF两大路线
- On-Policy (PPO)：每次训练都基于自己的生成模型Actor，通过教练Critic反馈奖励。好：效率高，坏：模型能力低。PPO共有4个模型（Actor, Critic, Reward, Reference），计算大。
- Off-Policy (DPO)：基于现有标注进行分析，可能样本与模型不匹配。好：可能达到模型上限，坏：效率低。
- GRPO=无需价值函数，与奖励模型的比较性质对齐，KL惩罚在损失函数中。DeepSeek GRPO避免了PPO用Critic Value Model近似，而是用同一问题下多个采样输出的平均奖励作基线，这样Actor（没了Critic）直接去对齐Reward，求均值后再去跟Policy求KL散度。


<!-- TOC --><a name="3-deepseek-v3r1"></a>
# 3. DeepSeek-V3/R1

https://www.bilibili.com/video/BV1DJwRevE6d/

<!-- TOC --><a name="deepseek-v3"></a>
## DeepSeek-V3
- Multi-Head Latent Attention (MLA)：引入潜在空间提高计算效率，并保持模型对输入数据复杂关系的捕捉能力。
- Mixture of Expert (MoE)：高效专家分配（负载均衡）和计算资源利用来降低成本。
- FP8量化（混合精度训练）+多token预测（并行推理，就像随机采样，加速解码过程），提高理解能力。
- 分阶段训练，性能提升依赖于算法升级（post-training, RL，知识蒸馏）
- 通信优化DulePipe双流水线并行优化


<!-- TOC --><a name="deepseek-r1"></a>
## DeepSeek-R1
- R1-Zero（探索RL+LLM）=基于规则的奖励（准确率奖励+思考过程格式奖励）+推理为中心的大规模强化学习（组相对策略优化GRPO+瞄准Reasoning推理任务）。
- R1（工程和数据调优）=V3+GRPO，好：自我进化，具有test-time reasoning。 坏：reasoning可读性差，中英混杂。
- 相比DeepSeek-v3-Base，增强了推理链可读性（用高质量数据冷启动让RL更稳定+推理为中心的RL）、提升通用能力和安全性（拒绝采样和全领域SFT+全领域All Scenario RL）。
- 无需监督微调（节省标注成本）的纯强化学习驱动（需要强大V3基座模型+GRPO强化学习优化+推理问题可以自动化标记验证）。基于强化学习的后训练Post-Training Scaling Law，有强大推理能力和长文本思考能力。 因为大模型预训练的边际收益递减，自回归的数学推理难以自我纠错。
- 涌现出检查、反思、长链推理。即使是稀疏奖励信号，模型也能自然探索出验证回溯总结反思。
- GRPO：构建多个模型输出的群组（多个回答），计算群组内相对奖励来估计基线。相比PPO的价值函数是大模型大计算，GRPO省略了Value Model而用群组相对方式计算优势值，将策略模型与参考模型的KL散度作为正则项加入损失函数（而非奖励函数），大幅降低RL计算成本。
- 四阶段交替训练：SFT、RL、再SFT、再RL，解决冷启动和收敛效率问题。涌现出检查、反思、长链推理。
- 冷启动数据+SFT给V3，由GRPO强化后得到R1，使用rejection sampling得到reasoning data再去微调V3，几轮post-training迭代后得到R1，蒸馏出小模型。

<!-- TOC --><a name="deepseek-janus"></a>
## DeepSeek Janus
- Janus：把理解和生成任务合并统一成自回归Transformer架构。understanding encoder（用SigLIP对图像编码）和generation encoder（用VQ-tokenizer）不同，二者编码后都经过adaptor进入自回归LLM架构，因此提升模型灵活性同时缓解生成-理解的冲突。
- Janus-Flow：不同于Janus，Generation部分基于Rectified Flow（如Stable Diffusion3）在LLM内融合两种架构，encoder-decoder迭代配对，统一视觉理解与文本生成。
- Janus-Pro：用ImageNet充分训练，用多模态数据后训练。


<!-- TOC --><a name="kimi-k15"></a>
## Kimi-K1.5
- 强化学习让模型试错。In-Context RL不训练模型规划，而是模拟规划approximate planning（将每个state和value都视为language tokens，建模成contextual bandit问题，用REINFORCE变种来优化，并用长度惩罚机制防止overthinking算力损耗）。
- 采样策略：课程学习循序渐进+优先采样做难题。
- 四阶段：Pretraining、SFT、Long CoT SFT、RL。
- 构造Vision Data（把文本内容转化为视觉格式）+Long2Short蒸馏（用够短思维链达到长思维链效果：模型融合+最短拒绝采样+DPO）。

技术对比
- Kimi K1.5与DeepSeek-R1对比：Kimi K1.5更多从In-Context RL出发，直接训练模型approximate planning，将思考过程建模到语言模型的next token prediction中，但复杂推理可能难以迭代。DeepSeek-R1从纯RL出发，用GPRO+Rule-based reward激活模型本身的推理能力。二者都根据答案对错给予奖励惩罚。
- 蒸馏与强化学习对比：蒸馏就是学生学习强大老师，短期内掌握复杂技能。强化学习是试错尝试，泛化性更好，因为SFT主要负责记忆而非理解故很难分布外泛化。DeepSeek用R1蒸馏出的小模型，因为R1强大，发现了高阶推理范式（小模型直接RL则难以发现因为预训练知识不足），蒸馏甚至超过RL的方法。

未来展望
- 长思维链的欺骗性推理In-Context Scheming、奖励篡改。要引入AI-driven监督机制、对比推理。
- 模态扩展+模态穿透（强推理+多模态），拓展模型推理边界。因为RLHF+DPO模态无感，用从语言反馈中学习Learning from Language Feedback，捕捉人类意图的多元偏好和复杂模态交互，实现any-to-any models与人类意图对齐。
- 强推理赋能Agentic，要反思、长程规划、工具调用。
- 大模型就像压缩器，因为弹性而抗拒对齐，用Deliberative Alignment审计对齐把宪法融入到模型推理过程。
- LLM受限于过程性推理任务，人类可以抽象出高维概念并细粒度反馈。


---

<!-- TOC --><a name="101-transformerlearning-rate"></a>
# 101. Transformer如何设定learning rate?

Learning rate是训练Transformer的超参数，决定了优化过程中每次迭代的步长大小，显著影响模型的收敛速度和最终性能。

设置策略

1. 网络搜索Grid Search
  -   验证集上实验一系列学习率
  -   计算成本高，对于大模型
2. 随机搜索
  -   指定范围内随机采样学习率
  -   比网络搜索更有效，对于高维超参数空间
3. 学习率调度
  -   预热：从较低学习率开始，逐渐增到峰值。有助于模型在训练早期稳定。
  -   衰减：逐渐降低学习率，避免过拟合。线性衰减、指数衰减、余弦衰减。
4. 技巧
  -   从保守的学习率开始，较小的学习率可以防止发散。
  -   监控训练损失，若损失增加，降低学习率。
  -   用学习率调度器，自动调整学习率。预热-线性衰减warm up-linear decay、余弦退火consine annealing（改善收敛的循环学习率调度）、polynomial decay（灵活调度，允许不同衰减率）
  -   用不同的优化器（Adam, AdamW）
  -   考虑批大小和模型大小，更大的批大小和模型需要更低的学习率。
  -   微调预训练模型，需要更低的学习率。

<!-- TOC --><a name="102-transformer-why-positional-encoding"></a>
# 102. Transformer: Why Positional Encoding?

Transformer缺乏对序列顺序的理解。与RNN、CNN不同，Transformer不会逐个处理输入序列中的元素，而是同时处理所有元素。并行处理虽然高效，但丧失了对序列中元素位置信息的感知。

Positional Encoding为输入序列中的每个元素添加独特的表示，指示其相对或绝对位置，这对于机器翻译、文本摘要、问答至关重要。

Why

1. 保留序列顺序，理解序列和元素的关系。
2. 捕获长距离依赖，理解复杂语言结构。对于机器翻译，一个词的含义可能取决于远的词，这样可以帮助模型理解这种关联。
3. 提升模型性能，让模型理解上下文。

How

对于位置i和维度d

$PE(pos, 2i) = \sin(pos / 10000^{2i/d_{model}})$

$PE(pos, 2i + 1) = \cos(pos / 10000^{2i/d_{model}})$

<!-- TOC --><a name="103-deploy-ml-applications"></a>
# 103. Deploy ML Applications?

用Nginx部署机器学习应用

1. 准备ML应用，用Flask或FastAPI开发接口，把模型预测功能暴露出来
   
```python
@app.route('/predict', methods=['POST'])
def predict():
  data = request.json
  result = model.predict([data['input']])
  return jsonify({'prediction': result[0]})
```

用Gunicorn运行

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

2. 安装Nginx

```bash
sudo apt update && sudo apt install nginx
sudo systemctl start nginx
```

3. 配置Nginx reverse proxy
```bash
sudo nano /etc/nginx/sites-available/ml_app
```
写入：
```nginx
server {
  listen 80;
  location/{
    proxy_pass http://127.0.0.1:5000;
  }
}
```
启用配置：
```bash
sudo ln -s/etc/nginx/sites-available/ml_app/etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

4. 配置HTTPS
用Certbot一键开启SSL：
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d <域名>
```

<!-- TOC --><a name="104-mlopsmodel-health-in-prod"></a>
# 104. MLOps：Model health in Prod?

1. 监控输入数据
  - 数据漂移：生产数据和训练数据分布是否一致，特征均值和方差
  - 数据质量：用自动化工具看缺失值、异常值问题

2. 跟踪模型表现
  - 预测漂移：模型输出是否变化 ，如分类概率分布偏移
  - 性能指标：评估准确率，特别是有标签反馈时
  - 响应时间：延迟报警，确保预测速度稳定

3. 资源监控

4. 配置告警和日志

5. 防止模型退化
  - 检查漂移（输入输出关系变化），检测模型是否需要更新
  - 构建自动化的数据-模型-系统全链路监控，可视化工具

<!-- TOC --><a name="105-rag"></a>
# 105. 优化RAG？

1. 优化Retrieval Component
  - 构建高质量知识库，确保知识库是最新高质量内容，通过筛选和过滤，提升检索相关性。
  - 微调检索器，使其更能理解和优化排序内容
  - 用领域特定的嵌入。预训练的嵌入模型通常缺乏专业领域特异性，可以对嵌入模型微调，用领域适配的语言模型，提供语义匹配精准度。
  - 添加上下文过滤器，通过标签、元数据等过滤条件，将检索范围限定在子领域，提升准确度，确保只获取相关知识库内容。
2. 增强generation component
  - 特定领域文本上微调生成器。生成模型微调，使其掌握语言风格、术语和表达方式，生成专业性更强、准确的回复。
  - 结合知识基础的训练，用特定领域问答进行训练，让模型准确引用检索内容，避免依赖通用知识。
  - 优化输出格式，如科学文献引用、分布指令。
3. 优化检索与生成的互动
  - 实现重排序技术。检索后，用交叉编码器或相关性评分对检索到的段落重排序，确保生成器获取最丰富信息的上下文。
  - 用查询扩展技术。用同义词或相关术语扩展查询，提高检索器捕捉重要内容的概率。
  - 调整检索到的文档或token数量，确保生成器既能获取上下文，又不至于超载，提高效率。
4. 优化索引技术
  - 分层索引策略。将知识库按领域分层索引，加快特定领域文档的检索速度，让检索器快速定位到子领域。
  - 动态索引更新。对知识频繁变化的部分，用动态索引或定期更新索引，确保内容最新。
  - 用向量索引加快语义搜索，如FAISS工具提高效率。

<!-- TOC --><a name="106-llm"></a>
# 106. LLM微调与优化？

微调

1. 有监督微调
    - 在特定标记数据集对LLM训练，数据集与目标任务相关，如文本分类、物体识别、问答。
    - 模型学习将其表示调整到特定数据集的细微差别，并保留预训练期间获得的知识。
2. 无监督微调
    - 过程。在缺乏标记数据情况下，用大量未标记的文本对LLM进一步训练，可以模型细化对语言和上下文的理解。
    - 增强模型生成或理解文本的能力，不用显式的标签。
3. 特定领域微调
    - 医疗、法律、技术的数据集对LLM微调，提高性能
    - 模型学习特定领域的属于，提高专业相关性和准确性
4. 少样本、零样本学习
    - 少样本学习，为模型提高少量任务实力，调整预测
    - 零样本学习，要求模型执行从未显式训练过的任务，利用通用语言理解能力

优化

1. 学习率调度
   - 训练过程中调整学习率，改善收敛性。用余弦退火或学习率预热技术。
   - 稳定训练，防止过度调整，允许模型更有效地收敛。
2. 梯度裁剪
   - 在反向传播过程中限制梯度，防止梯度爆炸，特别是深度模型中。
   - 保持稳定更新，避免数值不稳定性
3. 批量归一化和层归一化
   - 对每层输入进行归一化，稳定和加速训练
   - 减少内部协变量转变，提高收敛速度
4. 正则化
   - 训练中随机丢弃一些神经元，防止过拟合
   - 权重衰减，对权重大小添加乘法，防止过拟合

迁移学习

- 用预训练模型的权重作为新任务训练的起点
- 用LLM通用语言理解能力，在特定任务训练更快更高效

任务特定的头层 Task-specific head layers

- 预训练模型上添加特定任务的层，如classification head，这些层从头开始训练，或用预训练模型的权重进行初始化。
- 目标，将模型输出调整为特定任务，同时保持底层语言表示的完整性。

提示工程

- 为少样本或零样本任务设计有效提示，引导模型响应
- 提供更清晰指令或示例，改善模型在特定任务的表现

数据增强

- 同义词替换、反向翻译或改写等方法，创建额外训练样本
- 增加训练数据的多样性和数量
