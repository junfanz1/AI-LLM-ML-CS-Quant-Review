ML/LLM Cheat Sheet
---

[Contents](https://bitdowntoc.derlin.ch/)

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [1. Transformer如何设定learning rate?](#1-transformerlearning-rate)
- [2. Transformer: Why Positional Encoding?](#2-transformer-why-positional-encoding)
- [3. Deploy ML Applications?](#3-deploy-ml-applications)
- [4. MLOps：Model health in Prod?](#4-mlopsmodel-health-in-prod)
- [5. 优化RAG？](#5-rag)
- [6. LLM微调与优化？](#6-llm)

<!-- TOC end -->

# 0. Udemy LLM Project Keynotes

https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models/learn/lecture/

Building AI UIs with Gradio from HuggingFace using LLMs behind its scenes, implementing streaming responses

DALL-E-3, image generation model behind GPT-4o

Agent Framework, build multimodal (image, audio) AI assistant

Use HuggingFace pipelines, tokenizers and models, libraries: hub, datasets, transformers, peft (parameter efficient fine tuning), trl, accelerate

Use Frontier models/open source models to convert audio to text

Benchmarks comparing LLMs - HuggingFace Open LLM Leaderboard 

- ELO, evaluating Chats, results from head-to-head face-offs with other LLMs, as with ELO in Chess
- HumanEval, evaluating Python coding, 164 problems writing code based on docstrings
- MultiPL-E, evaluating broader coding, translation of HumanEval to 18 programming languages

## Metrics to train LLM
- Cross-entropy loss: -log(predicted probability of the thing that turned out to be actual next token)
- Perplexity: e^{Cross-entropy loss}, if = 1 then model is 100% correct, if = 2 then model is 50% correct, if = 4 then model is 25% correct. Higher perplexity: how many tokens would need to be to predict next token

## RAG

RAG (Retrieval Augmented Generation) uses vector embeddings and vector databases to add contexts to prompts, define LangChain and read/split documents. 

Convert chunks of text into vectors using OpenAI Embeddings, store vectors in Chroma (open source AI vector datastores) or FAISS, and visualize the vector store in 3D, and reduce the dimension of vectors to 2D using t-SNE.
- Autoregressive LLM: predict future token from the past
- Autoencoding LLM: produce output based on full input. Good at sentiment analysis, classification. (BERT, OpenAI Embeddings)

## LangChain

LangChain = LLM + Retriever (Chroma, vector storer) + Memory (list of dicts, history chats)
- Simplifies creation of applications using LLMs (AI assistants, RAG, summarization), fast time to market
- Wrapper code around LLMs makes it easy to swap models 
- As APIs for LLMs have matured, converged and simplified, need for unifying framework like LangChain has decreased 

LangChain’s decorative language LCEL

Professional private knowledge base can be vectorized in Chroma, vector datastore, and build conversational AI. Use libraries to connect to email history, Microsoft Office files, Google Drive (can map to Google Colab to vectorize all documents) and Slack texts in Chroma. Use RAG to get the 25 closest documents in the vector database. Use open source model BERT to do vectorization by myself. Use Llama.CPP library to vectorize all documents without the need to go to cloud. 

Use Transfer learning to train LLMs, take pretrained models as base, use additional training data to fine-tune for our task. 

Generate text and code with Frontier models including AI assistants with Tools and with open source models with HuggingFace transformers. Create advanced RAG solutions with LangChain. Make baseline model with traditional ML and making Frontier solution, and fine-tuning Frontier models.

## Fine-tuning open source model (smaller than Frontier model)

Llama 3.1 architecture
- 8B parameters, 32G memory, too large and costly to train.
- 32 groups of layers, each group = llama decoder layer

## LoRA (Low Rank Adaptation)

Freeze main model, come up with a bunch of smaller matrices with fewer dimensions, they’ll get trained and be applied using simple formulas to target modules. So we can make a base model that gets better as it learns because of the application of LoRA matrices.
- Freeze weights, we don’t optimize 8B weights (too many gradients), but we pick a few layers (target modules) that we think are key things we want to train. We create new matrices (Low Rank Adaptor) with fewer dimensions, and apply these matrices into target modules. So fewer weights are applied to target modules.
- Quantization (Q in QLoRA): Keep the number of weights but reduce precision of each weight. Model performance is worse, but impact is small. 

## 3 Hyperparameters for LoRA fine-tuning
- r, rank, how many dimensions in low-rank matrices. Start with 8, 16, 32 until diminishing returns 
- Alpha, scaling factor that multiplies the lower rank matrices. Alpha = 2 * r, the bigger the more effective.
- Target modules, which layers of NN are adapted. Target the attention head layers.

fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL), after quantized to 8 bit or 4 bit, model size reduced to 5000MB, after fine-tuned LoRA matrices applying to big model, size of weights reduced to 100MB.

## 5 Hyperparameters for QLoRA fine-tuning
- Target modules
- r, how many dimensions
- alpha, scaling factor to multiply up the importance of adaptor when applying to target modules, by default = 2 * r
- Quantization
- Dropout, regularization technique, to prevent overfitting

## Training
- Epochs, how many times we go through the entire dataset when training. At the end of each epoch, we save the model and the model gets better in each epoch before overfitting then gets worse; we pick the best model and that’s the result of training.
- Batch size, take a bunch of data together rather than step by step, it’s faster and better performance, because for multiple epochs, in each epoch the batch is different. 
- Learning rate, = 0.0001, predict the next token vs. the actual next token should be -> loss, how poorly it predicts the actual, use loss to do back propagation to figure out how to adjust weight to do better, the amount that it shifts the weights is learning rate. During the epochs we can gradually lower the learning rate, to make tiny adjustments as the model gets trained.
- Gradient accumulation. Improve speed of going through training. We can do forward pass and get the gradient, and don’t take a step, just do a second forward pass and add up the gradients and keep accumulating gradients and then take a step and optimize the network. Steps less frequently = faster. 
- Optimizer. Algorithm that updates NN to shift everything a bit to increase the prediction accuracy of the next token.

## 4 Steps in Training
- Forward pass, predict next token in training data
- Loss calculation, how different was it to the true token
- Backpropagation, how much (sensitivity) should we tweak parameters to do better next time (gradients)
- Optimization, update parameters a tiny step to do better next time

Loss function: cross-entropy loss = -log Prob(next true token), = 0 : 100% confident of right answer, higher number = lower confidence.

Carry out end-to-end process for selecting and training open source models to build proprietary verticalized LLM (deploy multiple models to production, including LLM on Modal, RAG workflow with Frontier model) to solve business problems that can outperform the Frontier model. 

Run inference on a QLoRA fine-tuned model.

## Capstone Project

Autonomous Agentic AI framework (watches for deals published online, estimate price of products, send push notifications when it’s opportunity)

modal.com to deploy LLM to production, serverless platform for AI teams, 

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

Build advanced RAG solution without LangChain

Build ensemble model 



# 1. Transformer如何设定learning rate?

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

# 2. Transformer: Why Positional Encoding?

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

# 3. Deploy ML Applications?

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

# 4. MLOps：Model health in Prod?

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

# 5. 优化RAG？

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

# 6. LLM微调与优化？

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
