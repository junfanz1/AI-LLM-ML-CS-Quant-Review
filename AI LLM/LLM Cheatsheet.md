Cheat Sheet
---

[Contents](https://bitdowntoc.derlin.ch/)

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [101. Transformer如何设定learning rate?](#101-transformerlearning-rate)
- [102. Transformer: Why Positional Encoding?](#102-transformer-why-positional-encoding)
- [103. Deploy ML Applications?](#103-deploy-ml-applications)
- [104. MLOps：Model health in Prod?](#104-mlopsmodel-health-in-prod)
- [105. 优化RAG？](#105-rag)
- [106. LLM微调与优化？](#106-llm)

<!-- TOC end -->

---

以下为高级 AI/ML 工程师岗位技术面试中常见核心问题及高水平参考答案，涵盖系统架构、模型调优、部署运维、评估监控等方面：

---

### 1. 如何设计一个支持多租户、多模型、多适配器的推理服务？如何隔离资源？
**答：** 基于 vLLM + Triton，在 Kubernetes 中为每个租户创建独立 Namespace 并配置 `limitRange`、`ResourceQuota` 与 GPU `nodeSelector`；请求通过 BentoML/Ray Serve 路由到指定模型与 Adapter；Prometheus + Grafana 监控各租户 QPS、token 用量与时延。

---

### 2. 如何在生产环境实现 token billing 与准实时用量监控？
**答：** 在推理中间件统计请求和响应的 token 使用（调用 HuggingFace tokenizer），将数据写入 Kafka；Flink 实时消费并存入 ClickHouse；Grafana 展示按租户、模型、时间粒度的 token 计费与错误率。

---

### 3. 如何无停机更新 embedding 模型并回填索引？
**答：** 采用双写入与双索引：新 embedding 写入 shadow index，同时保留旧 index；批量回填完成后，通过 feature flag 或蓝绿部署切换查询路由；canary 验证无误后剔除旧索引。

---

### 4. LoRA 多任务微调后如何实现 Adapter Routing 或合并？
**答：** 使用 PEFT 创建针对每个任务的 Adapter，并借助 Adapter Fusion 或 MergeKit 生成运行时路由表；通过前置分类器或 Prompt Meta 标签选取正确 Adapter，或将多个 Adapter 权重线性融合后部署。

---

### 5. 设计一个 RAG 系统：架构与性能优化。
**答：**  
- **Ingestion：** OCR/ETL → 文本清洗 → Sliding window chunking → Embedding (OpenAI/BGE)  
- **Index：** FAISS + metadata filter 支持 hybrid 批量检索  
- **Retriever：** Query rewrite → Dense retriever → Light-weight reranker (e.g. Cohere)  
- **Generator：** MapReduce Chain 拼接 top-k 文档  

优化：prompt compression、cache 热点查询、异步批量检索。

---

### 6. 如何检测与纠正 LLM 的 Hallucination？
**答：**  
- 在线：LLM-as-critic (GPT-4 judge) 给出 factuality score；  
- 离线：收集错误案例，构造对比训练集，用 DPO/SLiC 精调；  
- 结构化校验：Regex & JSON schema 验证 → 失败触发 retry 或 fallback。

---

### 7. 如何设计在线 A/B 实验评估模型迭代效果？
**答：** Prompt hash router 将请求按比例分配至 control/candidate；在线收集 latency、token use、human feedback；使用 Uplift Modeling 分析差异；通过 DataDog/Grafana 仪表盘实时监控。

---

### 8. BPE 与 WordPiece 区别与自定义 tokenizer 实践。
**答：**  
- BPE 基于贪心合并高频子词，WordPiece 最优化语言模型似然；  
- WordPiece 对低频语言兼容更好；  
- 实践：使用 HuggingFace tokenizers 训练领域特定 tokenizer（定制 vocab、添加专有名词、设置特殊 tokens），提升下游召回与生成质量。

---

### 9. 性能优化与成本控制策略。
**答：**  
- INT4 或 GPTQ 量化推理；  
- LoRA/QLoRA 代替全参微调；  
- Batch & KV Cache；  
- 异步任务与优先级队列；  
- Cache 热点 prompts。

---

### 10. 延迟瓶颈定位与诊断方法。
**答：**  
- 链路追踪 (OpenTelemetry)：拆分 Tokenization→Embedding→Retrieval→Generation latency；  
- GPU profiling (PyTorch Profiler)；  
- 网络与 I/O 分析 (Wireshark, iostat)；  
- 监控热点 slow requests。

---

### 11. Fine-tuned 模型业务价值评估。
**答：**  
- 定义核心 KPI：ROUGE/F1/Accuracy + 用户转化率；  
- 对比基线模型表现与真实业务指标变化；  
- 使用 RICE 排序与成本收益分析；  
- Online A/B 验证及统计显著性检验。

---

### 12. LoRA vs QLoRA vs 全参微调权衡。
**答：**  
- LoRA 轻量、快速且易部署；  
- QLoRA 适用于大模型与有限显存；  
- 全参微调在高质量私有数据上表现最佳；  
- 实践：小模型 LoRA，中模型 QLoRA，大模型全参或 Hybrid。

---

### 13. Embedding 压缩与索引优化。
**答：**  
- PCA 降维后使用 Product Quantization (Faiss PQ)；  
- Optimized Product Quantization (SOPQ)；  
- IVF+PQ 混合索引；  
- 倒排表结合 metadata filter 提升精度。

---

### 14. MLOps：CI/CD、版本管理、灰度与回滚。
**答：**  
- 使用 GitOps (ArgoCD)，模型产物在 MLflow Registry；  
- CI 触发单元/集成测试 + 模型校验；  
- Canary Release via Flagger；  
- 自动监控误差阈值触发 rollback。

---

### 15. 上线后评估指标设计。
**答：**  
- 技术指标：Latency, QPS, Outlier rate；  
- 质量指标：Factuality score, BLEU/ROUGE；  
- 业务指标：Click-through, Session length；  
- 可靠性：MTTR, MTBF。

---

### 16. 用户反馈采集与闭环训练。
**答：**  
- UI 侧显式反馈 (👍/👎, correction)；  
- 隐式反馈 (点击, 转化)；  
- 收集到 Data Lake, 定期构造 SFT 和 PPO 训练集；  
- 自动化 retraining pipeline (Airflow + Kubeflow)。

---

### 17. Fallback 机制设计。
**答：**  
- LLM 超时或失败 → 缓存回复 → 规则引擎回复；  
- 输出校验失败 → 限制 sampling 参数重试；  
- 频度监控触发流量削峰。

---

### 18. 无 LLM 场景解决方案。
**答：**  
- 经典 IR：Elasticsearch BM25 + TF-IDF；  
- 规则引擎：基于 DSL 的模板匹配；  
- 轻量 ML：sklearn TF-IDF + Logistic Regression；  
- 混合方案：规则+相似度+简单 Seq2Seq 微调。

---

### 19. 结构化 Prompt 与 Prompt Injection 防护。
**答：**  
- 使用 Function Calling schema 强制返回 JSON；  
- 添加 prompt guardrails (sanitize user input)；  
- 对可调参数使用 allowlist；  
- 监控 injection 历史案例并更新 blacklist。

---

### 20. 数据集构建与 Loss 设计。
**答：**  
- 从日志提取 `<prompt, response, feedback>` triplets；  
- 使用 Pairwise Ranking Loss 或 DPO Loss；  
- 对抗性样本增强与负例采样；  
- Loss 加权：质量反馈高的样本权重更大。

---

### 21. 向量数据库对比：Qdrant vs Weaviate vs Pinecone。
**答：**  
- Qdrant 性能高、Rust 实现、社区活跃；  
- Weaviate 支持 GraphQL 查询、多模态；  
- Pinecone SaaS 便捷、专有优化；  
- 选型考虑延迟、过滤能力、成本与运维投入。

---

### 22. Embedding Drift 检测与治理。
**答：**  
- 定期计算 embedding distribution stats；  
- 使用 KL divergence 或 Wasserstein 距离；  
- Drift 超阈值触发模型更新或人工审核；  
- 实时报警与可视化追踪。

---

### 23. 多模态系统设计：VLM/VLA 架构。
**答：**  
- Early Fusion：拼接图像与文本 embedding；  
- Late Fusion：独立编码后融合 logits 或 cross-attention；  
- CLIP+LLaVA Pipeline：CLIP 图像 encode → LLM cross-attend；  
- 提取 OCR、图像区域特征结合 Retrieval。

---

### 24. VQA 系统中的 Attention 联结原理。
**答：**  
- 使用 Visual Transformer 获得 patch-level embedding；  
- Text-Image Cross Attention 在 QKV 上计算 attention maps；  
- Align 图文 token via multi-head softmax weight；  
- 可视化 attention heatmap 辅助 debug。

---

### 25. 系统观察性：监控、Tracing、Logging。
**答：**  
- 全链路 trace ID（OpenTelemetry）  
- 结构化日志 (JSON) 写入 ELK  
- 指标采集 (Prometheus) + Dashboard(Grafana)  
- Alerts (Slack, PagerDuty)

---

### 26. 可确定性：Token Sampling 控制策略。
**答：**  
- 设置 top-k、top-p、temperature=0 限制随机性；  
- 强制 seed 与 deterministic operators；  
- Prompt 工具链固定化；  
- 对多步骤 chain-of-thought 输出进行校验。

---

### 27. Embedding 模型替换与无缝回填。
**答：**  
- 双版本 embedding 列：`embedding_v1` & `embedding_v2`；  
- 在检索阶段均查询两版结果，基于阈值动态合并；  
- 后台批量 backfill，监控插入速率与效果；  
- 切换时更新路由表，无需停机。

---

### 28. Tokenizer 与 Positional Encoding 底层原理。
**答：**  
- Tokenizer：BPE/WordPiece 基于最大似然或频率合并子词；  
- Positional Encoding：sinusoidal 通过不同频率编码位置信息，或 learnable embeddings；  
- 公式：  
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))


---

### 29. Fine-tuning 过程：Optimizer、LR、Layer Freezing。
**答：**  
- 使用 AdamW + weight decay；  
- Warmup + cosine decay LR schedule；  
- 冻结底层 layers 保留预训练知识，仅微调高层；  
- 使用 gradient accumulation 支持大 batch。

---

### 30. Function Calling 与 Tool-Use 策略。
**答：**  
- 利用 OpenAI function schema 声明接口；  
- Prompt 先调用判断函数，再拼接参数执行；  
- 异常 fallback 到自然语言模式；  
- 确保函数签名与 API 文档一致。

---

### 31. 多智能体系统设计：LangGraph vs CrewAI vs AutoGen。
**答：**  
- LangGraph 侧重反射与回溯，多节点图结构；  
- CrewAI 强调 Guardrail 安全与任务分工；  
- AutoGen 提供 Prompt templates + orchestration；  
- 选型基于任务耦合度、可解释性与扩展性。

---

### 32. Hallucination 评估：GPT-Judge vs 自动验证。
**答：**  
- GPT-Judge：批量给出评价标签；  
- Schema 验证：Regex/JSON检查；  
- 知识库对照检索，自动对齐事实；  
- 使用 True/False QA tasks 验证输出准确性。

---

### 33. RLHF / DPO 训练管道设计。
**答：**  
- 收集人类偏好数据或模拟评分数据；  
- SFT 初始微调，再用 DPO/PPO做策略优化；  
- 使用 trlX 管道管理训练任务；  
- Model registry 记录版本与评估结果。

---

### 34. 在线学习与增量更新。
**答：**  
- 微批量增量 fine-tune（Adapter 或少数层）；  
- Drift 检测触发自动训练；  
- 在线参数服务器（如 TorchServe warm reload）；  
- 半结构化日志驱动持续优化。

---

### 35. 灾难恢复与高可用性设计。
**答：**  
- 跨区部署推理与存储（multi-zone cluster）；  
- 定期备份 embedding、模型与配置；  
- 自动化健康检查 + 自愈（K8s liveness/readiness probes）；  
- Chaos engineering 验证故障恢复流程。  


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
