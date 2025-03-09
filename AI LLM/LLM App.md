# Contents

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [RAG: PDF读图](#rag-pdf)
- [推荐图书](#)
- [论文总结评估](#-1)
- [PEFT参数高效微调](#peft)
   * [HuggingFace PEFT框架，LoRA微调实战](#huggingface-peftlora)
   * [QLoRA](#qlora)
   * [OpenAI微调](#openai)
   * [Function Calling](#function-calling)
- [Sora](#sora)
- [模型追踪](#-2)
- [Acknowledgements](#acknowledgements)

<!-- TOC end -->

<!-- TOC --><a name="rag-pdf"></a>
# RAG: PDF读图

三个大模型
- GPT-4V：作为 GPT-4 的视觉版本，GPT-4V（gpt-4-vision-preview）在 RAG 系统中扮演了关键角色。我们利用 GPT-4V 对 PDF 中的图像进行理解和描述，通过精心设计的提示，引导模型提取图表信息、识别标题等。这充分发挥了 GPT-4V 在多模态理解方面的能力，使得 RAG 系统能够同时处理文本和图像信息。
- Embedding 模型：使用 OpenAI 的 “text-embedding-3-small” 模型对文本片段进行向量化表示。将文本映射到高维向量空间，使得语义相似的内容在向量空间中更加接近。这种语义表示方法是实现高效内容检索的基础，使 RAG 系统能够快速找到与查询最相关的内容。
- GPT-4：最后一步中使用 GPT-4 生成模型（gpt-4-turbo-preview），根据检索出的相关内容生成最终答案。这充分利用了 GPT-4 在自然语言生成方面的强大能力，能够根据上下文信息生成流畅、连贯的回复。同时，精心设计的系统提示也引导了 GPT-4 对相关性进行判断，提高了回复的质量。

步骤
- extract_text_from_doc，使用了 pdfminer 库的 extract_text 函数从 PDF 文件中提取文本。
- 使用了 pdf2image 库的 convert_from_path 函数将 PDF 文件转换为图像。定义了一个函数 get_img_uri，用于将图像转换为 base64 编码的数据 URI 格式，以便传入 GPT-V 模型。
- 函数 analyze_image 则将使用上面定义的提示，来调用 GPT-4V（即 gpt-4-vision-preview）模型进行图像分析。它将系统提示和图像 URL 作为消息发送给 API，并返回模型生成的内容描述。
- 将组合文本和图像分析结果，同时整理组合后的内容，并将整理后的内容转换为 DataFrame。
- 生成内容片段的嵌入向量，然后为 DataFrame 中的每个内容片段生成嵌入向量。使用 text-embedding-3-small 做嵌入的 Model。有了嵌入向量，文本检索就非常方便了。
- 函数 search_content 首先获取输入文本的嵌入向量，然后计算每个内容片段与输入文本的余弦相似度，并返回相似度最高的前 k 个结果。函数 get_similarity 则用于从 DataFrame 的一行中获取相似度得分。
- 根据检索结果生成回答。实现了多模态理解、语义检索、知识融合等关键功能。

<!-- TOC --><a name=""></a>
# 推荐图书

- 向量嵌入技术，如 Word2Vec 和 Sentence-BERT，以及最新的大型词嵌入模型，则可以将文本转化为密集的向量表示。通过嵌入，语义相似的文本会被映射到向量空间中相近的位置。我们可以计算图书描述嵌入之间的相似度（如余弦相似度），从而快速找到内容相似的图书。
- 定义获取嵌入向量的函数 get_embedding，它使用 OpenAI 的 text-embedding-ada-002 模型为给定的文本生成嵌入向量。我们将每本书的标题、作者和关键词拼接成一个字符串，然后调用此函数生成其嵌入向量。
- 根据输入文本搜索相似图书的函数 search_from_input_text。给定用户输入的查询文本，该函数首先使用 get_embedding 函数为查询文本生成嵌入向量。然后计算查询向量与数据集中每本书嵌入向量之间的余弦相似度，将结果存储在 similarity 列中，按相似度降序排列，取前 n 本书作为搜索结果返回。

<!-- TOC --><a name="-1"></a>
# 论文总结评估

摘要生成
- 导入 PyPDF2 包，打开论文文件，读取 PDF 文件内容。导入 anthropic 库，generate_summary 的函数，用于利用 Claude 模型生成研究论文的摘要。这个函数接受一个参数 text，即要总结的论文全文，然后会返回全文的总结。
- 创建了一个 Anthropic 客户端对象，用于与 Claude 模型进行交互。在系统提示中，我们告诉 Claude 它是一个用于总结研究论文的 AI 助手。构造了一个消息列表，其中包含一条用户消息，并在消息的内部中通过 {text} 变量嵌入了要总结的论文全文。我们还在消息中提供了一些指示，要求 Claude 关注论文的关键发现、方法和结论，并生成一个大约 150 字的简洁摘要。
- client.messages.create 方法会将所构造的消息列表、系统提示以及其他参数传递给 Claude。Claude 会根据这些信息生成论文的摘要，并返回结果。

评估
- ROUGE（Recall-Oriented Understudy for Gisting Evaluation）评估是一组用于自动化评估机器翻译和自动摘要系统性能的指标，要衡量生成摘要与参考摘要之间的重叠情况，特别关注回忆率（Recall）。
- ROUGE 评估指标能够从不同的角度评估生成摘要和参考摘要之间的相似度，涵盖了局部 N-gram 匹配（ROUGE-N）、全局序列信息（ROUGE-L）和灵活词对关系（ROUGE-S）。
  - ROUGE-N 是基于 N-gram 的匹配来评估生成摘要与参考摘要之间的相似度。N-gram 可以是单个词（unigram）、双词（bigram）、三词（trigram）等。常见的 ROUGE-N 指标包括 ROUGE-1 和 ROUGE-2，分别表示基于 unigram 和 bigram 的匹配情况。
  - ROUGE-L 基于最长公共子序列（Longest Common Subsequence, LCS）来评估生成摘要和参考摘要之间的相似度。通过计算生成摘要和参考摘要之间的最长公共子序列匹配情况，LCS 能够捕捉到全局序列信息，而不仅仅是局部的 N-gram 匹配。
  - ROUGE-S（Skip-Bigram）基于跳跃双词（skip-bigram）来评估生成摘要和参考摘要之间的相似度。通过计算生成摘要和参考摘要之间的跳跃双词匹配情况，跳跃双词允许在不改变词序的情况下跳过一些词，这样可以更灵活地捕捉摘要中的词对关系。
- BERTScore评估：通过计算生成的摘要和参考摘要的 BERT 嵌入表示之间的相似度来评估摘要的质量。相比于 ROUGE，BERTScore 利用了词向量的思想，因此就能够更好地捕捉语义上的相似度。

<!-- TOC --><a name="peft"></a>
# PEFT参数高效微调

- Adapter 微调：在预训练模型的每一层（或部分层）注入轻量级的 Adapter 模块。微调时只训练这些新加入的 Adapter 参数，冻结原模型参数。Adapter 充当了任务适配器的角色，以较小的参数量和计算代价实现了模型适配。
- LoRA 微调：以低秩分解的思想对预训练模型进行微调。在每个注意力模块中引入低秩矩阵，在前向传播时与原矩阵相加得到适配后的权重。LoRA 只需训练新引入的低秩矩阵，参数开销很小（新增参数量通常只有原模型的 0.1%~3%），但能在下游任务上取得不错的效果。
  - LoRA：在注意力矩阵上添加低秩分解矩阵。
  - LoHA：在注意力矩阵上添加基于 Hadamard 乘积的低秩分解。
  - LoKR：在注意力矩阵上添加基于 Kronecker 乘积的低秩分解。
  - AdaLoRA：自适应调整训练过程中的秩和约束力度。
- P-tuning v2：将连续的提示向量和 Adapter 的思想相结合，在每个 Transformer 层基础上引入可训练的提示嵌入。这些提示嵌入在前向传播时会注入到注意力矩阵和前馈层中。P-tuning v2 在保留预训练知识的同时，也能有效地进行任务适配。
- Diff Pruning：对微调后模型和原模型的参数差值进行剪枝。
- BitFit：只微调偏置参数，冻结其余部分。
- IA3：用少量可学习向量对模型的激活值（key、value、前馈的中间激活）进行缩放。

<!-- TOC --><a name="huggingface-peftlora"></a>
## HuggingFace PEFT框架，LoRA微调实战

- Alpaca 数据格式。这种数据是由 Meta AI 在发布 Llama 模型时一同提出的。Alpaca 数据集包含了 52K 个由 Llama 模型生成的指令 - 输出对，涵盖了问答、总结、创意写作等多种任务类型。这个数据集的格式对于后续的指令微调任务具有重要意义，成为了许多开源项目的基石。这些数据是通过自指令生成技术（Self-Instruct）生成的。每条数据都是一个字典，包含以下字段：instruction：描述模型需要执行的任务。input：任务的上下文或输入，约 40% 的例子包含此字段。output：由 text-davinci-003 生成的指令答案。
- `_init_` 方法会加载 JSON 文件，并初始化分词器和设备。_getitem_ 方法会获取并格式化这个数据样本，使用分词器将其转换为张量格式，并返回包含输入和标签的字典。format_example 方法会生成上下文和目标文本。有效利用当前 GPU 内存的方法，包括梯度检查点、多卡并行等等。
- 设置 LoRA 配置项，通过低秩分解，减少模型参数和内存占用，并使用 dropout 技术防止过拟合，同时仅在特定模块上应用 LoRA，从而在计算效率和模型性能之间取得平衡。
![image](https://github.com/user-attachments/assets/82e3f09a-1d69-4dad-99b6-4e392706fb86)
- 训练参数。训练跑 500 轮后，模型成功保存至本机目录。重新用本地目录加载微调后的 Qwen 模型，再次测试同样的问题。
![image](https://github.com/user-attachments/assets/4394139a-b1c2-4164-9717-c1ebb61bde78)

<!-- TOC --><a name="qlora"></a>
## QLoRA

- QLoRA 就是用 int4 或者 int8 级别的低精度去表示 LoRA 中引入的增量矩阵。这种方法大幅降低了 LoRA 的存储和带宽需求，使其更容易在资源受限的环境中部署。
- 每一步更新完增量矩阵后，QLoRA 就对其进行动态量化（Dynamic Quantization），将其转换为 int4。通过 QLoRA，我们在模型性能几乎无损的情况下，可以把 LoRA 增量矩阵的体积压缩到原来的 1/8（从 float32 到 int4），模型完成了瘦身（注意，这个瘦身只是针对于 LoRA 增量矩阵而言，模型原始权重矩阵 A 精度不变），推理速度也因此提升。当然，这是以损失一部分精度为代价的。
- bitsandbytes 是一个优化库，主要用于加速和优化大规模机器学习模型的训练和推理，在资源受限的环境下进行深度学习任务。 bitsandbytes 支持多种低精度格式（如 int8、int4），以显著减少模型的内存占用和计算需求。
- bitsandbytes 库通过以下方式实现 4 位（或 8 位）量化。权重量化：将模型的权重（通常是 32 位浮点数）转换为 4 位（或 8 位）整数表示。反量化：在进行计算时，将 4 位（或 8 位）整数权重转换回浮点数。优化计算：bitsandbytes 库针对低精度计算进行了优化，以提高计算效率。

<!-- TOC --><a name="openai"></a>
## OpenAI微调

- 准备符合要求的微调数据集。微调数据要采用 JSONL 格式，每行代表一个训练样本。每个样本是一个 JSON 对象，主要包含以下字段：messages：由角色（role）和内容（content）组成的对话数据。functions：可选。一个列表，其中列出了样本涉及的函数调用。
- 上传训练和验证数据。使用 Files API 把格式化好的 JSONL 文件分别上传到 OpenAI。上传完成后，会得到训练文件和验证文件各自的 ID。有了文件 ID，我们就可以启动微调任务了。
- 上传完训练数据后，就可以提交微调作业了。通过调用 OpenAI 库的 fine_tuning.jobs.create 方法，指定训练文件 ID、基础模型（这里使用 gpt-3.5-turbo）以及超参数（如训练轮数 n_epochs），即可创建一个微调作业。

<!-- TOC --><a name="function-calling"></a>
## Function Calling
- Function 的说明是一个 JSON 对象，用于向 Assistant 描述函数的元数据，如函数名称、参数类型等，但不包含函数的实现代码。在使用 Function calling 时，我们需要先提供函数的 JSON 描述，Assistant 了解函数的接口定义。然后在代码中实现对应的函数。当 Assistant 在对话中决定调用该函数时，它会根据 JSON 描述生成一个包含具体参数值的 JSON 对象——这也就是函数调用的元数据，我们的代码需要解析这个 JSON 对象，提取参数值，调用相应的函数，并将结果返回给 Assistant。
- 我们创建一个新的 Run，让 Assistant 处理这个 Thread。在 Assistant 处理 Thread 的过程中，需要轮询 Run 的状态，直到状态变为 requires_action（需要调用函数）或 completed（对话完成）。为了方便轮询，OpenAI 给出了一个 create_and_poll 函数。当 Run 的状态变为 requires_action 时，意味着 Assistant 需要调用一个函数来完成任务。我们在 Playground 的 Function 元数据中通过 JSON 格式定义的内容。这就是动态调用函数的基础！我们不需要硬性指定函数，Assitants 会动态的帮咱们选择函数，同时确定每一个参数的传入数据值。

Function Calling特定领域微调
- 允许我们在微调时定义一些“函数”，让模型学会理解并调用函数。Function Calling 微调的好处是，让你的微调模型拥有预定义的函数知识，否则，在每次调用 API 时都要包含一长串函数定义会消耗大量 Token，有时模型会产生幻觉或无法提供有效的 JSON 输出。因此，使用 Function Calling 微调模型可以让你在即使没有完整的函数定义时，也可以获得符合格式的响应，同时获得更准确、更一致的输出。
- messages 中引入了一个 function_call 角色，表示助手要调用名为 write_leave_letter 的函数，并传入员工姓名、部门等参数。而在 functions 中则详细定义了这个函数的名称、描述、参数列表、参数类型等信息。微调时模型会学习这种 function_call 的模式，之后再遇到类似的场景时，就能更准确地自动抽取出关键信息，组织成结构化的函数调用，而不是简单地生成一段文本。

<!-- TOC --><a name="sora"></a>
# Sora

- 传统的扩散模型通常使用 U-Net 作为骨干网，但在相关论文《Scalable Diffusion Models with Transformers》中，提出了 Diffusion Transformer（DiT）架构，这是一种基于 Transformer 架构的扩散模型，将 U-Net 替换为 Transformer。
- Sora 采用了 Diffusion Transformer（DiT）架构，通过将图像和不同时长、分辨率的视频统一表示为 patch，再用 Transformer 架构进行建模，因此能够灵活处理不同的视觉数据。Sora 通过从充满视觉噪声的帧开始，逐步细化和去噪，最终生成与文本指令紧密相关的视频内容。这种方法使 Sora 能够处理包括图像和视频在内的各种视觉数据，这些数据首先被压缩到一个较低维的隐空间，然后被分解成一系列时空 patch 作为 Transformer 模型的输入，使得 Sora 能够灵活处理不同时长、分辨率和宽高比的视频和图像。
- 在训练阶段，Sora 接受添加了高斯噪声的 patch；在推理阶段，输入的是符合高斯分布的随机噪声 patch 和特定的文本条件，通过迭代去噪过程生成视频。这种方法不仅提高了视频内容的生成质量，还允许模型并行处理连续视频帧，而非逐帧生成，从而生成时间上连贯的长视频。此外，Sora 在训练时使用了包含多种长度和分辨率的视频数据，进一步增强了模型对不同视频格式的适应性。

<!-- TOC --><a name="-2"></a>
# 模型追踪

- LangSmith 是一个用于构建生产级 LLM 应用程序的平台，可以帮你密切监控和评估应用程序，而且使用 LangSmith 不需要依赖 LangChain 包，也就是说这个工具可以独立于 LangChain 而存在。
- LangSmith 来记录任意的大模型调用过程，先使用 LangSmith 包装 OpenAI 客户端。其中，wrap_openai 函数用于将 OpenAI 客户端包装为 LangSmith 兼容的客户端，以便进行追踪和日志记录。之后，添加 traceable 装饰器。@traceable 装饰器用于将函数标记为可追踪的，这样 LangSmith 可以记录函数调用的输入和输出。程序运行之后，再度登录 LangSmith，将看到程序与大模型的交互细节都被记录。
- Weights & Biases (wandb) 这个工具。这是一个用于机器学习和深度学习项目的实验跟踪和可视化工具，它能够自动记录和跟踪实验参数、超参数、模型架构和训练指标，提供实时监控和可视化图表，支持团队协作，集成超参数优化工具，管理和部署模型，并生成详细的实验报告。这些功能可以帮助开发者和研究人员高效地管理、监控和优化模型及其实验过程。
- Weights & Biases 适用于记录和可视化训练过程中的各类参数和指标，而 LangSmith 更适合追踪大模型调用的详细信息，包括输入输出、调用次数、消耗的 Token 数等。除了 Weights & Biases 之外，TensorBoard 也是常用的机器学习可视化工具，尤其是用于监控和记录基于 TensorFlow 的模型训练过程中的指标、模型结构和数据流图。另一个机器学习开源监控平台是 MLflow，用于管理机器学习的生命周期，包括实验跟踪、项目管理、模型管理和部署。


<!-- TOC --><a name="acknowledgements"></a>
# Acknowledgements

[大模型应用开发实战-极客时间-黄佳](https://time.geekbang.org/column/intro/100764201)

![image](https://github.com/user-attachments/assets/4cd59ea7-6ff7-4a32-88b9-4a3efa6459c9)
