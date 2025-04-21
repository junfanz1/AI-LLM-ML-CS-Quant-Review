Efficient Training in PyTorch, by Ailing Zhang, 2024

<img src="https://github.com/user-attachments/assets/7295d359-d625-4930-802d-03f1ddcd8ed3" width="32%" height="32%">


# Contents

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [2. 硬件](#2-)
- [3. PyTorch](#3-pytorch)
- [5. 数据加载和预处理](#5-)
- [6. 单卡性能优化](#6-)
- [7. 单卡显存优化](#7-)
- [8. 分布式训练](#8-)
- [10. GPT-2优化](#10-gpt-2)

<!-- TOC end -->

<!-- TOC --><a name="2-"></a>
## 2. 硬件

- 算子：内存读取数据，调用CPU计算，计算结果写回内存。
- CPU执行指令效率由clock speed CPU频率衡量。内存读写速度慢，需要CPU多级缓存。
- GPU核心与显存VRAM的关系就像CPU核心与内存的关系。
- GPU核心 = L2缓存 + 流式多处理器Streaming Multiprocessors（= L1缓存 + 多个流式处理器（= CUDA core + tensor core + 寄存器 + warp scheduler线程束调度器））
- 线程：算法拆分得到个每个独立任务（如拆分张量加法），由warp scheduler将每32个线程打包成一个warp，每组线程束warp被调度到一个流式处理器上执行相同的GPU指令。
- 流式多处理器（SM）对应CUDA block线程块 = M * threads。流式处理器（SP)对应CUDA warp线程组 = 32 threads。
- 单机多卡：NVLink用于GPU之间通信，高带宽低延迟。多机多卡：Ethernet/InfiniBand.

<!-- TOC --><a name="3-pytorch"></a>
## 3. PyTorch

- 不同张量可以共享同一块底层存储，若不仅共享还存在重叠，则称一个张量是另一个张量的视图view。修改视图张量的数据时，原始张量的数据也被修改，因为指向同一块内存地址。可以避免新内存分配的时间成本并减少显存占用。
- 原位inplace，在方法名后加_直接修改输入张量并返回同一张量，无须创建新内存。
- 动态图dynamic graph，所见即所得，调试简单。可以根据张量的数值动态决定是调用加法还是乘法算子来计算。
- 自动微分。前向计算图是当场构建当场执行的，反向计算图是当场构建延迟执行的，直到`loss.backward()`时才执行反向计算图。autograd可以自定义算子。
- 异步执行机制：让CPU-GPU协同工作，CPU提交任务给GPU后，不等GPU任务完成而直接返回继续执行下一个CPU任务。

<!-- TOC --><a name="5-"></a>
## 5. 数据加载和预处理

- Dataset读取单个数据，输出单个张量。DataLoader可以批量读数据，包括BatchSize、预读取、多进程读取，输出一批张量。
- 迭代式数据集iterable style dataset可处理大日志文件、实时股票数据流。
- 实时预处理，CPU过载：增大DataLoader并行度、确保CPU上无过多线程如Numpy。

<!-- TOC --><a name="6-"></a>
## 6. 单卡性能优化

- 提高数据并行度：CPU预处理要够快，确保GPU有大量计算任务排队，始终有活干。GPU需要的数据能在执行前就传到显存，减少GPU空闲时间。
- 提高GPU效率：增大BatchSize（batch processing有效利用GPU并行核心）
- fused融合算子（例如把Linear和BatchNorm合并为新的Linear调用，减少调用次数和计算量），提升网络效率。
- 减少`torch.cuda.synchronize()`进行CPU-GPU之间同步、GPU直接张量创建减少额外开销、不必要的梯度计算、no_grad修饰。
- 用optimizer来参数更新，优化器对梯度进行加工。

<!-- TOC --><a name="7-"></a>
## 7. 单卡显存优化

- 缓存下来的显存耗尽时，PyTorch就向GPU申请新显存，显存是房子，GPU是房东，PyTorch显存池是二房东（对分配出来的若干显存段进行二次管理）。
- 训练过程中显存占用的峰值通常在反向传播过程的某个反向算子的计算中。
- 静态显存下放至CPU
- 动态显存优化：前向张量重算、跨批次梯度聚合、原位算子、张量共享显存、调整优化器模式

<!-- TOC --><a name="8-"></a>
## 8. 分布式训练

- 训练模型：每个GPU卡是一个生产车间
  - 模型并行：突破显存限制、要指挥不同车间生产小零、并通过通信开销来组装。模型初始化用broadcast，数据切分，梯度同步（反向传播完成后用allreduce对梯度求和）。
  - 数据并行（切分BatchSize）：一次训练更多数据、扩大产能生产数百个产品。
  - 流水线并行pipeline parallel（切分模型不同层）：将通信过程与GPU计算过程重叠，来掩盖通信的时间开销并减少延迟。把模型切分成顺序执行的几个阶段，每阶段计算完后通过send操作把结果发给下一阶段的节点，下一阶段的节点通过receive接收数据。
- 降低静态显存：单卡训练，让显存下放到CPU；分布式训练，ZeRO/FSDP（切分模型参数、优化器状态的存储：不要求每个节点存储整个模型的静态显存，而是将这些静态显存分割并分配到各GPU上存储，根据需要动态组合，可防止显存溢出out of memory）。
- 降低动态显存：单卡训练，可以即时重算；分布式训练，模型并行（每张卡负责模型一部分，这样静态和动态显存都被切分；大批次数据分成若干小批次，可以减少GPU空闲气泡，让每个阶段交替进行前向和反向计算，缩短中间节点等待时间）。
- 张量并行（切分单个层的参数和输入）：矩阵乘法超过单个GPU显存容量，因此将矩阵乘法分块计算，再用all gather拼接。

<!-- TOC --><a name="10-gpt-2"></a>
## 10. GPT-2优化

- GPT-mini = 前处理Embedding（n_embd）、Transformer Blocks（n_layers）核心算法逻辑、后处理 计算loss
- GPT-large = 774M参数，n_layers = 36，n_embd = 1280，n_heads = 20，BatchSize=32，显存=35GB
- 跨批次梯度累加：不变有效BatchSize，减小前向张量的尺寸，让每轮BatchSize减半、每两轮训练后参数更新，这样可以降低显存峰值，原先一个显存峰分裂成两个显存峰，前向传播结束处。
- 即时重算前向张量：要PyTorch减少对前向过程中间结果的缓存，右侧小峰值是优化器的梯度更新过程。
- FSDP：模型参数分散到多卡，分布式降低显存占用，可以自动分割模型参数而不需要太多手动调优。
- 显存优化
  - 初始模型 36272MB | 16.3s
  - 跨批次梯度累加x2 25194MB (-31%) | 17.2s (+6%) 
  - 即时重算前向张量 15394MB (-39%) | 23.1s (+34%)
  - 优化器for-loop模式 13108MB (-15%) | 23.5s (+2%)
  - FSDPx2 8502MB (-35%) | 17.1s (-27%)
  - 优化前H100的80GB显存只支持1795M模型，显存优化+双卡FSDP后，可以支持9183M（5倍）模型。
- 性能优化
  - 初始模型7.68s，瓶颈是GPU队列空闲率高
  - 提高BatchSize 2.28s (-70%)，瓶颈是GPU等待数据加载
  - 增加num_workers 1.87s (-18%)，瓶颈是GPU等待数据拷贝同步
  - non_blocking数据拷贝 1.81s (-3%)，若瓶颈在数据传输则用低精度数据、双重缓冲
  - 开启图优化 torch.compile 1.09s (-40%)，优化可以提高GPU计算效率
  - float16混合精度训练 0.73s (-33%)，其他优化方法如进一步提高BatchSize、手写CUDA算子
  - 分布式数据并行 DDPx2 0.48s (-35%)







