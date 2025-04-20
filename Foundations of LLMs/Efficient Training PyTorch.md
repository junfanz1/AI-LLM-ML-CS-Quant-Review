Efficient Training in PyTorch, by Ailing Zhang, 2024

<img src="https://github.com/user-attachments/assets/7295d359-d625-4930-802d-03f1ddcd8ed3" width="32%" height="32%">

## 2. 硬件

- 算子：内存读取数据，调用CPU计算，计算结果写回内存。
- CPU执行指令效率由clock speed CPU频率衡量。内存读写速度慢，需要CPU多级缓存。
- GPU核心与显存VRAM的关系就像CPU核心与内存的关系。
- GPU核心 = L2缓存 + 流式多处理器Streaming Multiprocessors（= L1缓存 + 多个流式处理器（= CUDA core + tensor core + 寄存器 + warp scheduler线程束调度器））
- 线程：算法拆分得到个每个独立任务（如拆分张量加法），由warp scheduler将每32个线程打包成一个warp，每组线程束warp被调度到一个流式处理器上执行相同的GPU指令。
- 流式多处理器（SM）对应CUDA block线程块 = M * threads。流式处理器（SP)对应CUDA warp线程组 = 32 threads。
- 单机多卡：NVLink用于GPU之间通信，高带宽低延迟。多机多卡：Ethernet/InfiniBand.

## 3. PyTorch

- 不同张量可以共享同一块底层存储，若不仅共享还存在重叠，则称一个张量是另一个张量的视图view。修改视图张量的数据时，原始张量的数据也被修改，因为指向同一块内存地址。可以避免新内存分配的时间成本并减少显存占用。
- 原位inplace，在方法名后加_直接修改输入张量并返回同一张量，无须创建新内存。
- 动态图dynamic graph，所见即所得，调试简单。可以根据张量的数值动态决定是调用加法还是乘法算子来计算。
- 自动微分。前向计算图是当场构建当场执行的，反向计算图是当场构建延迟执行的，直到`loss.backward()`时才执行反向计算图。autograd可以自定义算子。
- 异步执行机制：让CPU-GPU协同工作，CPU提交任务给GPU后，不等GPU任务完成而直接返回继续执行下一个CPU任务。

## 5. 数据加载和预处理

- Dataset读取单个数据，输出单个张量。DataLoader可以批量读数据，包括BatchSize、预读取、多进程读取，输出一批张量。
- 迭代式数据集iterable style dataset可处理大日志文件、实时股票数据流。
- 实时预处理，CPU过载：增大DataLoader并行度、确保CPU上无过多线程如Numpy。

## 6. 单卡性能优化

- 提高数据并行度：CPU预处理要够快，确保GPU有大量计算任务排队，始终有活干。GPU需要的数据能在执行前就传到显存，减少GPU空闲时间。
- 提高GPU效率：增大BatchSize（batch processing有效利用GPU并行核心）
- fused融合算子（例如把Linear和BatchNorm合并为新的Linear调用，减少调用次数和计算量），提升网络效率。
- 减少`torch.cuda.synchronize()`进行CPU-GPU之间同步、GPU直接张量创建减少额外开销、不必要的梯度计算、no_grad修饰。
- 用optimizer来参数更新，优化器对梯度进行加工。

## 7. 单卡显存优化

- 缓存下来的显存耗尽时，PyTorch就向GPU申请新显存，显存是房子，GPU是房东，PyTorch显存池是二房东（对分配出来的若干显存段进行二次管理）。
- 训练过程中显存占用的峰值通常在反向传播过程的某个反向算子的计算中。
- 静态显存下放至CPU
- 动态显存优化：前向张量重算、跨批次梯度聚合、原位算子、张量共享显存、调整优化器模式

## 8. 分布式训练

- 训练模型：每个GPU卡是一个生产车间
  - 模型并行：突破显存限制、要指挥不同车间生产小零、并通过通信开销来组装。模型初始化用broadcast，数据切分，梯度同步（反向传播完成后用allreduce对梯度求和）。
  - 数据并行：一次训练更多数据、扩大产能生产数百个产品。
  - 流水线并行pipeline parallel：将通信过程与GPU计算过程重叠，来掩盖通信的时间开销并减少延迟。把模型切分成顺序执行的几个阶段，每阶段计算完后通过send操作把结果发给下一阶段的节点，下一阶段的节点通过receive接收数据。











