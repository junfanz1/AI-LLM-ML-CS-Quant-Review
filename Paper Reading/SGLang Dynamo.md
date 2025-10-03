# 🚀 Insights from sgl-project x NVIDIA: #Dynamo for Inference Performance at Scale

# [LinkedIn Post](https://www.linkedin.com/posts/junfan-zhu_dynamo-sanfrancisco-ai-activity-7379780254493569024-pEBC?utm_source=share&utm_medium=member_desktop&rcm=ACoAABxP-p0BpUNGDf347aKh_1uJAPzG4er0As8)

Attended the event today in #SanFrancisco — a great mix of deep technical content and a fantastic networking opportunity with researchers, engineers, and system builders who are pushing the frontier of #AI #inference performance.

## Key Takeaways

### 🔧 Core Optimizations (Qiaolin Yu, Core Developer sgl-project)

- Kernel & Architecture: optimized attention kernels, efficient sparse MoE routing, and faster collective ops (all-reduce/all-to-all) to reduce communication bottlenecks in disaggregated systems.
- Seamless training-to-serving pipeline with RL frameworks (Slime, AReal, veRL).
- Speculative Decoding: low-latency draft-and-verify decoding significantly boosts throughput.
- Deterministic Inference: reproducible outputs even with temperature > 0, enabled by chunked prefill, CUDA Graphs, and radix cache.
- Distributed KV Cache: hierarchical caching (GPU ↔ system memory ↔ persistent storage) for better utilization and scalability.

### 🔬 Future Development (Baizhou Zhang, sgl-project, NVIDIA)
- Multi-Token Prediction (MTP) with Overlap Scheduler: improves decoding efficiency in small-batch or memory-bound cases, reducing latency while boosting throughput.

### 🌐 Distributed Inference Frontier (Ishan Dhanani, NVIDIA)
- SGLang + Dynamo: redefining high-throughput, low-latency deployment at scale.
- EP Fault Tolerance: GPU-initiated P2P communication, dynamic expert add/remove, and routing table updates that keep the system running under degraded conditions.

Big thanks to the sgl-project and NVIDIA teams for pushing the boundaries. Lots to take back and think about for applications in large-scale AI systems.

Curious to hear your thoughts on recent inspiring work <LoRA Without Regret> by Thinking Machines Lab: Could #RL + #LoRA (lightweight adaptation) be the bridge that allows inference systems to pivot objectives dynamically without full retraining — moving closer to "end-to-end" flexibility? Can deterministic inference and sampling diversity coexist in RL fine-tuning pipelines?
