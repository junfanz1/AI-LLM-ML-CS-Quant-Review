
# HiVA（Hierarchical Variable Agent）：Multi-Agent就像有生命的组织，从个体智能迈向组织智能，自组织新范式能演化出协作智能的涌现性吗？AGI存在于连接与组织方式？

自组织（不是固定结构优化参数，而是AI系统演化），共同进化两个层面相互促进（用文本梯度驱动）

1、Semantic：每个agent做什么，具有什么能力

2、Topology：每个Agent如何协作的团队架构

闭环：根据任务动态组队，执行拿到结果，评估反馈并追溯责任，各自改进并调整合作关系，清理维护组织架构。不断迭代持续优化。

技术细节

- Multi-Agent System不再是固定流程图，而是动态、自我生长、自我重塑的computational graph。系统内分化出不同分工就是在学习新的semantic，并优化拓扑，全流程是AI自我实现的全过程，驱动这个过程的引擎是STEV算法(Semantic Topological Evolution）。
- STEV算法的操作空间是Hybrid Space = semantic space (prompt, tools, preference) + topological space (how connected, how information flows)，二者共同进化（semantic space的变化可能促使系统思考团队合作方式调整，topological space的变化可能促使agents思考更适合自己的位置和学习技能）。
- 由于选择十分离散化，基于梯度的反向传播优化方法就失效。而是用Textual Gradient，是自然语言的优化指令，完成任务后输出被送到Textual Gradient Parser分析“环境”（编译器、 答案）评估反馈，翻译成结构化、具有指导意义的文本指令（Textual Gradient），例如“调整Agent 3 prompt让它关注一致性”。用LLM理解生成能力指导复杂混合空间优化方向，就像人类项目复盘，从结果出发往前回溯，用自然语言明确每个环节每个人员的改进地方。
- 迭代过程三步：（1）Forward Pass执行任务（由KABB动态路由机制，按需临时构建只包含相关agent的Execution Subgraph，指令从一个agent流向下一个agent，并由Aggregator agent汇总。动态方式高效，避免资源浪费）；（2）Textual Gradient Feedback（系统把输出交给环境评估得到Textual loss，文本梯度解析器生成全局“文本梯度”即总体改进方向，并沿着刚才信息流动的方向反向传播回去，并逐步分解成每个agent的局部文本梯度）；（3）Coordinated Update（每个agent收到改进方案后，同时协同地启动Semantic [修改prompt或者调整调用工具的方式] & Topological [系统根据梯度和agent贡献度决定是否修改下游连接：增加下游、删除连接、直连Aggregator、保持不变] updates）。最后，系统更新负责挑人的KABB路由机制的内部参数，让下次决策更明智。
- Repair topology：清除网络中无用的孤岛节点或有害的循环依赖，保持架构高效。
- KABB路由 Knowledge-Aware Bayesian-Bandit Routing：以MultiArmed Bandit为基础的调度员，综合考虑历史表现、任务相关性、团队协同效应（加入团队是否能1+1>2）。KABB本身也持续学习，每次任务后根据反馈更新对每个Agent评估，让未来路由决策更精准。
- 生物的自适应能力：可进化的工具子系统，不仅能优化人和组织，还能自己制造和修理工具。由成千上万Agent动态形成并不断自我优化的协作结构涌现出的集体智慧，AGI存在于连接与组织方式。
- Philosophy: Structure as Memory。Multi-Agent topology structure就编码了系统如何高效协作的集体记忆，记忆是分布式的、Hierarchical的：宏观上网络整体形态反应了系统沉淀下的已验证有效的模式（集体潜意识），中观上每个具体连接权重记录了这条路径的历史有效性，微观上每个Agent自身配置存储了个体知识。它的分布式特性就在于学到的知识就内生地长在系统组织结构中。
- 实验结论：性能又好又更省钱。具有持续学习稳步提升、改进自身的能力。
- Challenge：Agent互相冲突的时候（数学），Aggregator不知信谁。依赖高质量环境反馈，如果反馈模糊或有误，文本梯度就会带偏并陷入局部最优。优化迭代的计算成本是O(V^2), V是Agent数量。

<img width="2290" height="1080" alt="image" src="https://github.com/user-attachments/assets/5f778f05-a3d8-4a04-b05d-d74a64dcdfd5" />

I recently came across a new paper that I found truly inspiring—and honestly, pretty mind-blowing: HiVA: Self-organized Hierarchical Variable Agent via Goal-driven Semantic-Topological Evolution [1]. It rethinks how we look at Multi-Agent Systems, not just as a collection of tools or workflows, but as living organizations where intelligence evolves from the individual to the collective. The idea that AGI might not reside in a single model, but rather in the way agents connect and self-organize, really struck me.

## HiVA (Hierarchical Variable Agent)
Multi-Agent systems resemble living organisms: individual intelligence evolves toward organizational intelligence. Can self-organization as a new paradigm lead to the emergence of collaborative intelligence? Could AGI exist not in a single model, but in the ways agents connect and organize?

Here, self-organization means evolution of the AI system itself—not just tuning parameters in a fixed structure. Two levels co-evolve in a mutually reinforcing manner (driven by textual gradients):

- Semantic – what each agent does and what capabilities it has.
- Topology – how agents collaborate and how teams are structured.

Article content
This forms a closed loop: dynamically assemble teams for a task, execute, collect results, evaluate feedback, trace responsibility, improve individually, and adjust collaboration links—while pruning and maintaining the organizational structure. Continuous iteration drives ongoing optimization.

## Technical Details
In HiVA, a Multi-Agent System is no longer a fixed flowchart, but a dynamic, self-growing, self-remolding computational graph.

- Differentiated roles among agents amount to learning new semantics.
- Optimizing inter-agent structures amounts to learning topology. Together, this represents a process of AI self-realization, powered by the STEV algorithm (Semantic-Topological Evolution).

Hybrid Space of STEV = semantic space (prompts, tools, preferences) + topological space (connections, information flow). Both evolve together:

- A semantic shift may trigger changes in cooperation patterns.
- A topological shift may push agents to rethink their positions and acquire new skills.

Because choices are highly discrete, gradient-based backpropagation fails. Instead, HiVA uses Textual Gradient: natural-language optimization instructions. After completing a task, the system’s output is sent to a Textual Gradient Parser, which evaluates the environment (compiler, answers), and translates results into structured, instructive textual directives. Example: “Adjust Agent 3’s prompt to focus on consistency.”

In other words, LLM understanding and generation abilities guide optimization in this hybrid space—similar to how humans conduct project retrospectives: tracing back from results to identify improvement areas for each role.

## Iterative Process
Each iteration proceeds in three steps:

- Forward Pass – Execute the task via KABB dynamic routing, which constructs an Execution Subgraph containing only the relevant agents. Instructions flow agent-to-agent and results are aggregated by an Aggregator Agent. Dynamic routing improves efficiency and avoids waste.
- Textual Gradient Feedback – The system hands outputs to the environment, obtains a Textual Loss, and the parser generates a global textual gradient (overall improvement direction). This flows backward along the same path, decomposing into local gradients for each agent.
- Coordinated Update – Each agent receives its update plan, then synchronously performs:

Finally, the KABB routing parameters are updated for more informed agent selection in future tasks.

Additionally, topology repair cleans up useless isolated nodes or harmful dependency cycles to maintain efficiency.

## KABB Routing
Knowledge-Aware Bayesian-Bandit Routing extends Multi-Armed Bandits to act as a scheduler:

- Considers past performance, task relevance, and collaboration synergy (whether adding an agent yields >1+1=2).
- KABB itself continuously learns—after each task, updating its agent evaluations to make future routing more accurate.

## Biological Analogy
HiVA mimics biological adaptability with an evolving tool subsystem.

- It can not only optimize humans and organizations, but also build and repair its own tools.
- From thousands of dynamically assembled agents emerges collective intelligence.

AGI, in this view, lies in the way connections and organizations form and evolve.

## Philosophy: Structure as Memory
The Multi-Agent topology encodes distributed, hierarchical collective memory:

- Macro level – the overall network shape embodies validated, effective patterns (collective unconscious).
- Meso level – edge weights store the historical effectiveness of each path.
- Micro level – each agent’s configuration encodes individual knowledge.

Thus, knowledge is not external—it grows within the organizational structure itself.

## Experimental Findings
- Better performance, lower cost.
- Exhibits continual learning, steady improvement, and self-adaptation.

## Challenges
- Agent conflicts: When agents disagree (e.g., in math), the Aggregator may not know whom to trust.
- Feedback quality: If environment feedback is vague or wrong, textual gradients can mislead optimization and trap the system in local optima.
- Computation cost: Optimization iteration has complexity O(V²), where V = number of agents.

## References
- [HiVA: Self-organized Hierarchical Variable Agent via Goal-driven Semantic-Topological Evolution](https://arxiv.org/pdf/2509.00189)
- [LinkedIn: Co-Evolutionary Path Towards Organizational Intelligence: How Structure-as-Memory Network & Multi-Agent Emergence Could Unlock AGI?](https://www.linkedin.com/pulse/07-co-evolutionary-path-towards-organizational-intelligence-how-kpqjc/)
