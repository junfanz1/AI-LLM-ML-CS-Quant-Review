# Future of AI Agents = Agentic RL + Pretraining?

BrowserAgent还是需要有vision能力

浏览器和虚拟机sandbox的四种agent：

- 1、基于浏览器的agent；Browser认为，世界上所有网络服务都能在某个网页呈现，只要我的agent能看到网页就能去操作，用户可以可视化看agent动作，但很慢且tokens消耗大。
- 2、浏览器+虚拟机的agent（虚拟机代码+命令行）；写Python脚本在虚拟机跑然后完成任务，可以运行任何线下open source package，但很多情况下不能访问互联网（OAuth）。
- 3、虚拟机（虚拟机内部有很大限制的agent，主要通过LLM能力生成只能运行某类型的代码）；如Genspark，在限制环境下以LLM为主体，在sandbox写代码运行后生成可视化，用小环境执行。但这个sandbox is very limited，只有有限的三四个环境，无法临时下载package或修改，封闭工作流。
- 4、横跨多工具集成的agent。工作流内部每个节点和第三方服务直接集成，每个服务deliverable很可靠，但也不能全做到。

2、3的边界模糊，区别在于2的sandbox执行完后是开放sandbox，sandbox能力是运作主体；3是LLM是限制agent的主体。

- Manus：尽可能用一个sandbox + browser环境搭建一个万能的场景，LLM planning完后进入browser由另一个agent完成浏览器信息，总结信息后再给sandbox执行。（似乎Manus的sandbox环境比ChatGPT好）
- 但也被browser能力限制，比如修改google sheet cell（attention太小）、在某个button上传图片，而且也很慢，半小时一个task。
- Genspark：有superagent做任何事，但处理工具数量有限，开始做template（sheet, browser, AI call, slides都是agent标化工作流，在同一用例下把用户体验作为核心目标，就越来越不是通用agent而是人为选择的agent，但速度也快因为没有很大browser navigation且sandbox有限且节省token到细分应用场景，变成承载很多小任务的大平台就像微信小程序）
- Pokee速度提升四倍：不用复杂sandbox和tool calling，用第三方集成的SDK工具，通过Pokee的工具调用可以提速（没有MCP、tool calling long context）并削减tool calling成本一半，再加上context engineering。

- 未来工作流：不再是browser各个网页操作，而是ecommerce/搜索/视频网站的门户流量下降，入口变成各方向agent，有A2A交互（Agent入口）。现在MCP可用性差，市面上只有200/20000个MCP好用且难维护，目标是公司不再需要做MCP，直接提供SDK然后就获得额外流量。
- 创作者/SaaS生态的商业模式：Agent方每次调用，需要向知识产权平台方付款，广告由agent完成，用户选择不同agent，由ranking机制推荐哪个agent去做，就向那个agent公司收钱，广告在这时发生。而知识产权本身可以直接收费，因此创作者/SaaS生态变好了，不再需要谷歌投广告，而是agent直接向你付费。
- ranking-based推荐系统方向可能受到巨大挤压，在agent框架下，推荐系统仍是端到端multi-round decision making，但每次交互只是少量信息结果，决策线不再是ranking而是时间，因为人和agent交互时长有限，agent目标是推荐东西让你花费时间和它的回报成正比，这样推荐系统算法就不成立（原来：点击率与ranking成正比；现在：推荐的信息是你必点的东西但有第二轮交互，下一次交互时间内推荐的东西是最精准的从而有更多交互，因此是10条conversation且每一轮目标都是让你做下一轮交互，不再是rank而是sequential基于体验和探索的交互机制，在不损失未来opportunity cost情况下同等级别content内选择最多收入的内容）。


- 强化学习让agent决策可能不再是token-based，Deep Research是token by token，Agentic System可能工具是tokenized东西，多个工具一起解决问题，是目标驱动。而LLM可以supervised learning autoregressive训练，但agentic system很难：
- 单一tool calling可以通过数据完成，但变成工具链的时候，很难完成autoregressive training，因为市面上不存在数据而只能人为标注

RL pretraining而非RLHF：因为有很多任务是只有目标驱动的，大多数是没有数据的，而且问题非常泛化，需要生成小众的领域数据输出，通过ground-truth validator告诉对不对然后self-train，这种训练适用于有ground truth且能精准判断的用例并优化，适合RL。

AGI（产生知识是人类不拥有的）：如何提升verification机制泛化性？Agent输出偏离人类偏好时，如何让verifier adapt到新输出上使其完成更好verify？第一，如果能通过人类描述，从领域A推理向领域B的verification是什么，那么agent verification泛化性提高。第二，能否自我探索，基于现有知识grounding，完成对未来知识verification的延伸。
agent之间互相冲突所产生的矛盾，会引发政治吗？

当前verification泛化进展不大，是人类知识蒸馏的泛化提升。RL是counter-factual learning，可能出现能解决问题但人类看不懂的方案，这个弱点导致reward definition很重要，训练给什么incentive决定了训练出的agent是什么样。

- 数据标注在multi-modality（视频、图片）是逃不开的，因为verification基于视频和图片的Reinforce fine tuning，image input解析能力要求高且不能靠human rule完成，必须靠模型解析能力把图片和视频内容解析，在此之上人类才能写rule。但解析图片细节的能力很难。
- Reinforce fine tuning：不要reward model，就用大家共识或LLM-as-a-judge训练模型。
- Multi-modality：大量数据训练基础模型，然后RL fine tuning，然后怎样做标准化的judge/rule-based verifier，目前不存在，因为图像本身没有标准答案。目前是先通过数据训练reward model使得multi-modality能力最大，然后通过输入输出能力把它变成verifier然后再reinforce fine tuning。

Alignment是技术问题，robotics的行为很难有rubrics，因此multi-modal+actions的数据标注很难。

AI Agent未来形态：希望是像一行prompt调用API接口一样使用ChatGPT一样简单，而不用担心browser环境+infra去bypass agent工具的能力来压缩工具数量。browser是一个工具代替几千个工具，我们希望模型能力最强、工具把它铺开，让agent去操纵整个互联网，一个agent完成工具选择、规划、结果（而不需要infra之间来回跳转，而只需要prompt输入API就可以产生结果）。

垂直选一系列工具的推理模型：要直接access工具得到精确数据再分析的workflow，正常browser搞不定因为在训练中从未见过数据，需要foundation model开发来拓展workflow类型，而不只限定于购物和研究，很多专业性workflow还解决不了。

Model plasticity可塑性：模型训练到一定程度就会灾难性遗忘，需要continual learning解决。

<img width="1000" height="1500" alt="image" src="https://github.com/user-attachments/assets/955f6435-a3a2-491c-b86f-df2d7ff3ac34" />


# Future of AI Agents = Agentic RL + Pretraining?

As AI rapidly evolves, we’re witnessing a shift in how agents are designed, trained, and deployed. The discussion around BrowserAgents, sandboxed environments, and reinforcement learning (RL) is heating up — and for good reason. Here are my key takeaways and reflections.

## 🌐 Four Types of Browser + VM Agents
- Pure Browser Agents. View the web as the “world” — if it can be rendered in a browser, the agent can act on it. Pros: Fully visualizable; users can watch every action. Cons: Extremely slow, token-heavy, expensive.
- Browser + VM Hybrid Agents. Combines browser automation with code execution inside a virtual machine. Can run Python scripts, use offline open-source packages, but often blocked from internet access (e.g., OAuth issues).
- Sandbox-First Agents. Example: Genspark. The agent generates code inside a tightly restricted sandbox environment and runs it. Limitation: Can only use a fixed set of 3–4 environments, cannot dynamically download packages or modify the environment.
- Multi-Tool Integrated Agents. Workflow-based approach: each node is a direct integration with a service. Very reliable deliverables, but limited flexibility — not a truly general-purpose agent.

Key distinction between (2) and (3): 
- (2) = sandbox is the executor (open environment).
- (3) = LLM is the executor (sandbox is restrictive).

## 🏗️ Emerging Architectures: Manus, Genspark, Pokee
- Manus: Attempts to unify sandbox + browser into one universal agent framework. LLM does planning → passes plan to browser agent → collects info → executes in sandbox. Limitation: still bottlenecked by browser capabilities (e.g., editing Google Sheets, image upload).
- Genspark: “Superagent” approach. Standardized workflows (Sheets, Browser, AI Call, Slides) to improve UX. Faster because it avoids heavy browser navigation and token costs — optimized for targeted use cases.
- Future: Speed-optimized platform. Skips complex sandbox/tool-calling overhead by using third-party SDKs. Achieves 4× speedup and halves tool-calling costs via context engineering.

## 🔮 Future of Workflows: From Browser Ops to Agent-to-Agent (A2A)
- We may move away from browser-based navigation toward agents as the new entry points for e-commerce, search, and content discovery.
- MCP ecosystem challenge: today, only ~200 of 20,000 MCPs are reliable and maintainable.
- Vision: companies simply provide SDKs → agents gain native access → ecosystem grows.

## Business Model Implications
- Agents directly pay SaaS/creators for IP usage per call.
- Ads shift from Google-driven to agent-driven (ranking and selection).
- Recommendation systems transform: No longer just ranking for clicks. Shift to sequential decision-making — maximizing meaningful user-agent interactions within limited attention windows.

## 🧠 RL Pretraining (not just RLHF): Toward AGI
- Problem: Agentic systems can’t be trained purely with supervised autoregressive learning — multi-step toolchains lack training data.
- Solution: RL Pretraining (not just RLHF). Works best for goal-driven, data-scarce tasks where ground truth validators can tell “right vs wrong.” Enables self-training on synthetic data for long-horizon, generalized reasoning.

Most real-world tasks are goal-driven, not data-rich. They often lack demonstrations entirely, especially in niche or specialized domains. Instead of RLHF (which aligns a model with human preferences), we can use RL pretraining:

- Generate domain-specific outputs with a ground-truth validator.
- Self-train by reinforcing correct solutions and penalizing incorrect ones.
- Optimize for generalization in cases where no large dataset exists.

This is especially powerful when we have precise ways to determine correctness.

## ✅ Hard Problem: Verification & Alignment Challenges
- Key Research Question: How do we generalize verifiers to handle novel outputs and unseen domains? Can we reason from domain A to domain B’s verification logic using human-provided descriptions?
- Risks: RL may yield solutions humans can’t interpret (counterfactual learning). Reward definition becomes mission-critical — the agent you get is the agent you incentivize.
- Multi-Modality Barriers: Verification on images/videos requires strong perception models. Reinforcement Fine-Tuning (RFT) may replace reward models by using consensus signals or LLM-as-a-judge. But image/video verification still lacks standardized judge/rule-based frameworks.

Today, verification mechanisms are mostly distilled from human knowledge. The big open questions:

- How do we generalize verification? If an agent outputs something outside known human preferences, how do we adapt our verifier to evaluate it? (1) Can we reason across domains — infer what verification in Domain B should look like from Domain A’s rules? (2) Can verifiers self-explore and extrapolate based on existing grounding to handle future knowledge?

Agent disagreement is another open frontier — will conflicting agents create a form of “politics”?

## Multi-Modal RL Fine-Tuning: Why Data is Inevitable
When verification involves images or video, we cannot rely solely on hand-crafted human rules. We need models capable of:

- Parsing visual input accurately.
- Producing structured representations humans can reason about.
- Allowing rules or LLM-as-a-judge systems to operate on top of these representations.

Today, we train large multi-modal models first, then use reward modeling + RL fine-tuning to maximize perception capabilities. But there’s still no widely adopted standard for judge/rule-based verification in vision tasks — this is a key research gap.

## The Future of AI Agents
The ideal AI agent should feel as simple as sending a single API call — no complex browser environments, no juggling infrastructure just to make tools work.

- Browser as a meta-tool: replaces thousands of individual APIs, but we still need better workflow coverage.
- Workflow plasticity: foundation models must support direct tool access for precise data retrieval, analysis, and planning — not just shopping and research, but professional workflows too.
- 🧩 Rich Sutton's Model Plasticity & Continual learning: avoid catastrophic forgetting as agents adapt to new domains without losing prior skills..

Ultimately, we want goal-driven agents that can plan, act, and verify autonomously — with RL-pretraining making them robust in the absence of large-scale training data.

## 🚀 My Take
The future agent will feel as simple as calling an API — no more worrying about browser hacks or tool orchestration. Building AGI isn’t just about better LLMs — it’s about combining multi-modal understanding, RL-pretrained decision-making, and scalable verification systems that can keep up with agents generating knowledge humans don’t yet have. We are heading toward a world where:

- One agent can autonomously plan, select tools, execute tasks, and deliver results.
- RL pretraining becomes the norm for long-horizon decision-making.
- Verification generalization + multi-modal reasoning will be key unlocks for AGI.

This is the turning point where reinforcement learning becomes the backbone of intelligent, goal-driven agents.

## References

- [LinkedIn: Future of AI Agents = Agentic RL + Pretraining?](https://www.linkedin.com/pulse/future-ai-agents-agentic-rl-pretraining-jf-ai-p0nlc/)

- [硅谷101播客E201 feat. 朱哲清｜下一个AI前沿方向：强化学习预训练与AGI的转点时刻（上、下）](https://www.youtube.com/watch?v=3Kygpc3rCCo)(https://www.youtube.com/watch?v=52MNTycuS1o)
