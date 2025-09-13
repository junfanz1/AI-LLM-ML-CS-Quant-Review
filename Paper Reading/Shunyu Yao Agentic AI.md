# 30 Takeaways from Shunyu Yao's Talk on Agentic AI

Shunyu Yao has been one of the most influential researchers shaping the modern agentic AI movement. From ReAct to multi-agent collaboration, his work at OpenAI has deeply impacted how we think about reasoning, planning, and human-AI interaction. 

This post summarizes key takeaways from his latest in-depth 3-hour interview [https://www.youtube.com/watch?v=gQgKkUsx5q0&t=3245s], where he reflects on 6 years of agent research, the boundaries between humans and systems, and the future of a world that is simultaneously unified and pluralistic. 

A few thoughts before reading:

💬 How far can reasoning take agents beyond today’s chatbots?

🔑 Can we design rewards that are clear & hack-proof — or is reward hacking inevitable?

🔄 Will startups win by inventing new interaction modes or by building better data flywheels — or both?

👑 Will the future of AI be dominated by super-apps or decentralized multi-agent systems?

🔍 What “contrarian bets” could define the next scaling dimension of AI?

# English

## Definition of Agent
- **Agent**: interacts with the external world, makes decisions autonomously, and optimizes for reward.
- Early symbolic agents were rule-based, but rules can never capture every detail and edge case in the real world.
- **Deep Reinforcement Learning (RL)**, such as AlphaGo, leveraged infinite playthroughs in virtual environments, rewards, and general network architectures to learn like a black box.  
  - Limitation: environment-specific engineering (e.g., the game itself), poor generalization to new environments/games, and limited applicability in the real world.
- **Large model reasoning** allows agents to operate in new environments — coding, the internet, and the real world.
- **Language agents** are more fundamental than other agents:  
  - Language models provide powerful *priors*, enabling reasoning.  
  - Reasoning enables generalization — a requirement for thinking and problem-solving.

## Two Main Directions of Agents
- **Agents with their own rewards and exploration.**
- **Multi-agent systems** that form organizations.

### Improving Agent Capabilities
- Better context handling.
- Stronger memory.
- Lifelong online learning.

## Coding as an Affordance
- Coding is like the **hands** of a human — the most important affordance for AI.
- APIs are part of code. Future AGI may be API/code-based, human-defined, or a mixture of both.
- Many things don’t have APIs — it’s like asking whether we should make cars drive on all roads (agent adapts) or redesign all roads to fit cars (environment adapts).
- Ultimately, agents may be capable of doing almost everything.

## Generalization
- If pretraining includes all possible knowledge, RL is just activating skills.
- The optimal generalization may look like "overfitting reality" — if an agent is truly general, overfitting vs. generalizing stops mattering.
- Generalization comes from **reasoning**: transferable thinking skills across environments.
- Often, AI does not lack skills but lacks the **full context** to apply them.

## Second Half: Current State of Language Agents
- We now have general methods. The bottleneck is no longer training models but defining **good tasks** and **good environments**.
- **Key to RL success**: a well-defined task.
  - Rewards should be **outcome-based**, **computable**, and **not noisy**.
  - Process-based rewards or preference-based rewards risk **hacking** and do not solve the real problem.

### Types of Tasks
- **Reliability-oriented tasks** (customer support).
- **Creativity-oriented tasks** (writers, mathematicians).
- **Multi-agent tasks** emphasize breadth and long-term memory (like a company vs. a single person).
- **Design considerations**:
  - For code: measure *pass@1*, *pass@100* (at least one success).
  - For reliability tasks: measure probability of *never failing* or *failing at most once*.
- Current systems do not pay enough attention to **robustness** in simple tasks.

## Startups & Interfaces
- Startups must design **new interaction interfaces** *and* ensure the model is capable enough to power them. Both are necessary.
- If you only use old interfaces, ChatGPT will replace you.
- If you build new interfaces without better models, you will struggle.
- **Super Apps** are a double-edged sword: once you choose one interface, most of your resources will follow that path (e.g., Google’s search engine path dependency).

## Data Flywheel
- Most companies **do not have a data flywheel** and rely on better models.
- A data flywheel requires:
  - Training your own models.
  - Interaction-driven rewards to separate good data from bad data.
- **Good example: Midjourney**
  - Clear rewards (user likes/dislikes).
  - Aligned with the company’s business.
  - Creates a data flywheel — but note, this is not the main business for most companies.

## Pretraining & RL
- Pretraining may or may not be necessary — depends on the gap between open-source and closed-source models.
- Pretraining is a **non-trivial foundation for RL**:
  - Allows RL to work in language/prior-knowledge-based environments without "10^30 years" of learning.
- **Cost-value tradeoff**:
  - Current pretraining is expensive with limited value.
  - Justifiable when the environment is closed, high-value, and forms a full feedback loop (e.g., Google Ads).
  - But many real-world tasks are long-tail and require generalization, making pretraining more important.

## Research Directions for Agents
- **Long-term memory/context**: some contexts are distributed in the brain and cannot be written down.
- **Intrinsic rewards**: innovation requires intrinsic motivation, not just external rewards.
  - Babies explore through curiosity and internal drives.
  - Human games are closer to "word games," while baby games are physical exploration.
- **Scalable multi-agent systems**: creating new, great organizations.

## Human vs. Agent
- Humans can reason, machines are still lacking.
- Start from **first principles** and **utility** when designing systems.

## Crypto + Agents & Value-Based Business Models
- Human networks are becoming more centralized.
- Pareto principle (80/20) and Matthew effect (rich-get-richer) lead to stronger dominance of big companies.
- Centralization and diversity are not contradictory — both are accelerating.
- **Context limitation** is the ultimate barrier to centralization.
- Many people’s value lies in **information asymmetry** — maintaining privilege requires inventing distributed networks (e.g., traders sharing information via multi-agent exchange).

## Future Outlook
- Look for **contrarian opportunities** (different bets, unique interaction paradigms).
- You don’t need everyone’s consensus — just enough consensus.
- New **scaling dimensions** will emerge, but choosing the right scale for each application will be key.

## Language as the Core Tool
- Language is the most fundamental tool for generalization — like **fire** for humans.
- Specific domains (e.g., Go) may have better specialized reasoning systems, but language remains the **universal tool**.

## Scaling Up Agents
- The most important thing is to **find highly valuable applications**.
- Costs will naturally go down over time.

## Strong Agents & Interaction Modes
- There is no single definition of a "strong agent."
- Intelligence boundaries are interface-dependent, not model-dependent.
- Many interaction modes for assistants have not yet been invented.

## Manus & Killer Apps
- Manus represents a **general first-principle sense/interaction** with huge imaginative potential.
- Coexists with **killer apps** — doing one thing very well.

## DeepSeek & ChatGPT
- **DeepSeek**: long chains of thought are crucial — they give users a new experience.
- **ChatGPT**: not just technology, but memory forms a moat.  
  - More context → more stickiness → stronger moat.

## MCP & Context
- MCP is also memory.
- The world has a memory hierarchy — external environment included.
- Long context windows are a way to achieve long-term memory.

## Evaluation
- Necessary for measuring performance.
- Must be based on **real-world value**, not leaderboard scores.
- Ideally, evaluate across hundreds of tasks and aggregate rewards.
- The goal is to measure **how much better someone becomes after a year**, not just snapshot benchmarks.

## Future of Agents
- Expect **agentic interaction modes** and **new products like Cursor** built on larger, more general environments.
- Two directions:
  - **Model-based agents** (e.g., remote VMs).
  - Agents embedded into existing environments.

## Agentic Capabilities
- Need:
  - Rich interaction with digital environments (MCP, API).
  - Careful design, infrastructure, engineering.
  - An **ecosystem** for user intentions.

## Distributed vs. Centralized Agents
- In the next two years, agents may still be more centralized (super apps).
- Two directions:
  - Local user-environment optimization.
  - Creating brand-new environments (DeepResearch).

## High-Value Startup Opportunities
- Accumulating **user context** or building **specialized environments**.
- User context is like **oil before the invention of cars** — massive future value.
- If intelligence becomes ubiquitous, platforms like WeChat (environment + context) become powerful moats.

## Organizations: CEO vs. Scientist
- Organizational architecture can create many innovations.
- The path of organizational creation is different from that of scientific invention.

## Universal Mindset
- Focus on **high-ceiling projects**.
- Stay curious and imaginative.
- Creating something **more general than humans** is one of the most exciting goals.
- Suggested reading: *A Brief History of Intelligence*, biographies.

# 中文

Agent定义：能和外界交互、自我决策、optimize reward。

- 早期符号主义的基于硬规则的agent，但规则永远无法涵盖世界上所有细节和特殊情况。
- 深度强化学习如AlphaGo，有无穷次玩的虚拟环境、奖励、通用网络架构，像黑盒一样学习进步，但问题是环境-specific的工程（如游戏本身），无法泛化到其他环境、游戏，且很难真实世界应用。
- 大模型推理，可以有新环境如coding、互联网、真实世界。

语言agent比其他agent更本质：因为语言模型提供了强大的先验知识，先验知识可以推理，推理才能泛化。因为人可以思考，而没有推理做不到思考能力。

今天agent的两个方向：agent有自己的reward和探索、multi-agent形成组织。

提升agent能力：处理context, memory能力，并做lifelong online learning。

编程环境：coding就像人的手，是AI最重要的affordance。API也是code一部分，最终AGI是基于API或code，还是基于人的定义，可能是两者mixed的。很多事情并没有API，就像让车开在所有的路上，还是用人力改造所有的路去适应车（API），最终agent可能什么都做。

泛化：若pretraining包含所有世界上事情，RL只是激发出所有skill，maybe the optimal generalization is to overfit reality，如果是全能的那么讨论是overfit还是泛化就不重要了。但泛化的原因是能推理，即思考的技能可以迁移到不同环境，而不仅是技能多少。很多时候AI模型的技能不差，但缺乏的是完整的context。

Second Half：基于语言的agent正在转移，现在已有通用方法，bottleneck不是训模型和方法，而是怎么定义好的任务、好的环境，用通用方法解决什么问题。

RL成功关键是一个好任务：想定义一个reward是基于结果而非过程，是基于可计算的规则而非基于人或模型的黑盒偏好。RL任务最难的是定义白盒而非黑盒且不noisy的reward，任务有难度有价值。如果基于过程去定义reward就会出现hacking，如果优化人或机器的偏好也会出现hacking，不能解决问题。

定义不同任务：有的简单但注重reliability（客服），有的注重creativity（作家、数学家），multi-agent注重任务广度和长期记忆（一个人和一个公司的区别）。设计方法是pass@1, pass@100，起码成功一次的概率（代码），永远成功或最多失败一次的概率（客服），我们目前对简单任务的robustness还不够重视。

创业公司：创业公司要能设计不同interface和人机交互方式，即创造新的交互方式（如Cursor可以提示你写代码）&模型有溢出能力来赋能这个交互方式，二者缺一不可。如果只做旧的interface，那容易被ChatGPT取代；如果做新的interface但模型能力没有变好，那也难做。有super APP对公司来说是双刃剑，因为当你有一个交互方式的时候，公司的大部分资源必然形成路径依赖（例如谷歌依赖搜索引擎）。

数据飞轮：大部分公司没有数据飞轮，依赖于模型变好。数据飞轮需要自己训模型、通过交互有好reward来区分好数据坏数据。好案例Midjourney：有清晰reward（人喜欢哪张图）、应用与我公司业务align，这样可以数据飞轮，但数据飞轮也不主线。

Pretraining：会尝试训练模型，但未必pretraining，看情况，取决于开源与闭源模型的gap。Pretraining给RL做铺垫是non-trivial的，能在基于语言和先验知识的环境里做RL（而不需要学10^30年）。cost-value：当前cost很大而value不大，但也许不同应用有不同形态agent，很多交互方式需要不同模型的能力，如果能足够justify pretraining cost，就会合理。如果有封闭环境垂类价值足够大，数据能形成闭环，仅RL就可以（如谷歌ads）。但世界有很多长尾事情，需要generalization，需要像人类一样在线学习并适应环境，这时pretraining更重要因为需要泛化性。

Agent研究方向：long term memory/context（有些context只在大脑中基于分布式存在而无法写下，因此人无可或缺）, intrinsic reward（因为直到成功之前都没有reward/feedback，而是内在价值观和激励，这是创新者必要的。婴儿可以通过好奇或内在motivation做尝试，而人类社会的游戏更像文字游戏，与婴儿的物理游戏不同）, scalable multi-agent（创造新的伟大组织的人）

人与agent：人能推理而机器不够，基于第一性原理设计，从utility出发。

Crypto+agent，value-based商业模式：人类是网络，中心化程度会增加，贫富差距加大具有28定律和马太效应，大公司对世界掌控增加。不过另一个维度是中心化与diverse不矛盾，如今人们从网络边缘到中心的跃迁速度加快，普通人的机会也更多，产业组织形态也在增加。技术发展的趋势是两者都加剧，因为效率是根本原因。中心化的极限是context limitation，很多人的价值并非是某个单一技能的高超，而是掌握了别人没有的信息差，为了维持自己的previledge会发明出distributed network，例如每个trader都有自己的信息，用multi-agent交换信息来做交易。现在强巨头有motivation做中心化，而中心化以外的力量也会有motivation去做中心化，两种力量都会存在。

未来：要寻找反共识的事情，有different bet（多种super app的不同交互方式），不需要所有人的共识而只需要足够多人的共识就能做事。未来会有新scaling dimension出现，但基于不同应用如何选择scale比重是课题。

语言：是人为了实现泛化并完成事情的最本质的工具，就像火，而语言可以解决任何新任务。特定领域（如围棋）有比语言更好的思维方式，但语言不是为了处理特定任务（可能特定任务存在冗余性，但整体通用），而语言是通过强大先验知识可以打通各领域的通用工具。

Agent怎么scale up：找到很有价值的应用是最重要的，cost总会降低。

强agent：不同交互方式下有不同定义，不是单极的，需要不同系统去判断价值。智能边界取决于交互方式而非single model，比如做助手还有很多新的交互方式没诞生。

Manus：很通用的第一性感觉/交互方式/很多想象空间，与有killer app（做一件事好的应用），两者不矛盾。

DeepSeek：长思维链很重要，给人新体验。

ChatGPT：不仅是技术，而且memory可以形成壁垒，用户用的黏性强，有更多context就有更多黏性和壁垒（比如有效从很多用户对话中提炼出相关东西）。

MCP也是memory，context在软件里：世界有memory hierarchy，外部是世界环境。

Long context是实现long term memory的方式。

评估：是衡量好坏的必要条件，评估取决于现实世界的实际价值，而非刷榜与Benchmark。定义考试和游戏是简单well-defined reward，而世界很难是因为没有标准答案。评估可能要取决于几百个任务，把平行的数据加在一起进行reward，例如，评估人一年后能变得多好，而不是在100个平行宇宙中能变得多好。

Agent的未来：会有agentic交互方式，有新的cursor产品出现，但基于新的更大的环境copilot，两方面：基于模型的（如remote virtual machine），或者既有的环境场景中把agent引入。

Agentic增强能力：agent和数字世界的交互环境（MCP，API），人和agent交互是什么。需要很多设计、infra、工程。还有，如何构建user-intention生态系统。

Agent需要虚拟机吗：两年内可能还不会那么分布式，而是更中心化有很多super app。两方面：基于用户本地环境的优化，从头创造新的环境（DeepResearch）。

有价值的创业方向：积累user context，或构建特殊环境的公司，user context就像发明汽车之前的石油，有机会。如果intelligence可以普及，那么像腾讯微信这样拥有平台、环境、context会是壁垒。人类网络会变成啥样，取决于我们有更多人类朋友还是更多agent朋友。

CEO与科学家：组织架构也像通用方法可以创造很多东西（比如硅谷），它和科学家的发明创新路径有区别。

通用的Mindset：这个时代做上限更高的事情更好，想象力要丰富，什么都爱看，想变得通用，而创造比人更通用的东西更有意思。看《智能简史》、传记。

# Summary

🔭 Summarizing 30 key takeaways from Shunyu Yao's (OpenAI Researcher) latest in-depth 3-hour interview, where he reflects on #agent #research trends, the boundaries between humans and systems, and the future of an #AI world that is simultaneously unified and pluralistic.

🧠 He emphasized that agents must interact with the world, reason, and generalize — with language models providing powerful priors that enable this reasoning — and that the future of agents lies in both intrinsic-reward exploration and multi-agent organizational systems.

💡 The bottleneck is no longer model training but defining meaningful tasks, environments, and robust reward signals that avoid hacking, while improving context handling, memory, and lifelong learning capabilities.

🎯 He highlighted opportunities for startups to design new interaction modes, accumulate user context as a moat, and build valuable ecosystems, predicting that future agents will be integrated into larger environments, with language and long-term memory as the key to scaling.

Shunyu Yao’s perspective is both systematic and forward-looking, with several points worth highlighting:

Language as the Core of Reasoning 💬: He frames language-based agents as more fundamental than other agents because language carries prior knowledge — enabling reasoning, and thus generalization. LLMs are not just chat tools but universal reasoning engines, making language the “interface of interfaces” for future multimodal agents.

Reward Design > Bigger Models 🌉: Reinforcement learning’s real bottleneck is not the algorithm but designing well-defined tasks and white-box, low-noise rewards. Challenge is not to get a bigger model, but to define meaningful, computable goals that drive exploration and avoid reward hacking.

Bridging Technology and Business 🌌: He not only lays out the technical roadmap (memory, context, multi-agent, intrinsic motivation) but also stresses that startups must combine new interaction paradigms with model capability overflow, and build data flywheels as moats. He sees agents as a co-evolution of technology and business models, not just a research topic.

Pragmatism with Long-Term Vision 🧩: next few years may still be dominated by centralized super-apps, but also emphasizes the need for lifelong learning, generalization, and distributed systems in the long run — giving both near-term execution guidance and a north star for the future.

Encouraging Contrarian Thinking 🧐: Calling for “high-ceiling” projects and contrarian bets, urging researchers and entrepreneurs to focus on imagination, curiosity, and creating new interaction modes — rather than just chasing leaderboards.

# References

- [LinkedIn: 30 Takeaways from Shunyu Yao's Talk on Agentic AI](https://www.linkedin.com/pulse/30-takeaways-from-shunyu-yaos-talk-agentic-ai-jf-ai-dqz6c/)
- Full interview Video (in Chinese) 张小珺对OpenAI姚顺雨3小时访谈：6年Agent研究、人与系统、吞噬的边界、既单极又多元的世界: https://www.youtube.com/watch?v=gQgKkUsx5q0&t=3245s
- Shunyu Yao’s “The Second Half”: https://ysymyth.github.io/The-Second-Half/

