# 2025 Agentic AI Summit Berkeley − Technical & Industrial Insight

<div align="left">
  <marquee behavior="alternate" scrollamount="3">
    <strong>Views:</strong>
    <img src="https://komarev.com/ghpvc/?username=junfanz1&color=blue" alt="Profile Views" />
    &nbsp;•&nbsp;
    <strong>Followers:</strong>
    <img src="https://img.shields.io/github/followers/junfanz1?style=social" alt="GitHub Followers" />
    &nbsp;•&nbsp;
    <strong>Repo Stars:</strong>
    <img src="https://img.shields.io/github/stars/junfanz1/AI-LLM-ML-CS-Quant-Review?style=social" alt="Repository Stars" />
  </marquee>
</div>

---

Junfan Zhu

2025-08-05

[2025 Agentic AI Summit @ Berkeley Program Schedule](https://rdi.berkeley.edu/events/agentic-ai-summit)

<img width="2406" height="1202" alt="image" src="https://github.com/user-attachments/assets/2612341e-f25a-44c2-87d8-4e840f3086d8" />

Key Takeaways from Agentic AI Summit, UC Berkeley, 2025-08-02

If you like this repo, feel free to give a ⭐️

#Berkeley #AgenticAI


## Contents

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [Main Stage](#main-stage)
   * [1. Opening Keynote: Hardware for AI Agents](#1-opening-keynote-hardware-for-ai-agents)
   * [2. Keynote: Frameworks & Stacks for Agentic Systems](#2-keynote-frameworks-stacks-for-agentic-systems)
   * [3. Reflective Optimization of Agents with GEPA and DSPy](#3-reflective-optimization-of-agents-with-gepa-and-dspy)
   * [4. Visionary Stacks for Agentic Systems: Insights and Innovations Ahead Focus Talks](#4-visionary-stacks-for-agentic-systems-insights-and-innovations-ahead-focus-talks)
   * [5. Context Engineering and MCP for Document Workflows](#5-context-engineering-and-mcp-for-document-workflows)
   * [6. Multi-Turn Reinforcement Learning for LLM Agents](#6-multi-turn-reinforcement-learning-for-llm-agents)
   * [7. From Execution to Supervision: Scaling Productivity with Agents](#7-from-execution-to-supervision-scaling-productivity-with-agents)
   * [8. The Breakout year of Coding Agents](#8-the-breakout-year-of-coding-agents)
   * [9. Reliable AI Agents = Predictable + Aligned](#9-reliable-ai-agents-predictable-aligned)
   * [10. Building Scalable AI Companies: AI Hackers](#10-building-scalable-ai-companies-ai-hackers)
   * [11. Startup Pitch & Fireside Chats](#11-startup-pitch-fireside-chats)
- [Workshops](#workshops)
   * [101. Accelerating Agents: Scalable Efficient Inference with vLLM](#101-accelerating-agents-scalable-efficient-inference-with-vllm)
   * [102. MCP as a Foundational Protocol for Agents](#102-mcp-as-a-foundational-protocol-for-agents)
   * [103. Ray: Scalable Agent Infra](#103-ray-scalable-agent-infra)
   * [104. Training Intelligent AI Agents with RL](#104-training-intelligent-ai-agents-with-rl)
   * [105. A2A: AI Agent Collaboration](#105-a2a-ai-agent-collaboration)
   * [106. How to 10x AI Agent Price-Performance with Inference-Time Scaling](#106-how-to-10x-ai-agent-price-performance-with-inference-time-scaling)
   * [107. Multimodal Challenges, Neural Operating Systems and Deploying AI Agents by Lambda](#107-multimodal-challenges-neural-operating-systems-and-deploying-ai-agents-by-lambda)
   * [108. Vibe Coding with Goose: Building Apps with AI Agents and MCP by Block](#108-vibe-coding-with-goose-building-apps-with-ai-agents-and-mcp-by-block)
   * [109. AI Agents that Interact with the World ](#109-ai-agents-that-interact-with-the-world)
- [Frontier Stage ](#frontier-stage)
   * [201. MassGen: Frontier Multi-Agent Scaling in Open Source](#201-massgen-frontier-multi-agent-scaling-in-open-source)
   * [202. Open, Interoperable Infrastructure for the Internet of Agents](#202-open-interoperable-infrastructure-for-the-internet-of-agents)
   * [203. Transforming Agent-based Chatbots with Declarative Programming](#203-transforming-agent-based-chatbots-with-declarative-programming)
   * [204. Automating Scientific Research with AI Agents](#204-automating-scientific-research-with-ai-agents)
   * [205. AI Agent Scientific Reasoning in LLMs](#205-ai-agent-scientific-reasoning-in-llms)
   * [206. Grounding Foundation Models for Physical Science](#206-grounding-foundation-models-for-physical-science)
   * [207. AI for Science Lightning Talks](#207-ai-for-science-lightning-talks)
      + [Text-to-SQL using Multi-agent Collaborative Framework to find complex Autonomous Driving Scenarios at Zoox ](#text-to-sql-using-multi-agent-collaborative-framework-to-find-complex-autonomous-driving-scenarios-at-zoox)
      + [Provably Accountable AI Critical Open Questions](#provably-accountable-ai-critical-open-questions)
      + [Agentic Debugging Frameworks for LLMs](#agentic-debugging-frameworks-for-llms)
      + [Small VLM as Future of Agents ](#small-vlm-as-future-of-agents)
      + [Agentic Any Image to 4K Super-Resolution](#agentic-any-image-to-4k-super-resolution)
      + [Verl: Flexible and Efficient RL Framework for LLMs](#verl-flexible-and-efficient-rl-framework-for-llms)
   * [208. Building Autonomous Agents](#208-building-autonomous-agents)
   * [209. Agentic AI in Healthcare: Building Trustworthy Autonomy](#209-agentic-ai-in-healthcare-building-trustworthy-autonomy)
   * [210. AgentX Competition Winners](#210-agentx-competition-winners)
   * [211. ](#211)

<!-- TOC end -->

<!-- TOC --><a name="main-stage"></a>
# Main Stage

<!-- TOC --><a name="1-opening-keynote-hardware-for-ai-agents"></a>
## 1. Opening Keynote: Hardware for AI Agents

> Bill Dally, Chief Scientist @ NVIDIA

Inference 2 phases
- Prefill/Prompt phase: compute limited, latency insensitive
- Decode phase (token generation): bandwidth limited, latency sensitive

Tree of Thought

<!-- TOC --><a name="2-keynote-frameworks-stacks-for-agentic-systems"></a>
## 2. Keynote: Frameworks & Stacks for Agentic Systems

> Ion Stoica, Co-Founder @ Databricks & Anyscale, UC Berkeley

- Multi-Agent System failure Taxonomy (MAST)
  - Multi-agent system fail >50% on hard benchmarks, due to specification issues (improve multi-agent system design), Inter-agent misalignment (protocol + deeper social reasoning abilities from agents), Task verification (unit-testing + multi-level verification)
  - MAST Annotator Web, can analyze failures when you upload traces, understand why.
- LMArena: understand and quantify human preferences, key to reliable interaction with AI systems.
- Specifications is the key to build reliability, can verify system, debug system, decompose system, reuse components, plus automated decision making with no human in the loop.

<!-- TOC --><a name="3-reflective-optimization-of-agents-with-gepa-and-dspy"></a>
## 3. Reflective Optimization of Agents with GEPA and DSPy

> Matei Zaharia, CTO @ Databricks, UC Berkeley

- Teach AI new task: standard way in ML is to use weight updates with gradient descents, but requires huge data. As flops get cheaper, progress in AI capabilities will be limited by sample efficiency.
- Teach model more effectively: RL with verified rewards, instead of only 0/1 rewards, have a LM look at traces of all rollouts and reflect on what worked in them (using all intermediate outputs). Instead of only updating weights with small deltas, update a prompt: a single natural language update can give a large behavior change. → GEPA (Genetic-Pareto, evolutionary algorithm)

<img src="https://github.com/user-attachments/assets/66daae84-88c3-47a9-b01f-f4fd3c2c2890" width="50%" height="50%">

<img src="https://github.com/user-attachments/assets/7fab5c5a-c385-42fa-9228-fbfd6405f903" width="50%" height="50%">


- What is learned in GEPA? Prompt: Given summary, generate new query. So it can reflect on its past mistakes with hundreds of steps
- DSPy: framework for declarative programming with LLM, just describe workflow with high-level signatures, DSPy will optimize how to turn into prompts or weight-tune models to optimize performance. Could be the future of AI training.

<!-- TOC --><a name="4-visionary-stacks-for-agentic-systems-insights-and-innovations-ahead-focus-talks"></a>
## 4. Visionary Stacks for Agentic Systems: Insights and Innovations Ahead Focus Talks

> Sherwin Wu, Head of Engineering @ OpenAI

Ensembling agents together tends to improve performance, so use more inference compute!

> Chi Wang, Senior Staff Research Scientist @ Google DeepMind

Challenges: consensus, shared context, interop

<!-- TOC --><a name="5-context-engineering-and-mcp-for-document-workflows"></a>
## 5. Context Engineering and MCP for Document Workflows

> Jerry Liu, CEO @ LlamaIndex

Automate Knowledge Work
- Context (supply the right tooling so LLM can access external services and get output) 
  - Main limitation of adoption AI: You as a human is responsible for defining actual input, to tell AI what to do, API completely depend on user. If not good at prompting, then not using effectively. 
  - RAG should be reimagined as an MCP toolbox (retrieval, manipulation, structured querying).
- Workflow engineering Goal: define constrained steps of acts, you as user design prompts. Can encode the process as a DAG, need human in the loop in the end.

<!-- TOC --><a name="6-multi-turn-reinforcement-learning-for-llm-agents"></a>
## 6. Multi-Turn Reinforcement Learning for LLM Agents

> Sergey Levine, Co-Founder @ Physical Intelligence, UC Berkeley

How to train goal-conditioned value functions, use sub-optimal interactive data to get optimal strategy with RL: 
- Bellman equation (next action that is most likely to lead to the goal), so you don’t need optimal training data (states, actions), the data tells you what might happen and what transformation may occur, and then you ask if the best scenario what you’ll get, we can learn for many goals simultaneously. This is Q-learning with 0/1 rewards indicating goal reaching. 
- Offline RL: to train goal-conditioned value functions using our dataset, leverage representations from LLM as features on top of which to train value functions. Use LLM to make predictions that aid in optimal decision-making. In-domain data can help us learn expert level prediction. Offline RL can learn predictors for strategies that are more optimal than the ones in the data. Finally, we can combine all this to train agents using only suboptimal data that could outperform humans who generated data.

<!-- TOC --><a name="7-from-execution-to-supervision-scaling-productivity-with-agents"></a>
## 7. From Execution to Supervision: Scaling Productivity with Agents

> May Habib, CEO @ Writer

Action agent: let agent automatically solve problem, by supervising digital workforce. 
- Contain dynamic (not static) agent actions, Sandbox environment to secure agents
- Understand why (not ask what) happened, for reasoning tracability and guardrail alignment
- Design intelligent delegation paths (not simple permission), orchestration graph control.

<!-- TOC --><a name="8-the-breakout-year-of-coding-agents"></a>
## 8. The Breakout year of Coding Agents

> Michele Catasta, President @ Replit

- Frontier models with massive RL scaling will keep getting better on coherent, long, tool-using agent trajectories.
- Agent builders will create alpha on context management, quality of environment, stacks integrations search parallelism, etc.
- Gap between research and vibe coding
  - Human in the loop is really hard, as model is too sycophantic for non-tech users
  - Vibe coders want maintainable iterative environment, but issue → PR setup has already established code structure, and popular repos are extensible and SWE-bench rarely adds new components
- We need more evals. Capabilities with major product impact remain difficult to measure: app generation from scratch, frontend functionality, parallel tasks, long-term iterative development + maintenance
- Replit agent: less structure in the agent loop, more in the environment

<!-- TOC --><a name="9-reliable-ai-agents-predictable-aligned"></a>
## 9. Reliable AI Agents = Predictable + Aligned

> Karthik Narasimhan, Head of Research @ Sierra, Princeton

- Reliability is a measurable system property, to improve reliability: agent interfaces, self-evaluation + correction, memory, fine-tuning
- Future trends: proactive + self-improving agents, multi-agent networks (agents verify each other, adding a layer of trust)

<!-- TOC --><a name="10-building-scalable-ai-companies-ai-hackers"></a>
## 10. Building Scalable AI Companies: AI Hackers

> Snehal Antani, CEO @ Horizon3

- Agent 1: Iterative trail-and-error: agents explore step by step, using loops of reflection and retries. Each decision is influenced by the past. Flexible and useful for exploration, but expensive, non-deterministic, and inefficient at scale.
- Agent 2: Structured context for precision prompting: system builds context-facts, states and relationships stored in the graph. System queries graph to generate precise directives as prompts for targeted reasoning via LLM.
- NodeZero
  - Build knowledge graph, decision engine queries knowledge graph, queries generate comprehensive detailed prompts to increase precision
  - High value targeting, advanced data pilfering, explainable narratives, etc.
  - Amazon bedrock: find the best model to use, cheap for scalable business, with architectural flexibility
  - GOAD-HARD is hard, because of: multi-step exploits, conditional execution, discover & abuse trust, reason over file content, no prior knowledge, no human in the loop. Everything is patched, 5VM range, no legacy protocols.

<!-- TOC --><a name="11-startup-pitch-fireside-chats"></a>
## 11. Startup Pitch & Fireside Chats

> Alex Shan, Co-Founder @ Judgment Labs

- Agent PoCs are cool, but: agents break in production (infinite loops, improper tool calls, broken retry logic), broken CI/CD (lack of robust evals prevent regression tests, production monitoring, alerts), no route to optimization (no method for piping agent data into post-training workflows). → Judgement Labs: Open source post-building layer for agents.

> Vinod Khosla, Founding Partner @ Khosla Ventures

- We should trying fail, but don’t fail to try.

<!-- TOC --><a name="workshops"></a>
# Workshops

<!-- TOC --><a name="101-accelerating-agents-scalable-efficient-inference-with-vllm"></a>
## 101. Accelerating Agents: Scalable Efficient Inference with vLLM

> Michael Goin, Principle Engineer @ Red Hat

- vLLM can: efficiently schedule requests into the next forward pass, manage KV Cache context and runtime memory footprint
- Key challenge: handle multiple concurrent requests
  - Batching: not static, but continuous
  - KV Cache: takes up memory
- vLLM innovation: Paged attention, that allows for strong continuous keys/values in non-contiguous memory space.
  - Automatic Prefix Caching: reuse KV Cache blocks across requests, improve time-to-first-token by skipping prefill. Example: Multi-turn conversation.
  - Quantization: weight, activation, KV Cache
  - Speculative Decoding: accelerate decoding phase with speculation, by ngram, draft model, EAGLE, etc.
  - Distributed: multi-host, multi-device
  - Parallelism: mixture of tensor, pipeline, expert, data
  - Disaggregated Serving: partition the “time” dimension by separating instances for prompt processing & token generation. 

<!-- TOC --><a name="102-mcp-as-a-foundational-protocol-for-agents"></a>
## 102. MCP as a Foundational Protocol for Agents

> Jason Kim @ Anthropic

- Models are only as good as the context provided to them

<img src="https://github.com/user-attachments/assets/d7f5a358-c6cd-4394-8adf-cdcc1cf6b114" width="50%" height="50%">

<img src="https://github.com/user-attachments/assets/3f7d5811-adc2-4164-9e94-c33dfc57e7fa" width="50%" height="50%">

- Why build a client? Connect to network of MCP servers, reduce time to build and maintain integrations, maintain flexibility to switch between LLM providers and vendors
- Why build a server? Allow AI apps and agents to connect to your product, build and maintain a single interface for exposing your app’s data, simplify external integrations

<!-- TOC --><a name="103-ray-scalable-agent-infra"></a>
## 103. Ray: Scalable Agent Infra

> Sumanth Hegde @ Anyscale

- Post-training runs on Ray: OpenRLHF, VeRL, NeMo-RL, SkyRL

<img src="https://github.com/user-attachments/assets/f0768ab3-7222-4f83-8920-480d239ad593" width="50%" height="50%">

<img src="https://github.com/user-attachments/assets/6d0c6396-2c89-4a39-8dc9-72cea1d01ace" width="50%" height="50%">

<img src="https://github.com/user-attachments/assets/1f94485d-1422-4add-bf17-1002868b9666" width="50%" height="50%">

- Use case for Deploying Agents: use Ray Serve as flexible, scalable, efficient compute for online inference
- Use case for Deploying LLMs: automatic scaling & load balancing, multi-LoRA support, scale to massive models like DeepSeek R1

<img src="https://github.com/user-attachments/assets/5e896eac-e58a-4cd4-8d53-8573d5347b4f" width="50%" height="50%">

<!-- TOC --><a name="104-training-intelligent-ai-agents-with-rl"></a>
## 104. Training Intelligent AI Agents with RL

> Daniel Han-Chen, CEO @ Unsloth

<img src="https://github.com/user-attachments/assets/3570679f-64e8-4991-971c-c10e155d7592" width="50%" height="50%">

GRPO: reduce memory usage, no need reward model, use value model to estimate average rewards removed. Want to do action better than average.

- Other reward functions that can replace the blue box: LLM as a judge, regex check, executable code, format check.
- Inference sampling / rollout: use LLM itself to generate n answers and give the correct answer reward while others 0 reward, take statistics to get advantage (z-score) so no more value model.
- GRPO’s group relative: do above for many questions, not only 1 question

<img src="https://github.com/user-attachments/assets/b2d138b1-7c77-4eba-b8a0-06b3dfce2e42" width="50%" height="50%">

<!-- TOC --><a name="105-a2a-ai-agent-collaboration"></a>
## 105. A2A: AI Agent Collaboration

> Chitra Venkatesh @ Google

- Capability Discovery: A2A agents explicitly list their capabilities and skills through an AgentCard (like business card) that’s found at a well-known URL, so agents can find other agents to collaborate with automatically.
- UX negotiation: A2A agents allow for Multimodality negotiations during handoffs, agreeing on other agent’s critical parameters/outputs beyond just modality.
- Task and State management: A2A manages long-running operations using a Task object with unique ID/status, allowing for asynchronous communication. A2A supports streaming with SSE for real-time updates.
- Secure collaboration: Agents interact without needing to share internal memory, tools or proprietary logic, ensuring security and IP.
- `pip install a2a-sdk`

<!-- TOC --><a name="106-how-to-10x-ai-agent-price-performance-with-inference-time-scaling"></a>
## 106. How to 10x AI Agent Price-Performance with Inference-Time Scaling

> Jared Quincy Davis, CEO @ Foundry, Stanford

- Compound AI Systems: https://github.com/compound-ai-systems/awesome-compound-ai-systems
- Ember: inference-time scaling architecture framework. Eclipse quality or reliability frontier available via today’s frontier LLMs or to achieve comparable quality at 1/1000th cost or less.
- Routing: FrugalGPT cut cost 98%
- Ensembles with voting-based aggregation: revealing a rich, non-monotonic relationship between ensemble-size and performance
- Best-of-N graphs: shockingly simple and effective primitive
- Verifier-prover structures: showing how composing calls into verifier-prover structures provably and predictably boosts performance 
- Parallel thinking (Deep Think): Latest Gemini 2.5Pro system’s core technique
- Alex Demachus, Berkeley: conic decoding, take 5 DeepSeek R-1 calls, more thinking means more efforts in tree search, but when it’s stuck it’ll backtrack, the longer the thinking traces were, the more likely that the answer is wrong, because it wasn’t able to backtrack to the right path before it runs out of time. If you ask 5 times, the answer with the shortest length is more likely to be correct than the longer answers. Based on the fact, he creates an algorithm for early stopping to select the shortest answer among 5 replicas, by waiting for the first thinking token and stopping all other replicas and returning the answer. Therefore, faster, smarter, cheaper, even 5 clones save money.
- A committee of models gets better answers for easy questions, reducing variance for hallucination. But not good for hard questions, as it’s unlikely to keep consistency between long answers from different models, so you spend more money with worse answers.

<!-- TOC --><a name="107-multimodal-challenges-neural-operating-systems-and-deploying-ai-agents-by-lambda"></a>
## 107. Multimodal Challenges, Neural Operating Systems and Deploying AI Agents by Lambda

> Amir Zadeh, Staff Machine Learning Researcher @ Lambda

- NeuralOS: the neural operating system. 
- Multimodal learning, by 2030 we’ll run out of useful data points.
- 2 algorithms for scaling: autoregressive, diffusion (language community is exploring diffusion on text, e.g. D3PM, PLAID)

CMU: Diffusion beats Autoregressive in data-constrained settings.
- Diffusion creates random masks and train models to recover, get rewards/penalties if the prediction is correct/incorret. Autoregressive predicts next token, from 3 masks to 2 masks to 1 mask decreasingly.
- Pareto optimal tradeoff of validation loss and flops. Train 1 epoch only, as 1 epoch is always the same, which is true for autoregressive. There's a critical compute point where we can compute and predict, and diffusion is more data efficient using less data to do better, while ultimately autoregressive is data hungry, there’s a way to compensate flops issues.

<img src="https://github.com/user-attachments/assets/c69ef45c-819e-4b43-8eec-fcc5d43c4853" width="50%" height="50%">

- Half-life of data-reuse: at what point of data < 50% useful for us. After 31 epochs and 100 million tokens, autoregressive can’t extract more useful information, but diffusion’s half-life occurs at 493 epochs, you can still extract 50% useful information after 493 epochs. It’s useful in multimodality in that data will run out in the end, we need more scalable algorithms not bounded by data and algorithms.
- At what point can I expect new data is happening in the same epochs over and over again, no longer be considered as new data? After 4 epochs, data is no longer new for autoregressive, autoregressive has exhausted data with no new useful information, but for diffusion, that happens > 100. So with the same data, diffusion extracts more information than autoregressive.

<img src="https://github.com/user-attachments/assets/892fcdfd-0655-4cd1-aa05-34eb9a4b7c08" width="50%" height="50%">

<img src="https://github.com/user-attachments/assets/ecef029a-33bb-4277-a912-d652586015db" width="50%" height="50%">

- Conclusion: If you’re compute bottlenecked, use Autoregressive; if you’re data bottlenecked, use Diffusion. In robotics, simulation training, use Autoregressive; real-world training, use Diffusion.

<!-- TOC --><a name="108-vibe-coding-with-goose-building-apps-with-ai-agents-and-mcp-by-block"></a>
## 108. Vibe Coding with Goose: Building Apps with AI Agents and MCP by Block

> Angie Jones, Global Head of Developer Relations @ Block

- https://tinyurl.com/goose-subagents-workshop
- Subagent: main agent spins off so that other agents can orchestrate with it, can be created dynamically. If you want a math tutor agent, you can make it as a recipe and keep it for future reuse.
- There’re 5000 MCP servers, choose which one? We’re in the early days of MCP, projects on registry-as-a-service like Goose will call that service to have an official registry and figure out which MCP service you’ll need, and install it for you to do your task.
- How to measure and choose from which coding style for my needs? There’s a terminal benchmark that rates agents and gives a series of tests to do and ranks them. Depending on your task, you can make tests (when the model cooperates well with agents) by yourself and evaluate models any time. It’s becoming the standard of agent interactions.
- Goose can speed up development by splitting e.g. frontend and backend, as it’s wise enough to make API contracts connecting frontend and backend, not relying on any agent but working in parallel. 
- Every model has a context window, Goose can summarize your previous conversations giving you a new sheet of paper with highlights you can take away for new session.

<!-- TOC --><a name="109-ai-agents-that-interact-with-the-world"></a>
## 109. AI Agents that Interact with the World 

> Abhishek Hingnikar @ Auth0/Okta

<img src="https://github.com/user-attachments/assets/75ca035f-fbb6-4096-9daa-95fe3758a755" width="50%" height="50%">

<img src="https://github.com/user-attachments/assets/1f26658f-6d5a-40b6-be7f-3ca4e923393e" width="50%" height="50%">

- We can model as an internal MCP gateway, have a proxy MCP server, because protocol allows you to add a layer of MCP server combined in front of another MCP server. E.g. You can rely on your primary identity provider, which gives you a token for AI agent, that only grants access to MCP. (e.g. If Claude wants access to Jira, this credential is only good to call your MCP gateway. From MCP gateway to Jira, your primary IAM can facilitate this exchange, so that we ensure all OAuth credentials do not end up in third-party AI Agent clients, but they can be close to MCP server, and we don’t need to have OS-level assurance by container. If AI Agent wants to talk to MCP server, it will register its own credentials from OAuth client ID vault metadata, and OAuth won’t know who’s the agent, then this Agent can be allowed to VSCode, Claude, but not allowed for else, hence protecting internal resources.)
- We can have MCP server talk to another MCP server, and an MCP server can be a client, there can be a lot of nesting.


<!-- TOC --><a name="frontier-stage"></a>
# Frontier Stage 

<!-- TOC --><a name="201-massgen-frontier-multi-agent-scaling-in-open-source"></a>
## 201. MassGen: Frontier Multi-Agent Scaling in Open Source

> Chi Wang, Senior Staff Research Scientist @ Google DeepMind

- https://github.com/Leezekun/MassGen
- MassGen Orchestrator distributes tasks to different agents with different models in real-time collaboration, then share collaboration in the hub and reach consensus.
- Goal: cross-model/agent synergy for robust results; Multi-Agent system where agents evolve together. E.g. Grok Heavy uses the same model but multi-agent system, price goes up 10 times.
- Foundation: AutoGen orchestration, iterative refinement, 
- Insight: Reasoning isn’t linear, it’s messy, iterative, collaborative
- Features and methodology: cross-model synergy and harness strength, parallel processing, iterative refinement with nonlinear reasoning through cycles of improvements, intelligence sharing where agents share and learn each others’ work, consensus building with natural convergence through collaboration
- Challenges: shared memory, context, interoperability, conflicts when multi-agents not easily scalable simultaneously (e.g. double-booked flight tickets)
- Future: recursive agents bootstrapping intelligence, more agents, evaluation ability for each agent (e.g. context learning may have agent output poor examples blocking whole trajectory).

<!-- TOC --><a name="202-open-interoperable-infrastructure-for-the-internet-of-agents"></a>
## 202. Open, Interoperable Infrastructure for the Internet of Agents

> Papi Menon, CPO @ Outshift by Cisco

AGNTCY
- https://agntcy.org
- Why new Internet of Agents? We need new ways for agents to discover each other and assemble them to collaborate securely, cross-platform and reliably. We have a fragmented, heterogeneous deployment model (e.g. agents on cloud, agents in enterprise, and all servers need to communicate with each other). Due to probabilistic and multimodal, we need evaluation and observation when conflicts and loops.  → Goal: Semantic protocols above the Internet that enable agents to talk to each other.
- Ensembles of Agents need a secure collaboration layer. 
- Agent lifecycles: 
  - advocate themselves’ ability/skillset so that other agents discover them, 
  - how they stitch together and allow communication on semantic layer from different platforms (cloud, local) and be understood, 
  - how to deploy all to your cloud infra with message layer optimizing agent communication, 
  - how to reach low-latency with human-in-the-loop for multi-agent, 
  - all communication modes supported by the layer (e.g. a large state of exchange of 1-to-1, 1-to-many, many-to-many), 
  - how to describe identity of agent, 
  - agent with different abilities performs specific task depending on what they’re allowed to do, with a tuple (agentic identity, identity of human principle on who’s behalfing the agent is acting, task the agent is performing at any time)

<img src="https://github.com/user-attachments/assets/a0c2e6ef-b46f-4593-948e-a6020b270cfb" width="50%" height="50%">

<img src="https://github.com/user-attachments/assets/a2c1d1fc-ac15-45dc-bdd8-e7bb6f11d293" width="50%" height="50%">

<!-- TOC --><a name="203-transforming-agent-based-chatbots-with-declarative-programming"></a>
## 203. Transforming Agent-based Chatbots with Declarative Programming

> Xifeng Yan @ UCSB

- Most customer service chatbots are program-centric, not agent-centric, even OpenAI’s swarm environment can pack functions into agents, but agent definitions are buried in Python program. 
- Agent Declarative Language (ADL) puts all domain specific knowledge and business logic in one agent-centric file, to describe what agent can do and their relationship. So each agent can handle FAQ, recommendation, guardrail, workflow, and can be coordinated into 1 file.
- Why need? Declarative, no vendor lock-in. Separate logic and optimization. More natural language programming. Simplify chatbot maintenance and updates.

<!-- TOC --><a name="204-automating-scientific-research-with-ai-agents"></a>
## 204. Automating Scientific Research with AI Agents

> Samuel Rodriques, CEO @ FutureHouse

platform.futurehouse.org Already has AI published research papers with new discoveries in biology, human only needs to provide wet lab experiment results.

<img src="https://github.com/user-attachments/assets/663760e7-0d76-43a2-b795-36433e4ae6c8" width="50%" height="50%">

<!-- TOC --><a name="205-ai-agent-scientific-reasoning-in-llms"></a>
## 205. AI Agent Scientific Reasoning in LLMs

> Le Cong @ Stanford Medicine

- RL router: Have 2 layers of RL, because some models are good at A, some models good at B, having an RL over RL can understand what models are good at what specific task, then integrate like a MoE approach with open router, and route the challenging question to the best model.
- Build complex workflow to do analysis based on understanding of your research, agentic workflow can scale at almost any domain specific area, especially labor-intensive tasks.
- Self-evolving LLM agent: STELLA (Mengdi Wang @ Princeton), if an agent is given many tools and APIs and benchmarks, agents can improve by self-RL even without much fine-tuning. When agents are self-evolving, we can give them more data, give them cameras to let them see what happens using VLM to understand the real world.

<img src="https://github.com/user-attachments/assets/91b904ab-dc11-4f97-ae49-ce9a81ad1476" width="50%" height="50%">

<!-- TOC --><a name="206-grounding-foundation-models-for-physical-science"></a>
## 206. Grounding Foundation Models for Physical Science

> Rose Yu @ UCSD

Adapting While Learning (AWL): shouldn’t over-rely on tools without generalization to new tasks, models need to learn when/how to use tools, only when the model realizes its internal knowledge is insufficient to some tasks, then it’ll seek external resources. 

<!-- TOC --><a name="207-ai-for-science-lightning-talks"></a>
## 207. AI for Science Lightning Talks

<!-- TOC --><a name="text-to-sql-using-multi-agent-collaborative-framework-to-find-complex-autonomous-driving-scenarios-at-zoox"></a>
### Text-to-SQL using Multi-agent Collaborative Framework to find complex Autonomous Driving Scenarios at Zoox 

> Core Data Science Team @ Zoox 

<img src="https://github.com/user-attachments/assets/1f4c5d95-f39e-49a0-98d6-7c94fe39d7de" width="50%" height="50%">

- to query/search for scenarios encountered in driving logs, easy to interpret and scale for any scenario definition 

<!-- TOC --><a name="provably-accountable-ai-critical-open-questions"></a>
### Provably Accountable AI Critical Open Questions

- How to design efficient Zero Knowledge Proofs for complex, multi-step agentic reasoning?
- What are crypto-economic models for self-governing and resilient agent ecosystems?

<!-- TOC --><a name="agentic-debugging-frameworks-for-llms"></a>
### Agentic Debugging Frameworks for LLMs

> IBM Research

- Trajectory Analyzer: dynamically identify errors and then provide revise prompts within certain thresholds to identify failures (hallucination, improper tool call), use prompt validator to check static errors and provide structured ReAct LangChain format.

<!-- TOC --><a name="small-vlm-as-future-of-agents"></a>
### Small VLM as Future of Agents 

> Virginia Tech

- Small model is good for edge AI deployment for robots, and data privacy and model efficiency tasks.
- Enhance small VLM through self-evolution or reinforce gym environment.

<!-- TOC --><a name="agentic-any-image-to-4k-super-resolution"></a>
### Agentic Any Image to 4K Super-Resolution

> TAMU

<img src="https://github.com/user-attachments/assets/8a3f5d09-70d3-4ac6-9c7d-555245e1b46e" width="50%" height="50%">

<img src="https://github.com/user-attachments/assets/74c112dc-0ff4-4bd9-a5c7-d391920eeca8" width="50%" height="50%">

<!-- TOC --><a name="verl-flexible-and-efficient-rl-framework-for-llms"></a>
### Verl: Flexible and Efficient RL Framework for LLMs

> Amazon

<img src="https://github.com/user-attachments/assets/27bdf757-f482-49e8-a584-8696a33644bd" width="50%" height="50%">

Verl
- RL recipes: PPO, GRPO, RLOO, reinforce++, DAPO (beats R1 with fewer steps), Dr.GRPO
- Transformers integration: DeepSeek, Llama, Qwen, Gemma
- Inference engine: vLLM, SGLang
- Distributed Training Engine: FSDP, Megatron 
- System optimizations: seq packing, seq parallelism, fused entropy kernels 
- Hybrid controller programming: program distributed RL algorithm like a single threaded program

RL Training in Rufus (Amazon AI Shopping Assistant)
- Large scaled model: optimized actor training and rollout inference throughput, supported long-context and reasoning scenarios
- Shopping assistant: asynchronous API environment and rollout, standalone and efficient reward services (reward signals: verifiers, LLMaaJ, preference modeling)

<!-- TOC --><a name="208-building-autonomous-agents"></a>
## 208. Building Autonomous Agents

> Adam Kaplan, Senior Staff Engineer @ LinkedIn Core AI

- Hiring Assistant multi-agent

<img src="https://github.com/user-attachments/assets/46a74356-80e4-4a98-9a0f-512bc3ea684a" width="50%" height="50%">

- Autonomous agents react to environment signals, with 3 tiers of memory (in session immediate, episodic summaries, long term RAG)

<img src="https://github.com/user-attachments/assets/36157836-6ed1-4eb2-8300-76ccd1bc5691" width="50%" height="50%">

- Scoped Agentic Memory w/Task Sharing: Users are isolated due to trust/regulatory concerns, we introduce another type of common knowledge: shared task summaries, so that agent is aware of what it’s doing under other contexts, and integrate information into main task loop instead of elsewhere.

<img src="https://github.com/user-attachments/assets/97d991e2-91ba-4be8-9aa4-f0c75ab4f083" width="50%" height="50%">

- Modular Agentic Architecture: take agent system and adapt to human shape/style of work (when you hire someone you onboard and give them keys, behavior psychology can also adapt our agents to conform to infra rules like anti-jailbreak)

<!-- TOC --><a name="209-agentic-ai-in-healthcare-building-trustworthy-autonomy"></a>
## 209. Agentic AI in Healthcare: Building Trustworthy Autonomy

> Anu Trivedi, Head of GenAI @ Oracle Health HDI

- Tool Integration is Mandatory
  - Agents without deterministic tool access (ontology mapping, scheduling APIs, clinical rule engines) cannot deliver reliable outcomes.
- Auditable Planning
  - LLM-generated plans must be materialized, versioned, and reflected upon-self-reflection + plan revision is key for reliability.
- Modular Multi-Agent Flows
  - Decomposing tasks into specialized agents (e.g., retrieval, reasoning, execution) increases resilience and enables granular recovery.
- Hybrid Decisioning
  - Pure LLM outputs fail edge-case safety constraints
  - combining rule-based constraints with learned reasoning gives acceptable precision/recall trade-offs.
- State & Context Continuity
  - Long-lived state stores (shared memory) are essential for multi-step clinical/trial workflows to avoid information loss between interactions.

<!-- TOC --><a name="210-agentx-competition-winners"></a>
## 210. AgentX Competition Winners

- AgentSynth
  - Multistep agentic work (booking flights) is hard to collect training and evaluation data, sparse and hard to scale, and existing benchmarks rely heavily on expensive human examples. So we use scalable and low-cost pipeline by synthesizing high-quality tasks, using information symmetry, we generate simple subtasks and chain them into complex ones, then enabling control of task complexity. Given the diversity of task, we generate the initial task, then iteratively generate follow-up tasks, chaining/composing into final task, difficulty controllable. 

<img src="https://github.com/user-attachments/assets/daf70cfb-4ce7-43cc-b160-6637fa8cfc49" width="50%" height="50%">

- New Agent Paradigm: EvoGitt.
  - State of Agent is a graph, no scalar rewards, everything is partially ordered.

<img src="https://github.com/user-attachments/assets/8053e86f-4aa4-4fa1-a4fe-f01da3e08091" width="50%" height="50%">

<img src="https://github.com/user-attachments/assets/01480390-4403-4d24-800b-9b446c6676b2" width="50%" height="50%">

- SimuRA: Towards General Goal-Oriented Agent via Simulative Reasoning with World Model
  - Human reasoning is not just linear, autoregressive reasoning, but also forward-looking, simulation-based reasoning using an internal world model
  - General architecture for optimal goal-oriented agent (i.e., SimuRA), with each component implemented using LLM. Overcame limitation of LLM autoregressive reasoning by introducing world model for planning via simulation

- Decentralized Cooperative Planning & Multi-Agent
  - https://happyeureka.github.io/damcs  
  - Each agent maintains individual knowledge graph memory, communicate and share states with other agents
  - Benchmark to evaluate cooperations, introduce cooperation dependency and constraints

<img src="https://github.com/user-attachments/assets/2d24ba5e-e508-4ac2-8e54-6e6351973d6a" width="50%" height="50%">

- Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving
  - Existing works primarily focus on offline scenarios and struggle to adapt to online settings with high query volume and constrained token budgets, due to Computational Scalability, Deployment Scalability, Sequential Query Arrival.
  - We leverage Approximate Nearest Neighbor Search (ANNS) to efficiently estimate the performance and cost associated with each query.
  - Online routing: We formulate online LLM routing as a Mixed-Integer Linear Programming (MILP) problem. Through analysis of its dual formulation, we derive a simple yet effective routing rule parameterized by learnable routing weights over LLMs. These weights are learned from a small set of initial queries and then used to efficiently guide the routing of subsequent queries.

- Grounded, Memory-driven Cognition in LLM Agents
  - Hybridizing Memory with Latent Dynamics
    - Combining memory-driven systems with learned latent dynamics to enhance temporal coherence and adaptive performance in LLMs.
    - Selective State-Space Models: Integrating episodic and conceptual memory with dynamic latent representations for real-time learning and planning.
  - Vision for Cognitive LLMs: Transitioning from reactive models to cognitive models that reason, reflect, and adapt in real-time through memory and experience-driven architectures.
  - Key Takeaway: The future of AI systems lies in integrating structured cognition and artificial experience to build scalable architectures that can evolve with time and engage in continuous thought.

<img src="https://github.com/user-attachments/assets/ef0dc6ce-9435-4d0d-b0c0-4815e439f782" width="50%" height="50%">

- Cross-Tool Data Harvesting and Polluting in Multi-tool Empowered LLM Agents

<img src="https://github.com/user-attachments/assets/396c509a-daa9-4073-8f3e-8d412ee6576b" width="50%" height="50%">


<!-- TOC --><a name="211"></a>
## 211. 




