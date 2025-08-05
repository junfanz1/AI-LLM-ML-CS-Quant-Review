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

2025/08/05

## Contents



## 1. 

> Bill, Chief Scientist @ NVIDIA

Inference 2 phases
- Prefill/Prompt phase: compute limited, latency insensitive
- Decode phase (token generation): bandwidth limited, latency sensitive

Tree of Thought

## 2. 

> Ion Stoica, Co-Founder @ Databricks & Anyscale, UC Berkeley

- Multi-Agent System failure Taxonomy (MAST)
  - Multi-agent system fail >50% on hard benchmarks, due to specification issues (improve multi-agent system design), Inter-agent misalignment (protocol + deeper social reasoning abilities from agents), Task verification (unit-testing + multi-level verification)
  - MAST Annotator Web, can analyze failures when you upload traces, understand why.
- LMArena: understand and quantify human preferences, key to reliable interaction with AI systems.
- Specifications is the key to build reliability, can verify system, debug system, decompose system, reuse components, plus automated decision making with no human in the loop.

## 3. Reflective Optimization of Agents with GEPA and DSPy

> Matei Zaharia, Co-Founder, CTO @ Databricks, UC Berkeley

- Teach AI new task: standard way in ML is to use weight updates with gradient descents, but requires huge data. As flops get cheaper, progress in AI capabilities will be limited by sample efficiency.
- Teach model more effectively: RL with verified rewards, instead of only 0/1 rewards, have a LM look at traces of all rollouts and reflect on what worked in them (using all intermediate outputs). Instead of only updating weights with small deltas, update a prompt: a single natural language update can give a large behavior change. → GEPA (Genetic-Pareto, evolutionary algorithm)

<img width="1618" height="928" alt="image" src="https://github.com/user-attachments/assets/66daae84-88c3-47a9-b01f-f4fd3c2c2890" />

<img width="1712" height="912" alt="image" src="https://github.com/user-attachments/assets/7fab5c5a-c385-42fa-9228-fbfd6405f903" />

- What is learned in GEPA? Prompt: Given summary, generate new query. So it can reflect on its past mistakes with hundreds of steps
- DSPy: framework for declarative programming with LLM, just describe workflow with high-level signatures, DSPy will optimize how to turn into prompts or weight-tune models to optimize performance. Could be the future of AI training.

## 4. 

> Sherwin Wu, Head of Engineering @ OpenAI

Ensembling agents together tends to improve performance, so use more inference compute!


> Chi Wang, Senior Staff Research Scientist @ Google DeepMind

Challenges: consensus, shared context, interop

## 5. Automate Knowledge Work

> Jerry Liu, CEO @ LlamaIndex

- Context (supply the right tooling so LLM can access external services and get output) 
  - Main limitation of adoption AI: You as a human is responsible for defining actual input, to tell AI what to do, API completely depend on user. If not good at prompting, then not using effectively. 
  - RAG should be reimagined as an MCP toolbox (retrieval, manipulation, structured querying).
- Workflow engineering Goal: define constrained steps of acts, you as user design prompts. Can encode the process as a DAG, need human in the loop in the end.

## 6. Multi-Turn RL for LLM Agents

> Sergey Levine @ Berkeley

How to train goal-conditioned value functions, use sub-optimal interactive data to get optimal strategy with RL: 
- Bellman equation (next action that is most likely to lead to the goal), so you don’t need optimal training data (states, actions), the data tells you what might happen and what transformation may occur, and then you ask if the best scenario what you’ll get, we can learn for many goals simultaneously. This is Q-learning with 0/1 rewards indicating goal reaching. 
- Offline RL: to train goal-conditioned value functions using our dataset, leverage representations from LLM as features on top of which to train value functions. Use LLM to make predictions that aid in optimal decision-making. In-domain data can help us learn expert level prediction. Offline RL can learn predictors for strategies that are more optimal than the ones in the data. Finally, we can combine all this to train agents using only suboptimal data that could outperform humans who generated data.

## 7. 

> May Habib, CEO @ Writer

Action agent: let agent automatically solve problem, by supervising digital workforce. 
- Contain dynamic (not static) agent actions, Sandbox environment to secure agents
- Understand why (not ask what) happened, for reasoning tracability and guardrail alignment
- Design intelligent delegation paths (not simple permission), orchestration graph control.

## 8. Coding Agents

> Michele Catasta, President @ Replit

- Frontier models with massive RL scaling will keep getting better on coherent, long, tool-using agent trajectories.
- Agent builders will create alpha on context management, quality of environment, stacks integrations search parallelism, etc.
- Gap between research and vibe coding
  - Human in the loop is really hard, as model is too sycophantic for non-tech users
  - Vibe coders want maintainable iterative environment, but issue → PR setup has already established code structure, and popular repos are extensible and SWE-bench rarely adds new components
- We need more evals. Capabilities with major product impact remain difficult to measure: app generation from scratch, frontend functionality, parallel tasks, long-term iterative development + maintenance
- Replit agent: less structure in the agent loop, more in the environment

## 9. Reliable AI Agents = Predictable + Aligned

> Karthik Narasimhan @ Princeton

- Reliability is a measurable system property, to improve reliability: agent interfaces, self-evaluation + correction, memory, fine-tuning
- Future trends: proactive + self-improving agents, multi-agent networks (agents verify each other, adding a layer of trust)

## 10. AI Hackers

> Snehal Antani, CEO @ Horizon3.AI

- Agent 1: Iterative trail-and-error: agents explore step by step, using loops of reflection and retries. Each decision is influenced by the past. Flexible and useful for exploration, but expensive, non-deterministic, and inefficient at scale.
- Agent 2: Structured context for precision prompting: system builds context-facts, states and relationships stored in the graph. System queries graph to generate precise directives as prompts for targeted reasoning via LLM.
- NodeZero
  - Build knowledge graph, decision engine queries knowledge graph, queries generate comprehensive detailed prompts to increase precision
  - High value targeting, advanced data pilfering, explainable narratives, etc.
  - Amazon bedrock: find the best model to use, cheap for scalable business, with architectural flexibility
  - GOAD-HARD is hard, because of: multi-step exploits, conditional execution, discover & abuse trust, reason over file content, no prior knowledge, no human in the loop. Everything is patched, 5VM range, no legacy protocols.

## 11. Startup Pitch

- Alex Shan: Agent PoCs are cool, but: agents break in production (infinite loops, improper tool calls, broken retry logic), broken CI/CD (lack of robust evals prevent regression tests, production monitoring, alerts), no route to optimization (no method for piping agent data into post-training workflows). → Judgement Labs: Open source post-building layer for agents.
- We should trying fail, but don’t fail to try.

## 12. 

LinkedIn

- Hiring Assistant multi-agent
<img width="1746" height="974" alt="image" src="https://github.com/user-attachments/assets/46a74356-80e4-4a98-9a0f-512bc3ea684a" />

- Autonomous agents react to environment signals, with 3 tiers of memory (in session immediate, episodic summaries, long term RAG)

<img width="1638" height="946" alt="image" src="https://github.com/user-attachments/assets/36157836-6ed1-4eb2-8300-76ccd1bc5691" />

- Scoped Agentic Memory w/Task Sharing: Users are isolated due to trust/regulatory concerns, we introduce another type of common knowledge: shared task summaries, so that agent is aware of what it’s doing under other contexts, and integrate information into main task loop instead of elsewhere.

<img width="1738" height="988" alt="image" src="https://github.com/user-attachments/assets/97d991e2-91ba-4be8-9aa4-f0c75ab4f083" />

- Modular Agentic Architecture











