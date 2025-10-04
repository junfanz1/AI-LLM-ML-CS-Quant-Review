# 🚀 Frontier of Agentic AI: Agent Memory — Key Takeaways
Attended Google’s fascinating meetup on hashtag#Agent hashtag#Memory, frontiers in building truly Agentic AI systems. Big thank you to Ran Li, Warren(Weilun) Chen and our guest speakers for the insightful event!

# [LinkedIn Post](https://www.linkedin.com/posts/junfan-zhu_agent-memory-activity-7380113422757277696-oaAc?utm_source=share&utm_medium=member_desktop&rcm=ACoAABxP-p0BpUNGDf347aKh_1uJAPzG4er0As8)

## 1️⃣ Lenjoy Lin, CoFounder @ Genspark — Product-Driven Memory Architecture
- Challenges: context window & token budget remain bottlenecks.
- Memory extends reasoning beyond sessions.
- Session-level vs. Application-level: Session = short-term continuity; Application = RAG for longer-term relevance.
- Over-personalization isn’t ideal — generalization matters more for consumer AI.
- General vs. vertical: “Different products for different scenarios — the world will be more diverse.” We won’t have one general agent; specialized, application-layer agents will dominate.
- Vision: democratize professional tools, lower R&D barriers, empower creativity.

## 2️⃣ Deshraj Yadav, CTO & CoFounder @ Mem0 — Model-Agnostic, Self-Improving Memory
- Retrieval≠memory. Agent memory should be independent, model-agnostic, self-improving.
- Architecture:
  - Extraction phase: parse info from conversations or actions.
  - Update phase: decide what and where to store — into a knowledge graph or KV/vector DB.
- Knowledge graphs are optional, used only when structure matters.
- Vision: “Memory Passport” as universal as email — standard identity layer for all agents.
- Use cases: Patient Context Continuity, Adaptive Learning, Cross-Session Shopping, Contextual Issue Resolution, Relationship Memory.

## 3️⃣ Research Scientist @ Google DeepMind — Memory as Reasoning Skill
- Agent memory is not storage — but reasoning skill. Memory acts as cognitive scaffolding, not infra. Human memory generalizes; agents reuse procedural knowledge.
- Optimization paths: 
  - Context editing: carry fewer tokens next time.
  - Long-term memory: rebuild abstractions from past episodes.
- Not just retrieval — it’s distillation, indexing, reuse.
- Echoes Claude 4.5’s tool-aware models: can model understand and invoke memory tools correctly?
- Hallucination = not knowing when to use memory.
- Parametric vs. Non-parametric memory: weights vs. external stores.
- Key: what kind of data to store for reasoning. If a problem can be distilled, it can be retrieved and solved.
- Future: self-verifying, recursive agents that can critique or even submit PRs autonomously.
- AGI may emerge from 1000-step reasoning chains connecting abstract dots.

## 4️⃣ Roundtable
- Memory = infra + cognition.
- Avoid over engineering; stay flexible as models evolve.
- Trend: RAG → memory-aware models → reasoning-integrated agents.
- Frontier: teach models when & how to use memory tools.

## 💡 Thoughts
Agent Memory shifts from retrieval layer into reasoning substrate — bridging infrastructure, cognition, and autonomy. Can we teach models to learn their own “update policy” — deciding what to retain, compress, or discard, independent of the base LLM? Curious to hear your thoughts: where cognition meets infrastructure in agent design.
