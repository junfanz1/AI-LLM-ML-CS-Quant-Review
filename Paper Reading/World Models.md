# World Model: 5 Debates Between Eric Xing's PAN & Yann LeCun’s JEPA  

Are autoregressive LLMs truly doomed, as Yann LeCun suggests — or are they just one layer in a future hierarchical reasoning stack?

Should representation learning stay continuous (JEPA) or embrace discrete tokens as a human-evolved latent space?

Is model-predictive control the right direction, or will RL with policy reuse win out in real-world efficiency?

Inspired by (Eric) Zhang Chen 's recent sharing in the Journal Club at AI community https://www.ez-encoder.com/ - our discussion sparked deep dive into simulative reasoning, JEPA vs. PAN, and exploring how world models could take us beyond autoregressive LLMs. Would love to hear about your thoughts — architecture, objective, or paradigm.

## Definition  
LLMs predict the **next word**, while **world models** predict the **next world**.  

A world model is a generative model that simulates possible futures:  

> s’ ~ p(s’ | s, a)

It runs **thought experiments** by simulating actions, then selects the optimal one.  
This ability allows knowledge transfer to novel tasks.

---

## Simulative Reasoning & Thought Experiments  
**Joint Embedding Predictive Models (JEPA, by Yann LeCun)** represent one direction of research.  
While still mostly limited to toy experiments, JEPA explores how models can generalize through **next-representation prediction** rather than raw data prediction.  

Outside of JEPA, most “world models” focus primarily on **video generation** as a way to simulate reality — essentially using it as a sandbox for thought experiments.

---

## Yann LeCun’s JEPA Framework  
Key insights from Yann LeCun’s work:  


- Sensory inputs are superior to texts
- World state should be represented by continuous embeddings instead of discrete tokens
- Autoregressive, generative models (LLMs) are doomed, as they have mistakes and uncertainties.
- Probabilistic, data-reconstruction objectives (e.g. encoder-decoder scheme) don’t work, as they force the model to predict irrelevant details.
- World models should be used in MPC (model-predictive control), rather than RL which needs lots of trials. Next representation prediction (rather than next data prediction like LLM): Encoder-encoder architecture, where prediction (reconstruction loss) is done in the latent space.Next representation prediction (rather than next data prediction like LLM): Encoder-encoder architecture, where prediction (reconstruction loss) is done in the latent space.

<img width="1972" height="930" alt="image" src="https://github.com/user-attachments/assets/3651de9c-582f-40a2-935b-83d424188bce" />

Next representation prediction (rather than next data prediction like LLM): Encoder-encoder architecture, where prediction (reconstruction loss) is done in the latent space.

---

## Eric Xing’s Debate & Critique  
Eric Xing provides a fascinating counterpoint across **data, representation, architecture, objective, and usage**:  

### 1. Data  
- **Information density > volume**  
  Some information (memory, social, mental states) can only be expressed in text — not video.  
- Overemphasizing sensory modalities may introduce bias.  

Information density, not just volume. E.g. memory, social, mental information can only be expressed in text, not video. We shouldn’t overemphasize certain modality over others in a biased way. 


### 2. Representation  
- **Continuous vs. discrete?**  
  Discrete tokens have an advantage: language itself is a *latent space*, refined by evolution.  
- The challenge: can discrete tokens capture the richness of high-dimensional sensory data?  
- **Conclusion:** we need both — scale modality tokenizers *up* and language expression *out*.  

> *JF.AI’s note: This resonates with Wittgenstein’s idea that “the limits of my language mean the limits of my world.”*

Continuous, discrete, or both? Discrete tokens can be an advantage, because space of language is already a latent space (through human evolution), human cognition maps these concepts to discrete worlds. How can we ensure such discrete representations capture richness in high-dimensional continuous sensory data? As long as we can scale up (learn a large modality tokenizer) and scale out (find longer language expression, increase sequence length). Conclusion: we need both. [JF.AI’s note: Some concepts in human language have already been abstracted or folded to certain level. This is a profound thought that human language is exactly a latent space, relevant to Ludwig Wittgenstein’s philosophy.]

<img width="2166" height="866" alt="image" src="https://github.com/user-attachments/assets/eaee6370-3f82-49f6-ae72-a8d5054bbe35" />


### 3. Architecture  
- **Autoregression is not the enemy**  
  Yann argues autoregressive models absorb noise and accumulate errors.  
  Eric argues JEPA itself is autoregressive (via Dirac delta factorization), just not probabilistic.  

- **Encoder-decoder debate**  
  - Yann claims decoders mislead models into learning spurious correlations (collapse to zero vectors).  
  - Eric argues we still need decoder-based loss, and hierarchical abstractions are crucial:  
    - **High-level:** LLM reasoning in thought space  
    - **Mid-level:** next-token prediction (Transformer decoders)  
    - **Low-level:** next-embedding prediction (latent diffusion)

Autoregressive generation is not the enemy. Yann Lecun thinks there’s compounding mistakes for autoregressive generative models therefore absorbs signal variability, but Eric thinks that JEPA is fundamentally autoregressive and generative, by Dirac delta function refactorization, so in essence it’s generative in the functional sense (though not in probabilistic sense). [JF.AI’s Note: Unique perspective that I've never thought of. Dirac delta function is a very profound concept across Quantum Physics, PDE and Functional Analysis. Quote: "in the functional sense", I think it "almost surely" refers to "Functional Analysis" in mathematics.] Why is encoder-decoder architecture not good? Yann LeCun thinks it misleads the model to learn unstable or spurious correlations, as it learns unpredictable or irrelevant stuffs (e.g. model collapse, everything maps to 0 vector). But Eric thinks it’s unclear if removing the decoder is effective, plus latent space can’t evaluate model learning effectively. Eric thinks we should adopt hierarchical abstraction for world knowledge, such as high-level (LLM in thought space), intermediate level (next-token predictor, e.g. autoregressive Transformer decoder), low-level (next embedding predictions, e.g. latent diffusion model). 

<img width="2226" height="882" alt="image" src="https://github.com/user-attachments/assets/da17260f-2c3e-4300-9a3d-b1380d7734b0" />


### 4. Objective  
- Loss purely in latent space risks **collapse** (everything maps to zero but still reconstructs).  
- Latent space ≠ real world → misalignment risk.  
- **Conclusion:** compute loss in **observation space** for grounding.

Learn in data space or latent space? Eric thinks that JEPA loss calculation in latent space can cause collapse and cheat configuration (map every input to 0 vector, and 0 vector can rebuild original signal). Also, latent space is not equivalent to real world and can be misaligned. Conclusion: We still need encoder-decoder, and calculate loss in observation space.

<img width="2270" height="814" alt="image" src="https://github.com/user-attachments/assets/9dfbc727-0446-4266-8cc7-59ad5ee98351" />


### 5. Usage  
- **MPC vs RL**  
  - MPC must simulate every possible trajectory.  
  - RL can reuse policy networks, requiring fewer simulations.  
  - Eric argues RL is more efficient.

MPC or RL? MPC needs to simulate every possible action, but RL can reuse policy model to generate the best action, no need to simulate all trajectories. So RL is more effective.

---

## Eric’s PAN Model Proposal  
Eric proposes a **PAN model** that fuses **sensory** and **textual worlds**:  

- **Backbone:** enhanced LLM + diffusion-based embedding predictor  
- **Hierarchical latent representations** combined with **agentic reasoning**

> *JF.AI’s note: This is both technically elegant and philosophically deep.*

<img width="2166" height="784" alt="image" src="https://github.com/user-attachments/assets/6a4cbdfd-842d-4ac4-b111-a2724ecd8808" />

<img width="2028" height="1030" alt="image" src="https://github.com/user-attachments/assets/efed1287-c95c-48b1-8ca6-a9c07e6c31b7" />


---

## References  
- *Critiques of World Models*, Eric Xing et al. [[arXiv:2507.05169]](https://arxiv.org/pdf/2507.05169)

- *World Model: 5 Debates Between Eric Xing's PAN & Yann LeCun’s JEPA*, JF.AI. [LinkedIn Post](https://www.linkedin.com/pulse/world-model-5-debates-between-eric-xings-pan-yann-lecuns-jepa-jf-ai-8xigc/)
