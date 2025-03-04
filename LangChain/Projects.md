# Contents

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [1. LangChain](#1-langchain)
- [2. LangGraph](#2-langgraph)
   * [2.1 LangGraph Researcher Agent](#21-langgraph-researcher-agent)
   * [2.2 RAG Self-Reflection Workflow ](#22-rag-self-reflection-workflow)
- [3. Agent Framework](#3-agent-framework)
   * [3.1 Project](#31-project)
   * [3.2 RAG](#32-rag)
   * [3.3 LoRA (Low Rank Adaptation)](#33-lora-low-rank-adaptation)
   * [3.4 Training](#34-training)
- [4. Prompting Cursor to develop Full-Stack AI Application](#4-prompting-cursor-to-develop-full-stack-ai-application)
   * [4.1 Project Overview](#41-project-overview)
   * [4.2 Cursor Tips](#42-cursor-tips)
   * [4.3 Create SaaS Application by Next.js](#43-create-saas-application-by-nextjs)
   * [4.4 Backend functionality `Github Summarizer` with LangChain](#44-backend-functionality-github-summarizer-with-langchain)
   * [4.5 Authentication Flow for user login](#45-authentication-flow-for-user-login)
   * [4.6 Landing Page UI Web development with Shadcn and V0(Vercel)](#46-landing-page-ui-web-development-with-shadcn-and-v0vercel)
   * [4.7 RESTful API for API keys management](#47-restful-api-for-api-keys-management)
   * [4.8 Deployment](#48-deployment)

<!-- TOC end -->

---

<!-- TOC --><a name="1-langchain"></a>
# 1. LangChain

[Eden Marco: LangChain- Develop LLM powered applications with LangChain](https://www.udemy.com/course/langchain/?srsltid=AfmBOooPg0Xkc19q5W1430Dzq6MHGKWqHtq5a1WY4uUl9sQkrh_b_pej&couponCode=ST4MT240225B)

<img src="https://github.com/user-attachments/assets/545885af-9c0b-431c-b8d4-cc28a0b7d64f" width="50%" height="50%">

Projects
- https://github.com/junfanz1/Code-Interpreter-ReAct-LangChain-Agent
- https://github.com/junfanz1/LLM-Documentation-Chatbot

LangChain = LLM + Retriever (Chroma, vector storer) + Memory (list of dicts, history chats)
- LLM applications = RAG + Agents
- Simplifies creation of applications using LLMs (AI assistants, RAG, summarization), fast time to market
- Wrapper code around LLMs makes it easy to swap models 
- As APIs for LLMs have matured, converged and simplified, need for unifying framework like LangChain has decreased 

ReAct (Reason-Act)
- Paradigm that integrates language models with reasoning and acting capabilities, allowing for dynamic reasoning and interaction with external environments to accomplish complex tasks.
- Simplest agent is for-loop (ReAct), ReAct agents are flexible and any state is possible, but have poor reliability (eg. invoking the same tool always and stuck, due to hallucination, tool-misuse, task ambiguity, LLM non-determinism).

Autonomy in LLM applications (5 levels): 
- human code
- LLM call: one-step only
- Chain: multiple steps, but one-directional
- Router (LangChain): decide output of steps and steps to take (but no cycles), still human-driven (not agent executed)
- State Machine (LangGraph): Agent executed, where agent is a control flow controlled by LLM, use LLM to reason where to go in this flow and tools-calling to execute steps, agent can have cycles.


<!-- TOC --><a name="2-langgraph"></a>
# 2. LangGraph

[Eden Marco: LangGraph-Develop LLM powered AI agents with LangGraph](https://www.udemy.com/course/langgraph)

<img src="https://github.com/user-attachments/assets/0511a3b1-a5d4-4255-8916-fc9cb2d08e99" width="50%" height="50%">

Projects:
- https://github.com/junfanz1/Cognito-LangGraph-RAG
- https://github.com/junfanz1/LangGraph-Reflection-Researcher


LangGraph
- LangGraph is both reliable (like Chain, that architects our state machine) and flexible (like ReAct).
- Flow Engineering (planning+testing), can build highly customized agents. (AutoGPT can do long-term planning, but we want to define the flow.) 
- Controllability (we define control flow, LLM make decisions inside flow) + Persistence + Human-in-the-loop + Streaming. 
- LangChain Agent. Memory (shared state across the graph), tools (nodes can call tools and modify state), planning (edges can route control flow based on LLM decisions).

<!-- TOC --><a name="21-langgraph-researcher-agent"></a>
## 2.1 LangGraph Researcher Agent
https://github.com/assafelovic/gpt-researcher
- Implementing agent production-ready. There’re nodes and edges, but no cycles. We can integrate GPT Researcher (as a node under LangGraph graph) within Multi-Agent Architecture. (https://github.com/assafelovic/gpt-researcher/tree/master/multi_agents)
- Every agent in a multi-agent system can be a researcher, as part of workflow. e.g., `Technology` agent is talor-made for technological subjects, and is dynamically created/chosen
- Research automation needs to make a decision for a few deeper levels and iterate again again again until the optimal answer. Key difference here is not only width (in parallel then aggregation) but also depth

Reason for LangGraph in Multi-Agent Architecture
- LangGraph (Flow Engineering techniques addresses the tradeoff between agent freedom and our control) is more flexible in production than CrewAI (doesn’t have as much control of the flow)
- breaks down the problem into specific actions, like microservices, (1) with specialized tasks, we can control quality of nodes, (2) can scale up the application as it grows
- Customizability, creative framework
- Contextual compression is the best method for retrieving in RAG workflow
- Allow both web and local data indexing, with LangChain easy integration can embed anything
- Human-in-the-loop, let user decide how much feedback autonomy to interact with, especially useful when finding two knowledge sources that conflict or contradict each other. When this happens, AI needs human assistance.


<!-- TOC --><a name="22-rag-self-reflection-workflow"></a>
## 2.2 RAG Self-Reflection Workflow 

LangGraph Components
- Nodes (Python functions)
- Edges (connect nodes)
- Conditional Edges (make dynamic decisions to go to node A or B)

State Management: dictionary to track the graph’s execution result, chat history, etc.

Reflection Agents: prompt to improve quality and success rate of agents/AI systems.

Self-reflects on 
- Document we retrieve
- Curate documents and add new info
- Answers if grounded in documents

We also implement a routing element, routing our request to correct datastore with info of the answer.

RAG Idea Foundations
- Self-RAG: reflect on the answer the model generated, check if answer is grounded in the docs.
- Adaptive RAG: (1) taking the route to search on a website, then continuing downstream on the same logic (2) use RAG from the vector store. Use conditional entry points for routing.
- Corrective RAG: Take query, vector search + semantic search, retrieve all docs, start to self-reflect and critique the docs, determine whether they’re relevant or not. If relevant, send to LLM, if not relevant, filter out and perform external Internet search to get more info, to augment our prompt with real-time online info, then augment the prompt and send to LLM.

Further Improvements

- LangGraph has a persistence layer via checkpoint object (save the state after each node execution, in persistence storage, e.g. SQLite). Can interrupt the graph and checkpoint the state of the graph and stop to get human feedback, and resume graph execution from the stop point.
- Create conditional branches for parallel node execution
- Use Docker to deploy to LangGraph Cloud, or use LangGraph Studio, LangGraph API to build LLM applications without frontend


<!-- TOC --><a name="3-agent-framework"></a>
# 3. Agent Framework

[Ed Donnoer: LLM Engineering: Master AI, Large Language Models & Agents](https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models/learn/lecture/)

<img src="https://github.com/user-attachments/assets/e5bb6fb6-9c70-42e6-9d4a-7603d9646b26" width="50%" height="50%">

Building AI UIs with Gradio from HuggingFace using LLMs behind its scenes, implementing streaming responses

DALL-E-3, image generation model behind GPT-4o

Agent Framework, build multimodal (image, audio) AI assistant

Use HuggingFace pipelines, tokenizers and models, libraries: hub, datasets, transformers, peft (parameter efficient fine tuning), trl, accelerate

Use Frontier models/open source models to convert audio to text

Benchmarks comparing LLMs - HuggingFace Open LLM Leaderboard 

- ELO, evaluating Chats, results from head-to-head face-offs with other LLMs, as with ELO in Chess
- HumanEval, evaluating Python coding, 164 problems writing code based on docstrings
- MultiPL-E, evaluating broader coding, translation of HumanEval to 18 programming languages

Metrics to train LLM
- Cross-entropy loss: -log(predicted probability of the thing that turned out to be actual next token)
- Perplexity: e^{Cross-entropy loss}, if = 1 then model is 100% correct, if = 2 then model is 50% correct, if = 4 then model is 25% correct. Higher perplexity: how many tokens would need to be to predict next token

<!-- TOC --><a name="31-project"></a>
## 3.1 Project

Autonomous Agentic AI framework (watches for deals published online, estimate price of products, send push notifications when it’s opportunity)

modal.com to deploy LLM to production, serverless platform for AI teams 

Agent Architecture/Workflows: 7 Agents work together (GPT-4o model identify deals from RSS feed, frontier-busting fine-tuned model estimate prices, use Frontier model with massive RAG Chroma datastore)
- UI, with Gradio
- Agent Framework: with memory and logging
- Planning Agent: coordinate activities
- Scanner Agent: identify promising deals
- Ensemble Agent: estimate prices using multiple models, and collaborate with other 3 agents
- Messaging Agent: send push notifications
- Frontier Agent: RAG pricer (based on inventory of lots of products, good use case for RAG)
- Specialist Agent: estimate prices 
- Random Forest Agent: estimate prices (transformer architecture)

SentenceTransformer from HuggingFace maps sentences to 384 dimensional dense vector space and is ideal for semantic search. 


<!-- TOC --><a name="32-rag"></a>
## 3.2 RAG

RAG (Retrieval Augmented Generation) uses vector embeddings and vector databases to add contexts to prompts, define LangChain and read/split documents. 

Convert chunks of text into vectors using OpenAI Embeddings, store vectors in Chroma (open source AI vector datastores) or FAISS, and visualize the vector store in 3D, and reduce the dimension of vectors to 2D using t-SNE.
- Autoregressive LLM: predict future token from the past
- Autoencoding LLM: produce output based on full input. Good at sentiment analysis, classification. (BERT, OpenAI Embeddings)

LangChain’s decorative language LCEL

Professional private knowledge base can be vectorized in Chroma, vector datastore, and build conversational AI. Use libraries to connect to email history, Microsoft Office files, Google Drive (can map to Google Colab to vectorize all documents) and Slack texts in Chroma. Use RAG to get the 25 closest documents in the vector database. Use open source model BERT to do vectorization by myself. Use Llama.CPP library to vectorize all documents without the need to go to cloud. 

Use Transfer learning to train LLMs, take pretrained models as base, use additional training data to fine-tune for our task. 

Generate text and code with Frontier models including AI assistants with Tools and with open source models with HuggingFace transformers. Create advanced RAG solutions with LangChain. Make baseline model with traditional ML and making Frontier solution, and fine-tuning Frontier models.

Fine-tuning open source model (smaller than Frontier model)

Llama 3.1 architecture
- 8B parameters, 32G memory, too large and costly to train.
- 32 groups of layers, each group = llama decoder layer

<!-- TOC --><a name="33-lora-low-rank-adaptation"></a>
## 3.3 LoRA (Low Rank Adaptation)

Freeze main model, come up with a bunch of smaller matrices with fewer dimensions, they’ll get trained and be applied using simple formulas to target modules. So we can make a base model that gets better as it learns because of the application of LoRA matrices.
- Freeze weights, we don’t optimize 8B weights (too many gradients), but we pick a few layers (target modules) that we think are key things we want to train. We create new matrices (Low Rank Adaptor) with fewer dimensions, and apply these matrices into target modules. So fewer weights are applied to target modules.
- Quantization (Q in QLoRA): Keep the number of weights but reduce precision of each weight. Model performance is worse, but impact is small. 

3 Hyperparameters for LoRA fine-tuning
- r, rank, how many dimensions in low-rank matrices. Start with 8, 16, 32 until diminishing returns 
- Alpha, scaling factor that multiplies the lower rank matrices. Alpha = 2 * r, the bigger the more effective.
- Target modules, which layers of NN are adapted. Target the attention head layers.

fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL), after quantized to 8 bit or 4 bit, model size reduced to 5000MB, after fine-tuned LoRA matrices applying to big model, size of weights reduced to 100MB.

5 Hyperparameters for QLoRA fine-tuning
- Target modules
- r, how many dimensions
- alpha, scaling factor to multiply up the importance of adaptor when applying to target modules, by default = 2 * r
- Quantization
- Dropout, regularization technique, to prevent overfitting

<!-- TOC --><a name="34-training"></a>
## 3.4 Training
- Epochs, how many times we go through the entire dataset when training. At the end of each epoch, we save the model and the model gets better in each epoch before overfitting then gets worse; we pick the best model and that’s the result of training.
- Batch size, take a bunch of data together rather than step by step, it’s faster and better performance, because for multiple epochs, in each epoch the batch is different. 
- Learning rate, = 0.0001, predict the next token vs. the actual next token should be -> loss, how poorly it predicts the actual, use loss to do back propagation to figure out how to adjust weight to do better, the amount that it shifts the weights is learning rate. During the epochs we can gradually lower the learning rate, to make tiny adjustments as the model gets trained.
- Gradient accumulation. Improve speed of going through training. We can do forward pass and get the gradient, and don’t take a step, just do a second forward pass and add up the gradients and keep accumulating gradients and then take a step and optimize the network. Steps less frequently = faster. 
- Optimizer. Algorithm that updates NN to shift everything a bit to increase the prediction accuracy of the next token.

4 Steps in Training
- Forward pass, predict next token in training data
- Loss calculation, how different was it to the true token
- Backpropagation, how much (sensitivity) should we tweak parameters to do better next time (gradients)
- Optimization, update parameters a tiny step to do better next time

Loss function: cross-entropy loss = -log Prob(next true token), = 0 : 100% confident of right answer, higher number = lower confidence.

Carry out end-to-end process for selecting and training open source models to build proprietary verticalized LLM (deploy multiple models to production, including LLM on Modal, RAG workflow with Frontier model) to solve business problems that can outperform the Frontier model. 

Run inference on a QLoRA fine-tuned model.

<!-- TOC --><a name="4-prompting-cursor-to-develop-full-stack-ai-application"></a>
# 4. Prompting Cursor to develop Full-Stack AI Application

> [Eden Marco: Cursor Course: FullStack development with Cursor AI Copilot](https://www.udemy.com/course/cursor-ai-ide/)

<img src="https://github.com/user-attachments/assets/71c2bd39-a1a1-410c-a541-0615e4608995" width="50%" height="50%">

<!-- TOC --><a name="41-project-overview"></a>
## 4.1 Project Overview

- Build E2E Micro SaaS AI application, takes in Github urls, generate json reports with AI powered insights and repo stats, similar to Gitingest
- Next.js to write full stack app, v0 to generate UI components, shadcn/ui for UI components, Supabase to store data with PostgreSQL, Vercel to deploy code, LangChain.js to write backend code to interact with LLM

![image](https://github.com/user-attachments/assets/f5c8ae6f-3f86-49f0-831f-5404840f6350)

<!-- TOC --><a name="42-cursor-tips"></a>
## 4.2 Cursor Tips
- Chat, Composer
- Inline editing coding (Command + K), will open a prompt bar to fill in snippets, can add follow up instructions (debug…)
- (Control + I): refactor code and break down a few files 
- `.cursorrules` file, can prompt every time the rules (coding style, conventions, etc.) based on LLM sent to the cursor. Go to [Cursor website cursor.directory](https://cursor.directory/), find TypeScript, copy paste, it’s local to our codebase, different contexts for different stack.
- Cursor notepad for prompt engineering: can add context which changes when you’re doing. (See below: RESTful API for API keys management)
- Input Modal with Cursor tag, just like chatting with teammate
![image](https://github.com/user-attachments/assets/6a36b989-4aed-4721-b986-db24ae9b1aff)
- Break down huge prompts into smaller tasks, otherwise debugging will be a huge pain, don’t put one big feature in a single prompt.
- Bolt vs. Windsurf vs. Cursor copilots

<!-- TOC --><a name="43-create-saas-application-by-nextjs"></a>
## 4.3 Create SaaS Application by Next.js

```bash
npx create-next-app@latest
cd junfan (project name)
npm run dev
```

Cursor Prompt:
- `@page.js   Create a button that will redirect to /dashboards that will be the dashboard for managing api keys. it'll have a UI for CRUD API for api keys. Then implement a UI for CRUD for managing user api keys.`
- Give a screenshot to Cursor and say: `i like this design, make the UI like this. make when clicking "create" open a modal like this, make when clicking the eye icon to show the api key. when clicking on the copy icon, it is copied to the clipboard. add popups when I create and delete and edit api keys. add the sidebar like screenshot. change the title on the left sidebar from "Tavily" to "Junfan AI", and use my logo image.`

Connect to Supabase:
- `@page.js connect this CRUD API to a real database which is hosted on Supabase`
- Create a project and then table in Supabase, add columns: `{id: uuid, created_at: timestamp, "name": text, "value": text, "usage": int8}`, in Supabase/Settings/Data API, copy anon key and project url, and add file in `cursor-project/.env.local`, in this file input

```bash
NEXT_PUBLIC_SUPABASE_ANON_KEY=anon key
NEXT_PUBLIC_SUPABASE_URL=project url
```
Now we can create API keys and see their storage on Supabase.

Next, create API playground. Cursor Prompt:
- `@Sidebar.js when clicking on "API playground", go into a new page /playground where it will have a form to submit an API key. When submitting the form, it will go to /protected page and there we will validate that this is a valid api key. if it is  we will notify with @Notification.js a green popup of "valid api key, /protected can be assessed"; if api key is not valid we will popup a red popup saying "invalid api key". if it's not a valid api key, pop up red window "invalid api key", if it's a valid api key which is matching the supabase record, pop up green window "valid api key"`

![image](https://github.com/user-attachments/assets/47bfe6c5-cd9c-4a6f-bd05-1f0483a94c14)

Now we have a SaaS application that can be used via API keys and can be validated. 

Next, build the app in cloud which is scalable, deploy on Vercel.
- import the project from Github, and enter the Framework and Environment variables.

<img src="https://github.com/user-attachments/assets/151cc71d-a350-4a58-b07d-b16ccb2d71db" width="50%" height="50%">

- https://github.com/junfanz1/CursorApptest
- https://cursor-apptest.vercel.app/ 

<!-- TOC --><a name="44-backend-functionality-github-summarizer-with-langchain"></a>
## 4.4 Backend functionality `Github Summarizer` with LangChain

- Use Postman for backend SaaS service, to validate API key, `{input: url, process raw message: api_key = xxx, output: “valid api key”.}`.
- Cursor prompt: `generate LangChain chain from @LangChain-JS that will generate the prompt of “summarize this github repository from readme file content”, then inject readme content to the prompt. The chain will invoke an LLM, we want to get structured output to an object with field “summary”: str and “cool facts”: List[str]`
- Coding style: Python to Pydantic is what Java to Zod, using these tools can output more strictly with a schema, we can use function schema, function calling, output parsing to make our code robust. Cursor prompt: `use withStructuredOutput from @LangChain-JS and bind it to the model`, to make LangChain model return structured data.

<!-- TOC --><a name="45-authentication-flow-for-user-login"></a>
## 4.5 Authentication Flow for user login

- Download NextAuth.js, Cursor prompt: `implement entire Google SSO flow, show me step by step what I need to configure. Add a login button @page.js`
- Go to Google Cloud, right up Console, Build Project, left sidebar APIs & Services/OAuth consent screen, fill in and then left side Credentials/Create OAuth client ID, get Google_Client_ID, GOOGLE_CLIENT_SECRET, then put in `.env.local`, configure all variables
- Add profile pic, and Google SSO for user to sign in and sign out
- Supabase authentication. Create a matching supabase migration script to add the table ‘users’

<!-- TOC --><a name="46-landing-page-ui-web-development-with-shadcn-and-v0vercel"></a>
## 4.6 Landing Page UI Web development with Shadcn and V0(Vercel)

- V0 website prompt: `a landing page for junfan github analyzer. It’s a SasS application with a free tier which gives you an api and you will output with AI github open source repositories summaries, analyses, starts, important pull requests, etc. The landing page should have a sign up / login button and a pricing with free tier.` It’ll generate the web preview and tsx code. Then prompt: `show me how to install it and use it step by step.` It’ll guide you to install all packages and configs. On the right side, it shows Preview webpage and .tsx code.
- Shadcn UI, has many reusable components to add to our API demo component and intergrate into our landing page, beautifully designed and can do copy paste into our apps. Go to Blocks, click on Lift Mode, click on the black button below to open it, it opens the V0 chat window, we can customize our needs. 

![image](https://github.com/user-attachments/assets/02289376-6d0d-4b0e-8fb6-c17efcf91a69)

<!-- TOC --><a name="47-restful-api-for-api-keys-management"></a>
## 4.7 RESTful API for API keys management

- Create Cursor notepad “CRUD”, prompt: `implement a CRUD API for managing API keys, the original code is in @apiKeyOperations.js`. If we tag this notepad, notepad will be attached to every request Cursor is going to make with LLM with our prompt. We can write a lot of high-level descriptions, product requirements, for the task. (e.g. We can copy the Jira ticket screenshot to give context to LLM).
![image](https://github.com/user-attachments/assets/239e3d1b-15cb-4a85-9af0-9c75eb17e022)

- CRUD REST API, integrating with UI. `update @page.js to use CRUD rest endpoints in @api-keys, get the current logged in user JWT in the client and send it to the server to perform all CRUD operations. Remove the usage from @apiKeyOperation.js and delete this file.`
- Adding retaliating and quota enforcement. `implement rate limit protection to /github-summarizer api in route.js. Each time a user invokes an api key, we want to increment the api key usage column of the corresponding api key in supabase. Check if the user is lower than the limit of api key, if larger, return 429 response and say there’s a rate limit. Make code usable and easy to maintain by splitting CheckApiKeyUsage() and UpdateApiKeyUsage() functions.`
 
<!-- TOC --><a name="48-deployment"></a>
## 4.8 Deployment

- Configure a custom production domain, buy from GoDaddy, setup domain with Vercel in Project Settings/Domains, add domain www.junfan.cloud, add A Record (copy this configure IP, go to GoDaddy Domain/Manage DNS to add new DNS records, then go back to refresh) and CNAME, then the web is online. Go to Google Cloud console/APIs Services/Credentials to add in URIs the new domain to allow list in authentication client.
- Patching Vulnerability handling. Because the application is vulnerable to cache poisoning, `sudo yarn audit` will find 1 vulnerability, prompt: `how to upgrade next version with yarn, need to upgrade what packages`. Then `sudo yarn audit –fix` should fix it.
