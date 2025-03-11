# Build AI Agents and Multi-Agent Systems with CrewAI

## Contents

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [1. CrewAI](#1-crewai)
   * [1.1 CrewAI Whole Cycle](#11-crewai-whole-cycle)
   * [1.2 Multi-Agent System with CrewAI](#12-multi-agent-system-with-crewai)
- [2. Building and Orchestrating Multi-Agent Systems](#2-building-and-orchestrating-multi-agent-systems)
   * [2.1 Caching in Tools ](#21-caching-in-tools)
   * [2.2 Exception handling in CrewAI](#22-exception-handling-in-crewai)
   * [2.3 Building Hierarchical Agent Structures for Scalability](#23-building-hierarchical-agent-structures-for-scalability)
   * [2.4 Kicking Off Crews with Advanced Functionalities](#24-kicking-off-crews-with-advanced-functionalities)
      + [2.4.1 How to have multiple inputs for the Crew's kickoff](#241-how-to-have-multiple-inputs-for-the-crews-kickoff)
      + [2.4.2 How to kickoff the crew asynchronously](#242-how-to-kickoff-the-crew-asynchronously)
      + [2.4.3 How to replay execution from a specific task](#243-how-to-replay-execution-from-a-specific-task)
   * [2.5 Implementing Conditional Tasks for Smarter Multi-Agent Systems](#25-implementing-conditional-tasks-for-smarter-multi-agent-systems)
   * [2.6 Ensuring Agent Simplicity and Robustness](#26-ensuring-agent-simplicity-and-robustness)
      + [2.6.1 Test Crews](#261-test-crews)
      + [2.6.2 AgentOps](#262-agentops)
- [Acknowledgements](#acknowledgements)

<!-- TOC end -->


<!-- TOC --><a name="1-crewai"></a>
# 1. CrewAI

<!-- TOC --><a name="11-crewai-whole-cycle"></a>
## 1.1 CrewAI Whole Cycle

- Frameworks like CrewAI provide a higher level of abstraction, making it easier to define and manage AI agents. Instead of handling all the low-level details, you can focus on your agents’ high-level logic and coordination.
- They promote a modular approach, allowing you to create reusable components. Agents, tasks, and tools can be defined separately and reused across different projects, enhancing productivity and maintainability.
- They have built-in tools and seamless integrations with popular libraries like LangChain and LlamaIndex. This saves time and effort by providing ready-to-use functionalities for tasks like information retrieval, data processing, and more.
- Managing multiple agents and their interactions can become complex. Agent frameworks manage agent communication and coordination, allowing you to scale your system without dealing with complex details.

<!-- TOC --><a name="12-multi-agent-system-with-crewai"></a>
## 1.2 Multi-Agent System with CrewAI

- Agents: Individual entities with specific roles and goals that perform tasks.
```python
from crewai import Agent

# Define agents
venue_finder = Agent(
  role='Conference Venue Finder',
  goal='Find the best venue for the upcoming conference',
  backstory="You are an experienced event planner with a knack for finding the perfect venues.", 
            "Your expertise ensures that all conference requirements are met efficiently.",
  verbose = True,
  tools=[search_tool]
  )

```
- Tools: Resources that agents use to complete their tasks, like APIs or databases.
```python
from crewai_tools import SerperDevTool

# Create a search tool
search_tool = SerperDevTool()
```

- Tasks: Specific objectives or assignments that agents work on to achieve a goal.
```python
from crewai import Task

# Define tasks
find_venue_task = Task(
    description=(
        "Conduct a thorough search to find the best venue for the upcoming "
        "conference. Consider factors such as capacity, location, amenities, "
        "and pricing. Use online resources and databases to gather comprehensive "
        "information."
    ),
    expected_output=(
        "A list of 5 potential venues with detailed information on capacity, "
        "location, amenities, pricing, and availability."
    ),
    agent=venue_finder
)
```

- Crews: Groups of agents working together, coordinated to complete a series of tasks. There are different ways to manage these processes:
  - Sequential process: Agents complete their tasks one after the other, in a specified order. This is like following a checklist where each item must be completed before moving on to the next.
  - Hierarchical process: Agents may only perform tasks based on certain conditions or triggers. This is similar to how a team might decide on their next step based on the outcome of a previous task.

```python
event_planning_crew = Crew(
  agents=[venue_finder, venue_quality_assurance_agent],
  tasks=[find_venue_task, quality_assurance_review_task],
  verbose=True,
  memory=True
)
```

<!-- TOC --><a name="2-building-and-orchestrating-multi-agent-systems"></a>
# 2. Building and Orchestrating Multi-Agent Systems

<!-- TOC --><a name="21-caching-in-tools"></a>
## 2.1 Caching in Tools 

```python
from crewai_tools import tool

@tool("Name of my tool")
def my_tool(question: str) -> str:
    """Clear description for what this tool is useful for, your agent will need this information to use it."""
    # Function logic here
    return "Result from your custom tool"
```
The `@tool` decorator is like a shortcut to building your tool. You simply define a function, add the `@tool` decorator on top, and just like that, your function is ready to be used as a tool by your agent. This is especially handy when you want to create a quick and simple tool without the need to dive into subclassing. The function logic does the heavy lifting, and the decorator wraps it up nicely, making it available to your agents.

<!-- TOC --><a name="22-exception-handling-in-crewai"></a>
## 2.2 Exception handling in CrewAI
Now, let's get to the heart of why CrewAI is such an efficient framework: its approach to handling exceptions. Picture a ship sailing through a storm. The waves are high, the wind is howling, and suddenly, a rogue wave hits. Most ships might falter, but not this one. It’s designed to ride the waves, adjust its sails, and keep moving forward. That’s CrewAI.

In traditional frameworks, when an agent encounters an error, everything can come to a screeching halt. It’s as if the ship has hit a reef and is stuck there, unable to proceed. CrewAI, however, is built differently. It anticipates that the seas won’t always be calm and that agents might encounter rough patches along the way. So, it’s designed to handle these situations with "grace." This is a fancy term that means the system continues to function, even if not everything is working perfectly. For example, if one tool fails, the agent might use a less optimal but still functional alternative. The system doesn’t just throw its hands up and say, “I can’t do this!” It finds another way to get the job done.

Imagine you’ve set up a researcher agent in CrewAI to gather information from the web. Suddenly, the internet connection drops. In a less resilient system, this would be a major problem, and the whole operation might fail. But in CrewAI, the agent doesn’t panic. It recognizes the issue, retries the connection, or shifts its focus to another task while waiting for the internet to come back. The end result? The project keeps moving forward without a hitch.

In the notebook below, we’ll explore a scenario that tests the resilience of CrewAI. We’ve set up two agents—a Senior Researcher and a Senior Writer—each with specific roles and goals. Both are highly capable but share a limitation: they can't determine a topic on their own and must rely on external input or communication with each other. What happens when neither agent is given a clear topic? Instead of halting or throwing an error, the agents enter a loop, continually asking each other for the topic. This demonstrates how CrewAI maintains operation flow even when ideal conditions aren’t met.
<!-- TOC --><a name="23-building-hierarchical-agent-structures-for-scalability"></a>
## 2.3 Building Hierarchical Agent Structures for Scalability

In a hierarchy, some agents take on the role of supervisors or coordinators. Their job is to ensure that everything fits together nicely, just like a research project manager who keeps the whole team aligned and on target.

- The `manager_llm` in the `Crew` oversees the entire project, ensuring that each task is completed on time and meets the project’s objectives.
- The `research_analyst_agent` collects relevant data based on the project manager’s instructions, and takes the gathered data and analyzes it to extract useful insights.
- The `report_writer_agent` compiles the analyzed data into a clear, concise, and comprehensive report.
- The `report_editor_agent` reviews and refines the report for clarity and accuracy.

In a hierarchical system, the `manager_llm` not only assigns tasks but also monitors their progress, ensuring that each stage is completed to a high standard before moving on to the next. Once the project is underway, the `manager_llm` keeps an eye on everything. If the `research_analyst_agent` brings back data that isn’t quite right, it can send it back out to refine its search. The same goes for the `report_writer_agent` — if the report is too shallow, the manager can ask for a deeper dive. This dynamic adjustment ensures that each stage of the project meets the necessary standards before moving on to the next. It’s like a feedback loop.

<!-- TOC --><a name="24-kicking-off-crews-with-advanced-functionalities"></a>
## 2.4 Kicking Off Crews with Advanced Functionalities

<!-- TOC --><a name="241-how-to-have-multiple-inputs-for-the-crews-kickoff"></a>
### 2.4.1 How to have multiple inputs for the Crew's kickoff


Imagine you’ve got several datasets and want to analyze the ages of participants in each one as we did in the previous lesson. Instead of running the analysis manually for each dataset, CrewAI automates the process. To accomplish this, CrewAI provides the `kickoff_for_each()` method. This method is designed to efficiently execute the same crew for each item in a list.

<!-- TOC --><a name="242-how-to-kickoff-the-crew-asynchronously"></a>
### 2.4.2 How to kickoff the crew asynchronously

CrewAI provides the `kickoff_async()` method. This method initiates the crew execution in a separate thread, which allows the main thread—essentially the main flow of your program—to continue running other tasks. This non-blocking approach is particularly useful when you want to manage multiple tasks concurrently or when you don't want to wait for one crew to finish before moving on to the next. 

<!-- TOC --><a name="243-how-to-replay-execution-from-a-specific-task"></a>
### 2.4.3 How to replay execution from a specific task
Now, what if something goes wrong, or you need to rerun just part of the mission? CrewAI allows you to replay from a specific task within a kickoff, which is particularly handy if you want to correct a mistake or simply refine part of the process without starting everything from scratch. Imagine this like rewinding a bit in a movie to re-watch an important scene. You’re not starting the whole movie over—just replaying the part that needs another look. Here’s a simple way to do it. We can run the `crewai log-tasks-outputs` command to see the tasks that were executed in the most recent kickoff, and then use the task you want to replay with the `crewai replay -t <task_id>` command.

This feature allows you to focus on specific parts of your workflow, refining them without needing to rerun everything from scratch. It’s particularly handy in debugging scenarios or when experimenting with different inputs. By understanding how to replay tasks, you gain more control over your crew’s execution, ensuring that you can make adjustments efficiently and effectively.
<!-- TOC --><a name="25-implementing-conditional-tasks-for-smarter-multi-agent-systems"></a>
## 2.5 Implementing Conditional Tasks for Smarter Multi-Agent Systems

The `should_fetch_more_data` function checks if we’ve got enough events. If we don’t, it returns `False` and the `data_analyzer` steps to fetch more or process what’s there to ensure we reach our target. This function is critical because it determines whether the conditional task should be executed. In our system, the `data_collector` might not always gather enough events on the first try. This function checks the output from that task and decides if more work is needed. From a technical perspective, this is a safeguard. It ensures that the system doesn’t proceed with insufficient data. If the condition returns `True`, the system knows to take additional steps (like fetching more data) before moving on to the next task.

The `planning=True` gives our crew the ability to think ahead—a nice little feature when you’ve got multiple tasks to juggle. What this does is that at the very start of the execution, it will assess the agents and tasks, and then create a plan to ensure that we have our final expected output.

By designing the system in this way, we ensure that no single agent becomes a bottleneck, and the system as a whole can adapt to varying conditions and requirements effectively.

<!-- TOC --><a name="26-ensuring-agent-simplicity-and-robustness"></a>
## 2.6 Ensuring Agent Simplicity and Robustness


<!-- TOC --><a name="261-test-crews"></a>
### 2.6.1 Test Crews

With the command `crewai test`, we can run tests on our entire crew. Think of it like running scrimmage matches with your team to see how they work together. By default, the test runs two iterations (just like giving your team two practice rounds), but we can adjust that if we want to run more drills. But what if you want to see how they perform under more pressure? Easy—you increase the number of iterations.

During each iteration, CrewAI evaluates the agents' behavior based on how well they complete their tasks. Some common metrics behind the scenes include:

- Accuracy: This measures how closely the agent's output aligns with the task’s objective. Even when there's no fixed result, CrewAI can measure accuracy by comparing the agent's output to a predefined goal or checking how well the response meets the qualitative or quantitative criteria of the task. For instance, if the task is to generate a creative solution or perform a complex analysis, accuracy may involve assessing how relevant, informative, or aligned the output is to the problem at hand.
- Efficiency: This tracks how quickly the agent completes the task and how many resources (like API calls or computational power) are consumed. Agents that are fast and resource-efficient will score higher.
- Consistency: This looks at how reliably the agent performs across multiple iterations. Does the agent produce consistent outputs, or does its performance fluctuate from iteration to iteration?


CrewAI also tracks how agents handle errors, such as failing to retrieve data from an API or generating incorrect results. If an agent encounters issues but successfully resolves them, it might still score well. But the score will drop if it gets stuck or produces poor results repeatedly. After each iteration, CrewAI generates a score for each task. The score typically ranges from 1 to 10, with 10 being the best. The score is based on the average of how the task/crew scored on the metrics defined above. After the test is completed, CrewAI calculates the average score for each task across all iterations and provides an overall score for the entire crew, which helps you see how well the agents worked together and whether any improvements are needed.


<!-- TOC --><a name="262-agentops"></a>
### 2.6.2 AgentOps


- LLM cost tracking: Imagine your team has an energy drink budget, and every move they make drains some of that energy. In AgentOps, this budget is tracked by how much you're spending on large language models. We don’t want to burn out our team too quickly, right? AgentOps helps you track those costs in real time.
- Replay analytics: Now, imagine you can watch instant replays of the game to see exactly where a player made a mistake. Did they pass the ball too early? Did they miss an opportunity? With replay analytics, we can review each decision our agents made during the game and spot where things went wrong or right.





<!-- TOC --><a name="acknowledgements"></a>
# Acknowledgements

- [Educative: Build AI Agents and Multi-Agent Systems with CrewAI](https://www.educative.io/verify-certificate/k5m3gACoj1xDYoOq7c0Kjk4y2AoGTn)

<img src="https://github.com/user-attachments/assets/1ee7df7f-a39e-4377-9a3e-ec9db9d90263" width="50%" height="50%">
