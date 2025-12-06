## PRIME-RL Implementation Status (Dec 6, 2025)

All remaining gaps identified in `docs/gap_analysis_rl_take1.md` have been **implemented and closed** in PRIME-RL.

### ✅ 1. Application-Level Sandboxes

- **CRM Support Sandbox** (`src/prime_rl/sandboxes/crm/`)
  - Environment adapter with tickets, customers, and agents.
  - Rubric-based verifier with weighted scoring for correctness, tone, and policy adherence.
  - Configuration and documentation (`docs/enterprise_sandbox_recipe.md`, example configs).

- **Finance Reconciliation Sandbox** (`src/prime_rl/sandboxes/finance/`)
  - Environment adapter with transactions, ledgers, and reconciliation tasks.
  - Rubric-based verifier focused on constraint satisfaction and reconciliation accuracy.
  - Configuration and documentation (included in sandbox and enterprise sandbox docs).

### ✅ 2. Replication Training

- **PytestVerifierClient** (`src/prime_rl/integrations/replication/`)
  - Runs `pytest` (or configurable commands) in isolated environments.
  - Converts test results (pass/fail/coverage) into scalar rewards for RL.

- **Benchmark Suite** (`benchmarks/replication/project_a/`)
  - Example Python project with incomplete implementation.
  - Test suite and task definition for Mechanize-style replication.
  - Configuration and documentation (`docs/replication_training.md`, example configs).

### ✅ 3. Scenario Registry & Eval-Only CLI

- **Scenario Registry** (`src/prime_rl/registry/`)
  - Central catalog of scenarios with metadata (id, name, category, tags, config path).
  - Filtering by category, tag, or search.
  - Built-in scenarios (CRM, Finance, Replication, and others) registered.

- **CLI Commands** (`src/prime_rl/cli.py`)
  - `prime-rl list-scenarios`: List and filter registered scenarios.
  - `prime-rl eval`: Run evaluation on a trained model without further training.
  - Integrated with existing `prime-rl` CLI entrypoint.

### ✅ Deliverables & Quality

- **Deliverables**:
  - 2 enterprise sandboxes (CRM, Finance).
  - 2 rubric verifiers.
  - 1 replication benchmark.
  - 1 `PytestVerifierClient`.
  - Scenario registry module.
  - Eval-only CLI command.
  - Supporting configuration files and documentation.

- **Quality**:
  - Code follows PRIME-RL patterns and is fully type-hinted.
  - Integrated with existing training and evaluation flows.
  - Documented for external contributors and RLaaS-style deployments.

https://zhuanlan.zhihu.com/p/1954178498890560347

RL Scaling is propelling AI from the "human data era" to the "Agent experience era," bringing a new paradigm of agents truly capable of handling complex, long-chain tasks. This shift from static data to dynamic interaction in learning paradigms necessitates a completely new infrastructure, giving rise to a new wave of startups. The core value of RL Infra is bridging the "sim-to-real" gap between simulation training and the real world, allowing AI agents to undergo superhuman-level "stress testing" and "deliberate practice" before deployment, enabling them to move from laboratory demos to commercially viable applications.



We've outlined the RL Infra industry landscape into three main modules: RL Environments, RLaaS, and Data/Evaluation. At one end of this landscape is an environment platform dedicated to "high-fidelity" real-world workflows, and at the other end are RLaaS solutions deeply optimized for specific enterprise workflows. Data and evaluation serve as a crucial bridge connecting them. These two main paths also represent different business ambitions: the "horizontally platformized" RL Environment aims to become the " Unreal Engine " of the AI ​​era's software world; while the "vertically integrated" RLaaS aspires to become the winner-takes-all " AI-native Palantir " within specific industries .



As new trends evolve, we will witness the GPT-3 moment for RL, truly scaling up RL data to the pre-training level. From an investment perspective, the RL environment and data are two sides of the same coin, forming a hedge against the Era of Experience theme; while RLaaS has the potential to incubate new vertically dominant leaders in specific industries.



01.



Why is RL Infra needed?



1. Era of Experience: Creating dynamic environments, saying goodbye to static data.



Foundation Model is facing a bottleneck: the performance gains from relying solely on static, human-generated internet datasets are showing diminishing marginal returns. As two RL pioneers, Richard Sutton and David Silver, have stated, we are moving from the "era of human data" to the "era of agent experience."



In this new era, AI agents are no longer merely repeaters of knowledge, but rather agents that autonomously learn and grow through continuous interaction with their environment. Therefore, the industry is turning its attention to interaction with RL environments. By experimenting in simulated environments, models can learn capabilities that are difficult to acquire through pre-training + SFT, such as long-chain reasoning and complex decision-making, thereby improving reliability and generalization.



Under this trend, we believe that RL has the opportunity to usher in its own GPT-3 moment: model training will shift from fine-tuning on a few environments to large-scale training on multiple environments. By increasing the scale and diversity of interactive environments to orders of magnitude greater than current levels, we can bring RL models with strong generalization and rapid adaptation to new tasks.






This year, the AI ​​community has been abuzz with RLVR ( Reinforcement Learning and Reinforcement Virtualization), which refers to designing tasks and rewards that can automatically verify the correctness of results. This allows the reinforcement learning process to rely less on human feedback and achieve highly autonomous optimization. For RL infrastructure providers, the RLVR trend is their most direct and strongest growth driver.



Currently, many RL training datasets have relatively limited total data volume. For example, DeepSeek-R1's RL was trained on only about 600,000 math problems, equivalent to the workload of a human working continuously for 6 years; while GPT-3's training corpus reached 300 billion tokens, equivalent to tens of thousands of years of human writing. To achieve a RL training scale comparable to GPT-3's "experience," it might be necessary to scale RL scaling to the same order of computation as pretraining, requiring tens of thousands of years of interactive experience data equivalent to the task duration. A prerequisite for achieving this is likely significantly scaling the number and diversity of RL environments and enabling these environments to be automatically evaluated through RLVR (Real-Time Ratio Evaluation).



2. Inadequacies of the existing infrastructure and production environment



Realizing the aforementioned vision of RL still faces bottlenecks, primarily the scarcity and limitations of training environments. Current RL environments are very rudimentary, supporting a limited range of tasks and tools, far from simulating the complexity of real-world work. Imagine a software engineer unable to run virtual machines or Docker, unable to use Slack to search historical information and chat with colleagues, and unable to collaborate with more than two people—the RL environments available in most industries are still so limited. These limitations were understandable in the past because early AI agents were weak and couldn't withstand the test of complex environments. However, as model capabilities improve, emerging RLVR paradigms require richer environments; otherwise, progress will be severely hampered.



The Production Environment Paradox is a common problem in building RL environments. Theoretically, allowing AI agents to learn in real production environments is the most efficient way because it provides the most authentic feedback. However, this introduces many risks in practice: uncontrollable security and compliance risks (involving sensitive user data and policy restrictions), extremely low user tolerance for errors (especially the high cost of errors in multi-step interaction scenarios), budget constraints for online system incidents, and significant agent spending in payment systems, among others.



Secondly, the challenges of reward design remain significant. If the reward function is not precise enough, the agent often exploits loopholes to optimize the wrong objective (i.e., reward hacking), performing well in the training environment but learning the wrong objective, leading to failure when transferring to real-world scenarios. These challenges result in poor generalization of RL models outside the training environment, greatly limiting the effectiveness of RL in a wider range of tasks.



Recent industry trends also confirm this demand: for example, AI cloud service provider CoreWeave acquired OpenPipe, a startup specializing in RL training tools, this month, and began to incorporate its trainable interactive agent platform and team into its own products.






02.



RL Infra Mapping Framework



The emerging RL infrastructure startups can be broadly categorized into two types: those focusing on providing RL environments and those offering "RL-as-a-Service" solutions. Additionally, a small number of companies are approaching the field from a data perspective, providing high-quality feedback data and evaluation tools for RL.






• RL Environment Companies : These companies build simulated environments, providing AI labs and small to medium-sized teams that need to train agents with simulated training environments, task platforms, and benchmarks. The goal is to "simulate" real-world workflows, allowing AI agents to practice repeatedly in virtual environments, thereby generating scalable interaction data. Ideally, these environments are standardized, scalable, and can be reused by different models. If this model works, it will have significant economies of scale—one environment platform company can serve numerous agent developers, just as a game engine serves numerous game developers. However, creating realistic, general-purpose environments is extremely challenging and often has a strong research bet element: short-term commercialization is highly uncertain, but a technological breakthrough could give rise to new platform companies.



• Reinforcement Learning as a Service (RLaaS) Companies : These companies can be seen as the Palantir model within reinforcement learning technologies. They primarily target the specific needs of large enterprises, collaborating deeply with them to apply reinforcement learning to existing workflows to solve real business pain points. Similar to technical consulting or professional services: Deeply understanding the enterprise's business -> defining reinforcement learning tasks and rewards -> preparing data and simulation environments -> training a customized model -> continuous monitoring and iteration. Each enterprise's solution is highly customized, making standardization difficult in the short term, and expansion relies heavily on human resources. However, RLaaS companies directly create commercial value, and enterprise clients are willing to pay for results; individual contracts are often substantial (some startups have entered the market with contracts worth tens of millions of dollars). Therefore, this is a field more suitable for "observing traction and then investing heavily."



• Data/Evaluation Companies : With the rise of RL, some teams are focusing on providing high-quality data and evaluation tools. For example, companies like Mercor have begun developing benchmarks and datasets for the RL era, acting as "data arms dealers." These companies' entry point is the scarcity of the large amounts of interactive data and sophisticated scoring standards needed to build RL. They support the training and validation of RL models through expert annotation and automated tools. In the long run, environment and data are complementary: realistic environments generate valuable data, and rich data, in turn, can expand environmental scenarios. Therefore, in terms of investment, environment platforms and data tools may constitute a hedge combination (we will discuss this later).



In general, the RL Infrastructure field is still in its early exploratory stage, and various startups have subtly different approaches. Below, we will delve into the technical elements and representative company cases of the two main categories: RL Environment and RL-as-a-Service, to outline the landscape of this new field.



03.



RL Environment: Unreal Engine for building software



These companies focus on building high-fidelity simulation environments that allow AI agents to be trained under near-realistic conditions. Essentially, they are factories that can safely, massively, and reproducibly generate "experience data." Building such an environment is far more complex than it sounds, as it addresses challenges common in real-world environments such as reward sparsity and incomplete information. Only by providing rich and immediate feedback through the simulation environment, ensuring that each action of the model receives an evaluation signal, can it learn more effectively. A typical simulation environment includes at least three core elements:



1. State Management System : Used to record and update the state of the environment, defining the world that the agent can perceive and influence. This system includes an initial world snapshot at the start of the task, the transition logic of how the agent's behavior changes the world state, and accurate modeling of all relevant assets (such as simulated APIs, databases, and synthetic users).



2. Task Scenarios : The problem scenarios that the agent needs to solve, including task description, background information, success criteria, and constraints. For example, in a customer service environment, a task scenario might be a series of customer inquiries and account information, and the agent needs to ultimately complete the refund process while complying with business policies.



3. Reward/Evaluation System : This is the guiding principle for agent learning. It uses a validation function to determine whether a task has been completed correctly and provides intermediate reward signals (process rewards) for complex, multi-step tasks. This system can use an LLM (Local Level Management) as the referee, a traditional unit test, or industry-customized scoring rules.



The emerging RL environment platforms currently take several forms:



• Application-level sandboxes : Sandbox environments are built for specific software applications or workflows, such as simulators for CRM/ERP systems or customer service/ticketing systems. Within such sandboxes, the application's UI and functionality are reproduced, allowing agents to operate and complete business processes as if they were in real software. For example, Salesforce AI Research's recently launched CRMArena-Pro is a typical case, injecting highly realistic synthetic CRM data into the sandbox environment to simulate complex processes such as enterprise customer service and sales quoting. This demonstrates the potential for SaaS applications to be packaged into standardized benchmark environments.



• General Browser/Desktop Environment : This direction is closer to the computer use agent approach. It provides a virtual web browser, operating system, or office software environment for the AI ​​agent to practice web navigation, form filling, and document editing. Such environments typically need to simulate mouse and keyboard interactions, UI element recognition, and handle various uncertainties in the real internet (such as random pop-ups, CAPTCHAs, network latency, etc.).



• Environment-World Model : This is a more research-oriented approach. It utilizes a large amount of historical interaction data to train an environment model, simulating the real-world environment's response to agent actions. Simply put, another coding agent generates the environment, and the AI ​​agent interacts with this "model-imagined" world to gain near-realistic experience. This is similar to the "world model" approach, where the environment model makes rolling predictions about agent actions, providing the feedback needed for practice. The advantage of this method is that it eliminates the reliance on manually constructed environments, generating new environments driven by AI Synthtic Data.



Regarding the partner ecosystem, many environment platform companies choose to collaborate with computing power providers or model providers. For example, some startup teams are responsible for building the environment and training pipeline, but outsource model hosting and large-scale computing tasks to third parties (such as Fireworks, Together, etc.). This model allows startups to complete RL training for customers without having to build their own large infrastructure.



Case Study



Mechanize - Replication Learning Platform



This company represents a cutting-edge environmental platform. Its founders come from Stripe and Epoch AI, AI frontier banchmark startups, and it has received angel investment from Nat Fridman and Daniel Gross.



Mechanize proposes a new paradigm: Replication Training. Its core idea is to have an AI agent completely reproduce an existing software product or a specific function within it, using this as a training task. The success or failure of the task can be verified automatically, for example, by checking if the code generated by the agent passes all unit tests in the original repository. This method cleverly transforms an open-ended, ambiguous creative task into a reinforcement learning (RL) problem with clear, verifiable reward signals, thus solving the challenge of designing reward functions for long-term, complex tasks.



This approach leverages the vast amount of software already developed by humans as a blueprint, allowing for a continuous stream of training tasks, much like the richness of internet corpora used for pre-training. Software is ubiquitous on the internet, and AI can accumulate capabilities by replicating these programs. This makes it possible to scale up the RL environment.



While "copying" a program verbatim may not sound like a common engineering goal, it's something leading AI labs explicitly need, providing experimental environments for their AI agents. This task also hones a range of core AI skills: accurately reading and understanding long specifications, strictly following instructions to avoid errors, identifying and correcting mistakes in early steps, and maintaining consistency in projects with tens of thousands of lines of code.



Veris – A training ground with a stronger corporate focus



Veris AI has chosen a more pragmatic, enterprise-market-oriented GTM (Government-Training-Made) strategy. They recognize that the most pressing concerns for enterprises adopting agency AI are security, compliance, and reliability. Therefore, Veris's entry point is to provide an absolutely safe training ground for high-risk sectors such as finance, manufacturing, and human resources. They recently completed an $8.5 million seed funding round led by Decibel and Acrew to accelerate this process.



Veris' core differentiator lies in its "Mirror Your Stack" capability. They build a "digital twin" of each enterprise customer's production environment. This twin not only includes general-purpose software, but more importantly, it precisely replicates the customer's unique internal tools, APIs, data structures, and user interaction patterns. This highly customized environment minimizes the "sim-to-real gap," allowing enterprises to confidently train agents with their core, proprietary workflows in an isolated sandbox without worrying about any risks. Through this mechanism, Veris aims to address two major pain points for enterprises deploying AI agents: environmental security (providing a secure offline training environment to avoid problems caused by immature agents in production) and training effectiveness (enabling AI to learn the precise requirements of enterprise tasks).






Reports indicate that some companies in the manufacturing and Fintech industries are already using the Veris platform to train their AI agents with positive results. A representative example is the "Supplier Negotiation AI Agent": Enterprise customers wanted to automate procurement negotiations using AI. They partnered with Veris to simulate a Slack conversation and email exchange environment containing a wealth of business details and sensitive information, allowing the AI ​​agent to practice negotiating with suppliers. After training, it performed exceptionally well in terms of tone, questioning, and concessions.



Halluminate , computer use environment platform



Halluminate represents the approach of a computer use environment platform. They found that many current Browser Agents (browser automation agents) are performing poorly, and their solution is to simultaneously provide a "real sandbox" and data/evaluation services.



2. Proprietary Datasets and Evaluation : Halluminate has built its own benchmark tasks and datasets, complete with expert annotations, for rigorously evaluating agent performance. Their evaluation service helps clients identify where the agent makes the most mistakes, allowing for targeted optimization. This "data-driven failure mode analysis" significantly accelerates model development iterations.

Of course, there is also significant uncertainty surrounding the RL environment direction. Environment construction itself is costly and slow to yield results; achieving true universality and high fidelity is no easy feat. Thus, we see different startups adopting drastically different market positioning: small and fast vs. large and deep. One type of company leans towards productization and scaling, hoping to quickly develop a large number of medium-fidelity environments to cover various scenarios and serve long-tail customers; the other focuses on acquiring a few large clients, deeply customizing high-fidelity environment solutions (somewhat similar to consulting projects). The latter is the RLaaS direction we will discuss next.



04.



RLaaS: Building an AI-native Palantir



When many traditional enterprises recognized the value of reinforcement learning (RL) but struggled with a lack of relevant talent and technology, "Reinforcement Learning as a Service" (RLaaS) emerged. RLaaS providers offered hosted RL training platforms and professional support, enabling enterprises to directly apply RL to their critical workflows and proprietary data. Their service model typically encompasses the following aspects:



1. Reward Modeling



The RLaaS team first works with the enterprise to identify the key performance indicators (KPIs) most important to their business and then transforms them into calculable reward functions. Because enterprise goals are often abstract (such as "improving customer satisfaction"), they need to be broken down into concrete metrics. RL experts leverage domain expertise to continuously experiment and adjust, transforming vague goals into optimizable numerical signals for the agent. For example, in customer service scenarios, it might be necessary to consider multiple dimensions such as response accuracy, resolution time, and customer sentiment rating when designing rewards.



2. Automated Scoring (Auto Scorer)



With the reward model in place, the team builds an automated scoring pipeline to rigorously score the AI ​​Agent in each rollout within the simulated environment. This is typically achieved through a pre-prepared series of test cases or rules: for example, in customer service tasks, 100 standard question-and-answer processes can be prepared, and the Agent checks its actions against the ideal steps after each step; or in financial report auditing tasks, a series of reports with known errors can be designed for the Agent to identify, to see if it can catch all the key issues. This scorer acts as an automated referee for the AI, providing immediate and objective feedback signals, and is a crucial part of RL training. Many RLaaS companies have accumulated their own evaluation toolchains as a competitive advantage, enabling them to quickly customize effective scoring methods for different domains (this often involves rule scripts, pattern matching, and training an auxiliary discriminative model to evaluate the AI ​​output).



3. Model Customization and Reinforcement Fine-Tuning (RFT)



Once the environment and rewards are ready, the RLaaS team selects a suitable base model (which could be a large open-source model or a customer's own model) for customized fine-tuning, continuously allowing the model to interact with the environment and adjusting the policy based on reward signals. This process may use RL algorithms such as policy gradients and evolutionary policies, but these details are hidden from the customer; they only see the model's continuous improvement. It's worth noting that emerging inference platforms are also striving to automate the training pipeline. For example, Fireworks' reinforcement fine-tuning feature allows users to define an evaluation function with a piece of Python code, and the platform can then take over the rest of the training process. The emergence of such tools allows RLaaS to be delivered in a more product-oriented way, rather than as a completely labor-intensive project.



Case Study



• Fireworks AI , from Inference to RFT Standard Definition



Fireworks, a three-year-old AI Inference Infrastructure company, has recently excelled in Reinforcement Fine-tuning (RFT). Their platform allows users to easily train large open-source models using Reinforcement Learning (RL). Users only need to provide a scoring function; the platform handles everything else, including GPU resources, training loops, and experiment management. Fireworks claims that several early clients have used their platform to fine-tune open-source models, achieving performance comparable to or even surpassing top-tier closed-source models, with inference speeds increased by 10 to 40 times. For example, Fireworks collaborated with Vercel to train an auto-code-fixing model using RFT, achieving performance comparable to GPT-4 derivative models while running dozens of times faster. Essentially, Fireworks is lowering the barrier to entry for RL and leveraging its model inference customization business model to attract these users.






• Applied Compute , OpenAI's star team



Founded in 2025 by former OpenAI researchers, this company follows a high-profile approach within the RLaaS (Research, Application, and Service) model. During their pre-launch phase, they secured $20 million in seed funding led by Benchmark at a valuation of $100 million. They choose to deeply integrate with a few large enterprises, implementing RL solutions on a project-by-project basis, with each contract potentially worth tens of millions of dollars. This model is more akin to that of an AI consulting firm. While product details remain confidential, public information suggests they may initially focus on traditional industrial sectors such as energy and manufacturing, leveraging RL to optimize complex industrial processes.



• RunRL , a platform that allows developers to access services.



Both originating from Y Combinator, this startup's name clearly states its vision: to enable anyone to "run RL with one click." They handle all the complex tasks involved, such as underlying GPU cluster management, algorithm selection, and model warm-up; developers only need to provide initial prompts and a reward function. Their pricing model is also extremely transparent, charging per node-hour ($80/node-hour), making it clear and straightforward. RunRL represents a more democratized direction in the RLaaS field: opening up to enterprise developers in a low-barrier manner. Their platform states that most models below 14 bytes can run RL training on a single node.



05.



Future Outlook under the RL Trend



RL Environment vs. RL Data: One Wins and the Other Loses or They Complement Each Other?



"Environment" and "data" are often discussed in opposition: one view holds that instead of delving into building complex environments, it's better to invest manpower in collecting high-quality interaction data to train models; another believes that realistic environments can generate infinite data, which is the source of sustainable, exponential growth. Environment and data are indeed like two sides of the same coin: data companies generate observations from a world, while environmental companies simulate the world.



• Online learning (RL environment) : Its biggest advantage is the ability to generate perfect on-policy data, meaning the agent learns directly from its own actions, providing the most direct and effective feedback. However, its disadvantages are high cost and slow speed, as each data point requires running a complete simulated interaction. The recently released Online RL by Cursor is currently designed for the Tab scenario, which involves short links and high-frequency feedback; a better environment is needed to move towards online RL for long-horizon agentic tasks.






• Offline learning (RL data) : Its advantages are low cost and high speed, and it can directly utilize existing massive amounts of data. However, its fundamental drawback is that this data is "off-policy," meaning it was generated by other agents or humans. Learning directly from this data can easily lead to models learning spurious correlations, resulting in poor generalization ability and poor performance in real-world environments.



The Mercor team is also closely monitoring the latest developments in RL.



There's no consensus in the industry on how to most efficiently train top-tier agents; it's likely a combination of these two approaches. A sound investment strategy is to invest in both paths simultaneously. If the cost of building and running high-fidelity simulation environments proves to be the main bottleneck, companies with massive amounts of high-quality offline data will become extremely valuable. Conversely, if offline data proves insufficient for training truly general-purpose agents, then the only path to AGI must involve richer, more realistic interactive environments. This bets on the grand theme of Era of Experience itself, while hedging against the uncertainty of the specific implementation path.



Will RLaaS's Palantir model lead to vertical industry monopolies?



In the AI ​​era, more companies are emulating Palantir's model: using forward-deployed engineers to delve into vertical business areas and solve specific, high-value business problems. The execution path of this model is very clear:



1. Embedded Experts : Deploy top RL engineers to clients (such as an investment bank or a pharmaceutical company) to work closely with their business teams.



2. Solving core problems : Utilize the RLaaS platform to build a customized agent for clients. The optimization goals of this agent are directly linked to the client's most critical business metrics, such as transaction fraud detection rate and the success rate of new drug development.



3. Build a proprietary data flywheel : This agent runs within the customer's real business processes, continuously learning from the customer's unique, real-time operational data. The RLaaS platform manages and optimizes this learning loop.



4. Building a competitive moat : Over time, this agent, fed with proprietary data, will become increasingly adept at understanding the customer's business, and its performance will far surpass any general-purpose model. It is deeply embedded in the customer's workflow, creating extremely high replacement costs. At this point, the RLaaS provider's role has transformed from a software vendor into an indispensable strategic partner.



This model is highly likely to create a "winner-takes-all" situation in specific vertical industries. The first service provider to successfully integrate its RLaaS platform into a leading company in a particular industry will accumulate unparalleled data and domain knowledge advantages, making it difficult for later entrants to enter the market. Therefore, the future landscape of RLaaS may not be dominated by a single giant platform, but rather by a series of "mini-Palantirs" that hold monopolistic or oligopolistic positions in their respective vertical fields.






