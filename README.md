## Building a Tournament-Ready Q&A Agent with Unsloth on AMD Instinct MI300X

### TL;DR
- We built a competitive Q&A agent system (questioner + answerer) optimized for logic puzzles and analytical reasoning, trained and iterated with Synthetic-Data-Kit (Meta) and Unsloth on AMD Instinct MI300X GPUs.
- The MI300X’s 192 GB HBM3 and Unsloth’s ROCm optimizations let us run Llama-3.3-70B-Instruct with LoRA r=64 at BF16, large batch sizes, and fast iteration.
- Data focus: seating arrangements, blood relations, hybrid seating-relations, and logic reasoning. We generated/curated CoT-rich datasets, filtered with an LLM judge, finetuned with SFT and performed prompt finetuning.
- Prompt strategy: multi-level CoT prompting, adversarial question design patterns, and role-aware templates for asking vs answering. Final agent emphasizes correctness, clarity, and time-to-solution under 6 seconds.

---

### Challenge
Use Unsloth with AMD Instinct MI300X GPUs to build a question-and-answer AI agent through SFT, RL, or prompt engineering that competes in a bracket-style tournament by asking and answering questions.

---

### Hardware and Training Setup (Unsloth on AMD Instinct MI300X)
- **Hardware**: 1 × AMD Instinct MI300X (192 GB HBM3)
- **Base Model**: `meta-llama/Llama-3.3-70B-Instruct`
- **Precision**: LoRA (16-bit BF16)
- **LoRA**: `r = 64`, `alpha = 64`
- **Seq Len**: 1024
- **Batch Size**: 128
- **Epochs**: 1 (default)
- **Optimizer**: `adamw_8bit`
- **Learning Rate**: `1e-4`
- **Grad Checkpointing**: Enabled (unsloth mode)
- **Grad Accumulation**: 4
- **Weight Decay**: 0.01
- **Scheduler**: Linear
- **Warmup Steps**: 5

What this enabled:
- A 70B model with r=64 LoRA at BF16, seq len 1024, and batch size 128 fitting comfortably on a single MI300X.
- Stable, memory-efficient training via Unsloth’s gradient offloading, kernel fusions, and ROCm optimizations.
- High throughput for rapid ablations, letting us iterate on prompts, data curation, and configs quickly.

---

### System Overview

- **Questioning Agent** (`agents/question_agent.py`):
  - Generates adversarial, high-quality MCQs from topic taxonomies.
  - Enforces strict JSON format, balanced distractors, and difficulty. Supports in-context exemplars.
  - Includes post-generation JSON validation and self-repair extraction.

- **Answering Agent** (`agents/answer_agent.py`):
  - Answers MCQs with concise reasoning (<100 words), outputs strict JSON with `answer` and `reasoning`.
  - Two alternative system prompts: a concise exam-style and a richer analytical variant.
  - Includes output validation, format repair, and token/time reporting hooks.

- **Configs**:
  - Inference generation defaults for answerer: `configs/agen.yaml`.
  - Synthetic-Data-Kit workflow for ingestion → generation → curation → formatting: `configs/tutorial_config_team.yaml`.

---

### Data Pipeline (Synthetic-Data-Kit)
We used Meta's Synthetic-Data-Kit to ingest, generate, curate, and format training data focused on logical reasoning:

1) Ingest and Parse
- Sources: logic/aptitude repositories, seating arrangements, blood relations, and academic resources. Text was normalized into chunked `.txt` inputs.

2) Generate
- Prompts targeted step-by-step deduction, pattern-structured reasoning, and progressive difficulty. We generated both QA pairs and explicit CoT traces.

3) Curate (LLM-as-Judge)
- A judge rates QA/CoT pairs on competitive balance, technical correctness, and training value.
- Kept only high-quality items above a strict threshold.

4) Format
- Converted to Alpaca-style JSON with metadata for Unsloth SFT.

Key configuration highlights from `configs/tutorial_config_team.yaml`:
- Provider: vLLM server for fast batched generations.
- Parameters tuned for reasoning: `temperature=0.7`, `top_p=0.95`, chunking `4000/300`, and `num_pairs=300` per chunk with CoT samples.
- Curation threshold set high (`8.5`) to prefer clean, systematic reasoning examples.

Representative open datasets we referenced or adapted:
- Logic Problems Reasoning Dataset — varied truth/logic puzzles. [Hugging Face]
- LogiQA2.0 — logical reasoning over text. [GitHub]
- BiRdQA — bilingual riddles with distractors. [arXiv]
- Brain-Teasers — lightweight verbal puzzles. [Hugging Face]
- QD-Logic-Puzzles — logic grid/constraint puzzles (CC-BY). [OSF]
- Open-Platypus Logical Reasoning — general logical reasoning. [Kaggle]
- CLUTRR — family relation reasoning; useful for blood relations patterning. [Hugging Face]

We additionally curated domain-specific content for:
- Blood relations (family trees, kinship chains).
- Seating arrangements (linear and circular), including hybrid problems combining relations + seating constraints.

---

### Our Unique Data Collection Strategy

**1. Structured PDF-to-JSON Learning Approach**

One of our key innovations was presenting training data in a structured JSON format that the model could easily parse and learn from. We converted PDFs containing logic puzzles into well-structured JSON examples with explicit difficulty levels, making it easier for the model to recognize patterns and difficulty gradients.

**Example Structured Format:**
```json
{
  "Question": "Eight individuals are seated around a round table. A sits immediately to the left of B, and B is positioned third to the left of C. D occupies a seat between C and E, while F is immediately to the right of E. Who is seated immediately to the left of D?",
  "Answer": "C",
  "Reasoning": "1. Place B and note A is immediately left of B. 2. From B, third left is C, so positions: A left of B, then two seats, then C. 3. D between C and E, so after C is D then E. 4. F immediately right of E, so after E is F. 5. The remaining seat after F leads back to A and B. 6. Thus, immediately left of D is C.",
  "Topic": "Seating Arrangement",
  "Difficulty": "Hard",
  "Source": "https://www.geeksforgeeks.org/aptitude/seating-arrangement-aptitude/",
  "License": "paraphrased from public educational resource"
}
```

This structured approach provided several advantages:
- **Clear difficulty progression**: Models learned to recognize Easy → Medium → Hard patterns
- **Consistent reasoning format**: Standardized step-by-step structure enabled pattern matching
- **Rich metadata**: Topic, source, and difficulty tags helped the model contextualize problem types
- **Easier curriculum learning**: We could organize training by difficulty, starting with simpler problems and progressively increasing complexity

**2. Diverse Source Collection Strategy**

We didn't just rely on a single dataset. Instead, we collected from multiple diverse sources:

- **Research Papers**: We studied papers on hard reasoning problems and model performance, which led us to focus on the Llama series. After analyzing reasoning benchmarks, **Llama-3.3-70B-Instruct stood out** for its balance of capability and efficiency, making it ideal for our tournament constraints.

- **Competitive Exam Resources**: Sourced from Indian competitive exam platforms (GeeksforGeeks, IndiaBIX, Careericons, Byju's) known for high-quality logical reasoning puzzles

- **Academic Datasets**: Integrated CLUTRR for family relations, research papers on seating arrangement complexity ("Optimal Seat Arrangement: What Are the Hard and Easy Cases?"), and reasoning surveys

- **Open Educational Resources**: Leveraged multiple free PDFs from educational institutions, ensuring broad coverage of problem types while maintaining educational licensing

This diversity ensured our model saw a wide variety of problem formulations, constraint types, and reasoning patterns, reducing overfitting to a single style.

---

### Why Chain-of-Thought Over Simple QA Pairs

After extensive experimentation, we chose **Chain-of-Thought (CoT) generation** over simple question-answer pairs for several critical reasons:

1. **Systematic Reasoning Patterns**: CoT teaches models to break down complex problems into structured steps (constraint extraction → enumeration → systematic testing → verification), making solutions reproducible and learnable rather than relying on pattern memorization.

2. **Competitive Edge**: In a tournament setting, the ability to reason through problems step-by-step under time pressure is crucial. CoT training ensures our agent can handle novel problem formulations by applying learned reasoning patterns rather than failing when faced with variations.

3. **Training Efficiency**: CoT examples provide richer signal—each example teaches not just the answer, but the reasoning methodology. This means fewer examples needed for the same level of competence.

4. **Error Recovery**: When our agent encounters a challenging problem, the internal reasoning structure (learned from CoT) helps it work through even partially understood scenarios, whereas simple QA pairs offer no intermediate reasoning scaffold.

---

### Prompting Strategy and CoT Levels

Our prompting strategy was built around **ACM ICPC-level competitive programming principles**—we treated this as an adversarial reasoning competition, not just a knowledge test. Here's how we designed our prompts:

**1. ELITE-Level System Prompts**

From `configs/tutorial_config_team.yaml`, our CoT generation prompt frames the task as:
```
You are an ELITE LOGICAL REASONING ARCHITECT and COMPETITIVE AI TRAINER 
specializing in chain-of-thought problem solving for adversarial competitions
with expertise in:
- ACM ICPC problem design
- Olympiad-grade logical reasoning assessment
- Adversarial question generation for AI battles
- Format compliance for automated evaluators
```

This positioning wasn't just for show—it directed the model to generate problems with:
- **Goldilocks Zone difficulty**: Hard enough that untrained opponents fail (50-70% failure rate), but systematic enough that our trained agent solves correctly (80-90% success rate) in under 6 seconds
- **Learnable patterns**: 35% Constraint Elimination, 25% Position Locking, 15% Circular Reference, 15% Relationship Chain, 10% Hybrid Cascade
- **Structured reasoning**: Every problem follows a 6-step structure (Extract Constraints → Identify Most Restrictive → Enumerate → Test → Verify → Final Answer)

**2. Structured CoT Format**

We enforced a strict 6-step reasoning template that models could learn and replicate:
```
STEP 1 - EXTRACT ALL CONSTRAINTS: List every constraint explicitly
STEP 2 - IDENTIFY MOST RESTRICTIVE: Find the constraint that limits possibilities most
STEP 3 - ENUMERATE POSSIBILITIES: List all potential configurations
STEP 4 - TEST SYSTEMATICALLY: Validate each possibility against constraints
STEP 5 - VERIFY SOLUTION: Check all constraints are satisfied (✓ marks)
STEP 6 - FINAL ANSWER: Provide answer with key insight
```

This structure ensured:
- **Consistency**: Models learned a repeatable methodology
- **Debuggability**: We could trace reasoning failures to specific steps
- **Efficiency**: The structure optimized for sub-6-second solutions

**3. Domain-Specific Prompting Strategies**

**Seating Arrangements (Linear & Circular):**
- Key patterns: Position Locking, Adjacency Cascade, Most Restrictive First, Systematic Elimination
- Constraint types: Absolute position, relative position, adjacency, non-adjacency, conditional, directional
- Target: 4-5 people, 4-5 constraints, 5-6 reasoning steps

**Blood Relations & Family Trees:**
- Key patterns: Relationship Chain, Gender Inference, Symmetric Relations, Generation Counting, Reverse Deduction
- Target: 5-6 family members, 3-4 relationship hops, 2-3 generations max

**Hybrid Problems (Our Secret Weapon):**
- Combined seating + blood relations in single questions
- 15% of our training data to maximize opponent confusion
- Example: "A family sits around a circular table. Father sits opposite daughter. Son sits adjacent to mother. Uncle sits 2 seats from nephew..."

**4. Adversarial Rating System**

Our LLM-as-Judge prompt (`qa_rating` in config) evaluated on three dimensions:

- **Competitive Balance (0-4 points)**: Does it hit the Goldilocks Zone?
- **Technical Correctness (0-3 points)**: Is the logic flawless?
- **Training Value (0-3 points)**: Will this teach the agent to win?

Only examples scoring ≥8.5/10 made it into training, ensuring every example had maximum competitive value.

**5. Role-Aware Templates**

- **Questioner**: Enforces topic alignment, single correct option, plausible distractors, strict JSON schema. Advanced system prompt positions it as an "expert-level examiner" designing "highly challenging and conceptually rigorous MCQs."

- **Answerer**: Two prompt variants—concise exam-style and richer analytical. Both emphasize step-by-step reasoning, option elimination, and confidence in selection.

**6. Chain-of-Thought (CoT) Sampling Levels**

We generated multiple CoT levels:
- **Explicit step lists**: Full 6-step reasoning traces (for training)
- **Concise rationale**: Brief but complete reasoning (<100 words, for inference)

During inference, we keep outputs concise while ensuring the internal reasoning structure has been learned, allowing fast answers (1-2 seconds for medium difficulty) without sacrificing accuracy.

**7. Difficulty Shaping and Distribution**

- **40% Basic**: 3-4 constraints, 4-5 steps—fast, high confidence
- **45% Intermediate**: 4-5 constraints, 5-6 steps—competitive zone
- **15% Advanced**: 5-6 constraints, 6-7 steps—elite differentiation

Target: Goldilocks zone where our model succeeds 80-90% of the time while opponents fail 50-70%.

---

### Supervised Fine-Tuning (SFT) with Unsloth

**SFT Objective**: Teach reliable patterns for both asking and answering, ensuring our agent can seamlessly switch between roles while maintaining high accuracy and speed.

**Training Configuration:**
- Model: Llama-3.3-70B-Instruct with LoRA (r=64, alpha=64) at BF16
- Optimizer: `adamw_8bit`, LR `1e-4`, linear scheduler, warmup 5 steps
- Regularization: weight decay 0.01, gradient checkpointing enabled
- Training data: Curated QA pairs and CoT traces across seating/blood-relations/hybrids
- Outputs validated to ensure strict JSON formats for downstream automation

**Key SFT Strategies:**

1. **Diverse Training Data Collection**
   - Used Synthetic-Data-Kit to generate large custom Q&A datasets from PDFs and educational content
   - Generated complex reasoning questions with detailed CoT solutions requiring multiple steps, assumption testing, and explicit explanations
   - Ensured training data covered **both roles**—asking questions (questioner) and answering them (answerer)—so the model could seamlessly switch between modes

2. **Quality Filtering with LLM-as-Judge**
   - Applied strict quality thresholds (≥8.5/10) using our competitive rating system
   - Filtered out trivial, ambiguous, or logically flawed problems
   - Mirrored DeepSeekMath approach: generated ~800k+ synthetic examples, then used LLM judge to reject incorrect or low-quality responses

3. **Difficulty-Aware Training**
   - Focused on generating difficult questions that could stump other agents: multi-step reasoning, constraint satisfaction, and hybrid domain problems
   - Trained agent to both solve AND ask tricky problems through curriculum learning (basic → intermediate → advanced)
   - Incorporated chain-of-thought in answers to teach step-by-step reasoning, improving accuracy on complex problems

4. **Memory-Efficient Training on MI300X**
   - Leveraged Unsloth's optimizations (~30% less memory usage) to enable larger batch sizes
   - Mixed-precision (bf16) and gradient checkpointing allowed us to fit the full 70B model with LoRA in GPU memory
   - Fast iteration cycles enabled rapid experimentation with hyperparameters and training data variations

**Self-Play and Iterative Refinement:**

We attempted self-play Q&A matches where the model competed with past versions, but due to time constraints, we focused primarily on:

- **Offline Evaluation**: Quizzed the model on known QA benchmarks and sample questions across domains (seating arrangements, blood relations, hybrid problems)
- **Mock Matches**: Simulated bracket scenarios by alternating asking and answering for multiple rounds
- **Weak Spot Identification**: When the model failed at specific problem types, we generated additional similar problems via the synthetic data pipeline and fine-tuned on them to patch gaps
- **Prompt Finetuning**: With the AMD GPU's fast iteration capability, we ran multiple ablations—testing different prompt variants, CoT inclusion/exclusion, and difficulty distributions—rapidly converging on the best configuration

**Evaluation Results:**

Thanks to the rapid iteration enabled by AMD's MI300X, we were able to extensively test and refine our model. Our final evaluation showed:

- **Medium difficulty questions**: Solved in **1-2 seconds** consistently
- **Hard difficulty questions**: Solved in **3-4 seconds** on average, well under the 6-second tournament constraint
- **Accuracy**: 85-90% on our curated test set of logical reasoning problems
- **Question quality**: Generated questions that consistently stumped baseline models while remaining solvable by our trained agent

**Philosophy: "Sometimes Simple is the Best"**

Under strict time constraints, we found that a well-executed simple approach often outperformed complex multi-stage pipelines. Our focus on:
- Clean, structured data formats
- Clear reasoning patterns
- Strict quality filtering
- Efficient training configuration

proved more effective than trying to implement every advanced technique. The combination of excellent hardware (MI300X), efficient framework (Unsloth), powerful data toolkit (Synthetic-Data-Kit), and strategic simplification allowed us to build a highly competitive agent within the hackathon timeline.

---

### Short RL Loop (PPO/GRPO) for Competitive Edge
- Reward signals: correctness, clarity, and “stump value” of generated questions.
- Self-play simulations: weaker model attempts answers; failures increase reward for question difficulty.
- GRPO (via Unsloth) used for memory-efficient reasoning refinement on select batches.
- while we were not able to implement this to the full extend, we beleive with our testing this would still give better results as it reasons and improves every iteration

---

### Advanced Prompt Engineering Strategies

Beyond basic CoT, we implemented a comprehensive suite of prompt engineering techniques optimized for tournament performance:

**1. Clear System Prompts with Competitive Context**

We crafted system prompts that explicitly defined the agent's role and objectives:

**Example System Prompt (Answering):**
```
"You are a champion question-answering AI. Your goal is to win a Q&A debate. 
On your turn, ask the most challenging, knowledge-packed question you can. 
When answering, provide a correct, well-reasoned answer. Always be clear 
and confident."
```

This competitive framing:
- Sets a winning mindset from the start
- Reduces uncertainty and hesitation in responses
- Encourages confident, decisive answers (no revealing uncertainty to opponents)
- Maintains focus on the competition context

**2. Chain-of-Thought Prompting (Explicit vs Implicit)**

For answering questions, we prepended prompts with reasoning triggers:
- **Explicit CoT**: "Let's think this through step by step." followed by the question
- **Implicit CoT**: Instructed model to reason internally and output final answer when context length was limited
- **Scratchpad approach**: Some templates used a "scratchpad" section where the model could show work, then output the final answer

The key insight: Even though our model was fine-tuned on CoT, an explicit nudge during inference could trigger more systematic reasoning, leading to better accuracy on complex questions.

**3. Few-Shot Examples and In-Context Learning**

We incorporated few-shot demonstrations of ideal Q&A behavior:

**Questioner Examples:**
- Showed mini-dialogs where Agent A asks a hard question
- Agent B (opponent) answers incorrectly or incompletely
- Agent A provides the correct answer, demonstrating the "stumping" strategy

**Answerer Examples:**
- Demonstrated step-by-step elimination of distractors
- Showed how to handle ambiguous or trick questions
- Illustrated confident, complete answers vs. hesitant responses

These examples implicitly taught the model:
- What constitutes a "hard question" (for the questioner)
- What constitutes a "good answer" (for the answerer)
- Competitive tactics without explicit instruction

Note: We balanced example count with context length limits and fairness considerations.

**4. Persona and Theming**

We experimented with themed personas to complement performance:

- **"Professor" Persona**: Knowledgeable, authoritative tone making answers more convincing and questions sound more challenging
- **Competitive AI Persona**: Focused on winning, maintaining consistent strategic behavior

**Implementation:**
- Integrated persona into system prompt (e.g., "You are Professor Quantum, a brilliant polymath AI...")
- Ensured persona enhanced rather than detracted from clarity and correctness
- Focused on clear, correct Q&A first; theme as secondary stylistic enhancement

**Key insight**: A consistent persona helped the model maintain steady tone and not be thrown off by opponent's style variations.

**5. Formatting and Precision Optimization**

We optimized answer formatting for judge evaluation:

**Answer Format:**
```
Direct answer first (for immediate clarity)
↓
Then explanation (to demonstrate knowledge depth)
```

This two-part structure ensured:
- Judges see correct answer immediately (for speed scoring)
- Extra knowledge demonstration still visible (for impressiveness)
- Best of both worlds: speed + depth

**Question Format:**
- Clear phrasing with implicit difficulty hints
- Example: "In [domain], which [hard-to-recall fact]...?" 
- Made non-triviality obvious to both judges and opponents

**6. Dynamic Role Handling**

Critical for tournament: seamless switching between asking and answering.

**Approach 1: Special Tokens**
- Used `<Role=Questioner>` vs `<Role=Answerer>` tokens in prompts
- Clear signal for mode switching

**Approach 2: Dual Prompt Templates**
- Separate templates for each role:
  - Questioner template: "Now ask a very difficult question for your opponent."
  - Answerer template: "Answer the following question with clear reasoning: [opponent's question]"

**Testing**: We prepared both prompt patterns and verified correct behavior in each scenario to prevent mode confusion (e.g., answering when should ask, vice versa).

**7. "Win at All Costs" Tactics**

Within tournament rules, we employed strategic tactics:

- **Multi-part questions**: Questions requiring thorough answers increased chance of incomplete opponent responses
- **Trap detection**: Trained agent to recognize ambiguous/trick questions and clarify or consider multiple interpretations rather than walking into traps
- **Strategic difficulty**: Questions calibrated to be just hard enough to stump opponents but still solvable by our trained model

**8. Utilization of MI300X Resources**

The powerful MI300X hardware enabled rapid experimentation:

- **Ablation Studies**: Ran multiple fine-tuning runs with different configurations:
  - CoT answers vs. non-CoT answers
  - Different prompt templates
  - Various difficulty distributions
  - Different system prompt variants

- **Fast Convergence**: Within tight hackathon timeline, we could try multiple small tweaks and quickly identify the best configuration
- **Hyperparameter Optimization**: Tested different learning rates, batch sizes, and LoRA ranks, converging on optimal settings faster than traditional hardware would allow

**9. Testing and Refinement Pipeline**

Our iterative refinement process:

1. **Offline Evaluation**: Tested on known QA benchmarks across domains (seating, blood relations, math, logic)
2. **Weak Spot Analysis**: Identified failure patterns (e.g., specific constraint types, domain gaps)
3. **Targeted Data Generation**: Created additional problems in weak areas via synthetic data pipeline
4. **Rapid Re-training**: Fast GPU cycles allowed quick model updates to patch identified gaps
5. **Mock Matches**: Simulated bracket scenarios against baseline agents and earlier model versions
6. **Prompt Iteration**: Refined prompts based on evaluation results, testing variations until optimal performance

This iterative loop—powered by MI300X's speed—was critical to our success.

---

### What Worked vs What Didn’t
- Worked
  - MI300X + Unsloth delivered stable, high-throughput 70B finetuning with large batches and fast iteration cycles.
  - Strict JSON schemas and self-repair prompts kept pipelines automated and robust.
  - Hybrid domain training increased “opponent failure” rates in internal tests.

- Didn’t Work (or de-emphasized)
  - Early GRPO runs without curated CoT were unstable; we first solidified SFT + curation, then reintroduced RL.
  - Overly verbose CoT at inference hurt latency; we trained with rich CoT but infer concisely.

---

### How to Reproduce (Sketch)

1) Prepare data with Synthetic-Data-Kit using `configs/tutorial_config_team.yaml`.
2) Run question generation and curation to produce high-quality JSON data.
3) SFT with Unsloth on MI300X using the curated dataset; adopt LoRA r=64 BF16 config above.
4) Optionally build DPO pairs and run a DPO stage.
5) Optionally run short PPO/GRPO with an AI judge and self-play for question difficulty shaping.
6) Use `agents/question_agent.py` to generate tournament questions and `agents/answer_agent.py` to answer opponents.

Example run snippets:

```bash
# Generate questions (batching, strict JSON, post-repair)
python -m agents.question_agent --num_questions 50 --output_file outputs/questions.json --batch_size 5 --verbose

# Answer questions (concise rationales, strict JSON)
python -m agents.answer_agent --input_file outputs/filtered_questions.json --output_file outputs/answers.json --batch_size 5 --verbose
```

---

### Acknowledgements
- Immense thanks to **AMD** — the Instinct MI300X made 70B-class experimentation genuinely practical: big memory, fast throughput, and smooth ROCm support.
- **Unsloth** — streamlined 70B finetuning via gradient checkpointing, kernel fusion, and efficient LoRA.
- **Meta’s Synthetic-Data-Kit** — a powerful engine for generating, curating, and formatting high-quality reasoning datasets at scale.
- And thanks to **Eda**, **Sanyam**, **Daniel** and other organizers for an amazing experience and the opportunity to build something competitive, fast, and fun.


