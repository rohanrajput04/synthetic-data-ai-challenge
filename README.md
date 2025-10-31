# synthetic-data-ai-challenge

## 🧠 Overview
This repository presents a **comprehensive pipeline** for constructing and fine-tuning datasets aimed at **complex relational reasoning** tasks such as:

- 🧬 Family & Kinship reasoning  
- 🩸 Blood relation deduction  
- 🪑 Seating arrangement (linear & circular)  
- 🧩 Multi-step logical reasoning  

We integrate multiple benchmark datasets, apply **prompt engineering and validation**, synthesize **high-difficulty Q&A pairs**, and fine-tune a large language model (LLM) using **supervised fine-tuning (SFT)**.

---

## 📚 Dataset Sources

| Dataset | Focus Area | Reference |
|----------|-------------|------------|
| [RiddleBench](https://huggingface.co/datasets/ai4bharat/RiddleBench) | Logical riddles (includes “Blood Relations”) | ai4bharat |
| [CLUTRR](https://github.com/facebookresearch/clutrr) | Family tree reasoning and relational inference | Facebook Research |
| [Kinship](https://github.com/juanshernandez/kinship) | Kinship relation triples for reasoning tasks | Open Source |

These were unified, normalized, and cleaned to form a **rich reasoning corpus**.

## ⚙️ Data Processing Pipeline

### 🧩 Step 1: Prompt Engineering & Validation
We employed multiple **prompt patterns** to increase linguistic variety and complexity:
- **Reformulation prompts** → Diverse expression of relations  
- **Constraint-injection prompts** → Add reasoning steps and ambiguity  
- **Self-consistency validation** → Filter logically coherent samples  

✅ *Result:* Highly challenging but solvable examples.

---

### 🤖 Step 2: Synthetic Q&A Generation
We used Meta’s [Synthetic Data Kit](https://github.com/meta-llama/synthetic-data-kit) to transform reasoning texts into question–answer pairs:

```bash
synthetic-data-kit -c configs/reasoning_config.yaml create ./data/parsed/ --type qa --num-pairs 50
