# GRPO-CoT: Structure-Aware Reinforcement Learning on Qwen2.5-3B (in the spirit of DeepSeek)

Pretrained checkpoints are available on Hugging Face:
- **Merged 16-bit (drop-in Transformers):** [srikar-v05/Qwen2.5-3B-GRPO-16bit](https://huggingface.co/srikar-v05/Qwen2.5-3B-GRPO-16bit)
- **LoRA adapters (PEFT/composable):** [srikar-v05/Qwen2.5-3B-GRPO-LoRA](https://huggingface.co/srikar-v05/Qwen2.5-3B-GRPO-LoRA)

Both variants were trained with GRPO on GSM8K using an XML reasoning schema, in the spirit of DeepSeek’s GRPO-based reasoning training.


> A lightweight, reproducible pipeline that fine-tunes **Qwen2.5-3B-Instruct** with **LoRA** and trains it with **GRPO** (Group Relative Policy Optimization) on **GSM8K**, enforcing an XML reasoning schema for easy evaluation—conceptually similar to DeepSeek’s GRPO-based reasoning training.

---

## TL;DR

* **Base model:** Qwen/Qwen2.5-3B-Instruct
* **Method:** 4-bit **QLoRA** + **GRPO** (online RL with multi-objective rewards)
* **Runtime:** **Unsloth** for efficient LoRA training; **vLLM** for fast candidate generation during RL
* **Task/data:** GSM8K (grade-school math)
* **Output schema:** XML tags `<reasoning>` and `<answer>` to make grading trivial
* **Artifacts:** Save LoRA, merge to 16-bit, and push to Hub

Why “similar to DeepSeek”? We adopt **GRPO** (critic-free, group-baseline PPO variant) to improve reasoning, the same RL algorithm family DeepSeek introduced in **DeepSeekMath** and later used in **DeepSeek-R1**’s reasoning-focused training.&#x20;

---

## 1) Background & Motivation

* **GRPO for reasoning.** DeepSeek introduced **GRPO**, which replaces the value function with a **group baseline** over multiple samples from the same prompt—cutting memory and complexity while keeping PPO-style stability. This was shown to boost math reasoning and later underpinned R1’s reasoning training.&#x20;
* **DeepSeek-R1 pipeline.** R1-Zero was trained **purely with RL (no SFT)**; R1 added **multi-stage training** and **RL** to improve readability and performance—establishing a strong precedent for RL-driven reasoning. Our project is **inspired by** this approach (not an exact replica). ([arXiv][1])
* **Why LoRA / QLoRA.** **LoRA** adapts only small rank-decomposed matrices—massively reducing trainable params. **QLoRA** enables **4-bit** fine-tuning while preserving 16-bit quality, making the project feasible on modest GPUs. ([arXiv][2])
* **Why Unsloth & vLLM.** **Unsloth** speeds up LoRA/QLoRA fine-tuning; **vLLM** provides high-throughput generation that plugs into TRL’s online RL (GRPO) loop. ([GitHub][3], [VLLM Documentation][4])

---

## 2) What This Repo/Notebook Does

* Loads **Qwen2.5-3B-Instruct** (3B) and wraps it with **LoRA** adapters for attention & MLP projections. ([Hugging Face][5])
* Prepares **GSM8K** with a **system prompt** that enforces an XML schema. ([arXiv][6])
* Defines **five reward functions**:

  * XML structure (strict/soft), XML token-shape (count), integer-answer heuristic, and **exact-match correctness**.
* Runs **TRL’s GRPOTrainer** with **vLLM** to sample **multiple generations per prompt** (group) and train the policy using group-relative advantages + KL to a reference policy—**a GRPO hallmark**. ([Hugging Face][7])
* Saves the **LoRA**, evaluates before/after, merges to **16-bit**, and optionally **pushes to the Hub** for easy reuse.

> The overall recipe mirrors the GRPO-driven **reasoning** emphasis in DeepSeekMath and R1: online RL, multi-candidate sampling, group-relative baselining, and KL regularization to a reference policy.&#x20;

---

## 3) Environment & Dependencies

* **Unsloth** (fast LoRA/QLoRA fine-tuning), **vLLM** (fast generation), **TRL** (GRPO trainer), **Transformers**, **PEFT**, **bitsandbytes**. ([GitHub][8], [VLLM Documentation][9], [Hugging Face][7])
* Quantized LoRA (**QLoRA**) relies on 4-bit NF4 / double-quantization with **bitsandbytes**. ([arXiv][10], [Hugging Face][11])

> The notebook includes a Colab-aware installer and GPU checks.

---

## 4) Data & Prompting

* **GSM8K**: 8.5k grade-school math word problems; we parse the gold label after `####` and wrap prompts in a **system schema** enforcing `<reasoning>`/`<answer>`. ([arXiv][6], [Hugging Face][12])
* **Why XML schema?**

  * Easy to **parse & grade**.
  * Lets us **separate** CoT (**reasoning**) from the final **answer** for targeted rewards.

---

## 5) Rewards (Multi-Objective)

1. **XML structure (strict / soft)** — pushes completions to respect the schema.
2. **XML token-shape counter** — stabilizes early training by rewarding partial structural compliance.
3. **Integer-answer check** — cheap sanity for numeric tasks.
4. **Exact-match correctness** — aligns the `<answer>` text to the gold label.

> Combining structure + outcome signals is a lightweight proxy for outcome/process rewards that DeepSeek explored at scale; we keep it simple but **schema-aligned**.&#x20;

---

## 6) Training with GRPO (via TRL + vLLM)

* **Online RL**: generate multiple candidates per prompt with **vLLM**; compute **group-relative advantages**; update the policy while **regularizing to a reference policy** (KL). ([Hugging Face][7])
* **No critic/value model**: GRPO uses the **group mean** as a baseline, reducing memory vs PPO—exactly the DeepSeek idea.&#x20;
* **Config highlights:**

  * `num_generations` (group size), `max_completion_length`, conservative LR, cosine schedule, gradient clipping, checkpointing.

> For context on GRPO’s design and usage within TRL and vLLM, see the TRL GRPO docs and vLLM-TRL integration notes. ([Hugging Face][7], [VLLM Documentation][4])

---

## 7) Inference, A/B Checks, and Publishing

* **Before/after**: Query the model (“How many r’s in strawberry?”) pre-adapter vs. with the saved LoRA to verify the adapter actually moves behavior.
* **Save & share**:

  * Save **LoRA** (adapter-only) for PEFT workflows. ([Hugging Face][13])
  * **Merge to 16-bit** and push a single checkpoint for drop-in Transformers use.
* **Why both?** Adapters enable composition; merged weights simplify downstream deployment.

---

## 8) Reproducing the Notebook

1. **Install** (Colab/local): run the first cell to set up **Unsloth**, **vLLM**, **TRL**, etc. ([GitHub][8], [VLLM Documentation][9])
2. **Load model**: Qwen2.5-3B-Instruct with 4-bit loading + LoRA targets (Q/K/V/O, gate/up/down). ([Hugging Face][5])
3. **Load GSM8K** and map prompts to the XML schema with gold answers. ([Hugging Face][12])
4. **Define rewards** as in the code.
5. **Configure GRPO** (see the `GRPOConfig` cell) and **train**. ([Hugging Face][7])
6. **Run sanity inference**, **save LoRA**, **re-load LoRA for A/B**, then **merge** and **push**.

---

## 9) How This Echoes DeepSeek’s Training

* **Same RL family:** We use **GRPO** to improve reasoning—**the algorithm introduced by DeepSeek** for math reasoning and carried into **R1**’s reasoning pipeline.&#x20;
* **Online, multi-sample training:** Generate **groups** of outputs per prompt and compute **relative advantages**—a core GRPO idea.&#x20;
* **Reasoning-first objective:** While our rewards are simpler (structure + correctness), the spirit matches R1’s focus on **reinforcement-driven reasoning quality**. ([arXiv][1])

> Important: We do **not** claim parity with DeepSeek’s scale, datasets, or exact multi-stage recipes; this is a **lightweight, reproducible** adaptation of the **same RL principle** on a 3B model.

---

## 10) Evaluation Tips

* **Exact-match** on `<answer>` for GSM8K; optionally add **stop sequences** after `</answer>`.
* Track: **format adherence**, **accuracy**, and **length** of `<reasoning>`.
* For deeper rigor, compare with a pure **SFT** baseline and an **offline RFT/DPO** variant.&#x20;

---

## 11) Limitations & Ethics

* **Small scale:** 3B with GSM8K is great for demos but not state-of-the-art.
* **Rewards are brittle:** Exact-match favors formatting; consider numeric tolerance and unit handling.
* **Schema lock-in:** Over-constraining output can hurt fluency; keep a balance.
* **Safety:** Do not deploy for high-stakes decisions without thorough red-teaming and guardrails.

---

## 12) References & Further Reading

* **DeepSeek / GRPO / R1**

  * *DeepSeekMath*: Introduces **GRPO** (critic-free PPO variant) and shows reasoning gains.&#x20;
  * *DeepSeek-R1*: RL-centric reasoning training; R1-Zero trained **without SFT**, R1 uses **multi-stage + RL**. ([arXiv][1])
  * Plain-language overviews of **GRPO** (HF course; community explainers). ([Hugging Face][14], [Oxen.ai][15])
* **GRPO in TRL & vLLM**

  * TRL **GRPOTrainer** docs; **vLLM** for fast online sampling in RL. ([Hugging Face][7], [VLLM Documentation][4])
* **Models & Libraries**

  * **Qwen2.5-3B-Instruct** model card / blog. ([Hugging Face][5], [Qwen][16])
  * **Unsloth** (repo/wiki; 2× inference notes). ([GitHub][8])
  * **PEFT** docs and library. ([Hugging Face][13], [GitHub][17])
  * **vLLM** docs. ([VLLM Documentation][18])
* **Data / Quantization**

  * **GSM8K** paper + dataset. ([arXiv][6], [Hugging Face][12])
  * **LoRA** paper; **QLoRA** paper/blog. ([arXiv][2], [Hugging Face][19])

---

### Acknowledgements

* Thanks to the **DeepSeek** team for releasing work that popularized **GRPO** for reasoning.&#x20;
* Thanks to **Qwen**, **Hugging Face (Transformers, TRL, PEFT)**, **Unsloth**, **vLLM** maintainers, and the **GSM8K** authors for open resources. ([Hugging Face][5], [GitHub][8], [VLLM Documentation][18], [arXiv][6])


[1]: https://arxiv.org/abs/2501.12948 "[2501.12948] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
[2]: https://arxiv.org/abs/2106.09685?utm_source=chatgpt.com "LoRA: Low-Rank Adaptation of Large Language Models"
[3]: https://github.com/unslothai/unsloth/wiki/Home/f961aac2ad938b243fe5ed58d1c3f8a2c9b8f128?utm_source=chatgpt.com "Home · unslothai/unsloth Wiki · GitHub"
[4]: https://docs.vllm.ai/en/v0.9.1/training/trl.html?utm_source=chatgpt.com "Transformers Reinforcement Learning - vLLM"
[5]: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct?utm_source=chatgpt.com "Qwen/Qwen2.5-3B-Instruct"
[6]: https://arxiv.org/abs/2110.14168?utm_source=chatgpt.com "Training Verifiers to Solve Math Word Problems"
[7]: https://huggingface.co/docs/trl/main/en/grpo_trainer?utm_source=chatgpt.com "GRPO Trainer"
[8]: https://github.com/unslothai/unsloth?utm_source=chatgpt.com "unslothai/unsloth"
[9]: https://docs.vllm.ai/en/stable/index.html?utm_source=chatgpt.com "Welcome to vLLM"
[10]: https://arxiv.org/abs/2305.14314?utm_source=chatgpt.com "QLoRA: Efficient Finetuning of Quantized LLMs"
[11]: https://huggingface.co/docs/transformers/en/quantization/bitsandbytes?utm_source=chatgpt.com "Bitsandbytes"
[12]: https://huggingface.co/datasets/openai/gsm8k?utm_source=chatgpt.com "openai/gsm8k · Datasets at Hugging Face"
[13]: https://huggingface.co/docs/peft/en/index?utm_source=chatgpt.com "PEFT"
[14]: https://huggingface.co/learn/llm-course/en/chapter12/3?utm_source=chatgpt.com "Understanding the DeepSeek R1 Paper"
[15]: https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/?utm_source=chatgpt.com "Why GRPO is Important and How it Works"
[16]: https://qwenlm.github.io/blog/qwen2.5/?utm_source=chatgpt.com "Qwen2.5: A Party of Foundation Models! | Qwen"
[17]: https://github.com/huggingface/peft?utm_source=chatgpt.com "PEFT: State-of-the-art Parameter-Efficient Fine-Tuning."
[18]: https://docs.vllm.ai/?utm_source=chatgpt.com "vLLM"
[19]: https://huggingface.co/blog/4bit-transformers-bitsandbytes?utm_source=chatgpt.com "Making LLMs even more accessible with bitsandbytes, 4- ..."
