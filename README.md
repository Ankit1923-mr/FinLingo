# FinLingo: Financial Instruction-Tuned LLM using QLoRA

![Comprehensive FinLingo Overview](assets\image-1.png)

FinLingo is a parameter-efficient fine-tuning (PEFT) project that adapts a large language model for financial instruction-following and question-answering tasks. 

By utilizing **QLoRA (Quantized Low-Rank Adaptation)** on a 4-bit quantized Llama-3 base model, this project demonstrates how to execute large language model training on highly constrained consumer-grade hardware, specifically an NVIDIA T4 GPU. The repository represents a production-style LLM training pipeline complete with modular code, reproducible experiments, hardware-specific optimizations, and evaluation metrics.

## 🎯 Project Objectives

* **Fine-tune** a large language model for domain-specific financial tasks.
* **Demonstrate** QLoRA training on limited consumer GPU hardware (NVIDIA T4).
* **Build** a clean, modular ML engineering pipeline.
* **Implement** evaluation pipelines using ROUGE metrics.
* **Track** experimental configurations and model runs using Weights & Biases (W&B).

---

## 🏗️ Project Architecture

```text
01-finlingo/
│
├── data/
│   └── prepare_dataset.py
│
├── training/
│   └── train_qlora.py
│
├── evaluation/
│   └── evaluate_model.py
│
├── processed_data/
│
├── finlingo_outputs/
│
├── requirements.txt
│
└── README.md


```

## 📊 Dataset & Prompt Formatting

This project uses the [Finance Alpaca Dataset](https://huggingface.co/datasets/gbharti/finance-alpaca), which contains thousands of financial instruction-response pairs designed specifically for training and fine-tuning LLMs.

### Example Prompt Transformation

The dataset is transformed into a strict instruction-following format to ensure compatibility with instruction-tuned LLM architectures:


``` text 
### Instruction:

What is a stock dividend?

### Input:

<context if available>

### Response:

A stock dividend is a payment made by a corporation to its shareholders.

```

* * * * *

⚙️ Installation & Requirements

------------------------------

Clone the repository and install the dependencies:

Bash

```

git clone [https://github.com/Ankit1923-mr/FinLingo.git](https://github.com/Ankit1923-mr/FinLingo.git)

cd 01-finlingo

pip install -r requirements.txt

```

### Dependencies

-   `transformers==4.40.0`

-   `peft==0.10.0`

-   `bitsandbytes==0.43.1`

-   `datasets==2.19.0`

-   `trl==0.8.6`

-   `wandb`, `accelerate`, `scipy`, `evaluate`, `rouge_score`, `matplotlib`

* * * * *

🛠️ Pipeline Execution
----------------------

### 1\. Data Preparation

Run the dataset preparation script to download, format, filter, and store the dataset locally.

Bash

```
python data/prepare_dataset.py

```

*Note: The processed dataset is stored in `processed_data/finlingo_train`. Saving the dataset locally ensures that training scripts do not bottleneck on repeated network requests.*

### 2\. Model Training (QLoRA)

Execute the main training loop:

Bash

```
python training/train_qlora.py

```

#### Training Configuration

| **Parameter** | **Value** |
| --- | --- |
| **Base Model** | Llama-3 8B |
| **Quantization** | 4-bit NF4 |
| **Adapter Method** | LoRA |
| **LoRA Rank** | 8 |
| **Batch Size** | 2 |
| **Gradient Accumulation** | 2 |
| **Optimizer** | paged_adamw_8bit |
| **Max Steps** | 40 |

### 3. Model Evaluation
Evaluate the fine-tuned adapter against test queries, calculating ROUGE metrics.
```bash
python evaluation/evaluate_model.py

```

**Example Output:**

> **Instruction:** What is a stock dividend?
>
> **Generated Output:** A stock dividend is a distribution of additional shares issued by a corporation to its shareholders instead of cash payments.

🧠 Why QLoRA?
-------------

Training an 8B parameter model in full precision requires tens of gigabytes of GPU memory. QLoRA solves this hardware bottleneck by combining:

1.  **4-bit Quantization:** The base model weights are compressed to 4-bit precision, reducing memory usage dramatically.

2.  **LoRA Adapters:** Instead of training the full model, only small, low-rank matrices are trained.

3.  **Frozen Base Model:** The original model parameters remain frozen, drastically reducing computational cost and allowing training on a single T4 GPU.

* * * * *

📈 LoRA Rank Experiments: Memory vs. Performance
------------------------------------------------

To determine the optimal system tradeoff between resource utilization and output quality, experiments were run across different LoRA ranks (`r`).

| **LoRA Rank (r)** | **Training Loss (Epoch 2)** | **Peak GPU Memory** | **ROUGE-L Impact** |
| --- | --- | --- | --- |
| **4** | 2.655730 | 3.67 GB | Baseline |
| **8** | 2.652314 | 3.67 GB | Minimal change |
| **16** | 2.645862 | 3.68 GB | Minimal change |

**Conclusion:** Increasing the LoRA rank from 4 to 16 slightly decreased training loss (2.655 down to 2.645) with a negligible peak memory increase (~10MB difference). Output quality (ROUGE-L score) remained highly stable across all ranks, indicating that a rank of 8 is a highly efficient sweet spot for this specific financial instruction dataset.

🧠 Why QLoRA?
-------------

Training an 8B parameter model in full precision requires tens of gigabytes of GPU memory. QLoRA solves this hardware bottleneck by combining:

1.  **4-bit Quantization:** The base model weights are compressed to 4-bit precision, reducing memory usage dramatically.

2.  **LoRA Adapters:** Instead of training the full model, only small, low-rank matrices are trained.

3.  **Frozen Base Model:** The original model parameters remain frozen, drastically reducing computational cost and allowing training on a single T4 GPU.

* * * * *

📈 LoRA Rank Experiments: Memory vs. Performance
------------------------------------------------

To determine the optimal system tradeoff between resource utilization and output quality, experiments were run across different LoRA ranks (`r`).

| **LoRA Rank (r)** | **Training Loss (Epoch 2)** | **Peak GPU Memory** | **ROUGE-L Impact** |
| --- | --- | --- | --- |
| **4** | 2.655730 | 3.67 GB | Baseline |
| **8** | 2.652314 | 3.67 GB | Minimal change |
| **16** | 2.645862 | 3.68 GB | Minimal change |

**Conclusion:** Increasing the LoRA rank from 4 to 16 slightly decreased training loss (2.655 down to 2.645) with a negligible peak memory increase (~10MB difference). Output quality (ROUGE-L score) remained highly stable across all ranks, indicating that a rank of 8 is a highly efficient sweet spot for this specific financial instruction dataset.

🐛 Handling Hardware Constraints: The NVIDIA T4 `bfloat16` Bug
--------------------------------------------------------------

During the QLoRA training initialization, a critical hardware compatibility issue specific to the **NVIDIA T4 GPU** was encountered.

**The Problem:** The base Llama-3 model expects `bfloat16` precision, which is natively supported on newer GPU architectures (Ampere and later). However, the NVIDIA T4 utilizes the Turing architecture, which lacks native hardware support for `bfloat16` gradient scaling. Consequently, when PyTorch attempted mixed-precision training during backpropagation, the loop failed with a `NotImplementedError` originating from the PyTorch `GradScaler`.

**The Solution:** To resolve this, a custom parameter initialization loop was implemented during model setup.

-   The loop iterates through the PEFT model parameters and explicitly casts **only the trainable LoRA adapter tensors** from `bfloat16` to `float32`.

-   Because QLoRA drastically reduces the trainable parameter footprint (approx. 6 million parameters vs. the 8 billion base parameters), computing gradients in full 32-bit precision introduced virtually zero VRAM overhead.

-   This bypassed the T4's mixed-precision hardware limitation completely, stabilizing the training loop and enabling successful fine-tuning on a single consumer GPU.

## 🚀 Key Technologies
* **PyTorch** & **HuggingFace Transformers**
* **PEFT** (Parameter-Efficient Fine-Tuning)
* **QLoRA** & **BitsAndBytes**
* **TRL** (Transformer Reinforcement Learning)
* **Weights & Biases (W&B)**

---

## 🔮 Future Improvements
* [ ] Train on larger, more complex financial datasets (e.g., earnings call transcripts).
* [ ] Integrate **Retrieval-Augmented Generation (RAG)** to ground answers in real-time market data.
* [ ] Add formal financial reasoning benchmarks (e.g., FinQA).
* [ ] Deploy the fine-tuned model as a microservice using **FastAPI** or HuggingFace Inference Endpoints.

---

## 👨‍💻 Author
**Ankit Kumar** GitHub: [@Ankit1923-mr](https://github.com/Ankit1923-mr)