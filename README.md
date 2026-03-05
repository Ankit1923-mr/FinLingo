FinLingo

========



### Financial Instruction-Tuned LLM using QLoRA



FinLingo is a **parameter-efficient fine-tuning project** that adapts a large language model for **financial instruction-following and question answering**.



The model is trained using **QLoRA (Quantized Low Rank Adaptation)** on a **4-bit quantized Llama-3 base model**, allowing large language model training on **limited GPU resources such as an NVIDIA T4**.



The goal of this project is to demonstrate a **production-style LLM training pipeline** with modular code, reproducible experiments, and evaluation metrics.



* * * * *



Project Objectives

==================



-   Fine-tune a large language model for **financial domain tasks**



-   Demonstrate **QLoRA training on consumer GPU hardware**



-   Build a **clean, modular ML engineering pipeline**



-   Implement **evaluation using ROUGE metrics**



-   Track experiments using **Weights & Biases**



* * * * *



Project Architecture

====================



finlingo/

│

├── data/

│   └── prepare_dataset.py

│

├── training/

│   └── train_qlora.py

│

├── evaluation/

│   └── evaluate_model.py

│

├── processed_data/

│

├── finlingo_outputs/

│

├── requirements.txt

│

└── README.md



* * * * *



Dataset

=======



This project uses the **Finance Alpaca Dataset**:



<https://huggingface.co/datasets/gbharti/finance-alpaca>



The dataset contains financial instruction-response pairs designed for training LLMs.



Example:



Instruction: What is a stock dividend?\

Response: A stock dividend is a payment made by a corporation to its shareholders.



* * * * *



Prompt Formatting

=================



The dataset is transformed into a strict **instruction-following format**:



### Instruction:\

<Question>



### Input:\

<context if available>



### Response:\

<answer>



This formatting ensures compatibility with **instruction-tuned LLM architectures**.



* * * * *



Installation

============



Clone the repository:



git clone https://github.com/Ankit1923-mr/01-finlingo.git\

cd 01-finlingo



Install dependencies:



pip install -r requirements.txt



* * * * *



Requirements

============



transformers==4.40.0\

peft==0.10.0\

bitsandbytes==0.43.1\

datasets==2.19.0\

trl==0.8.6\

wandb\

accelerate\

scipy\

evaluate\

rouge_score\

matplotlib



* * * * *



Data Preparation

================



Run the dataset preparation script:



python data/prepare_dataset.py



This script performs:



1.  Dataset download



2.  Prompt formatting



3.  Data filtering



4.  Local dataset storage



The processed dataset is stored in:



processed_data/finlingo_train



Saving the dataset locally ensures that **training scripts do not repeatedly download the dataset**.



* * * * *



Model Training (QLoRA)

======================



Training is implemented in:



training/train_qlora.py



Run training using:



python training/train_qlora.py



### Training Configuration



| Parameter | Value |

| --- | --- |

| Base Model | Llama-3 8B |

| Quantization | 4-bit NF4 |

| Adapter Method | LoRA |

| LoRA Rank | 8 |

| Batch Size | 2 |

| Gradient Accumulation | 2 |

| Optimizer | paged_adamw_8bit |

| Max Steps | 40 |

| GPU | NVIDIA T4 |



* * * * *



Why QLoRA?

==========



Training an 8B parameter model normally requires **tens of gigabytes of GPU memory**.



QLoRA solves this by combining:



### 4-bit Quantization



The base model weights are compressed to **4-bit precision**, reducing memory usage dramatically.



### LoRA Adapters



Instead of training the full model, **only small low-rank matrices are trained**.



### Frozen Base Model



The original model parameters remain frozen, drastically reducing computation cost.



This allows training **large models on a single T4 GPU**.



* * * * *



Experiment Tracking

===================



Training experiments are logged using **Weights & Biases**.



Metrics tracked include:



-   training loss



-   step progress



-   learning rate



Initialization occurs automatically during training:



wandb.init(project="finlingo-experiment")



* * * * *



Model Evaluation

================



Evaluation is implemented in:



evaluation/evaluate_model.py



Run evaluation:



python evaluation/evaluate_model.py



The evaluation pipeline:



1.  Loads the trained LoRA adapter



2.  Generates responses for test queries



3.  Calculates **ROUGE metrics**



Example test query:



What is a stock dividend?



* * * * *



Example Model Output

====================



Prompt:



### Instruction:\

What is a stock dividend?



### Response:



Generated Output:



A stock dividend is a distribution of additional shares issued by a corporation to its shareholders instead of cash payments.



* * * * *



Handling Hardware Constraints & Architecture Bugs

=================================================



During the QLoRA training initialization, I encountered a critical **hardware compatibility issue specific to the NVIDIA T4 GPU**.



The base **Llama-3 model expects `bfloat16` precision**, which is commonly supported on newer GPU architectures (such as Ampere and later). However, the **NVIDIA T4 GPU uses the Turing architecture**, which **does not provide native hardware support for `bfloat16` gradient scaling**.



Because of this limitation, when PyTorch attempted to perform mixed-precision training during backpropagation, the training loop failed with a **`NotImplementedError` originating from PyTorch's `GradScaler`**, which could not execute gradient operations on `bfloat16` tensors on the T4.



The Solution

------------



To resolve the issue, I implemented a **custom parameter initialization loop** during model setup.



This loop iterates through the PEFT model parameters and **explicitly casts only the trainable LoRA adapter tensors from `bfloat16` to `float32`**.



Since **QLoRA dramatically reduces the number of trainable parameters**, the LoRA adapters contained **approximately 6 million parameters** compared to the **8 billion parameters in the base model**.



Because of this drastic reduction, **computing gradients in full 32-bit precision introduced negligible VRAM overhead**, while completely bypassing the T4's mixed-precision hardware limitation.



This modification **stabilized the training loop and allowed QLoRA fine-tuning to run successfully on a single NVIDIA T4 GPU**.



* * * * *



Key Technologies

================



-   PyTorch



-   HuggingFace Transformers



-   PEFT (Parameter Efficient Fine Tuning)



-   QLoRA



-   BitsAndBytes



-   TRL (Transformer Reinforcement Learning)



-   Weights & Biases



* * * * *



Future Improvements

===================



Possible extensions of this project include:



-   Training on larger financial datasets



-   Increasing LoRA rank for improved performance



-   Integrating **Retrieval Augmented Generation (RAG)**



-   Adding financial reasoning benchmarks



-   Deploying the model using **FastAPI or HuggingFace Inference API**



* * * * *



Author

======



**Ankit Kumar**



GitHub:\

<https://github.com/Ankit1923-mr>



