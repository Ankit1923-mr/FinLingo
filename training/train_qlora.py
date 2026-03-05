"""
QLoRA Training Module for FinLingo
Trains a 4-bit quantized Llama-3 model using LoRA adapters on a T4 GPU.
"""
import torch
import wandb
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def main():
    # 1. Load the processed data
    print("Loading processed dataset...")
    dataset = load_from_disk("../processed_data/finlingo_train")
    tiny_dataset = dataset.select(range(10)) # Sanity check subset

    # 2. Configure Hardware Data Types (The T4 Fixes)
    compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype 
    )

    model_id = "NousResearch/Meta-Llama-3-8B"

    # 3. Load Tokenizer & Model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token 

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype 
    )

    # 4. Attach LoRA Adapters
    print("Configuring LoRA...")
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    peft_model = get_peft_model(model, lora_config)

    # 5. The Nuclear Hardware Fix
    for name, param in peft_model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float32)

    # 6. Initialize Trainer
    wandb.init(project="finlingo-experiment", name="sanity-check-script")

    sft_config = SFTConfig(
        output_dir="../finlingo_outputs",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        max_steps=40,
        logging_steps=5,
        report_to="wandb",
        fp16=False,  # Bypass GradScaler bug on T4
        bf16=False,
        dataset_text_field="formatted_text" 
    )

    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=tiny_dataset,
        args=sft_config
    )

    print("Starting training...")
    trainer.train()
    
    # Save the final adapter weights
    trainer.save_model("../finlingo_outputs/final_adapter")
    print("Training complete. Adapters saved.")

if __name__ == "__main__":
    main()