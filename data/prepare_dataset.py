"""
Data Preparation Module for FinLingo
Downloads, formats, and cleans the finance-alpaca dataset.
"""
from datasets import load_dataset
import os

def format_prompt(sample):
    """Converts raw JSON into a strict instruction-following string."""
    if sample['input']:
        text = f"### Instruction: {sample['instruction']}\n### Input: {sample['input']}\n### Response: {sample['output']}"
    else:
        text = f"### Instruction: {sample['instruction']}\n### Response: {sample['output']}"
    sample['formatted_text'] = text
    return sample

def main():
    print("Loading raw dataset...")
    dataset = load_dataset("gbharti/finance-alpaca", split="train")
    
    print("Formatting prompts...")
    formatted_dataset = dataset.map(format_prompt)
    
    print("Filtering garbage data...")
    filtered_dataset = formatted_dataset.filter(
        lambda x: x['output'] is not None and len(x['output']) >= 10
    )
    
    print(f"Final dataset size: {len(filtered_dataset)} samples.")
    
    # Save locally so the training script can use it
    os.makedirs("./processed_data", exist_ok=True)
    filtered_dataset.save_to_disk("./processed_data/finlingo_train")
    print("Dataset saved to ./processed_data/finlingo_train")

if __name__ == "__main__":
    main()