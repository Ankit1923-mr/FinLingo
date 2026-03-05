"""
Evaluation Module for FinLingo
Calculates ROUGE scores and tests model generations.
"""
import evaluate
import torch

def generate_answer(query, model, tokenizer):
    """Generates an answer from the fine-tuned model."""
    prompt = f"### Instruction: {query}\n### Response:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response.split("### Response:")[1].strip()

def run_evaluation(model, tokenizer):
    print("Running evaluation suite...")
    test_questions = ["What is a stock dividend?"]
    ground_truths = ["A stock dividend is a payment made by a corporation to its shareholders."]
    
    rouge = evaluate.load('rouge')
    predictions = [generate_answer(q, model, tokenizer) for q in test_questions]
    
    results = rouge.compute(predictions=predictions, references=ground_truths)
    print(f"ROUGE Metrics: {results}")

# Note: In a full pipeline, you would load the saved base model and your 
# saved adapter from '../finlingo_outputs/final_adapter' here before running eval.