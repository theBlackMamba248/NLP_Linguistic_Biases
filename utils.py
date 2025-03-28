import torch
import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from datasets import load_metric

# Download NLTK data

nltk.download('punkt')

bleu_metric = load_metric("bleu")

# --------------------------------------------------
# 1. Batch-Optimized Utils (utils.py)
# --------------------------------------------------
def load_model_and_tokenizer(model_name="gpt2", device="cuda"):
    """Load model with FP16 support"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()
    return model, tokenizer, device
    
def compute_bleu(reference, candidate):
   
    results = bleu_metric.compute(
        predictions=[candidate.split()],
        references=[[reference.split()]]
        )
    return results["bleu"
        
def batch_log_likelihood(texts, model, tokenizer, device, batch_size=8):
    """Compute log-likelihood for a batch of texts"""
    input_ids = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=128
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**input_ids, labels=input_ids["input_ids"])
    
    # Calculate per-sequence loss (adjust for padding)
    losses = -outputs.loss * torch.sum(input_ids["attention_mask"], dim=1)
    return losses.tolist()

def batch_generate(prompts, model, tokenizer, device, batch_size=8, max_new_tokens=50):
    """Generate outputs for a batch of prompts"""
    outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(device)
        
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return outputs

def batch_bertscore(references, hypotheses, lang="hi", batch_size=16):
    """Compute BERTScore in batches"""
    scores = []
    for i in tqdm(range(0, len(references), batch_size), desc="BERTScore"):
        batch_refs = references[i:i+batch_size]
        batch_hyps = hypotheses[i:i+batch_size]
        _, _, F1 = bert_score(batch_hyps, batch_refs, lang=lang)
        scores.extend(F1.tolist())
    return scores

# --------------------------------------------------
# 2. Main Analysis Pipeline
# --------------------------------------------------
def analyze_scenarios(scenarios, model, tokenizer, device, batch_size=8):
    results = []
    
    # Process English texts (generate Hindi outputs)
    en_texts = [text for category in scenarios["English"].values() for text in category]
    hi_outputs = batch_generate(en_texts, model, tokenizer, device, batch_size)
    
    # Get Hindi references (aligned with English prompts)
    hi_refs = [
        scenarios["Hindi"][category][i % len(scenarios["Hindi"][category])] 
        for i, category in enumerate(
            [cat for cat in scenarios["English"] for _ in scenarios["English"][cat]]
        )
    ]
    
    # Batch compute metrics
    en_lls = batch_log_likelihood(en_texts, model, tokenizer, device, batch_size)
    hi_lls = batch_log_likelihood(hi_refs, model, tokenizer, device, batch_size)
    bleu_scores = [
        sentence_bleu([ref.split()], hyp.split(), smoothing_function=SmoothingFunction().method1)
        for ref, hyp in zip(hi_refs, hi_outputs)
    ]
    bert_scores = batch_bertscore(hi_refs, hi_outputs)
    
    # Compile results
    idx = 0
    for category in scenarios["English"]:
        for en_text, hi_output, hi_ref in zip(
            scenarios["English"][category],
            hi_outputs[idx:idx+len(scenarios["English"][category])], 
            hi_refs[idx:idx+len(scenarios["English"][category])]
        ):
            results.append({
                "category": category,
                "en_text": en_text,
                "hi_output": hi_output,
                "hi_reference": hi_ref,
                "en_ll": en_lls[idx],
                "hi_ll": hi_lls[idx],
                "bleu": bleu_scores[idx],
                "bert_f1": bert_scores[idx],
                "bias_score": hi_lls[idx] - en_lls[idx]  # Higher = more bias
            })
            idx += 1
    
    return results

# --------------------------------------------------
# 3. Execute Pipeline
# --------------------------------------------------
def main():
    # Load model
    model, tokenizer, device = load_model_and_tokenizer("gpt2")
    print(f"Model loaded on {device}")
    
    # Load data
    with open("scenarios.json", "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    
    # Run analysis
    results = analyze_scenarios(scenarios, model, tokenizer, device, batch_size=8)
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("batch_processed_results.csv", index=False)
    print(f"Saved results for {len(df)} samples")

if __name__ == "__main__":
    main()
