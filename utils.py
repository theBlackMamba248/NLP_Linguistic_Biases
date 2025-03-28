import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_metric, evaluate
import numpy as np

# Initialize BLEU metric
try:
    # New way (preferred)
    bleu_metric = evaluate.load("bleu")
except:
    # Fallback to old way
    bleu_metric = load_metric("bleu")

def load_model_and_tokenizer(model_name="gpt2", offload_folder="./offload"):
    """Loads model and tokenizer"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device

def compute_bleu(reference, candidate):
    """Computes BLEU score between texts"""
    results = bleu_metric.compute(
        predictions=[candidate.split()],
        references=[[reference.split()]]
    )
    return results["bleu"]

def compute_log_likelihood(text, model, tokenizer, device):
    """Computes log likelihood for text"""
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return -loss.item() * input_ids.size(1)

def compute_bias_factor(text, tokenizer):
    """Calculates tokenization error rate"""
    tokens = tokenizer.tokenize(text)
    if not tokens:
        return 0.0
    split_errors = sum(1 for token in tokens[1:] if not token.startswith("Ġ"))
    return split_errors / len(tokens)
