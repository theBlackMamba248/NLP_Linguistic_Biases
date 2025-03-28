import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_metric
import numpy as np

# Initialize BLEU metric
try:
    bleu_metric = load_metric("bleu")
except:
    bleu_metric = None

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
    if bleu_metric is None:
        return 0.0  # Fallback if metric couldn't load
    try:
        results = bleu_metric.compute(
            predictions=[candidate.split()],
            references=[[reference.split()]]
        )
        return results.get("bleu", 0.0)
    except:
        return 0.0

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
