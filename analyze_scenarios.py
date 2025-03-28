import json
import pandas as pd
from tqdm import tqdm
from utils import (
    load_model_and_tokenizer,
    batch_log_likelihood,
    batch_generate,
    batch_bertscore,
    compute_bleu,
    compute_bias_factor
)

def analyze_scenarios(scenarios, model, tokenizer, device, batch_size=8):
    results = []
    
    # Process all English texts (generate Hindi outputs)
    en_texts = [text for category in scenarios["English"].values() for text in category]
    hi_outputs = batch_generate(en_texts, model, tokenizer, device, batch_size)
    
    # Get aligned Hindi references
    hi_refs = [
        scenarios["Hindi"][category][i % len(scenarios["Hindi"][category])]
        for i, category in enumerate([cat for cat in scenarios["English"] 
                                   for _ in scenarios["English"][cat]])
    ]
    
    # Batch compute all metrics
    en_lls = batch_log_likelihood(en_texts, model, tokenizer, device, batch_size)
    hi_lls = batch_log_likelihood(hi_refs, model, tokenizer, device, batch_size)
    bert_scores = batch_bertscore(hi_refs, hi_outputs, lang="hi", batch_size=batch_size)
    
    # Compile results with progress tracking
    idx = 0
    for category in tqdm(scenarios["English"], desc="Processing categories"):
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
                "bleu": compute_bleu(hi_ref, hi_output),
                "bert_f1": bert_scores[idx],
                "bias_score": compute_bias_factor(hi_output, tokenizer)
            })
            idx += 1
    
    return pd.DataFrame(results)

def main():
    model, tokenizer, device = load_model_and_tokenizer("gpt2")
    with open("scenarios.json", "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    
    results = analyze_scenarios(scenarios, model, tokenizer, device, batch_size=8)
    results.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()
