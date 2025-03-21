import json
from utils import load_model_and_tokenizer, compute_log_likelihood, compute_bias_factor

def load_scenarios(filename="scenarios.json"):
    """
    Loads the generated scenarios from a JSON file.
    """
    with open(filename, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    return scenarios

def analyze_scenarios(scenarios, language, model, tokenizer, device):
    """
    Computes log likelihood scores for each scenario text and prints the results.
    """

    print(f"\n----- Analyzing scenarios in {language} -----")
    results = {}
    for scenario, texts in scenarios.items():
        print(f"Scenario: {scenario}")
        scenario_results = []
        for text in texts:
            ll = compute_log_likelihood(text, model, tokenizer, device)
            print(f"Text: {text}\nLog Likelihood: {ll:.2f}\n")
            scenario_results.append(ll)
        results[scenario] = scenario_results
        print("-" * 50)

    for text in texts:
        ll = compute_log_likelihood(text, model, tokenizer, device)
        beta = compute_bias_factor(text, tokenizer)
        adjusted_ll = ll - beta  # B(s_i^L) = LL - β
        print(f"Text: {text}\nRaw LL: {ll:.2f}, β: {beta:.2f}, Adjusted LL: {adjusted_ll:.2f}\n")
        scenario_results.append(adjusted_ll)
    return results

def main():
    model, tokenizer, device = load_model_and_tokenizer()
    scenarios = load_scenarios("scenarios.json")
    
    # Analyze English scenarios.
    print("Analyzing English scenarios:")
    english_results = analyze_scenarios(scenarios["English"], "English", model, tokenizer, device)
    
    # Analyze Hindi scenarios.
    print("Analyzing Hindi scenarios:")
    hindi_results = analyze_scenarios(scenarios["Hindi"], "Hindi", model, tokenizer, device)
    
    # Optional: Summary comparison of corresponding scenarios.
    print("\n--- Summary Comparison ---")
    mapping = {"Hotel": "होटल", "Restaurant": "रेस्तरां", "Academic": "अकादमिक"}
    for eng_key, hin_key in mapping.items():
        eng_values = english_results.get(eng_key, [])
        hin_values = hindi_results.get(hin_key, [])
        if eng_values and hin_values:
            print(f"Scenario: {eng_key} vs {hin_key}")
            for i, (eng_val, hin_val) in enumerate(zip(eng_values, hin_values)):
                print(f"Example {i+1}: English LL = {eng_val:.2f}, Hindi LL = {hin_val:.2f}")
            print("-" * 50)

if __name__ == "__main__":
    main()
