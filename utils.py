import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# def load_model_and_tokenizer(model_name="EleutherAI/gpt-j-6B", offload_folder="./offload"):
#     """
#     Loads the model and tokenizer from Hugging Face using the given model name.
#     The model is loaded on GPU if available, with torch.float16 for efficiency.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         device_map="auto",
#         torch_dtype=torch.float16,
#         offload_folder=offload_folder
#     )
#     model.eval()
#     return model, tokenizer, device
def load_model_and_tokenizer(model_name="gpt2", offload_folder="./offload"):
    """
    Loads the model and tokenizer from Hugging Face using the given model name.
    The model is loaded on GPU if available, with torch.float16 for efficiency.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device

def compute_log_likelihood(text, model, tokenizer, device):
    """
    Computes the total log likelihood for a given text.
    """
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    total_ll = -loss.item() * input_ids.size(1)
    return total_ll

def compute_bias_factor(text, tokenizer):
    """
    Computes tokenization errors for Hindi text by counting subword splits.
    Split indicator: Tokens *without* "Ġ" (except the first token).
    """
    tokens = tokenizer.tokenize(text)
    if not tokens:
        return 0.0
    split_errors = 0
    for i in range(1, len(tokens)):  # Skip first token
        if not tokens[i].startswith("Ġ"):
            split_errors += 1
    return split_errors / len(tokens)  # Normalize by total tokens

def clean_generated_text(prompt, generated_text):
    """
    Cleans the generated text by removing the prompt (or parts of it) if present.
    """
    # If the output starts with the prompt, remove it.
    if generated_text.startswith(prompt):
        return generated_text[len(prompt):].strip()
    # Otherwise, if the output contains "Answer:", return only the answer part.
    if "Answer:" in generated_text:
        return generated_text.split("Answer:")[-1].strip()
    return generated_text.strip()

def generate_text(prompt, model, tokenizer, device, max_new_tokens=500, num_return_sequences=2, clean_prompt=True):
    """
    Generates texts given a prompt using the model.
    If clean_prompt is True, the output is post-processed to remove the prompt text.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,  # Generate new tokens beyond the prompt.
        attention_mask=attention_mask,
        do_sample=True,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id
    )
    texts = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
    if clean_prompt:
        texts = [clean_generated_text(prompt, text) for text in texts]
    return texts
