from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.cuda.amp import autocast

# Load model and tokenizer globally (optional: wrap into a function if needed)
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    model = torch.compile(model)
    return tokenizer, model, device

# Inference function
def generate_next_paragraph(input_paragraph, tokenizer, model, device):
    prompt = f"Continue the story with a new paragraph based on the following content:\n{input_paragraph.strip()}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        with autocast():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                no_repeat_ngram_size=3,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True
            )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    next_paragraph = output_text.replace(prompt.strip(), "").strip()

    # Make sure the output is a full sentence ending with proper punctuation
    if not next_paragraph.endswith(('.', '!', '?')):
        next_paragraph = next_paragraph.rsplit('.', 1)[0] + '.'

    return next_paragraph