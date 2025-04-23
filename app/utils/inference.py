from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.cuda.amp import autocast
from peft import PeftModel

# Load model and tokenizer globally (optional: wrap into a function if needed)
def load_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    
    return tokenizer, model, device

# Inference function
def generate_next_paragraph(input_paragraph, tokenizer, model, device):
    prompt = (
        "You are a creative writer. Read the paragraph below carefully and continue the story "
        "with a coherent and engaging next paragraph that maintains the same tone, style, and characters. "
        "Only write the next paragraph, and do not repeat or summarize the input.\n\n"
        f"Previous paragraph:\n{input_paragraph.strip()}\n\nNext paragraph:"
    )

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