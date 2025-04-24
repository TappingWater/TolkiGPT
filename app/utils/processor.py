import torch
from .model import get_gen_model, get_gen_tokenizer
import app.config as config
from torch.cuda.amp import autocast

def generate_paragraph(text):
    prompt = f"Generate the next paragraph for this story based by continuing the text from the previous paragraph: '''{text.strip()}'''"
    tokenizer = get_gen_tokenizer()
    model = get_gen_model()
    device = config.DEFAULT_INFERENCE_DEVICE
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        with autocast(device):
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
