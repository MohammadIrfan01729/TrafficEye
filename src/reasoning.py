from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate_reasoning(violations, plate_text):

    violation_text = ", ".join(violations)

    if plate_text == "No Plate":
        plate_info = "No number plate is visible. Manual verification is required."
    elif plate_text == "Unreadable":
        plate_info = "The number plate is detected but unreadable."
    else:
        plate_info = f"The detected number plate is {plate_text}."

    prompt = f"""
A traffic monitoring system detected: {violation_text}.
{plate_info}
Explain clearly in one short sentence.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.7
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response.strip()