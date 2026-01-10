from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
from PIL import Image

# Test avec modèle de base (non fine-tuné)
processor = AutoProcessor.from_pretrained("LiquidAI/LFM2.5-VL-1.6B", trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    "LiquidAI/LFM2.5-VL-1.6B",
    torch_dtype=torch.float16,
    trust_remote_code=True
).to("mps")

image = Image.open("./data/lines/page_029_line_024.png").convert("RGB")

conversation = [
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Transcris ce texte manuscrit."},
    ]},
]

inputs = processor.apply_chat_template([conversation], add_generation_prompt=True, return_tensors="pt", return_dict=True, tokenize=True)
inputs = {k: v.to("mps") for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=False)

result = processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
print(result)