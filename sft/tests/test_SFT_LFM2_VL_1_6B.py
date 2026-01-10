import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

base_model_id = "LiquidAI/LFM2.5-VL-1.6B"
adapter_path = "./models/lfm25-adapter" 
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"🚀 Chargement sur : {device}")

processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True,size={"height": 336, "width": 336})

# Chargement du modèle de base + adapteur lora
# Note: utiliser float32 sur MPS si float16 pose problème
base_model = AutoModelForImageTextToText.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    trust_remote_code=True
).to(device)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# Préparation de l'image
image_path = "./data/lines/page_031_line_003.png"
image = Image.open(image_path).convert("RGB")

# Formatage STRICT (exactement comme dans le fichier d'entraînement)
system_message = (
    "Tu es un système OCR spécialisé dans la transcription de manuscrits français anciens. "
    "Transcris fidèlement le texte manuscrit visible dans l'image."
)

conversation = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Transcris ce texte manuscrit."},
    ]},
]

text = processor.tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True
)

inputs = processor(
    text=text,
    images=[image],
    return_tensors="pt",
)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=2048, 
        do_sample=False, # on reste déterministe
        num_beams=3,     # aide le modèle à ne pas choisir le premier mot venu
        # repetition_penalty=1.2 # évite les répétitions
    )

result = processor.batch_decode(
    outputs[:, inputs["input_ids"].shape[1]:], 
    skip_special_tokens=True
)[0]

print("-" * 30)
print(image_path)
print(result)
print("-" * 30)