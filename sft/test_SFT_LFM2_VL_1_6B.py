# GOOGLE COLAB GPU T4
# VOIR LE FICHIER test_LFM2_VL_1_6B.ipynb

from transformers import pipeline

MODEL_PATH = "./lfm2-manuscrit"  # ou "/content/drive/MyDrive/lfm2-manuscrit"
IMAGE_PATH = "/content/page_011.png"

pipe = pipeline("image-text-to-text", model=MODEL_PATH)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": IMAGE_PATH},
            {"type": "text", "text": "Transcris ce texte manuscrit."}
        ]
    },
]

result = pipe(text=messages)
print(result)