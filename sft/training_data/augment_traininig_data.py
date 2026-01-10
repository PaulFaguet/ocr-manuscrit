import json
import cv2
import numpy as np
from pathlib import Path
import random

INPUT_JSONL = Path("./data/transcription.jsonl")
OUTPUT_JSONL = Path("./data/transcription_augmented.jsonl")
LINES_DIR = Path("./data/lines")
AUG_DIR = Path("./data/lines_augmented")


def augment_image(img, aug_type: str):
    """Applique une augmentation à l'image."""
    h, w = img.shape[:2]
    
    if aug_type == "rotation_left":
        M = cv2.getRotationMatrix2D((w/2, h/2), 1.5, 1)
        return cv2.warpAffine(img, M, (w, h), borderValue=255)
    
    elif aug_type == "rotation_right":
        M = cv2.getRotationMatrix2D((w/2, h/2), -1.5, 1)
        return cv2.warpAffine(img, M, (w, h), borderValue=255)
    
    elif aug_type == "brightness_up":
        return cv2.convertScaleAbs(img, alpha=1.1, beta=10)
    
    elif aug_type == "brightness_down":
        return cv2.convertScaleAbs(img, alpha=0.9, beta=-10)
    
    elif aug_type == "noise":
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        return cv2.add(img, noise)
    
    return img


def run_augmentation():
    AUG_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        original_data = [json.loads(line) for line in f if line.strip()]
        
    augmented_data = []
    augmentations = ["rotation_left", "rotation_right", "brightness_up", "brightness_down"]
    
    for item in original_data:
        img_name = item["image"]
        img_path = LINES_DIR / img_name
        
        if not img_path.exists():
            print(f"  Image manquante: {img_name}")
            continue
        
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        # Copie l'original dans le dossier augmenté
        cv2.imwrite(str(AUG_DIR / img_name), img)
        augmented_data.append(item)
        
        # 2 augmentations aléatoires par image
        for aug_type in random.sample(augmentations, 2):
            aug_img = augment_image(img, aug_type)
            
            stem = Path(img_name).stem
            aug_name = f"{stem}_{aug_type}.png"
            
            cv2.imwrite(str(AUG_DIR / aug_name), aug_img)
            
            augmented_data.append({
                "image": aug_name,
                "text": item["text"]
            })
    
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for item in augmented_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_augmentation()