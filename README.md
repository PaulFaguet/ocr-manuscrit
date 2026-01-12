# 📜 OCR Manuscrit - Mémoires de Guerre 1914-1918

Digitalisation d'un manuscrit familial de ~500 pages en utilisant un Vision-Language Model fine-tuné sur l'écriture spécifique de l'auteur.

## 🛠 Stack technique

| Composant | Technologie |
|-----------|-------------|
| Modèle | [LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B) (Liquid AI) |
| Fine-tuning | LoRA + TRL (SFTTrainer) |
| Segmentation | OpenCV (projection horizontale) |
| Conversion PDF | pdf2image + Poppler |
| Training | Google Colab (A100) |

## 📁 Structure du projet

```
├── sft/
│   ├── SFT_LFM2_5_VL_1_6B.ipynb              # Notebook d'entraînement (Colab)
│   ├── training_data/
│   │   ├── extract_training_data.py          # Segmentation pages → lignes
│   │   └── augment_training_data.py          # Augmentation (rotation et luminosité)
│   └── tests/
│       ├── test_LFM2_5_base_model.py         # Test du modèle de base
│       └── test_SFT_LFM2_VL_1_6B.py          # Test du modèle fine-tuné
├── models/                                   # (non versionné) Modèle et Adapters
└── data/                                     # (non versionné) Dataset et transcriptions (jsonl)
└── sample/                     
    ├── lines_sample/                         # Sample des lignes PNG de la page 6
    ├── page_006.png                          # Exemple de PNG d'une page
    └── transcription_sample.png              # Sample des transcriptions des lignes augmentées
```

## 🔄 Pipeline

```
PDF ──→ Images (300 DPI) ──→ Lignes (OpenCV) ──→ Fine-tuning LoRA ──→ Inférence
```

1. **Conversion** : PDF → PNG haute résolution
2. **Segmentation** : Découpage des pages en lignes individuelles
3. **Labélisation** : Transcription manuelle des lignes
4. **Augmentation** : Variations de luminosité pour robustesse
5. **Fine-tuning** : Entraînement LoRA sur l'écriture manuscrite
6. **Inférence** : OCR batch ligne par ligne

## 📊 Résultats

**Dataset** : 2000 lignes labélisées (avec augmentation)

| Epoch | Training Loss | Validation Loss | Token Accuracy |
|-------|---------------|-----------------|----------------|
| 1 | 0.55 | 0.50 | 89.4% |
| 3 | 0.18 | 0.27 | 94.4% |
| 5 | 0.11 | 0.22 | **95.7%** |

## ♻️ Mettre à jour le fine-tuning

### 1. Installation

```bash
pip install torch transformers peft trl pillow opencv-python pdf2image
```

### 2. Préparer les données d'entraînement

```bash
# 1. Segmenter les pages en lignes
python sft/training_data/extract_training_data.py

# 2. Labéliser manuellement → transcription.jsonl
# Format : {"image": "page_001_line_001.png", "text": "transcription ici"}

# 3. Augmenter le dataset (variations de luminosité)
python sft/training_data/augment_training_data.py
```

### 3. Fine-tuning (Google Colab)

1. Uploader `sft/SFT_LFM2_5_VL_1_6B.ipynb` sur Colab
2. Uploader le dataset (`lines/` + `transcription_augmented.jsonl`)
3. Sélectionner GPU (A100 recommandé)
4. Exécuter le notebook

### 4. Récupérer le modèle fine-tuné

Après l'entraînement, télécharger depuis Google Drive :
```
checkpoint-xxx/
├── adapter_config.json      # Config LoRA
└── adapter_model.safetensors # Poids (~14 MB)
```

Placer dans `./models/lfm25-adapter/`

### 5. Inférence locale

```bash
# Télécharger le modèle de base (une seule fois)
huggingface-cli download LiquidAI/LFM2.5-VL-1.6B --local-dir ./models/lfm25-base

# Tester le modèle fine-tuné
python sft/tests/test_SFT_LFM2_VL_1_6B.py
```

### 6. Ré-entraînement (améliorer le modèle)

```
Nouvelles données → extract → augment → Colab → télécharger adapter_*.safetensors
```

Seuls les fichiers `adapter_config.json` et `adapter_model.safetensors` changent (~14 MB).
Le modèle de base (~3 GB) reste identique.

## 📝 Licence

Projet personnel — code partagé à titre éducatif.

---

*Projet réalisé dans le cadre de la préservation d'un patrimoine familial.*