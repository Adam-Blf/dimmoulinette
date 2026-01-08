# Guide d'entraînement LLM sur Acer Nitro 5

## 1. Vérifier votre GPU

Ouvrez un terminal PowerShell et exécutez :
```powershell
nvidia-smi
```

Vous devriez voir quelque chose comme :
- **GTX 1650** : 4 GB VRAM → Modèle léger (TinyLlama, Phi-2)
- **RTX 3050** : 4-6 GB VRAM → Modèle léger (Phi-2)
- **RTX 3060** : 6 GB VRAM → Modèle moyen (Phi-2, Mistral-7B avec optimisations)
- **RTX 3070+** : 8+ GB VRAM → Mistral-7B complet

## 2. Installer les prérequis

### A. Drivers NVIDIA + CUDA

1. Téléchargez les derniers drivers : https://www.nvidia.com/drivers
2. Installez CUDA Toolkit 12.x : https://developer.nvidia.com/cuda-downloads
3. Redémarrez votre PC

### B. Python et dépendances

```bash
# Créer un environnement virtuel
python -m venv venv_training
venv_training\Scripts\activate

# Installer PyTorch avec CUDA (IMPORTANT!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Vérifier CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Installer les dépendances de fine-tuning
pip install transformers datasets peft trl bitsandbytes accelerate
pip install polars loguru fastapi uvicorn jinja2 python-multipart
```

## 3. Copier le projet DIM

Copiez tout le dossier `DIM` sur votre Acer Nitro 5.

## 4. Préparer les données d'entraînement

```bash
cd DIM

# Option A: Générer des données synthétiques
python setup_llm.py --create-data

# Option B: Utiliser vos vrais épisodes
python setup_llm.py --create-data --from-episodes output/fichier_episodes.csv
```

Le fichier `output/train_dataset.jsonl` sera créé.

## 5. Lancer le fine-tuning

### Via ligne de commande
```bash
python finetune_lora.py --model mistralai/Mistral-7B-Instruct-v0.3
```

### Via l'interface web
```bash
python app.py
# Ouvrir http://localhost:8080
# Section IA -> Fine-tuning GPU
```

## 6. Configuration selon votre VRAM

### 4 GB VRAM (GTX 1650, RTX 3050)
Modifiez `finetune_lora.py` :
```python
class TrainingConfig:
    MODEL_ID = "microsoft/phi-2"  # Modèle léger
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 8
    MAX_SEQ_LENGTH = 256
```

### 6 GB VRAM (RTX 3060)
```python
class TrainingConfig:
    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 8
    MAX_SEQ_LENGTH = 384
    LORA_R = 8  # Réduire le rang LoRA
```

### 8+ GB VRAM (RTX 3070+)
```python
class TrainingConfig:
    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION = 4
    MAX_SEQ_LENGTH = 512
```

## 7. Utiliser le modèle entraîné

Après l'entraînement, les adaptateurs LoRA sont dans `models/adapters/final_adapter/`.

### Tester localement
```bash
python finetune_lora.py --predict "Patient: P001, UM: PSY-AMB, Dates: 2024-01-15 -> 2024-01-20, Duree: 5 jours"
```

### Créer un modèle Ollama avec les adaptateurs
```bash
# Fusionner les adaptateurs (optionnel, nécessite plus de VRAM)
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3', torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, './models/adapters/final_adapter')
merged = model.merge_and_unload()
merged.save_pretrained('./models/merged_model')
"

# Puis convertir en GGUF pour Ollama (voir llama.cpp)
```

## 8. Durée estimée de l'entraînement

| GPU | Dataset 100 samples | Dataset 1000 samples |
|-----|---------------------|----------------------|
| GTX 1650 | ~30 min | ~4-5 heures |
| RTX 3050 | ~20 min | ~3 heures |
| RTX 3060 | ~15 min | ~2 heures |
| RTX 3070 | ~10 min | ~1 heure |

## 9. Troubleshooting

### "CUDA out of memory"
- Réduisez `BATCH_SIZE` à 1
- Réduisez `MAX_SEQ_LENGTH` à 256
- Utilisez un modèle plus léger (Phi-2)

### "Torch not compiled with CUDA"
- Réinstallez PyTorch avec CUDA :
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Entraînement très lent
- Vérifiez que le GPU est utilisé : `nvidia-smi` doit montrer une utilisation
- Fermez les autres applications utilisant le GPU
