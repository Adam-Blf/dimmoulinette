# README Developpeur - Moulinettes DIM

## Documentation Technique pour le Starter Kit PMSI Psychiatrie

---

## Architecture du Projet

```
DIM/
├── app.py                  # Backend FastAPI + API REST
├── processor.py            # Pipeline ETL avec Polars
├── finetune_lora.py        # Fine-tuning QLoRA 4-bit
├── requirements.txt        # Dependances Python
├── .gitignore              # Exclusions Git (donnees sensibles)
├── templates/
│   └── index.html          # Dashboard HTML5/CSS3
├── output/                 # Resultats du traitement
│   ├── train_dataset.jsonl # Dataset pour fine-tuning
│   └── episodes_consolides.csv
├── models/
│   ├── cache/              # Cache HuggingFace
│   └── adapters/           # Adaptateurs LoRA sauvegardes
├── MODE_D_EMPLOI_DIM.md    # Guide utilisateur
└── README_DEV.md           # Ce fichier
```

---

## Installation

### Prerequis

- **Python** : 3.10+ (recommande : 3.11)
- **GPU** : NVIDIA avec CUDA 11.8+ (optionnel mais recommande)
- **RAM** : 16 GB minimum
- **VRAM** : 8 GB minimum pour le fine-tuning

### Etape 1 : Creer l'environnement virtuel

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Etape 2 : Installer les dependances

```bash
pip install -r requirements.txt
```

### Etape 3 : Installer PyTorch avec CUDA (si GPU disponible)

```bash
# Pour CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Pour CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU uniquement
pip install torch torchvision torchaudio
```

### Etape 4 : Creer le dossier source

```bash
mkdir -p "C:\Users\adamb\Downloads\frer"
```

---

## Commandes de Lancement

### Dashboard Web (mode developpement)

```bash
python app.py
# Serveur disponible sur http://localhost:8000
```

### Pipeline ETL seul

```bash
# Execution standard
python processor.py

# Forcer le retraitement de tous les fichiers
python processor.py --force

# Scanner uniquement (sans traitement)
python processor.py --scan-only
```

### Fine-tuning IA seul

```bash
# Verifier le systeme
python finetune_lora.py --check

# Lancer l'entrainement
python finetune_lora.py

# Utiliser un modele specifique
python finetune_lora.py --model "microsoft/phi-2"

# Faire une prediction avec les adaptateurs
python finetune_lora.py --predict "Patient X, 3 jours consecutifs au SAU"
```

---

## Configuration

### Modifier le chemin des donnees source

Dans `processor.py`, ligne ~35 :

```python
class Config:
    SOURCE_DIR = Path(r"C:\Users\adamb\Downloads\frer")  # Modifier ici
```

### Parametres de fine-tuning

Dans `finetune_lora.py`, classe `TrainingConfig` :

```python
class TrainingConfig:
    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # Modele de base
    LORA_R = 16          # Rang LoRA (plus eleve = plus de parametres)
    LORA_ALPHA = 32      # Scaling factor
    BATCH_SIZE = 2       # Reduire si OOM
    NUM_EPOCHS = 3       # Nombre d'epoques
    MAX_SEQ_LENGTH = 512 # Longueur max des sequences
```

### Modeles recommandes selon VRAM

| VRAM | Modele | ID HuggingFace |
|------|--------|----------------|
| 12+ GB | Mistral 7B | `mistralai/Mistral-7B-Instruct-v0.3` |
| 8-12 GB | Phi-2 | `microsoft/phi-2` |
| 6-8 GB | TinyLlama | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| < 6 GB | CPU recommande | - |

---

## API REST

### Endpoints disponibles

| Methode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Dashboard HTML |
| GET | `/api/files` | Liste des fichiers source |
| GET | `/api/system` | Info systeme (GPU, etc.) |
| POST | `/api/etl/run` | Lancer le pipeline ETL |
| POST | `/api/training/run` | Lancer le fine-tuning |
| GET | `/api/logs` | Recuperer les logs |
| DELETE | `/api/logs` | Effacer les logs |
| GET | `/api/results/etl` | Resultats ETL |
| GET | `/api/results/training` | Resultats training |

### Exemple d'appel API

```bash
# Lancer l'ETL
curl -X POST http://localhost:8000/api/etl/run

# Recuperer les fichiers
curl http://localhost:8000/api/files
```

---

## Structure des Donnees

### Format d'entree (RPS/RAA)

Le pipeline detecte automatiquement les colonnes suivantes :

| Nom standard | Variantes acceptees |
|--------------|---------------------|
| `patient` | NO_PATIENT, NUM_PATIENT, IPP, NIP |
| `date_debut` | DATE_DEBUT, DATE_ENT, DATE_ENTREE |
| `date_fin` | DATE_FIN, DATE_SOR, DATE_SORTIE |
| `um` | UM, UNITE_MED, CODE_UM |
| `site` | SITE, ETABLISSEMENT, FINESS |

### Format de sortie (train_dataset.jsonl)

```json
{"instruction": "Detecter anomalie parcours", "input": "Patient X, ...", "output": "Anomalie probable"}
{"instruction": "Analyser coherence sejour", "input": "Sejour du ...", "output": "Sejour normal"}
```

---

## Securite

### Mesures implementees

1. **Verification d'integrite SHA-256**
   - Chaque fichier est hashe avant traitement
   - Detecte les modifications non autorisees

2. **Mode Read-Only**
   - Les fichiers source ne sont jamais modifies
   - Ouverture en mode lecture seule

3. **Sanitization des chemins**
   - Protection contre les attaques path traversal
   - Validation stricte des chemins utilisateurs

4. **Exclusion Git**
   - `.gitignore` complet pour les donnees sensibles
   - Pas de commit accidentel de donnees de sante

### Checklist securite production

- [ ] Deplacer les donnees hors du dossier `Downloads`
- [ ] Activer BitLocker sur le disque
- [ ] Creer un compte Windows dedie
- [ ] Configurer un firewall pour le port 8000
- [ ] Mettre en place un certificat HTTPS

---

## Depannage

### Erreur : `ModuleNotFoundError: No module named 'torch'`

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Erreur : `CUDA out of memory`

1. Reduire `BATCH_SIZE` dans `TrainingConfig`
2. Utiliser un modele plus petit (Phi-2 ou TinyLlama)
3. Fermer les autres applications GPU

### Erreur : `FileNotFoundError: train_dataset.jsonl`

```bash
# Executer d'abord le pipeline ETL
python processor.py
```

### Erreur : `bitsandbytes` sur Windows

```bash
# Installer la version Windows
pip install bitsandbytes-windows
```

---

## Tests

### Verifier l'installation

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import polars; print('Polars OK')"
python -c "import transformers; print('Transformers OK')"
```

### Test du pipeline ETL

```bash
# Creer un fichier de test
echo "NO_PATIENT;DATE_DEBUT;UM" > C:\Users\adamb\Downloads\frer\test_RPS.csv
echo "P001;2024-01-15;PSY01" >> C:\Users\adamb\Downloads\frer\test_RPS.csv

# Lancer le pipeline
python processor.py --scan-only
```

---

## Contribution

### Style de code

- PEP 8 pour le formatage
- Type hints obligatoires
- Docstrings pour les fonctions publiques

### Pre-commit

```bash
pip install black isort
black *.py
isort *.py
```

---

## Licence

Projet interne - Donnees de sante confidentielles (RGPD/HDS)
