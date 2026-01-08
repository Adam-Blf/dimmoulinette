# DIM - Data Intelligence Medicale

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20AI-purple.svg)](https://ollama.ai)

**Outil de traitement des donnees PMSI avec IA locale pour la psychiatrie francaise.**

> :lock: 100% local - Aucune donnee ne quitte votre machine

## Fonctionnalites

- **ETL PMSI** : Transformation des fichiers RAA, RPS, VIDHOSP en CSV exploitables
- **Episodes ambulatoires** : Reconstruction des parcours patients psychiatriques
- **Detection d'anomalies** : Identification automatique des durees excessives
- **IA locale** : LLM fine-tune pour l'analyse PMSI (Ollama/Mistral/Phi-2)
- **Visualisation GHT** : Graphes de structure hospitaliere
- **Interface web** : Dashboard intuitif avec analyse en temps reel

## Installation rapide

### Prerequis

- Python 3.10+
- [Ollama](https://ollama.ai/download) (pour l'IA locale)
- GPU NVIDIA (optionnel, pour le fine-tuning)

### Installation

```bash
# Cloner le repo
git clone https://github.com/VOTRE_USERNAME/DIM.git
cd DIM

# Installer les dependances
pip install -r requirements.txt

# Configurer l'IA locale (telecharge Mistral-7B)
python setup_llm.py --setup
```

### Lancer l'application

```bash
python app.py
```

Ouvrez **http://localhost:8080** dans votre navigateur.

## Utilisation

### Interface Web

1. **Deposez** vos fichiers PMSI (drag & drop)
2. **Lancez** les moulinettes ETL
3. **Generez** les episodes ambulatoires
4. **Analysez** avec l'IA

### Ligne de commande

```bash
# Traitement ETL
python processor.py --input fichier.txt --output output/

# Configuration IA
python setup_llm.py --setup

# Fine-tuning GPU (RTX 3050+)
python finetune_lora.py
```

## Fine-tuning sur GPU

Le projet supporte le fine-tuning QLoRA pour adapter le modele a vos donnees.

### Configuration par GPU

| GPU | VRAM | Modele | Script |
|-----|------|--------|--------|
| GTX 1650 | 4GB | TinyLlama-1.1B | `finetune_lora.py` |
| **RTX 3050** | 4-6GB | **Phi-2 (2.7B)** | `train_rtx3050.bat` |
| RTX 3060 | 6GB | Mistral-7B | `finetune_lora.py` |
| RTX 3070+ | 8GB+ | Mistral-7B | `finetune_lora.py` |

### Installation CUDA (Windows)

```bash
# Executer le script d'installation
install_cuda.bat

# Ou manuellement
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft bitsandbytes trl datasets accelerate
```

### Lancer l'entrainement

```bash
# Windows - RTX 3050
train_rtx3050.bat

# Tous OS
python finetune_lora.py
```

## Formats PMSI supportes

| Type | Description | Annee |
|------|-------------|-------|
| **RAA** | Resume d'Activite Ambulatoire | 2024 |
| **RPS** | Resume Par Sequence (hospit. PSY) | 2024 |
| **VIDHOSP** | Chainage sejours | 2024 |
| **FICHCOMP** | Fichiers complementaires | 2024 |
| **RSFACE** | Facturation | 2024 |
| **RUMRSS** | MCO (Resumes de sejour) | 2024 |

## Structure du projet

```
DIM/
├── app.py                 # Application FastAPI (interface web)
├── processor.py           # Pipeline ETL principal
├── etl_processor.py       # Parseur universel PMSI
├── psy_logic.py          # Logique episodes psychiatrie
├── ai_manager.py         # Gestionnaire IA (Ollama/llama-cpp)
├── finetune_lora.py      # Fine-tuning QLoRA
├── setup_llm.py          # Configuration LLM simplifiee
├── ficom_viz.py          # Visualisation structure GHT
├── configs/              # Formats PMSI (JSON)
│   ├── format_raa_2024.json
│   ├── format_rps_2024.json
│   └── ...
├── models/               # Modeles et Modelfiles (gitignore)
├── templates/            # Interface web HTML
├── output/               # Resultats (gitignore)
├── train_rtx3050.bat     # Script entrainement RTX 3050
└── install_cuda.bat      # Installation PyTorch CUDA
```

## API REST

| Methode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/api/files` | Liste les fichiers source |
| `POST` | `/api/etl/run` | Lance le pipeline ETL |
| `POST` | `/api/episodes/run` | Genere les episodes |
| `POST` | `/api/ai/init` | Initialise l'IA |
| `POST` | `/api/ai/create-model` | Cree le modele dim-pmsi |
| `GET` | `/api/ai/status` | Statut de l'IA |
| `GET` | `/api/ai/models` | Liste les modeles Ollama |
| `POST` | `/api/ai/training/start` | Lance le fine-tuning |

## Securite des donnees

Ce projet est concu pour traiter des donnees de sante sensibles (HDS/RGPD) :

- :lock: **100% local** : Aucune API cloud, tout reste sur votre machine
- :robot: **IA embarquee** : Les modeles tournent localement via Ollama
- :no_entry: **.gitignore strict** : Les donnees ne sont jamais commitees
- :page_facing_up: **Pas de logs sensibles** : Aucune donnee patient dans les logs

## Dependances principales

```
polars          # Traitement donnees haute performance
fastapi         # API REST et interface web
transformers    # Modeles HuggingFace
peft            # Fine-tuning LoRA
bitsandbytes    # Quantification 4-bit
trl             # Training LLM
loguru          # Logging
```

## Contribution

Les contributions sont les bienvenues !

```bash
# Fork et clone
git clone https://github.com/VOTRE_USERNAME/DIM.git

# Creer une branche
git checkout -b feature/ma-feature

# Commit et push
git commit -m "Add ma feature"
git push origin feature/ma-feature

# Ouvrir une Pull Request
```

## License

MIT License - Voir [LICENSE](LICENSE)

---

*Developpe pour les DIM de psychiatrie - 2024*
