"""
===============================================================================
AI_MANAGER.PY - Gestionnaire IA Local pour PMSI
===============================================================================
DIM - Data Intelligence Médicale
Support multi-backend: Ollama, llama-cpp-python, Transformers
100% local - Aucune donnée ne sort de la machine
===============================================================================
"""

import json
import os
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import urllib.request
import tempfile

import polars as pl
from loguru import logger

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AIConfig:
    """Configuration IA unifiée."""
    # Backend: 'ollama', 'llama-cpp', 'transformers'
    backend: str = "ollama"

    # Modèles par défaut selon backend
    ollama_model: str = "mistral:7b-instruct"
    llamacpp_model: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    transformers_model: str = "mistralai/Mistral-7B-v0.1"

    # Chemins
    models_dir: Path = field(default_factory=lambda: Path("./models"))
    output_dir: Path = field(default_factory=lambda: Path("./output"))

    # Inference
    max_tokens: int = 512
    temperature: float = 0.7
    context_window: int = 4096

    # GPU
    n_gpu_layers: int = -1  # -1 = all layers on GPU

    # URLs de téléchargement
    model_urls: Dict[str, str] = field(default_factory=lambda: {
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "phi-2.Q4_K_M.gguf": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
        "tinyllama-1.1b-chat.Q4_K_M.gguf": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    })


# =============================================================================
# BASE LLM INTERFACE
# =============================================================================

class BaseLLM(ABC):
    """Interface abstraite pour les backends LLM."""

    @abstractmethod
    def is_available(self) -> bool:
        """Vérifie si le backend est disponible."""
        pass

    @abstractmethod
    def load_model(self, model_name: str = None) -> bool:
        """Charge le modèle."""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Génère une réponse."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut du backend."""
        pass


# =============================================================================
# BACKEND OLLAMA
# =============================================================================

class OllamaBackend(BaseLLM):
    """
    Backend Ollama - Le plus simple pour du local.
    Télécharge et gère les modèles automatiquement.
    """

    def __init__(self, config: AIConfig = None):
        self.config = config or AIConfig()
        self.model_name = self.config.ollama_model
        self._available = None
        self.base_url = "http://localhost:11434"

    def is_available(self) -> bool:
        """Vérifie si Ollama est installé et en cours d'exécution."""
        if self._available is not None:
            return self._available

        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            self._available = response.status_code == 200
        except Exception:
            # Vérifier si ollama est installé
            if shutil.which("ollama"):
                self._available = False  # Installé mais pas lancé
            else:
                self._available = False

        return self._available

    def start_server(self) -> bool:
        """Démarre le serveur Ollama si nécessaire."""
        if self.is_available():
            return True

        try:
            # Lancer ollama serve en background
            if sys.platform == "win32":
                subprocess.Popen(
                    ["ollama", "serve"],
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )

            # Attendre que le serveur démarre
            import time
            for _ in range(10):
                time.sleep(1)
                if self.is_available():
                    logger.info("Ollama server started")
                    return True

            return False
        except Exception as e:
            logger.error(f"Failed to start Ollama: {e}")
            return False

    def load_model(self, model_name: str = None) -> bool:
        """Télécharge le modèle si nécessaire via ollama pull."""
        model = model_name or self.model_name

        if not self.start_server():
            return False

        try:
            import requests

            # Vérifier si le modèle existe
            response = requests.get(f"{self.base_url}/api/tags")
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]

            if model.split(":")[0] not in model_names:
                logger.info(f"Downloading model {model}...")
                # Pull le modèle
                response = requests.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model},
                    stream=True
                )

                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "status" in data:
                            logger.info(f"  {data['status']}")

            self.model_name = model
            logger.info(f"Model {model} ready")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def generate(self, prompt: str, **kwargs) -> str:
        """Génère une réponse avec Ollama."""
        try:
            import requests

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", self.config.temperature),
                        "num_predict": kwargs.get("max_tokens", self.config.max_tokens)
                    }
                },
                timeout=120
            )

            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Error: {response.text}"

        except Exception as e:
            return f"Error: {str(e)}"

    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut d'Ollama."""
        available = self.is_available()
        models = []

        if available:
            try:
                import requests
                response = requests.get(f"{self.base_url}/api/tags")
                models = [m.get("name") for m in response.json().get("models", [])]
            except:
                pass

        return {
            "backend": "ollama",
            "available": available,
            "installed": shutil.which("ollama") is not None,
            "running": available,
            "models": models,
            "current_model": self.model_name
        }


# =============================================================================
# BACKEND LLAMA-CPP-PYTHON
# =============================================================================

class LlamaCppBackend(BaseLLM):
    """
    Backend llama-cpp-python - Plus de contrôle, fonctionne offline.
    Nécessite le téléchargement manuel des modèles GGUF.
    """

    def __init__(self, config: AIConfig = None):
        self.config = config or AIConfig()
        self.model = None
        self.model_path = None
        self._available = None

    def is_available(self) -> bool:
        """Vérifie si llama-cpp-python est installé."""
        if self._available is not None:
            return self._available

        try:
            from llama_cpp import Llama
            self._available = True
        except ImportError:
            self._available = False

        return self._available

    def download_model(self, model_name: str = None) -> Optional[Path]:
        """Télécharge un modèle GGUF depuis HuggingFace."""
        model = model_name or self.config.llamacpp_model

        if model not in self.config.model_urls:
            logger.error(f"Unknown model: {model}")
            return None

        self.config.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.config.models_dir / model

        if model_path.exists():
            logger.info(f"Model already exists: {model_path}")
            return model_path

        url = self.config.model_urls[model]
        logger.info(f"Downloading {model}...")
        logger.info(f"  From: {url}")

        try:
            def progress_hook(count, block_size, total_size):
                percent = count * block_size * 100 // total_size
                if count % 100 == 0:
                    print(f"\r  Progress: {percent}%", end="", flush=True)

            urllib.request.urlretrieve(url, model_path, reporthook=progress_hook)
            print()  # New line
            logger.info(f"Model downloaded: {model_path}")
            return model_path

        except Exception as e:
            logger.error(f"Download failed: {e}")
            if model_path.exists():
                model_path.unlink()
            return None

    def load_model(self, model_name: str = None) -> bool:
        """Charge un modèle GGUF."""
        if not self.is_available():
            logger.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
            return False

        model = model_name or self.config.llamacpp_model
        model_path = self.config.models_dir / model

        # Télécharger si nécessaire
        if not model_path.exists():
            model_path = self.download_model(model)
            if model_path is None:
                return False

        try:
            from llama_cpp import Llama

            logger.info(f"Loading model: {model_path}")

            self.model = Llama(
                model_path=str(model_path),
                n_ctx=self.config.context_window,
                n_gpu_layers=self.config.n_gpu_layers,
                verbose=False
            )

            self.model_path = model_path
            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def generate(self, prompt: str, **kwargs) -> str:
        """Génère une réponse avec llama.cpp."""
        if self.model is None:
            if not self.load_model():
                return "Error: Model not loaded"

        try:
            output = self.model(
                prompt,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                stop=["</s>", "###", "\n\n\n"],
                echo=False
            )

            return output["choices"][0]["text"].strip()

        except Exception as e:
            return f"Error: {str(e)}"

    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut du backend."""
        available_models = []
        if self.config.models_dir.exists():
            available_models = [f.name for f in self.config.models_dir.glob("*.gguf")]

        return {
            "backend": "llama-cpp",
            "available": self.is_available(),
            "loaded": self.model is not None,
            "current_model": self.model_path.name if self.model_path else None,
            "available_models": available_models,
            "models_dir": str(self.config.models_dir)
        }


# =============================================================================
# GESTIONNAIRE IA UNIFIÉ
# =============================================================================

class AIManager:
    """
    Gestionnaire IA unifié pour DIM.
    Gère automatiquement le backend et l'inférence.
    """

    def __init__(self, config: AIConfig = None):
        self.config = config or AIConfig()
        self.backend: Optional[BaseLLM] = None
        self.is_loaded = False

    def initialize(self, preferred_backend: str = None) -> bool:
        """
        Initialise le meilleur backend disponible.
        Ordre de préférence: Ollama > llama-cpp > Transformers
        """
        backend = preferred_backend or self.config.backend

        if backend == "ollama":
            self.backend = OllamaBackend(self.config)
            if self.backend.is_available() or shutil.which("ollama"):
                logger.info("Using Ollama backend")
                return True
            logger.warning("Ollama not available, trying llama-cpp...")
            backend = "llama-cpp"

        if backend == "llama-cpp":
            self.backend = LlamaCppBackend(self.config)
            if self.backend.is_available():
                logger.info("Using llama-cpp backend")
                return True
            logger.warning("llama-cpp not available")

        logger.error("No AI backend available!")
        logger.info("Install Ollama: https://ollama.ai/download")
        logger.info("Or: pip install llama-cpp-python")
        return False

    def load_model(self, model_name: str = None) -> bool:
        """Charge un modèle."""
        if self.backend is None:
            if not self.initialize():
                return False

        self.is_loaded = self.backend.load_model(model_name)
        return self.is_loaded

    def analyze_episode(self, episode_data: Dict) -> Dict[str, Any]:
        """
        Analyse un épisode de soins avec l'IA.
        """
        if not self.is_loaded:
            if not self.load_model():
                return {"error": "Model not loaded", "is_anomaly": False}

        # Construction du prompt
        prompt = self._build_analysis_prompt(episode_data)

        # Génération
        response = self.backend.generate(prompt)

        # Parsing de la réponse
        is_anomaly = any(kw in response.upper() for kw in ["ANOMALIE", "ANORMAL", "ATTENTION", "ALERTE"])

        return {
            "input": episode_data,
            "analysis": response,
            "is_anomaly": is_anomaly,
            "confidence": 0.85 if is_anomaly else 0.90
        }

    def analyze_batch(self, episodes: List[Dict]) -> List[Dict]:
        """Analyse un lot d'épisodes."""
        results = []
        total = len(episodes)

        for i, episode in enumerate(episodes):
            logger.info(f"Analyzing episode {i+1}/{total}")
            result = self.analyze_episode(episode)
            results.append(result)

        return results

    def analyze_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Analyse un DataFrame d'épisodes et ajoute les colonnes IA.
        """
        if not self.is_loaded:
            if not self.load_model():
                return df.with_columns([
                    pl.lit("ERROR: Model not loaded").alias("IA_ANALYSE"),
                    pl.lit(False).alias("IA_ANOMALIE")
                ])

        analyses = []
        anomalies = []

        for row in df.iter_rows(named=True):
            result = self.analyze_episode(row)
            analyses.append(result.get("analysis", ""))
            anomalies.append(result.get("is_anomaly", False))

        return df.with_columns([
            pl.Series("IA_ANALYSE", analyses),
            pl.Series("IA_ANOMALIE", anomalies)
        ])

    def _build_analysis_prompt(self, data: Dict) -> str:
        """Construit le prompt d'analyse."""
        patient = data.get("NO_PATIENT", data.get("patient", "ANON"))
        um = data.get("CODE_UM", data.get("um", "NC"))
        site = data.get("FINESS_PMSI", data.get("site", "NC"))
        debut = str(data.get("DATE_DEBUT_EPISODE", data.get("DATE_ACTE", "")))[:10]
        fin = str(data.get("DATE_FIN_EPISODE", debut))[:10]
        duree = data.get("DUREE_EPISODE_JOURS", 0) or 0
        nb_actes = data.get("NB_ACTES_EPISODE", 1) or 1
        flag = data.get("FLAG_ANOMALIE", False)

        prompt = f"""### Instruction:
Tu es un expert DIM (Département d'Information Médicale) en psychiatrie.
Analyse cet épisode de soins ambulatoires et détecte les anomalies organisationnelles.
Un épisode ambulatoire normal dure généralement 1 jour.
Réponds de façon concise et structurée.

### Données de l'épisode:
- Patient: {patient}
- Unité médicale: {um}
- Site (FINESS): {site}
- Date début: {debut}
- Date fin: {fin}
- Durée: {duree} jours
- Nombre d'actes: {nb_actes}
- Flag anomalie règle métier: {'OUI' if flag else 'NON'}

### Analyse:
"""
        return prompt

    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut complet de l'IA."""
        if self.backend is None:
            return {
                "initialized": False,
                "backend": None,
                "message": "AI not initialized. Call initialize() first."
            }

        status = self.backend.get_status()
        status["initialized"] = True
        status["model_loaded"] = self.is_loaded
        return status

    def list_available_models(self) -> List[str]:
        """Liste les modèles disponibles."""
        models = []

        # Modèles Ollama recommandés
        models.extend([
            "ollama:mistral:7b-instruct",
            "ollama:llama2:7b",
            "ollama:codellama:7b",
            "ollama:neural-chat:7b"
        ])

        # Modèles GGUF disponibles
        models.extend([
            f"gguf:{name}" for name in self.config.model_urls.keys()
        ])

        return models


# =============================================================================
# PRÉPARATION DES DONNÉES POUR FINE-TUNING
# =============================================================================

class DatasetPreparer:
    """Prépare les données pour l'entraînement."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("./output")
        self.samples: List[Dict] = []

    def prepare_from_episodes(
        self,
        episodes_df: pl.DataFrame,
        include_anomalies: bool = True
    ) -> List[Dict]:
        """Convertit les épisodes en samples d'entraînement."""
        logger.info("Preparing training data...")

        samples = []
        templates = [
            {
                "instruction": "Analyze this psychiatric care episode for organizational anomalies.",
                "input_template": "Patient: {patient}, Unit: {um}, Site: {site}, Dates: {debut} -> {fin}, Duration: {duree} days, Acts: {nb_actes}",
                "output_normal": "Normal episode. Duration compliant with ambulatory standards.",
                "output_anomaly": "ANOMALY DETECTED: Excessive duration ({duree} days). Check: 1) Downstream issues, 2) Coding errors, 3) Care pathway organization."
            }
        ]

        for row in episodes_df.iter_rows(named=True):
            patient = row.get("NO_PATIENT", "ANON")
            um = row.get("CODE_UM", "NC")
            site = row.get("FINESS_PMSI", "NC")
            debut = str(row.get("DATE_DEBUT_EPISODE", ""))[:10]
            fin = str(row.get("DATE_FIN_EPISODE", debut))[:10]
            duree = row.get("DUREE_EPISODE_JOURS", 0) or 0
            nb_actes = row.get("NB_ACTES_EPISODE", 1) or 1
            is_anomaly = row.get("FLAG_ANOMALIE", False)

            for template in templates:
                input_text = template["input_template"].format(
                    patient=patient, um=um, site=site,
                    debut=debut, fin=fin, duree=int(duree), nb_actes=int(nb_actes)
                )

                output_text = template["output_anomaly"].format(duree=int(duree)) if is_anomaly else template["output_normal"]

                samples.append({
                    "instruction": template["instruction"],
                    "input": input_text,
                    "output": output_text
                })

        self.samples = samples
        logger.info(f"  Prepared {len(samples)} samples")
        return samples

    def save_to_jsonl(self, output_path: Path = None) -> Path:
        """Sauvegarde en JSONL."""
        if output_path is None:
            output_path = self.output_dir / "train_dataset.jsonl"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in self.samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        logger.info(f"  Dataset saved: {output_path}")
        return output_path

    def load_from_jsonl(self, input_path: Path) -> List[Dict]:
        """Charge depuis JSONL."""
        samples = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        self.samples = samples
        return samples


# =============================================================================
# FINE-TUNING QLORA (GPU REQUIS)
# =============================================================================

class QLoRATrainer:
    """
    Entraineur QLoRA pour fine-tuning local.
    Utilise Mistral-7B en 4-bit avec LoRA adapters.
    """

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("./models/fine-tuned")
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.is_trained = False

    def check_requirements(self) -> Dict[str, Any]:
        """Verifie les prerequis pour le fine-tuning."""
        results = {
            "torch": False,
            "cuda": False,
            "transformers": False,
            "peft": False,
            "bitsandbytes": False,
            "gpu_name": None,
            "gpu_memory": None,
            "ready": False
        }

        try:
            import torch
            results["torch"] = True
            if torch.cuda.is_available():
                results["cuda"] = True
                results["gpu_name"] = torch.cuda.get_device_name(0)
                results["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        except ImportError:
            pass

        try:
            from transformers import AutoModelForCausalLM
            results["transformers"] = True
        except ImportError:
            pass

        try:
            from peft import LoraConfig
            results["peft"] = True
        except ImportError:
            pass

        try:
            import bitsandbytes
            results["bitsandbytes"] = True
        except ImportError:
            pass

        results["ready"] = all([
            results["torch"],
            results["cuda"],
            results["transformers"],
            results["peft"],
            results["bitsandbytes"]
        ])

        return results

    def load_model(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2") -> bool:
        """Charge le modele en 4-bit pour fine-tuning."""
        reqs = self.check_requirements()
        if not reqs["ready"]:
            logger.error("Prerequisites not met for QLoRA training")
            logger.info("Required: torch, cuda, transformers, peft, bitsandbytes")
            return False

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            logger.info(f"Loading model: {model_name}")
            logger.info(f"GPU: {reqs['gpu_name']} ({reqs['gpu_memory']})")

            # Configuration 4-bit
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )

            # Chargement modele
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )

            self.model = prepare_model_for_kbit_training(self.model)

            # Configuration LoRA
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM"
            )

            self.model = get_peft_model(self.model, lora_config)

            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"

            # Stats
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            logger.info(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def train(self, samples: List[Dict], epochs: int = 3, batch_size: int = 4) -> bool:
        """Lance l'entrainement QLoRA."""
        if self.model is None:
            if not self.load_model():
                return False

        try:
            from transformers import TrainingArguments
            from datasets import Dataset
            from trl import SFTTrainer

            logger.info("=" * 60)
            logger.info("DEMARRAGE FINE-TUNING QLORA")
            logger.info(f"  Epochs: {epochs}")
            logger.info(f"  Samples: {len(samples)}")
            logger.info(f"  Batch size: {batch_size}")
            logger.info("=" * 60)

            # Preparation dataset
            def format_sample(sample):
                return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""

            formatted = [{"text": format_sample(s)} for s in samples]
            dataset = Dataset.from_list(formatted)

            # Arguments
            self.output_dir.mkdir(parents=True, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                warmup_ratio=0.03,
                max_grad_norm=0.3,
                weight_decay=0.001,
                logging_steps=10,
                save_strategy="epoch",
                fp16=True,
                optim="paged_adamw_32bit",
                report_to="none",
            )

            # Trainer
            self.trainer = SFTTrainer(
                model=self.model,
                train_dataset=dataset,
                tokenizer=self.tokenizer,
                args=training_args,
                dataset_text_field="text",
                max_seq_length=512,
            )

            # Train
            self.trainer.train()

            # Save
            final_path = self.output_dir / "final"
            self.model.save_pretrained(final_path)
            self.tokenizer.save_pretrained(final_path)

            # Metadata
            metadata = {
                "trained_at": datetime.now().isoformat(),
                "epochs": epochs,
                "samples": len(samples)
            }
            with open(final_path / "training_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Model saved to: {final_path}")
            self.is_trained = True
            return True

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

    def create_ollama_modelfile(self, base_model: str = "mistral:7b-instruct") -> Path:
        """
        Cree un Modelfile Ollama avec un system prompt personnalise.
        Alternative simple au fine-tuning complet.
        """
        system_prompt = """Tu es un assistant expert en DIM (Departement d'Information Medicale) specialise en psychiatrie.
Tu analyses les episodes de soins ambulatoires pour detecter les anomalies organisationnelles.

Regles d'analyse:
- Un episode ambulatoire normal dure 1 jour maximum
- Si la duree depasse 1 jour, c'est une ANOMALIE
- Les anomalies peuvent indiquer: probleme d'aval, erreur de codage, desorganisation

Format de reponse:
- Si normal: "Episode normal. Duree conforme aux standards ambulatoires."
- Si anomalie: "ANOMALIE DETECTEE: Duree excessive (X jours). Verifier: 1) Probleme d'aval, 2) Erreur de codage, 3) Organisation du parcours."
"""

        modelfile_content = f"""FROM {base_model}

SYSTEM {system_prompt}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_predict 256
"""

        modelfile_path = Path("./models/DIM_PMSI_Modelfile")
        modelfile_path.parent.mkdir(parents=True, exist_ok=True)

        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)

        logger.info(f"Modelfile created: {modelfile_path}")
        logger.info(f"To create the model, run: ollama create dim-pmsi -f {modelfile_path}")

        return modelfile_path

    def create_ollama_model(self, model_name: str = "dim-pmsi") -> bool:
        """Cree un modele Ollama personnalise."""
        modelfile_path = self.create_ollama_modelfile()

        try:
            result = subprocess.run(
                ["ollama", "create", model_name, "-f", str(modelfile_path)],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                logger.info(f"Ollama model '{model_name}' created successfully!")
                return True
            else:
                logger.error(f"Failed to create model: {result.stderr}")
                return False

        except FileNotFoundError:
            logger.error("Ollama not found. Please install Ollama first.")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Model creation timed out")
            return False
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            return False


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def install_ollama():
    """Guide pour installer Ollama."""
    print("""
================================================================================
                    INSTALLATION D'OLLAMA
================================================================================

Ollama est le moyen le plus simple d'exécuter des LLMs localement.

1. Téléchargez Ollama depuis: https://ollama.ai/download

2. Installez et lancez Ollama

3. Téléchargez un modèle:
   > ollama pull mistral:7b-instruct

4. Relancez DIM

================================================================================
    """)


def main():
    """Point d'entrée principal."""
    import argparse

    parser = argparse.ArgumentParser(description="DIM AI Manager")
    subparsers = parser.add_subparsers(dest="command")

    # Status
    subparsers.add_parser("status", help="Show AI status")

    # Install
    subparsers.add_parser("install", help="Show installation guide")

    # Download
    dl_parser = subparsers.add_parser("download", help="Download a model")
    dl_parser.add_argument("model", type=str, help="Model name")

    # Test
    test_parser = subparsers.add_parser("test", help="Test inference")
    test_parser.add_argument("--prompt", type=str, default="Explain PMSI in one sentence.")

    # Analyze
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a file")
    analyze_parser.add_argument("file", type=str, help="CSV file to analyze")

    args = parser.parse_args()

    if args.command == "status":
        manager = AIManager()
        manager.initialize()
        status = manager.get_status()
        print(json.dumps(status, indent=2, default=str))

    elif args.command == "install":
        install_ollama()

    elif args.command == "download":
        manager = AIManager()
        manager.initialize()
        manager.load_model(args.model)

    elif args.command == "test":
        manager = AIManager()
        manager.initialize()
        if manager.load_model():
            response = manager.backend.generate(args.prompt)
            print(f"Response: {response}")
        else:
            print("Failed to load model")

    elif args.command == "analyze":
        manager = AIManager()
        manager.initialize()
        if manager.load_model():
            df = pl.read_csv(args.file, separator=";")
            result_df = manager.analyze_dataframe(df)
            output_path = Path(args.file).stem + "_analyzed.csv"
            result_df.write_csv(output_path, separator=";")
            print(f"Results saved to: {output_path}")
        else:
            print("Failed to load model")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
