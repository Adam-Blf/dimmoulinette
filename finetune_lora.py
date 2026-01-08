"""
===============================================================================
FINETUNE_LORA.PY - Fine-tuning QLoRA 4-bit pour LLM Medical
===============================================================================
Moulinettes DIM - Entrainement sur donnees PMSI Psychiatrie
Auteur: Adam B. | Detection automatique du materiel
===============================================================================
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION DYNAMIQUE (AUTO-DETECTEE)
# ============================================================================

class TrainingConfig:
    """Configuration du fine-tuning - Auto-detectee selon le materiel."""

    # Chemins (fixes)
    DATASET_PATH = Path("./output/train_dataset.jsonl")
    OUTPUT_DIR = Path("./models/adapters")
    CACHE_DIR = Path("./models/cache")

    # Valeurs par defaut (seront ecrasees par auto-detection)
    MODEL_ID = "microsoft/phi-2"
    MODEL_FALLBACK = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 8
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    MAX_SEQ_LENGTH = 256
    WARMUP_RATIO = 0.03

    USE_4BIT = True
    USE_CPU = False
    BNB_4BIT_COMPUTE_DTYPE = "float16"
    BNB_4BIT_QUANT_TYPE = "nf4"

    # Info hardware detecte
    HARDWARE_PROFILE = "unknown"
    GPU_NAME = "N/A"
    GPU_VRAM = 0.0

    @classmethod
    def from_auto_detection(cls):
        """Cree une config basee sur la detection automatique du materiel."""
        try:
            from hardware_detector import detect_and_configure

            logger.info("Detection automatique du materiel...")
            system_info, config = detect_and_configure(save_config=True, verbose=True)

            # Mise a jour des parametres
            cls.MODEL_ID = config["model"]["model_id"]
            cls.USE_4BIT = config["model"]["use_4bit"]
            cls.USE_CPU = config["model"]["use_cpu"]

            cls.BATCH_SIZE = config["training"]["batch_size"]
            cls.GRADIENT_ACCUMULATION = config["training"]["gradient_accumulation_steps"]
            cls.MAX_SEQ_LENGTH = config["training"]["max_seq_length"]

            cls.LORA_R = config["lora"]["r"]
            cls.LORA_ALPHA = config["lora"]["alpha"]

            cls.HARDWARE_PROFILE = config["hardware"]["profile_name"]
            cls.GPU_NAME = config["hardware"]["gpu_name"]
            cls.GPU_VRAM = config["hardware"]["gpu_vram_gb"]

            logger.info(f"Configuration chargee: {cls.HARDWARE_PROFILE}")
            return cls

        except ImportError:
            logger.warning("Module hardware_detector non trouve - utilisation des valeurs par defaut")
            return cls
        except Exception as e:
            logger.warning(f"Erreur detection auto: {e} - utilisation des valeurs par defaut")
            return cls

    @classmethod
    def from_json(cls, json_path: str = "./training_config.json"):
        """Charge la configuration depuis un fichier JSON."""
        try:
            with open(json_path, 'r') as f:
                config = json.load(f)

            cls.MODEL_ID = config["model"]["model_id"]
            cls.USE_4BIT = config["model"]["use_4bit"]
            cls.USE_CPU = config["model"]["use_cpu"]

            cls.BATCH_SIZE = config["training"]["batch_size"]
            cls.GRADIENT_ACCUMULATION = config["training"]["gradient_accumulation_steps"]
            cls.MAX_SEQ_LENGTH = config["training"]["max_seq_length"]

            cls.LORA_R = config["lora"]["r"]
            cls.LORA_ALPHA = config["lora"]["alpha"]

            cls.HARDWARE_PROFILE = config["hardware"]["profile_name"]
            cls.GPU_NAME = config["hardware"]["gpu_name"]
            cls.GPU_VRAM = config["hardware"]["gpu_vram_gb"]

            logger.info(f"Configuration chargee depuis {json_path}")
            return cls

        except FileNotFoundError:
            logger.warning(f"Fichier {json_path} non trouve - lancement de l'auto-detection")
            return cls.from_auto_detection()
        except Exception as e:
            logger.warning(f"Erreur chargement config: {e}")
            return cls

    @classmethod
    def print_config(cls):
        """Affiche la configuration actuelle."""
        print("\n" + "=" * 50)
        print("  CONFIGURATION D'ENTRAINEMENT")
        print("=" * 50)
        print(f"  Profil      : {cls.HARDWARE_PROFILE}")
        print(f"  GPU         : {cls.GPU_NAME} ({cls.GPU_VRAM:.1f} GB)")
        print(f"  Modele      : {cls.MODEL_ID}")
        print(f"  Batch size  : {cls.BATCH_SIZE}")
        print(f"  Grad accum  : {cls.GRADIENT_ACCUMULATION}")
        print(f"  Seq length  : {cls.MAX_SEQ_LENGTH}")
        print(f"  LoRA r/alpha: {cls.LORA_R}/{cls.LORA_ALPHA}")
        print(f"  4-bit quant : {cls.USE_4BIT}")
        print(f"  Mode CPU    : {cls.USE_CPU}")
        print("=" * 50 + "\n")


# ============================================================================
# VÉRIFICATION DU SYSTÈME
# ============================================================================

class SystemChecker:
    """Vérifie les prérequis système."""

    @staticmethod
    def check_gpu() -> dict:
        """Vérifie la disponibilité et les specs du GPU."""
        gpu_info = {
            "cuda_available": False,
            "device_name": "CPU",
            "vram_gb": 0,
            "recommended_model": TrainingConfig.MODEL_FALLBACK
        }

        try:
            import torch
            if torch.cuda.is_available():
                gpu_info["cuda_available"] = True
                gpu_info["device_name"] = torch.cuda.get_device_name(0)
                gpu_info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                # Choix du modèle selon VRAM
                if gpu_info["vram_gb"] >= 12:
                    gpu_info["recommended_model"] = TrainingConfig.MODEL_ID
                elif gpu_info["vram_gb"] >= 6:
                    gpu_info["recommended_model"] = "microsoft/phi-2"
                else:
                    gpu_info["recommended_model"] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

                logger.info(f"✓ GPU détecté: {gpu_info['device_name']} ({gpu_info['vram_gb']:.1f} GB)")
            else:
                logger.warning("⚠ Pas de GPU CUDA détecté - Entraînement sera lent")

        except ImportError:
            logger.error("✗ PyTorch non installé")

        return gpu_info

    @staticmethod
    def check_dependencies() -> bool:
        """Vérifie que toutes les dépendances sont installées."""
        required = [
            ("torch", "PyTorch"),
            ("transformers", "Transformers"),
            ("peft", "PEFT (LoRA)"),
            ("datasets", "Datasets"),
            ("trl", "TRL (Trainer)"),
            ("bitsandbytes", "BitsAndBytes (Quantization)")
        ]

        all_ok = True
        for module, name in required:
            try:
                __import__(module)
                logger.info(f"  ✓ {name}")
            except ImportError:
                logger.error(f"  ✗ {name} manquant - pip install {module}")
                all_ok = False

        return all_ok


# ============================================================================
# GESTIONNAIRE DE MODÈLES
# ============================================================================

class ModelManager:
    """Gestion du téléchargement et chargement des modèles."""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def check_or_download_model(self, model_id: str) -> bool:
        """
        Vérifie si le modèle est en cache, sinon le télécharge.
        """
        from huggingface_hub import snapshot_download, HfApi
        import os

        cache_path = self.config.CACHE_DIR / model_id.replace("/", "_")

        # Vérification du cache local
        if cache_path.exists() and any(cache_path.iterdir()):
            logger.info(f"✓ Modèle trouvé en cache: {cache_path}")
            return True

        # Téléchargement depuis HuggingFace
        logger.info(f"⬇ Téléchargement du modèle: {model_id}")
        logger.info("  (Cela peut prendre plusieurs minutes...)")

        try:
            # Création du dossier cache
            cache_path.mkdir(parents=True, exist_ok=True)

            # Téléchargement
            snapshot_download(
                repo_id=model_id,
                local_dir=str(cache_path),
                local_dir_use_symlinks=False,
                resume_download=True
            )

            logger.info(f"✓ Modèle téléchargé: {cache_path}")
            return True

        except Exception as e:
            logger.error(f"✗ Erreur téléchargement: {e}")
            return False

    def load_model_4bit(self, model_id: str):
        """
        Charge le modèle en quantification 4-bit avec BitsAndBytes.
        """
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig
        )

        logger.info(f"Chargement du modèle en 4-bit: {model_id}")

        # Configuration de la quantification
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.config.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True  # Double quantification pour économiser la mémoire
        )

        # Chargement du tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            padding_side="right"
        )

        # Ajout du token de padding si absent
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Chargement du modèle quantifié
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        # Préparation pour l'entraînement
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        logger.info(f"✓ Modèle chargé en 4-bit")
        logger.info(f"  Paramètres: {model.num_parameters():,}")

        return model, tokenizer


# ============================================================================
# PRÉPARATION DES DONNÉES
# ============================================================================

class DatasetPreparer:
    """Préparation du dataset pour le fine-tuning."""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def load_jsonl_dataset(self):
        """Charge le dataset JSONL généré par le pipeline ETL."""
        from datasets import Dataset

        if not self.config.DATASET_PATH.exists():
            raise FileNotFoundError(
                f"Dataset non trouvé: {self.config.DATASET_PATH}\n"
                "Exécutez d'abord le pipeline ETL: python processor.py"
            )

        # Chargement des données
        data = []
        with open(self.config.DATASET_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        logger.info(f"✓ Dataset chargé: {len(data)} samples")

        return Dataset.from_list(data)

    def format_prompt(self, example: dict) -> str:
        """
        Formate un exemple au format instruction-tuning.
        Compatible avec le format Alpaca/Mistral.
        """
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        if input_text:
            prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            prompt = f"""### Instruction:
{instruction}

### Response:
{output}"""

        return prompt

    def prepare_dataset(self, tokenizer):
        """Prépare le dataset tokenisé pour l'entraînement."""
        dataset = self.load_jsonl_dataset()

        def tokenize_function(examples):
            prompts = [self.format_prompt({"instruction": i, "input": inp, "output": o})
                      for i, inp, o in zip(examples["instruction"],
                                           examples["input"],
                                           examples["output"])]

            tokenized = tokenizer(
                prompts,
                truncation=True,
                max_length=self.config.MAX_SEQ_LENGTH,
                padding="max_length",
                return_tensors=None
            )

            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        # Tokenisation
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenisation"
        )

        # Split train/eval
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

        logger.info(f"✓ Dataset préparé:")
        logger.info(f"  Train: {len(split_dataset['train'])} samples")
        logger.info(f"  Eval: {len(split_dataset['test'])} samples")

        return split_dataset


# ============================================================================
# ENTRAÎNEUR QLORA
# ============================================================================

class QLoRATrainer:
    """Entraîneur QLoRA pour le fine-tuning."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model_manager = ModelManager(config)
        self.data_preparer = DatasetPreparer(config)

    def setup_lora(self, model):
        """Configure les adaptateurs LoRA sur le modèle."""
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        # Préparation du modèle quantifié pour l'entraînement
        model = prepare_model_for_kbit_training(model)

        # Configuration LoRA
        lora_config = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            lora_dropout=self.config.LORA_DROPOUT,
            target_modules=self.config.TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Application de LoRA
        model = get_peft_model(model, lora_config)

        # Stats des paramètres entraînables
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        logger.info(f"✓ LoRA configuré:")
        logger.info(f"  Paramètres entraînables: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        logger.info(f"  Paramètres totaux: {total_params:,}")

        return model

    def train(self, model_id: Optional[str] = None) -> dict:
        """
        Lance l'entraînement complet.

        Returns:
            dict: Résultats de l'entraînement
        """
        import torch
        from transformers import TrainingArguments, DataCollatorForLanguageModeling
        from trl import SFTTrainer

        # Sélection du modèle
        if model_id is None:
            gpu_info = SystemChecker.check_gpu()
            model_id = gpu_info["recommended_model"]

        logger.info("=" * 60)
        logger.info("DÉMARRAGE DU FINE-TUNING QLORA")
        logger.info(f"  Modèle: {model_id}")
        logger.info("=" * 60)

        # Téléchargement si nécessaire
        self.model_manager.check_or_download_model(model_id)

        # Chargement du modèle en 4-bit
        model, tokenizer = self.model_manager.load_model_4bit(model_id)

        # Configuration LoRA
        model = self.setup_lora(model)

        # Préparation du dataset
        dataset = self.data_preparer.prepare_dataset(tokenizer)

        # Création du dossier output
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Arguments d'entraînement
        training_args = TrainingArguments(
            output_dir=str(self.config.OUTPUT_DIR),
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION,
            learning_rate=self.config.LEARNING_RATE,
            warmup_ratio=self.config.WARMUP_RATIO,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            fp16=True,
            optim="paged_adamw_8bit",
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            max_seq_length=self.config.MAX_SEQ_LENGTH,
        )

        # Entraînement
        logger.info("\n🚀 Lancement de l'entraînement...")
        train_result = trainer.train()

        # Sauvegarde des adaptateurs uniquement
        adapter_path = self.config.OUTPUT_DIR / "final_adapter"
        model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))

        logger.info(f"\n✓ Adaptateurs sauvegardés: {adapter_path}")

        # Rapport final
        result = {
            "status": "success",
            "model_id": model_id,
            "adapter_path": str(adapter_path),
            "train_loss": train_result.training_loss,
            "epochs": self.config.NUM_EPOCHS,
            "samples_trained": len(dataset["train"])
        }

        logger.info("\n" + "=" * 60)
        logger.info("FINE-TUNING TERMINÉ")
        logger.info(f"  Loss finale: {train_result.training_loss:.4f}")
        logger.info(f"  Adaptateurs: {adapter_path}")
        logger.info("=" * 60)

        return result


# ============================================================================
# INFÉRENCE AVEC ADAPTATEURS
# ============================================================================

class InferenceEngine:
    """Moteur d'inférence avec les adaptateurs LoRA."""

    def __init__(self, adapter_path: str):
        self.adapter_path = Path(adapter_path)
        self.model = None
        self.tokenizer = None

    def load(self):
        """Charge le modèle avec les adaptateurs."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        logger.info(f"Chargement des adaptateurs: {self.adapter_path}")

        # Lecture de la config pour trouver le modèle de base
        adapter_config_path = self.adapter_path / "adapter_config.json"
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)

        base_model_id = adapter_config.get("base_model_name_or_path", TrainingConfig.MODEL_ID)

        # Quantification 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        # Chargement du modèle de base
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # Application des adaptateurs
        self.model = PeftModel.from_pretrained(base_model, str(self.adapter_path))
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.adapter_path))

        logger.info("✓ Modèle chargé avec adaptateurs")

    def predict(self, instruction: str, input_text: str = "") -> str:
        """
        Génère une prédiction.

        Args:
            instruction: L'instruction pour le modèle
            input_text: Le contexte/input optionnel

        Returns:
            La réponse générée
        """
        import torch

        if self.model is None:
            self.load()

        # Formatage du prompt
        if input_text:
            prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        else:
            prompt = f"""### Instruction:
{instruction}

### Response:
"""

        # Tokenisation
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Génération
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Décodage
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extraction de la réponse uniquement
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()

        return response


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

def main():
    """Point d'entree principal."""
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tuning QLoRA pour PMSI - Detection auto du materiel")
    parser.add_argument("--check", action="store_true", help="Verifier le systeme uniquement")
    parser.add_argument("--detect", action="store_true", help="Detecter le materiel et afficher la config optimale")
    parser.add_argument("--auto-config", action="store_true", help="Utiliser la detection auto pour configurer l'entrainement")
    parser.add_argument("--config", type=str, default=None, help="Charger la config depuis un fichier JSON")
    parser.add_argument("--model", type=str, default=None, help="ID du modele HuggingFace (ecrase l'auto-detection)")
    parser.add_argument("--predict", type=str, default=None, help="Faire une prediction avec les adaptateurs")
    parser.add_argument("--adapter-path", type=str, default="./models/adapters/final_adapter",
                       help="Chemin vers les adaptateurs")
    args = parser.parse_args()

    if args.check:
        # Verification systeme
        print("\n=== VERIFICATION DU SYSTEME ===\n")
        print("Dependances:")
        SystemChecker.check_dependencies()
        print("\nGPU:")
        gpu_info = SystemChecker.check_gpu()
        print(f"\nModele recommande: {gpu_info['recommended_model']}")

    elif args.detect:
        # Detection materiel uniquement
        try:
            from hardware_detector import detect_and_configure
            detect_and_configure(save_config=True, verbose=True)
        except ImportError:
            logger.error("Module hardware_detector.py non trouve")

    elif args.predict:
        # Mode inference
        engine = InferenceEngine(args.adapter_path)
        response = engine.predict(
            instruction="Detecter une anomalie de parcours patient en psychiatrie",
            input_text=args.predict
        )
        print(f"\nReponse:\n{response}")

    else:
        # Mode entrainement
        # Configuration selon les arguments
        if args.config:
            config = TrainingConfig.from_json(args.config)
        elif args.auto_config:
            config = TrainingConfig.from_auto_detection()
        else:
            # Par defaut: auto-detection
            config = TrainingConfig.from_auto_detection()

        # Afficher la configuration
        config.print_config()

        trainer = QLoRATrainer(config)

        # Verification des prerequis
        if not SystemChecker.check_dependencies():
            logger.error("Installez les dependances manquantes avant de continuer")
            return

        # Lancement de l'entrainement
        # Si --model specifie, il ecrase la detection auto
        model_to_use = args.model if args.model else config.MODEL_ID
        result = trainer.train(model_id=model_to_use)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
