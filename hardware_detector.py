"""
===============================================================================
HARDWARE_DETECTOR.PY - Detection automatique du materiel pour l'entrainement
===============================================================================
Detecte GPU, VRAM, CPU, RAM et choisit la configuration optimale
Auteur: Adam B. | DIM - Data Intelligence Medicale
===============================================================================
"""

import os
import sys
import json
import subprocess
import platform
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATIONS PRE-DEFINIES PAR NIVEAU DE HARDWARE
# =============================================================================

HARDWARE_PROFILES = {
    # =====================
    # NIVEAU 1: Ultra-leger (CPU only ou GPU < 4GB)
    # =====================
    "cpu_only": {
        "name": "CPU Only",
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "model_size": "1.1B",
        "batch_size": 1,
        "gradient_accumulation": 16,
        "max_seq_length": 128,
        "lora_r": 4,
        "lora_alpha": 8,
        "use_4bit": False,  # Pas de quantification GPU sur CPU
        "use_cpu": True,
        "estimated_vram": 0,
        "estimated_time_factor": 10.0,  # 10x plus lent que GPU
    },

    # =====================
    # NIVEAU 2: GPU Entry-level (4GB VRAM) - GTX 1650, etc.
    # =====================
    "gpu_4gb": {
        "name": "GPU 4GB (GTX 1650, etc.)",
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "model_size": "1.1B",
        "batch_size": 1,
        "gradient_accumulation": 8,
        "max_seq_length": 256,
        "lora_r": 8,
        "lora_alpha": 16,
        "use_4bit": True,
        "use_cpu": False,
        "estimated_vram": 3.5,
        "estimated_time_factor": 3.0,
    },

    # =====================
    # NIVEAU 3: GPU Mid-range (6GB VRAM) - RTX 3050, RTX 2060, etc.
    # =====================
    "gpu_6gb": {
        "name": "GPU 6GB (RTX 3050, RTX 2060)",
        "model_id": "microsoft/phi-2",
        "model_size": "2.7B",
        "batch_size": 1,
        "gradient_accumulation": 8,
        "max_seq_length": 256,
        "lora_r": 8,
        "lora_alpha": 16,
        "use_4bit": True,
        "use_cpu": False,
        "estimated_vram": 5.0,
        "estimated_time_factor": 2.0,
    },

    # =====================
    # NIVEAU 4: GPU Standard (8GB VRAM) - RTX 3060, RTX 3070, etc.
    # =====================
    "gpu_8gb": {
        "name": "GPU 8GB (RTX 3060, RTX 3070)",
        "model_id": "microsoft/phi-2",
        "model_size": "2.7B",
        "batch_size": 2,
        "gradient_accumulation": 4,
        "max_seq_length": 512,
        "lora_r": 16,
        "lora_alpha": 32,
        "use_4bit": True,
        "use_cpu": False,
        "estimated_vram": 7.0,
        "estimated_time_factor": 1.5,
    },

    # =====================
    # NIVEAU 5: GPU Pro (12GB VRAM) - RTX 3080, RTX 4070, etc.
    # =====================
    "gpu_12gb": {
        "name": "GPU 12GB (RTX 3080, RTX 4070)",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "model_size": "7B",
        "batch_size": 2,
        "gradient_accumulation": 4,
        "max_seq_length": 512,
        "lora_r": 16,
        "lora_alpha": 32,
        "use_4bit": True,
        "use_cpu": False,
        "estimated_vram": 10.0,
        "estimated_time_factor": 1.0,
    },

    # =====================
    # NIVEAU 6: GPU Haute performance (16GB+ VRAM) - RTX 4080, A4000, etc.
    # =====================
    "gpu_16gb": {
        "name": "GPU 16GB+ (RTX 4080, A4000)",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "model_size": "7B",
        "batch_size": 4,
        "gradient_accumulation": 2,
        "max_seq_length": 1024,
        "lora_r": 32,
        "lora_alpha": 64,
        "use_4bit": True,
        "use_cpu": False,
        "estimated_vram": 14.0,
        "estimated_time_factor": 0.8,
    },

    # =====================
    # NIVEAU 7: GPU Pro+ (24GB+ VRAM) - RTX 4090, A5000, etc.
    # =====================
    "gpu_24gb": {
        "name": "GPU 24GB+ (RTX 4090, A5000)",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "model_size": "7B",
        "batch_size": 8,
        "gradient_accumulation": 1,
        "max_seq_length": 2048,
        "lora_r": 64,
        "lora_alpha": 128,
        "use_4bit": True,  # Peut etre False si besoin de precision
        "use_cpu": False,
        "estimated_vram": 20.0,
        "estimated_time_factor": 0.5,
    },
}


# =============================================================================
# CLASSES DE DETECTION
# =============================================================================

@dataclass
class GPUInfo:
    """Information sur le GPU detecte."""
    available: bool = False
    name: str = "N/A"
    vram_gb: float = 0.0
    cuda_version: str = "N/A"
    driver_version: str = "N/A"
    compute_capability: str = "N/A"
    is_nvidia: bool = False
    is_amd: bool = False


@dataclass
class CPUInfo:
    """Information sur le CPU."""
    name: str = "Unknown"
    cores: int = 1
    threads: int = 1


@dataclass
class RAMInfo:
    """Information sur la RAM."""
    total_gb: float = 0.0
    available_gb: float = 0.0


@dataclass
class SystemInfo:
    """Information complete du systeme."""
    gpu: GPUInfo
    cpu: CPUInfo
    ram: RAMInfo
    os_name: str = ""
    os_version: str = ""
    python_version: str = ""
    recommended_profile: str = "cpu_only"
    profile_config: dict = None


class HardwareDetector:
    """Detecteur automatique de materiel."""

    def __init__(self):
        self.gpu_info = GPUInfo()
        self.cpu_info = CPUInfo()
        self.ram_info = RAMInfo()

    def detect_all(self) -> SystemInfo:
        """Detecte tout le materiel et retourne la config optimale."""
        self._detect_gpu()
        self._detect_cpu()
        self._detect_ram()

        # Determine le profil optimal
        profile_name = self._select_optimal_profile()

        return SystemInfo(
            gpu=self.gpu_info,
            cpu=self.cpu_info,
            ram=self.ram_info,
            os_name=platform.system(),
            os_version=platform.version(),
            python_version=platform.python_version(),
            recommended_profile=profile_name,
            profile_config=HARDWARE_PROFILES[profile_name]
        )

    def _detect_gpu(self):
        """Detecte le GPU (NVIDIA ou AMD)."""
        # Essayer d'abord avec PyTorch CUDA
        if self._detect_nvidia_pytorch():
            return

        # Sinon essayer nvidia-smi directement
        if self._detect_nvidia_smi():
            return

        # Essayer AMD ROCm
        if self._detect_amd():
            return

        # Pas de GPU trouve
        self.gpu_info = GPUInfo(available=False, name="Aucun GPU detecte")

    def _detect_nvidia_pytorch(self) -> bool:
        """Detecte GPU NVIDIA via PyTorch."""
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_info.available = True
                self.gpu_info.is_nvidia = True
                self.gpu_info.name = torch.cuda.get_device_name(0)
                self.gpu_info.vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.gpu_info.cuda_version = torch.version.cuda or "N/A"

                # Compute capability
                props = torch.cuda.get_device_properties(0)
                self.gpu_info.compute_capability = f"{props.major}.{props.minor}"

                return True
        except Exception as e:
            logger.debug(f"PyTorch CUDA detection failed: {e}")
        return False

    def _detect_nvidia_smi(self) -> bool:
        """Detecte GPU NVIDIA via nvidia-smi."""
        try:
            # Chemins possibles pour nvidia-smi
            nvidia_smi_paths = [
                "nvidia-smi",
                "C:\\Windows\\System32\\nvidia-smi.exe",
                "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe",
                "/usr/bin/nvidia-smi"
            ]

            for nvidia_smi in nvidia_smi_paths:
                try:
                    result = subprocess.run(
                        [nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )

                    if result.returncode == 0:
                        parts = result.stdout.strip().split(", ")
                        if len(parts) >= 3:
                            self.gpu_info.available = True
                            self.gpu_info.is_nvidia = True
                            self.gpu_info.name = parts[0].strip()
                            self.gpu_info.vram_gb = float(parts[1].strip()) / 1024  # MB to GB
                            self.gpu_info.driver_version = parts[2].strip()
                            return True
                except FileNotFoundError:
                    continue
                except subprocess.TimeoutExpired:
                    continue
        except Exception as e:
            logger.debug(f"nvidia-smi detection failed: {e}")
        return False

    def _detect_amd(self) -> bool:
        """Detecte GPU AMD via ROCm."""
        try:
            # Verifier rocm-smi
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                self.gpu_info.available = True
                self.gpu_info.is_amd = True
                self.gpu_info.name = "AMD GPU (ROCm)"

                # Parser la sortie pour extraire la VRAM
                for line in result.stdout.split('\n'):
                    if 'Total Memory' in line:
                        # Extraire la valeur en MB
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.isdigit():
                                self.gpu_info.vram_gb = float(part) / 1024
                                break
                return True
        except Exception as e:
            logger.debug(f"ROCm detection failed: {e}")
        return False

    def _detect_cpu(self):
        """Detecte les informations CPU."""
        self.cpu_info.cores = os.cpu_count() or 1
        self.cpu_info.threads = self.cpu_info.cores

        # Nom du CPU
        if platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        self.cpu_info.name = lines[1].strip()
            except:
                pass
        elif platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            self.cpu_info.name = line.split(":")[1].strip()
                            break
            except:
                pass
        elif platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    self.cpu_info.name = result.stdout.strip()
            except:
                pass

    def _detect_ram(self):
        """Detecte les informations RAM."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            self.ram_info.total_gb = mem.total / (1024**3)
            self.ram_info.available_gb = mem.available / (1024**3)
        except ImportError:
            # Fallback sans psutil
            if platform.system() == "Windows":
                try:
                    result = subprocess.run(
                        ["wmic", "OS", "get", "TotalVisibleMemorySize"],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if len(lines) > 1:
                            kb = int(lines[1].strip())
                            self.ram_info.total_gb = kb / (1024**2)
                except:
                    pass
            elif platform.system() == "Linux":
                try:
                    with open("/proc/meminfo", "r") as f:
                        for line in f:
                            if "MemTotal" in line:
                                kb = int(line.split()[1])
                                self.ram_info.total_gb = kb / (1024**2)
                            elif "MemAvailable" in line:
                                kb = int(line.split()[1])
                                self.ram_info.available_gb = kb / (1024**2)
                except:
                    pass

    def _select_optimal_profile(self) -> str:
        """Selectionne le profil optimal selon le materiel detecte."""

        # Pas de GPU -> CPU only
        if not self.gpu_info.available:
            return "cpu_only"

        vram = self.gpu_info.vram_gb

        # Selection basee sur la VRAM
        if vram >= 24:
            return "gpu_24gb"
        elif vram >= 16:
            return "gpu_16gb"
        elif vram >= 12:
            return "gpu_12gb"
        elif vram >= 8:
            return "gpu_8gb"
        elif vram >= 6:
            return "gpu_6gb"
        elif vram >= 4:
            return "gpu_4gb"
        else:
            return "cpu_only"


# =============================================================================
# GENERATEUR DE CONFIGURATION
# =============================================================================

class ConfigGenerator:
    """Genere la configuration d'entrainement optimale."""

    def __init__(self, system_info: SystemInfo):
        self.system_info = system_info
        self.profile = system_info.profile_config

    def generate_training_config(self) -> dict:
        """Genere la configuration complete pour l'entrainement."""
        return {
            "model": {
                "model_id": self.profile["model_id"],
                "model_size": self.profile["model_size"],
                "use_4bit": self.profile["use_4bit"],
                "use_cpu": self.profile["use_cpu"],
            },
            "training": {
                "batch_size": self.profile["batch_size"],
                "gradient_accumulation_steps": self.profile["gradient_accumulation"],
                "max_seq_length": self.profile["max_seq_length"],
                "learning_rate": 2e-4,
                "num_epochs": 3,
                "warmup_ratio": 0.03,
                "fp16": not self.profile["use_cpu"],
                "optim": "paged_adamw_8bit" if not self.profile["use_cpu"] else "adamw_torch",
            },
            "lora": {
                "r": self.profile["lora_r"],
                "alpha": self.profile["lora_alpha"],
                "dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            "hardware": {
                "profile_name": self.system_info.recommended_profile,
                "profile_display_name": self.profile["name"],
                "gpu_name": self.system_info.gpu.name,
                "gpu_vram_gb": self.system_info.gpu.vram_gb,
                "cpu_cores": self.system_info.cpu.cores,
                "ram_gb": self.system_info.ram.total_gb,
                "estimated_vram_usage": self.profile["estimated_vram"],
                "estimated_time_factor": self.profile["estimated_time_factor"],
            }
        }

    def save_config(self, path: str = "./training_config.json"):
        """Sauvegarde la configuration dans un fichier JSON."""
        config = self.generate_training_config()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return config


# =============================================================================
# AFFICHAGE DES RESULTATS
# =============================================================================

def print_detection_results(system_info: SystemInfo):
    """Affiche les resultats de la detection de maniere formatee."""

    print("\n" + "=" * 70)
    print("   DETECTION AUTOMATIQUE DU MATERIEL - DIM")
    print("=" * 70)

    # GPU
    print("\n[GPU]")
    if system_info.gpu.available:
        print(f"  Nom           : {system_info.gpu.name}")
        print(f"  VRAM          : {system_info.gpu.vram_gb:.1f} GB")
        if system_info.gpu.is_nvidia:
            print(f"  Type          : NVIDIA CUDA")
            if system_info.gpu.cuda_version != "N/A":
                print(f"  CUDA          : {system_info.gpu.cuda_version}")
            if system_info.gpu.compute_capability != "N/A":
                print(f"  Compute Cap.  : {system_info.gpu.compute_capability}")
        elif system_info.gpu.is_amd:
            print(f"  Type          : AMD ROCm")
    else:
        print(f"  Status        : Aucun GPU detecte")
        print(f"  Mode          : CPU uniquement (plus lent)")

    # CPU
    print("\n[CPU]")
    print(f"  Nom           : {system_info.cpu.name}")
    print(f"  Coeurs        : {system_info.cpu.cores}")

    # RAM
    print("\n[RAM]")
    print(f"  Total         : {system_info.ram.total_gb:.1f} GB")
    if system_info.ram.available_gb > 0:
        print(f"  Disponible    : {system_info.ram.available_gb:.1f} GB")

    # Systeme
    print("\n[SYSTEME]")
    print(f"  OS            : {system_info.os_name} {system_info.os_version[:30]}...")
    print(f"  Python        : {system_info.python_version}")

    # Configuration recommandee
    profile = system_info.profile_config
    print("\n" + "-" * 70)
    print("   CONFIGURATION OPTIMALE DETECTEE")
    print("-" * 70)
    print(f"\n  Profil        : {profile['name']}")
    print(f"  Modele        : {profile['model_id']} ({profile['model_size']})")
    print(f"  Batch size    : {profile['batch_size']}")
    print(f"  Grad accum    : {profile['gradient_accumulation']}")
    print(f"  Seq length    : {profile['max_seq_length']}")
    print(f"  LoRA rank     : {profile['lora_r']}")
    print(f"  Quantif 4-bit : {'Oui' if profile['use_4bit'] else 'Non'}")
    print(f"  Mode CPU      : {'Oui' if profile['use_cpu'] else 'Non'}")

    if profile['estimated_vram'] > 0:
        print(f"\n  VRAM estimee  : ~{profile['estimated_vram']:.1f} GB")

    # Avertissements
    if system_info.gpu.available and system_info.gpu.vram_gb < profile['estimated_vram']:
        print(f"\n  [!] ATTENTION: VRAM limitee ({system_info.gpu.vram_gb:.1f} GB)")
        print(f"      Risque d'erreur OOM. Reduisez batch_size ou max_seq_length si necessaire.")

    print("\n" + "=" * 70)


def get_optimal_config() -> dict:
    """
    Fonction principale pour obtenir la configuration optimale.
    Retourne un dictionnaire avec tous les parametres.
    """
    detector = HardwareDetector()
    system_info = detector.detect_all()
    generator = ConfigGenerator(system_info)
    return generator.generate_training_config()


def detect_and_configure(save_config: bool = True, verbose: bool = True) -> Tuple[SystemInfo, dict]:
    """
    Detecte le materiel et genere la configuration optimale.

    Args:
        save_config: Sauvegarder la config dans un fichier JSON
        verbose: Afficher les resultats de detection

    Returns:
        Tuple (SystemInfo, config_dict)
    """
    detector = HardwareDetector()
    system_info = detector.detect_all()

    if verbose:
        print_detection_results(system_info)

    generator = ConfigGenerator(system_info)
    config = generator.generate_training_config()

    if save_config:
        config_path = Path("./training_config.json")
        generator.save_config(str(config_path))
        if verbose:
            print(f"\nConfiguration sauvegardee: {config_path}")

    return system_info, config


# =============================================================================
# POINT D'ENTREE
# =============================================================================

def main():
    """Point d'entree principal."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detection automatique du materiel pour l'entrainement IA"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Sortie en format JSON uniquement"
    )
    parser.add_argument(
        "--save",
        type=str,
        default="./training_config.json",
        help="Chemin pour sauvegarder la configuration"
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=list(HARDWARE_PROFILES.keys()),
        help="Forcer un profil specifique"
    )

    args = parser.parse_args()

    detector = HardwareDetector()
    system_info = detector.detect_all()

    # Forcer un profil si specifie
    if args.profile:
        system_info.recommended_profile = args.profile
        system_info.profile_config = HARDWARE_PROFILES[args.profile]

    generator = ConfigGenerator(system_info)
    config = generator.generate_training_config()

    if args.json:
        print(json.dumps(config, indent=2, ensure_ascii=False))
    else:
        print_detection_results(system_info)
        generator.save_config(args.save)
        print(f"\nConfiguration sauvegardee: {args.save}")
        print("\nPour lancer l'entrainement avec ces parametres:")
        print("  python finetune_lora.py --auto-config")


if __name__ == "__main__":
    main()
