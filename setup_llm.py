"""
===============================================================================
SETUP_LLM.PY - Configuration et entrainement du LLM DIM-PMSI
===============================================================================
DIM - Data Intelligence Medicale
Script pour configurer et personnaliser le modele IA local
===============================================================================
"""

import json
import os
import subprocess
import sys
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
import urllib.request

# Configuration des couleurs pour le terminal
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_success(text: str):
    print(f"{Colors.GREEN}[OK] {text}{Colors.END}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}[!] {text}{Colors.END}")


def print_error(text: str):
    print(f"{Colors.RED}[X] {text}{Colors.END}")


def print_info(text: str):
    print(f"{Colors.BLUE}[i] {text}{Colors.END}")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LLMConfig:
    """Configuration du LLM."""
    # Modeles de base recommandes (du plus leger au plus lourd)
    BASE_MODELS = {
        "tiny": "tinyllama:1.1b",      # 1.1B params - Tres rapide
        "small": "phi:2.7b",            # 2.7B params - Bon compromis
        "medium": "mistral:7b",         # 7B params - Recommande
        "large": "llama2:13b"           # 13B params - Meilleure qualite
    }

    # Modele par defaut
    DEFAULT_MODEL = "mistral:7b-instruct"

    # Nom du modele personnalise
    CUSTOM_MODEL_NAME = "dim-pmsi"

    # Chemins
    MODELS_DIR = Path("./models")
    OUTPUT_DIR = Path("./output")

    # System prompt specialise DIM/PMSI
    SYSTEM_PROMPT = """Tu es un assistant expert en DIM (Departement d'Information Medicale) specialise en psychiatrie francaise.
Tu analyses les donnees PMSI (Programme de Medicalisation des Systemes d'Information) pour detecter les anomalies organisationnelles.

CONTEXTE PMSI PSYCHIATRIE:
- RAA = Resume d'Activite Ambulatoire (actes de jour)
- RPS = Resume Par Sequence (hospitalisation psychiatrique)
- UM = Unite Medicale (service)
- FINESS = Identifiant unique de l'etablissement
- GHT = Groupement Hospitalier de Territoire

REGLES D'ANALYSE DES EPISODES AMBULATOIRES:
1. Un episode ambulatoire NORMAL dure 1 jour maximum
2. Si la duree depasse 1 jour = ANOMALIE DETECTEE
3. Types d'anomalies possibles:
   - DUREE_EXCESSIVE: Episode trop long (> seuil)
   - CHEVAUCHEMENT: Dates incohérentes
   - MULTI_SITE: Patient sur plusieurs sites simultanement
   - CODAGE_SUSPECT: Patterns de codage inhabituels

CAUSES POSSIBLES DES ANOMALIES:
- Probleme d'aval (pas de place disponible)
- Erreur de codage PMSI
- Desorganisation du parcours patient
- Attente de transfert inter-etablissement

FORMAT DE REPONSE:
Pour chaque analyse, reponds de facon structuree:
- STATUT: NORMAL ou ANOMALIE
- TYPE: Type d'anomalie si applicable
- DUREE: Duree observee
- ANALYSE: Explication concise
- RECOMMANDATION: Action a entreprendre

Tu reponds toujours en francais, de maniere professionnelle et concise."""


# =============================================================================
# VERIFICATEUR SYSTEME
# =============================================================================

class SystemChecker:
    """Verifie les prerequis du systeme."""

    @staticmethod
    def check_ollama_installed() -> bool:
        """Verifie si Ollama est installe."""
        return shutil.which("ollama") is not None

    @staticmethod
    def check_ollama_running() -> bool:
        """Verifie si Ollama est en cours d'execution."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    @staticmethod
    def start_ollama() -> bool:
        """Demarre Ollama si necessaire."""
        if SystemChecker.check_ollama_running():
            return True

        print_info("Demarrage d'Ollama...")

        try:
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

            # Attendre que le serveur demarre
            for i in range(15):
                time.sleep(1)
                print(f"  Attente... {i+1}/15", end='\r')
                if SystemChecker.check_ollama_running():
                    print()
                    print_success("Ollama demarre")
                    return True

            print()
            print_error("Timeout: Ollama n'a pas demarre")
            return False

        except Exception as e:
            print_error(f"Erreur demarrage Ollama: {e}")
            return False

    @staticmethod
    def get_available_models() -> List[str]:
        """Liste les modeles disponibles."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m.get("name", "") for m in models]
        except:
            pass
        return []

    @staticmethod
    def check_gpu() -> Dict[str, Any]:
        """Verifie la presence d'un GPU."""
        gpu_info = {
            "available": False,
            "name": "CPU Only",
            "memory_gb": 0
        }

        try:
            import torch
            if torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["name"] = torch.cuda.get_device_name(0)
                gpu_info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            pass

        return gpu_info


# =============================================================================
# GESTIONNAIRE DE MODELES
# =============================================================================

class ModelManager:
    """Gere le telechargement et la creation des modeles."""

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self.config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def pull_base_model(self, model_name: str = None) -> bool:
        """Telecharge un modele de base via Ollama."""
        model = model_name or self.config.DEFAULT_MODEL

        print_info(f"Telechargement du modele: {model}")
        print_info("(Cela peut prendre plusieurs minutes selon votre connexion)")

        try:
            result = subprocess.run(
                ["ollama", "pull", model],
                capture_output=False,
                timeout=1800  # 30 minutes max
            )

            if result.returncode == 0:
                print_success(f"Modele {model} telecharge")
                return True
            else:
                print_error(f"Echec du telechargement")
                return False

        except subprocess.TimeoutExpired:
            print_error("Timeout: Le telechargement a pris trop de temps")
            return False
        except Exception as e:
            print_error(f"Erreur: {e}")
            return False

    def create_modelfile(self, base_model: str = None) -> Path:
        """Cree le Modelfile personnalise pour DIM-PMSI."""
        base = base_model or self.config.DEFAULT_MODEL

        modelfile_content = f'''FROM {base}

SYSTEM """{self.config.SYSTEM_PROMPT}"""

# Parametres optimises pour l'analyse medicale
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 512
PARAMETER repeat_penalty 1.1
PARAMETER stop "###"
PARAMETER stop "---"

# Template de conversation
TEMPLATE """{{{{ if .System }}}}### System:
{{{{ .System }}}}

{{{{ end }}}}### User:
{{{{ .Prompt }}}}

### Assistant:
"""
'''

        modelfile_path = self.config.MODELS_DIR / "DIM_PMSI_Modelfile"

        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)

        print_success(f"Modelfile cree: {modelfile_path}")
        return modelfile_path

    def create_custom_model(self, model_name: str = None, base_model: str = None) -> bool:
        """Cree un modele Ollama personnalise."""
        name = model_name or self.config.CUSTOM_MODEL_NAME

        # Creer le Modelfile
        modelfile_path = self.create_modelfile(base_model)

        print_info(f"Creation du modele personnalise: {name}")

        try:
            result = subprocess.run(
                ["ollama", "create", name, "-f", str(modelfile_path)],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                print_success(f"Modele '{name}' cree avec succes!")
                return True
            else:
                print_error(f"Echec: {result.stderr}")
                return False

        except Exception as e:
            print_error(f"Erreur: {e}")
            return False

    def test_model(self, model_name: str = None) -> bool:
        """Teste le modele avec une requete simple."""
        name = model_name or self.config.CUSTOM_MODEL_NAME

        print_info(f"Test du modele {name}...")

        test_prompt = """Analyse cet episode ambulatoire:
- Patient: P001
- Unite Medicale: PSY-AMB
- Date debut: 2024-01-15
- Date fin: 2024-01-18
- Duree: 3 jours
- Nombre d'actes: 5"""

        try:
            import requests

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": name,
                    "prompt": test_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 256
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json().get("response", "")

                print_success("Test reussi!")
                print("\n" + "-"*50)
                print("REPONSE DU MODELE:")
                print("-"*50)
                print(result[:500] + ("..." if len(result) > 500 else ""))
                print("-"*50 + "\n")

                # Verification basique de la reponse
                if "ANOMALIE" in result.upper() or "DUREE" in result.upper():
                    print_success("Le modele detecte correctement les anomalies")
                    return True
                else:
                    print_warning("Le modele repond mais la detection peut etre amelioree")
                    return True
            else:
                print_error(f"Erreur API: {response.text}")
                return False

        except Exception as e:
            print_error(f"Erreur test: {e}")
            return False


# =============================================================================
# PREPARATEUR DE DONNEES D'ENTRAINEMENT
# =============================================================================

class TrainingDataPreparer:
    """Prepare les donnees pour le fine-tuning futur."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("./output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_training_examples(self) -> List[Dict]:
        """Cree des exemples d'entrainement synthetiques."""
        examples = []

        # Exemples d'episodes normaux
        normal_examples = [
            {
                "instruction": "Analyse cet episode de soins ambulatoires psychiatriques.",
                "input": "Patient: P001, UM: PSY-AMB-01, Site: 330000001, Dates: 2024-03-15 -> 2024-03-15, Duree: 0 jours, Actes: 2",
                "output": "STATUT: NORMAL\nTYPE: -\nDUREE: 0 jours (meme jour)\nANALYSE: Episode ambulatoire standard. Le patient a ete vu et traite dans la journee.\nRECOMMANDATION: Aucune action requise."
            },
            {
                "instruction": "Analyse cet episode de soins ambulatoires psychiatriques.",
                "input": "Patient: P002, UM: CMP-ADULTE, Site: 330000002, Dates: 2024-03-20 -> 2024-03-20, Duree: 0 jours, Actes: 1",
                "output": "STATUT: NORMAL\nTYPE: -\nDUREE: 0 jours\nANALYSE: Consultation ambulatoire classique en CMP. Duree conforme aux standards.\nRECOMMANDATION: Aucune action requise."
            }
        ]

        # Exemples d'anomalies
        anomaly_examples = [
            {
                "instruction": "Analyse cet episode de soins ambulatoires psychiatriques.",
                "input": "Patient: P003, UM: PSY-AMB-02, Site: 330000001, Dates: 2024-03-10 -> 2024-03-15, Duree: 5 jours, Actes: 8",
                "output": "STATUT: ANOMALIE\nTYPE: DUREE_EXCESSIVE\nDUREE: 5 jours (seuil: 1 jour)\nANALYSE: Episode ambulatoire anormalement long. La duree de 5 jours depasse largement le seuil de 1 jour pour les soins ambulatoires.\nRECOMMANDATION: Verifier le codage PMSI. Si correct, evaluer s'il s'agit d'un probleme d'aval (attente de place en hospitalisation) ou d'une desorganisation du parcours patient."
            },
            {
                "instruction": "Analyse cet episode de soins ambulatoires psychiatriques.",
                "input": "Patient: P004, UM: HDJ-PSY, Site: 330000003, Dates: 2024-02-01 -> 2024-02-10, Duree: 9 jours, Actes: 15",
                "output": "STATUT: ANOMALIE\nTYPE: DUREE_EXCESSIVE\nANALYSE: Episode de 9 jours en hopital de jour. Bien que l'HDJ permette des sejours plus longs, cette duree necessite verification.\nRECOMMANDATION: 1) Verifier si le codage RPS (hospitalisation) ne serait pas plus adapte. 2) Evaluer le parcours patient pour optimisation."
            },
            {
                "instruction": "Analyse cet episode de soins ambulatoires psychiatriques.",
                "input": "Patient: P005, UM: URGENCES-PSY, Site: 330000001, Dates: 2024-04-01 -> 2024-04-03, Duree: 2 jours, Actes: 3",
                "output": "STATUT: ANOMALIE\nTYPE: DUREE_EXCESSIVE\nDUREE: 2 jours aux urgences\nANALYSE: Sejour prolonge aux urgences psychiatriques. Cela indique probablement un probleme d'aval (pas de lit disponible en unite d'hospitalisation).\nRECOMMANDATION: 1) Verifier la disponibilite des lits en aval. 2) Signaler au DIM pour suivi des tensions de capacite. 3) Envisager convention avec etablissements partenaires."
            }
        ]

        examples.extend(normal_examples)
        examples.extend(anomaly_examples)

        return examples

    def export_to_jsonl(self, examples: List[Dict] = None) -> Path:
        """Exporte les exemples au format JSONL pour fine-tuning."""
        if examples is None:
            examples = self.create_training_examples()

        output_path = self.output_dir / "train_dataset.jsonl"

        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        print_success(f"Dataset d'entrainement cree: {output_path}")
        print_info(f"  {len(examples)} exemples")

        return output_path

    def load_from_episodes(self, episodes_file: Path) -> List[Dict]:
        """Charge des exemples depuis un fichier d'episodes."""
        import polars as pl

        if not episodes_file.exists():
            print_error(f"Fichier non trouve: {episodes_file}")
            return []

        df = pl.read_csv(episodes_file, separator=";")
        examples = []

        for row in df.iter_rows(named=True):
            patient = row.get("NO_PATIENT", "ANON")
            um = row.get("CODE_UM", "NC")
            site = row.get("FINESS_PMSI", "NC")
            debut = str(row.get("DATE_DEBUT_EPISODE", ""))[:10]
            fin = str(row.get("DATE_FIN_EPISODE", debut))[:10]
            duree = row.get("DUREE_EPISODE_JOURS", 0) or 0
            nb_actes = row.get("NB_ACTES_EPISODE", 1) or 1
            is_anomaly = row.get("FLAG_ANOMALIE", False)

            input_text = f"Patient: {patient}, UM: {um}, Site: {site}, Dates: {debut} -> {fin}, Duree: {int(duree)} jours, Actes: {int(nb_actes)}"

            if is_anomaly:
                output = f"STATUT: ANOMALIE\nTYPE: DUREE_EXCESSIVE\nDUREE: {int(duree)} jours\nANALYSE: Episode ambulatoire depassant le seuil de duree normale. Necessite verification.\nRECOMMANDATION: Verifier le codage et le parcours patient."
            else:
                output = f"STATUT: NORMAL\nTYPE: -\nDUREE: {int(duree)} jours\nANALYSE: Episode conforme aux standards ambulatoires.\nRECOMMANDATION: Aucune action requise."

            examples.append({
                "instruction": "Analyse cet episode de soins ambulatoires psychiatriques.",
                "input": input_text,
                "output": output
            })

        print_success(f"Charge {len(examples)} exemples depuis {episodes_file.name}")
        return examples


# =============================================================================
# SETUP COMPLET
# =============================================================================

class LLMSetup:
    """Orchestrateur de l'installation complete du LLM."""

    def __init__(self):
        self.config = LLMConfig()
        self.checker = SystemChecker()
        self.manager = ModelManager(self.config)
        self.data_preparer = TrainingDataPreparer()

    def run_full_setup(self, model_size: str = "medium") -> bool:
        """Execute l'installation complete."""
        print_header("INSTALLATION DU LLM DIM-PMSI")

        # 1. Verification Ollama
        print_info("Etape 1/5: Verification d'Ollama")

        if not self.checker.check_ollama_installed():
            print_error("Ollama n'est pas installe!")
            print_info("Telechargez Ollama: https://ollama.ai/download")
            print_info("Puis relancez ce script.")
            return False

        print_success("Ollama est installe")

        # 2. Demarrage Ollama
        print_info("Etape 2/5: Demarrage d'Ollama")

        if not self.checker.start_ollama():
            print_error("Impossible de demarrer Ollama")
            return False

        # 3. Telechargement du modele de base
        print_info("Etape 3/5: Telechargement du modele de base")

        base_model = self.config.BASE_MODELS.get(model_size, self.config.DEFAULT_MODEL)

        # Verifier si deja present
        available = self.checker.get_available_models()
        if any(base_model.split(":")[0] in m for m in available):
            print_success(f"Modele de base deja present: {base_model}")
        else:
            if not self.manager.pull_base_model(base_model):
                print_error("Echec du telechargement du modele")
                return False

        # 4. Creation du modele personnalise
        print_info("Etape 4/5: Creation du modele DIM-PMSI")

        if not self.manager.create_custom_model(base_model=base_model):
            print_error("Echec de la creation du modele personnalise")
            return False

        # 5. Test
        print_info("Etape 5/5: Test du modele")

        self.manager.test_model()

        # 6. Preparation des donnees d'entrainement
        print_info("Bonus: Creation du dataset d'entrainement")
        self.data_preparer.export_to_jsonl()

        # Resume
        print_header("INSTALLATION TERMINEE")
        print_success(f"Modele '{self.config.CUSTOM_MODEL_NAME}' pret a l'emploi!")
        print_info("")
        print_info("Utilisation:")
        print_info(f"  ollama run {self.config.CUSTOM_MODEL_NAME}")
        print_info("")
        print_info("Ou via l'interface web:")
        print_info("  python app.py")
        print_info("  -> Ouvrir http://localhost:8080")
        print_info("  -> Section 'IA' -> Initialiser")

        return True

    def quick_setup(self) -> bool:
        """Installation rapide avec le modele le plus leger."""
        return self.run_full_setup(model_size="small")


# =============================================================================
# POINT D'ENTREE
# =============================================================================

def main():
    """Point d'entree principal."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Configuration du LLM DIM-PMSI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python setup_llm.py --setup          Installation complete (modele medium)
  python setup_llm.py --quick          Installation rapide (modele leger)
  python setup_llm.py --test           Tester le modele existant
  python setup_llm.py --create-data    Creer le dataset d'entrainement
        """
    )

    parser.add_argument("--setup", action="store_true",
                       help="Installation complete du LLM")
    parser.add_argument("--quick", action="store_true",
                       help="Installation rapide (modele leger)")
    parser.add_argument("--size", choices=["tiny", "small", "medium", "large"],
                       default="medium", help="Taille du modele")
    parser.add_argument("--test", action="store_true",
                       help="Tester le modele existant")
    parser.add_argument("--create-data", action="store_true",
                       help="Creer le dataset d'entrainement")
    parser.add_argument("--from-episodes", type=str,
                       help="Creer le dataset depuis un fichier d'episodes")

    args = parser.parse_args()

    setup = LLMSetup()

    if args.setup:
        setup.run_full_setup(model_size=args.size)

    elif args.quick:
        setup.quick_setup()

    elif args.test:
        if not SystemChecker.check_ollama_running():
            SystemChecker.start_ollama()
        setup.manager.test_model()

    elif args.create_data:
        if args.from_episodes:
            examples = setup.data_preparer.load_from_episodes(Path(args.from_episodes))
            setup.data_preparer.export_to_jsonl(examples)
        else:
            setup.data_preparer.export_to_jsonl()

    else:
        # Menu interactif
        print_header("CONFIGURATION DU LLM DIM-PMSI")
        print("Choisissez une option:\n")
        print("  1. Installation complete (recommande)")
        print("  2. Installation rapide (modele leger)")
        print("  3. Tester le modele existant")
        print("  4. Creer le dataset d'entrainement")
        print("  5. Quitter")
        print()

        choice = input("Votre choix [1-5]: ").strip()

        if choice == "1":
            setup.run_full_setup()
        elif choice == "2":
            setup.quick_setup()
        elif choice == "3":
            if not SystemChecker.check_ollama_running():
                SystemChecker.start_ollama()
            setup.manager.test_model()
        elif choice == "4":
            setup.data_preparer.export_to_jsonl()
        elif choice == "5":
            print("Au revoir!")
        else:
            print_error("Choix invalide")


if __name__ == "__main__":
    main()
