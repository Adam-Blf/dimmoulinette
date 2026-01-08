"""
===============================================================================
SETUP_ENV.PY - Audit Sécurité & Configuration Environnement
===============================================================================
DIM - Data Intelligence Médicale
Vérifie la conformité de l'environnement pour le traitement de données HDS
===============================================================================
"""

import os
import sys
import hashlib
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, field

# Configuration Rich pour l'affichage console
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    rprint = print


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AuditConfig:
    """Configuration de l'audit de sécurité."""
    source_dir: Path = field(default_factory=lambda: Path(r"C:\Users\adamb\Downloads\frer"))
    project_dir: Path = field(default_factory=lambda: Path(__file__).parent.resolve())
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent / "output")
    configs_dir: Path = field(default_factory=lambda: Path(__file__).parent / "configs")
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent / "models")
    templates_dir: Path = field(default_factory=lambda: Path(__file__).parent / "templates")


# =============================================================================
# AUDIT DE SÉCURITÉ
# =============================================================================

class SecurityAuditor:
    """Auditeur de sécurité pour environnement HDS."""

    def __init__(self, config: AuditConfig = None):
        self.config = config or AuditConfig()
        self.audit_results: List[Dict] = []
        self.console = Console() if RICH_AVAILABLE else None

    def log(self, level: str, message: str, details: str = ""):
        """Log un résultat d'audit."""
        icons = {"OK": "✅", "WARN": "⚠️", "ERROR": "❌", "INFO": "ℹ️"}
        icon = icons.get(level, "•")

        result = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "details": details
        }
        self.audit_results.append(result)

        if RICH_AVAILABLE:
            color = {"OK": "green", "WARN": "yellow", "ERROR": "red", "INFO": "blue"}.get(level, "white")
            self.console.print(f"[{color}]{icon} {message}[/{color}]")
            if details:
                self.console.print(f"   [dim]{details}[/dim]")
        else:
            print(f"{icon} {message}")
            if details:
                print(f"   {details}")

    def check_source_directory(self) -> bool:
        """Vérifie l'existence et l'accessibilité du dossier source."""
        self.log("INFO", "Vérification du dossier source des données PMSI...")

        if not self.config.source_dir.exists():
            self.log("ERROR",
                     f"Dossier source INTROUVABLE: {self.config.source_dir}",
                     "Créez le dossier ou vérifiez le chemin dans la configuration")
            return False

        if not self.config.source_dir.is_dir():
            self.log("ERROR",
                     f"Le chemin n'est pas un dossier: {self.config.source_dir}")
            return False

        # Vérifie les permissions de lecture
        try:
            list(self.config.source_dir.iterdir())
            self.log("OK", f"Dossier source accessible: {self.config.source_dir}")
        except PermissionError:
            self.log("ERROR",
                     "Permissions insuffisantes sur le dossier source",
                     "Vérifiez les droits d'accès en lecture")
            return False

        # Compte les fichiers de données
        data_files = list(self.config.source_dir.glob("*"))
        data_count = len([f for f in data_files if f.is_file()])
        self.log("INFO", f"Fichiers trouvés dans le dossier source: {data_count}")

        return True

    def check_gitignore(self) -> bool:
        """Vérifie et met à jour le .gitignore pour la sécurité HDS."""
        self.log("INFO", "Vérification du .gitignore pour protection des données HDS...")

        gitignore_path = self.config.project_dir / ".gitignore"

        # Règles critiques obligatoires
        critical_rules = [
            "# === DONNÉES SENSIBLES HDS (CRITIQUE) ===",
            "frer/",
            "**/Downloads/frer/",
            "*.csv",
            "*.txt",
            "*.tsv",
            "*.xlsx",
            "*.xls",
            "*.jsonl",
            "# === MODÈLES IA ===",
            "models/",
            "*.safetensors",
            "*.bin",
            "*.pt",
            "*.pth",
            "*.gguf",
            "# === CACHE ===",
            "__pycache__/",
            ".cache/",
            "output/",
            "*.log",
            "# === SECRETS ===",
            ".env",
            "*.pem",
            "*.key",
        ]

        existing_rules = set()
        if gitignore_path.exists():
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                existing_rules = set(line.strip() for line in f if line.strip() and not line.startswith('#'))

        # Vérifie les règles manquantes (hors commentaires)
        required_patterns = [r for r in critical_rules if not r.startswith('#')]
        missing_rules = [r for r in required_patterns if r not in existing_rules]

        if missing_rules:
            self.log("WARN",
                     f"Règles .gitignore manquantes: {len(missing_rules)}",
                     "Exécutez setup_env.py --fix pour corriger")
            return False

        self.log("OK", ".gitignore correctement configuré pour HDS")
        return True

    def fix_gitignore(self) -> bool:
        """Crée/Met à jour le .gitignore avec les règles de sécurité HDS."""
        gitignore_path = self.config.project_dir / ".gitignore"

        gitignore_content = """# ============================================
# .gitignore - Moulinettes DIM
# SÉCURITÉ DONNÉES DE SANTÉ (PMSI/HDS)
# ============================================

# ==== DONNÉES SENSIBLES (CRITIQUE) ====
# Dossier source des données PMSI - JAMAIS dans Git
C:/Users/adamb/Downloads/frer/
**/Downloads/frer/
frer/

# Fichiers de données brutes
*.csv
*.txt
*.tsv
*.xlsx
*.xls

# Fichiers de données transformées
*.jsonl
*.json
!package.json
!tsconfig.json
!configs/*.json

# ==== MODÈLES IA (VOLUMINEUX + SENSIBLES) ====
*.safetensors
*.bin
*.pt
*.pth
*.ckpt
*.gguf
*.ggml
models/
adapters/
checkpoints/
*.h5

# ==== CACHE & BYTECODE PYTHON ====
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# ==== ENVIRONNEMENTS VIRTUELS ====
.env
.venv/
env/
venv/
ENV/
.conda/
*.local

# ==== HUGGINGFACE CACHE ====
.cache/
~/.cache/huggingface/

# ==== LOGS & TEMPORAIRES ====
*.log
logs/
*.tmp
*.temp
*.swp
*.swo
*~

# ==== IDE ====
.vscode/
!.vscode/extensions.json
.idea/
*.sublime-*

# ==== SÉCURITÉ ====
# Fichiers de hash/intégrité (contiennent des métadonnées)
.data_manifest.json
security_audit.log

# Clés et secrets
*.pem
*.key
secrets.yaml
.secrets/

# ==== OS ====
.DS_Store
Thumbs.db
desktop.ini

# ==== OUTPUTS DU PROJET ====
output/
results/
exports/
train_dataset.jsonl
"""

        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write(gitignore_content)

        self.log("OK", f".gitignore mis à jour: {gitignore_path}")
        return True

    def create_directory_structure(self) -> bool:
        """Crée la structure de dossiers du projet."""
        self.log("INFO", "Création de la structure de dossiers...")

        directories = [
            self.config.output_dir,
            self.config.configs_dir,
            self.config.models_dir,
            self.config.templates_dir,
        ]

        for dir_path in directories:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                self.log("OK", f"Dossier créé/vérifié: {dir_path.name}/")
            except Exception as e:
                self.log("ERROR", f"Impossible de créer {dir_path}: {e}")
                return False

        return True

    def check_python_environment(self) -> bool:
        """Vérifie l'environnement Python."""
        self.log("INFO", "Vérification de l'environnement Python...")

        # Version Python
        py_version = sys.version_info
        if py_version < (3, 10):
            self.log("WARN",
                     f"Python {py_version.major}.{py_version.minor} détecté",
                     "Python 3.10+ recommandé pour les performances")
        else:
            self.log("OK", f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")

        # Vérification des dépendances critiques
        critical_packages = ['polars', 'fastapi', 'torch', 'transformers']
        missing = []

        for pkg in critical_packages:
            try:
                __import__(pkg)
                self.log("OK", f"Package '{pkg}' installé")
            except ImportError:
                missing.append(pkg)
                self.log("WARN", f"Package '{pkg}' manquant")

        if missing:
            self.log("INFO",
                     "Installez les dépendances manquantes:",
                     "pip install -r requirements.txt")

        return len(missing) == 0

    def check_network_isolation(self) -> bool:
        """Vérifie que les données ne risquent pas de fuir vers Internet."""
        self.log("INFO", "Vérification de l'isolation réseau...")

        # Note: Cette vérification est indicative
        # Dans un vrai environnement HDS, il faudrait des contrôles plus stricts

        self.log("WARN",
                 "Vérification réseau: Mode indicatif uniquement",
                 "En production HDS, utilisez un réseau isolé ou un VPN sécurisé")

        # Vérifie qu'aucun proxy HTTP n'est configuré qui pourrait logger les données
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
        proxies_found = [v for v in proxy_vars if os.environ.get(v)]

        if proxies_found:
            self.log("WARN",
                     f"Proxies détectés: {proxies_found}",
                     "Vérifiez que le proxy ne log pas les données sensibles")
        else:
            self.log("OK", "Aucun proxy HTTP configuré")

        return True

    def run_full_audit(self, fix: bool = False) -> Dict:
        """Exécute l'audit de sécurité complet."""
        if RICH_AVAILABLE:
            self.console.print(Panel.fit(
                "[bold blue]AUDIT DE SÉCURITÉ - DIM[/bold blue]\n"
                "Vérification de l'environnement pour données HDS",
                border_style="blue"
            ))
        else:
            print("=" * 60)
            print("AUDIT DE SÉCURITÉ - DIM")
            print("Vérification de l'environnement pour données HDS")
            print("=" * 60)

        results = {
            "timestamp": datetime.now().isoformat(),
            "platform": platform.system(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "checks": {}
        }

        # Exécute les vérifications
        results["checks"]["source_directory"] = self.check_source_directory()
        results["checks"]["gitignore"] = self.check_gitignore()
        results["checks"]["python_env"] = self.check_python_environment()
        results["checks"]["network"] = self.check_network_isolation()

        # Création de la structure si demandé
        if fix:
            self.log("INFO", "Mode FIX activé - Application des corrections...")
            self.fix_gitignore()
            self.create_directory_structure()

        # Résumé
        passed = sum(1 for v in results["checks"].values() if v)
        total = len(results["checks"])
        results["summary"] = {
            "passed": passed,
            "total": total,
            "status": "OK" if passed == total else "WARN" if passed > 0 else "ERROR"
        }

        if RICH_AVAILABLE:
            status_color = {"OK": "green", "WARN": "yellow", "ERROR": "red"}[results["summary"]["status"]]
            self.console.print(f"\n[{status_color}]Résultat: {passed}/{total} vérifications passées[/{status_color}]")
        else:
            print(f"\nRésultat: {passed}/{total} vérifications passées")

        return results


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main():
    """Point d'entrée principal."""
    import argparse

    parser = argparse.ArgumentParser(
        description="DIM - Audit de sécurité et configuration environnement HDS"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Applique les corrections automatiques (.gitignore, dossiers)"
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Chemin personnalisé vers le dossier source des données"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Mode silencieux (sortie JSON uniquement)"
    )

    args = parser.parse_args()

    # Configuration
    config = AuditConfig()
    if args.source:
        config.source_dir = Path(args.source)

    # Audit
    auditor = SecurityAuditor(config)
    results = auditor.run_full_audit(fix=args.fix)

    if args.quiet:
        import json
        print(json.dumps(results, indent=2))

    # Code de sortie
    sys.exit(0 if results["summary"]["status"] == "OK" else 1)


if __name__ == "__main__":
    main()
