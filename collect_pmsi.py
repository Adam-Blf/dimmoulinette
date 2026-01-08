"""
===============================================================================
COLLECT_PMSI.PY - Collecteur de fichiers PMSI selon critères ATIH
===============================================================================
DIM - Data Intelligence Médicale
Recherche et valide les fichiers PMSI selon les formats ATIH 2024
===============================================================================
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Critères ATIH 2024/2025 pour la psychiatrie (PSY)
# Note: Les formats PSY sont différents des formats MCO
ATIH_CRITERIA = {
    "RAA": {
        "description": "Résumé d'Activité Ambulatoire (R3A PSY)",
        "patterns": ["RAA", "R3A"],
        "filename_patterns": ["raa_", "r3a_", "raa.", "r3a."],
        "line_length": 96,  # Format PSY 2024/2025
        "tolerance": 10,
        "required": True,
        "priority": 1
    },
    "RPS": {
        "description": "Résumé Par Séquence (hospitalisation PSY)",
        "patterns": ["RPS"],
        "filename_patterns": ["rps_", "rps.", "rimp"],
        "line_length": 154,  # Format PSY 2024/2025
        "tolerance": 20,
        "required": True,
        "priority": 2
    },
    "VIDHOSP": {
        "description": "Chaînage des séjours",
        "patterns": [],  # Pas de pattern générique
        "filename_patterns": ["vidhosp"],
        "line_length": 514,  # Format PSY réel
        "tolerance": 50,
        "required": False,
        "priority": 3,
        "min_lines": 100
    },
    "ANOHOSP": {
        "description": "Chaînage anonyme",
        "patterns": [],
        "filename_patterns": ["anohosp"],
        "line_length": 1709,  # Format réel observé
        "tolerance": 100,
        "required": False,
        "priority": 4,
        "min_lines": 100
    },
    "FICHCOMP_ISO": {
        "description": "Isolement",
        "patterns": [],
        "filename_patterns": ["iso_"],  # Plus strict - underscore obligatoire
        "line_length": 113,  # Format réel observé
        "tolerance": 20,
        "required": False,
        "priority": 5,
        "min_lines": 50
    },
    "FICHCOMP_TRANSPORT": {
        "description": "Transport",
        "patterns": [],
        "filename_patterns": ["fichcomp"],
        "line_length": 100,
        "tolerance": 50,
        "required": False,
        "priority": 6,
        "min_lines": 50
    },
    "UM": {
        "description": "Unités Médicales",
        "patterns": [],  # Pas de pattern générique pour éviter faux positifs
        "filename_patterns": ["um_"],  # Strict - underscore obligatoire
        "line_length": 50,
        "tolerance": 30,
        "required": False,
        "priority": 7,
        "min_lines": 10
    },
    "IPP": {
        "description": "Identifiant Patient Permanent",
        "patterns": [],
        "filename_patterns": ["ipp_"],  # Strict
        "line_length": 135,  # Format réel observé
        "tolerance": 20,
        "required": False,
        "priority": 8,
        "min_lines": 100
    }
}

# Chemins de recherche par défaut
DEFAULT_SEARCH_PATHS = [
    Path.home() / "Downloads",
    Path.home() / "Documents",
    Path.home() / "Desktop",
    Path("C:/PMSI"),
    Path("D:/PMSI"),
    Path("C:/Users/adamb/Downloads/frer"),
]


@dataclass
class PMSIFile:
    """Représente un fichier PMSI détecté."""
    path: Path
    file_type: str
    description: str
    line_count: int
    line_length: int
    is_valid: bool
    validation_message: str


class PMSICollector:
    """Collecteur de fichiers PMSI selon critères ATIH."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("./data/pmsi")
        self.found_files: List[PMSIFile] = []
        self.search_paths: List[Path] = []

    def add_search_path(self, path: Path):
        """Ajoute un chemin de recherche."""
        if path.exists() and path.is_dir():
            self.search_paths.append(path)
            print(f"  + Chemin ajouté: {path}")

    def detect_file_type(self, filepath: Path) -> Tuple[str, str]:
        """Détecte le type de fichier PMSI."""
        filename = filepath.name.lower()
        filename_upper = filepath.name.upper()

        for file_type, criteria in ATIH_CRITERIA.items():
            # Priorité aux patterns de nom de fichier (plus précis)
            if "filename_patterns" in criteria:
                for pattern in criteria["filename_patterns"]:
                    if filename.startswith(pattern) or pattern in filename:
                        return file_type, criteria["description"]

            # Fallback aux patterns généraux (début du nom seulement)
            for pattern in criteria["patterns"]:
                if filename_upper.startswith(pattern):
                    return file_type, criteria["description"]

        return "UNKNOWN", "Type non reconnu"

    def validate_file(self, filepath: Path, file_type: str) -> Tuple[bool, str, int, int]:
        """
        Valide un fichier selon les critères ATIH.
        Retourne: (is_valid, message, line_count, avg_line_length)
        """
        if file_type not in ATIH_CRITERIA:
            return False, "Type non supporté", 0, 0

        criteria = ATIH_CRITERIA[file_type]
        expected_length = criteria["line_length"]
        tolerance = criteria["tolerance"]
        min_lines = criteria.get("min_lines", 1)

        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

            if not lines:
                return False, "Fichier vide", 0, 0

            # Analyse des lignes
            line_count = len(lines)
            line_lengths = [len(line.rstrip('\n\r')) for line in lines if line.strip()]

            if not line_lengths:
                return False, "Aucune ligne valide", 0, 0

            # Vérification du nombre minimum de lignes
            if line_count < min_lines:
                return False, f"Trop peu de lignes ({line_count} < {min_lines})", line_count, 0

            avg_length = sum(line_lengths) / len(line_lengths)

            # Vérification de la longueur
            if abs(avg_length - expected_length) <= tolerance:
                return True, f"Format ATIH valide ({line_count} lignes)", line_count, int(avg_length)
            else:
                return False, f"Longueur incorrecte ({int(avg_length)} vs {expected_length})", line_count, int(avg_length)

        except Exception as e:
            return False, f"Erreur lecture: {str(e)}", 0, 0

    def scan_directory(self, directory: Path, recursive: bool = True) -> List[PMSIFile]:
        """Scanne un répertoire à la recherche de fichiers PMSI."""
        found = []

        if not directory.exists():
            return found

        # Extensions à rechercher
        extensions = {'.txt', '.csv', '.dat', ''}

        # Pattern de recherche
        pattern = "**/*" if recursive else "*"

        for filepath in directory.glob(pattern):
            if not filepath.is_file():
                continue

            # Vérifier l'extension
            if filepath.suffix.lower() not in extensions:
                continue

            # Détecter le type
            file_type, description = self.detect_file_type(filepath)

            if file_type == "UNKNOWN":
                continue

            # Valider le fichier
            is_valid, message, line_count, line_length = self.validate_file(filepath, file_type)

            pmsi_file = PMSIFile(
                path=filepath,
                file_type=file_type,
                description=description,
                line_count=line_count,
                line_length=line_length,
                is_valid=is_valid,
                validation_message=message
            )

            found.append(pmsi_file)

        return found

    def collect(self, search_paths: List[Path] = None) -> Dict[str, List[PMSIFile]]:
        """
        Recherche et collecte tous les fichiers PMSI.
        Retourne un dictionnaire par type de fichier.
        """
        paths_to_search = search_paths or self.search_paths or DEFAULT_SEARCH_PATHS

        print("\n" + "=" * 60)
        print("  COLLECTE FICHIERS PMSI - CRITÈRES ATIH 2024")
        print("=" * 60)
        print("\nRecherche dans:")

        all_found = []

        for search_path in paths_to_search:
            if search_path.exists():
                print(f"  - {search_path}")
                found = self.scan_directory(search_path)
                all_found.extend(found)

        # Grouper par type
        by_type: Dict[str, List[PMSIFile]] = {}
        for pmsi_file in all_found:
            if pmsi_file.file_type not in by_type:
                by_type[pmsi_file.file_type] = []
            by_type[pmsi_file.file_type].append(pmsi_file)

        # Afficher le résumé
        print("\n" + "-" * 60)
        print("  FICHIERS DÉTECTÉS")
        print("-" * 60)

        for file_type, files in sorted(by_type.items()):
            valid_count = sum(1 for f in files if f.is_valid)
            print(f"\n  {file_type} ({ATIH_CRITERIA.get(file_type, {}).get('description', 'N/A')}):")

            for f in files:
                status = "[OK]" if f.is_valid else "[X]"
                print(f"    {status} {f.path.name}")
                print(f"        {f.line_count} lignes | {f.validation_message}")

        self.found_files = all_found
        return by_type

    def copy_valid_files(self, destination: Path = None) -> List[Path]:
        """Copie les fichiers valides vers le répertoire de destination."""
        dest = destination or self.output_dir
        dest.mkdir(parents=True, exist_ok=True)

        copied = []

        print("\n" + "-" * 60)
        print(f"  COPIE VERS: {dest}")
        print("-" * 60)

        for pmsi_file in self.found_files:
            if not pmsi_file.is_valid:
                continue

            dest_path = dest / pmsi_file.path.name

            try:
                shutil.copy2(pmsi_file.path, dest_path)
                copied.append(dest_path)
                print(f"  [OK] {pmsi_file.path.name}")
            except Exception as e:
                print(f"  [X] {pmsi_file.path.name}: {e}")

        print(f"\n  Total: {len(copied)} fichiers copiés")
        return copied

    def get_summary(self) -> Dict:
        """Retourne un résumé de la collecte."""
        summary = {
            "total_found": len(self.found_files),
            "valid_files": sum(1 for f in self.found_files if f.is_valid),
            "invalid_files": sum(1 for f in self.found_files if not f.is_valid),
            "by_type": {}
        }

        for pmsi_file in self.found_files:
            if pmsi_file.file_type not in summary["by_type"]:
                summary["by_type"][pmsi_file.file_type] = {
                    "total": 0,
                    "valid": 0,
                    "files": []
                }

            summary["by_type"][pmsi_file.file_type]["total"] += 1
            if pmsi_file.is_valid:
                summary["by_type"][pmsi_file.file_type]["valid"] += 1
            summary["by_type"][pmsi_file.file_type]["files"].append(str(pmsi_file.path))

        return summary

    def check_required_files(self) -> Tuple[bool, List[str]]:
        """Vérifie que tous les fichiers requis sont présents."""
        missing = []

        for file_type, criteria in ATIH_CRITERIA.items():
            if not criteria.get("required", False):
                continue

            # Chercher un fichier valide de ce type
            has_valid = any(
                f.file_type == file_type and f.is_valid
                for f in self.found_files
            )

            if not has_valid:
                missing.append(f"{file_type} ({criteria['description']})")

        return len(missing) == 0, missing


def main():
    """Point d'entrée principal."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Collecteur de fichiers PMSI selon critères ATIH"
    )
    parser.add_argument(
        "--search", "-s",
        type=str,
        nargs="+",
        help="Chemins de recherche supplémentaires"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data/pmsi",
        help="Répertoire de destination"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copier les fichiers valides vers le répertoire de destination"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Afficher le résumé en JSON"
    )

    args = parser.parse_args()

    # Initialisation
    collector = PMSICollector(output_dir=Path(args.output))

    # Chemins de recherche
    search_paths = DEFAULT_SEARCH_PATHS.copy()
    if args.search:
        search_paths.extend(Path(p) for p in args.search)

    # Collecte
    by_type = collector.collect(search_paths)

    # Vérification des fichiers requis
    all_required, missing = collector.check_required_files()

    if not all_required:
        print("\n" + "!" * 60)
        print("  ATTENTION: Fichiers requis manquants")
        print("!" * 60)
        for m in missing:
            print(f"  - {m}")

    # Copie si demandée
    if args.copy:
        collector.copy_valid_files()

    # Sortie JSON si demandée
    if args.json:
        summary = collector.get_summary()
        print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("  COLLECTE TERMINÉE")
    print("=" * 60)

    return 0 if all_required else 1


if __name__ == "__main__":
    exit(main())
