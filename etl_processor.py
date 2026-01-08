"""
===============================================================================
ETL_PROCESSOR.PY - Moulinettes à Formats PMSI
===============================================================================
DIM - Data Intelligence Médicale
Classe UniversalParser pour la transformation de fichiers positionnels en CSV
===============================================================================
"""

import json
import os
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import polars as pl
from loguru import logger

# Configuration du logger
logger.add(
    "logs/etl_processing.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

class FileType(Enum):
    """Types de fichiers PMSI supportés par ordre de priorité."""
    RPS = 1       # Priorité 1 - Psychiatrie hospitalisation
    RAA = 2       # Priorité 2 - Psychiatrie ambulatoire
    VID_HOSP = 3  # Priorité 3 - Chaînage
    RSF_ACE = 4   # Actes et consultations externes
    FICHCOMP = 5  # Fichiers complémentaires
    RUM_RSS = 6   # MCO
    VID_IPP = 7   # Chaînage IPP
    UNKNOWN = 99


@dataclass
class ETLConfig:
    """Configuration du processeur ETL."""
    source_dir: Path = field(default_factory=lambda: Path("./data/pmsi"))
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    configs_dir: Path = field(default_factory=lambda: Path("./configs"))
    csv_separator: str = ";"
    encoding: str = "utf-8"
    year: str = "2024"


# =============================================================================
# UNIVERSAL PARSER - Cœur ETL
# =============================================================================

class UniversalParser:
    """
    Parser universel pour fichiers PMSI positionnels.
    Transforme les fichiers plats en CSV structurés.
    """

    def __init__(self, config: ETLConfig = None):
        self.config = config or ETLConfig()
        self.format_configs: Dict[str, dict] = {}
        self._load_format_configs()

        # Statistiques de traitement
        self.stats = {
            "files_processed": 0,
            "lines_processed": 0,
            "errors": [],
            "warnings": []
        }

    def _load_format_configs(self) -> None:
        """Charge les configurations de format depuis les fichiers JSON."""
        if not self.config.configs_dir.exists():
            logger.warning(f"Dossier configs non trouvé: {self.config.configs_dir}")
            return

        for config_file in self.config.configs_dir.glob(f"format_*_{self.config.year}.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    format_config = json.load(f)
                    file_type = format_config.get("type", "UNKNOWN")
                    self.format_configs[file_type] = format_config
                    logger.info(f"Format chargé: {file_type} ({config_file.name})")
            except Exception as e:
                logger.error(f"Erreur chargement config {config_file}: {e}")

    def detect_file_type(self, filepath: Path) -> FileType:
        """
        Détecte le type de fichier PMSI basé sur le nom et le contenu.
        """
        filename = filepath.name.upper()

        # Détection par pattern dans le nom (flexible)
        patterns = {
            FileType.RPS: ["RPS", "RIMP"],
            FileType.RAA: ["RAA", "RPSA", "R3A"],
            FileType.VID_HOSP: ["VIDHOSP", "VID_HOSP", "VID-HOSP", "ANOHOSP", "ANO"],
            FileType.RSF_ACE: ["RSF", "RSFACE", "ACE"],
            FileType.FICHCOMP: ["FICHCOMP", "ISO", "TRANSPORT", "CONTENTION"],
            FileType.RUM_RSS: ["RUM", "RSS"],
            FileType.VID_IPP: ["VIDIPP", "VID_IPP", "VID-IPP", "IPP", "UM"],
        }

        for file_type, keywords in patterns.items():
            if any(kw in filename for kw in keywords):
                return file_type

        # Détection par analyse du contenu (longueur de ligne - format PSY/MCO)
        try:
            with open(filepath, 'r', encoding=self.config.encoding, errors='replace') as f:
                first_line = f.readline().strip()

            # Analyse de la longueur de ligne (format PSY 2024/2025)
            line_lengths = {
                154: FileType.RPS,    # PSY
                96: FileType.RAA,     # PSY (R3A)
                520: FileType.RPS,    # MCO fallback
                400: FileType.RAA,    # MCO fallback
                514: FileType.VID_HOSP,  # PSY
                150: FileType.VID_HOSP,  # MCO
                300: FileType.RSF_ACE,
                113: FileType.FICHCOMP,  # ISO PSY
                200: FileType.FICHCOMP,
                550: FileType.RUM_RSS,
            }

            closest_type = FileType.UNKNOWN
            min_diff = float('inf')

            for length, ftype in line_lengths.items():
                diff = abs(len(first_line) - length)
                if diff < min_diff and diff < 100:  # Tolérance large de 100 caractères
                    min_diff = diff
                    closest_type = ftype

            # Si toujours UNKNOWN mais fichier .txt, tenter détection par extension
            if closest_type == FileType.UNKNOWN and filepath.suffix.lower() == '.txt':
                # Fichier texte non reconnu - accepter comme FICHCOMP générique
                return FileType.FICHCOMP

            return closest_type

        except Exception as e:
            logger.warning(f"Impossible de détecter le type de {filepath.name}: {e}")
            return FileType.UNKNOWN

    def _parse_positional_line(
        self,
        line: str,
        fields: List[dict]
    ) -> Dict[str, Any]:
        """
        Parse une ligne positionnelle selon la définition des champs.
        """
        record = {}

        for field_def in fields:
            name = field_def["name"]
            start = field_def["start"] - 1  # Conversion 1-indexé vers 0-indexé
            end = field_def["end"]
            field_type = field_def.get("type", "string")

            # Extraction de la valeur
            try:
                raw_value = line[start:end].strip() if len(line) >= end else ""
            except IndexError:
                raw_value = ""

            # Conversion selon le type
            if field_type == "integer":
                try:
                    record[name] = int(raw_value) if raw_value else None
                except ValueError:
                    record[name] = None
            elif field_type == "decimal":
                try:
                    # Gestion du format français (virgule décimale)
                    raw_value = raw_value.replace(',', '.')
                    record[name] = float(raw_value) if raw_value else None
                except ValueError:
                    record[name] = None
            elif field_type == "date":
                record[name] = raw_value if raw_value and raw_value != "00000000" else None
            else:
                record[name] = raw_value if raw_value else None

        return record

    def parse_file(
        self,
        filepath: Path,
        file_type: FileType = None
    ) -> Optional[pl.DataFrame]:
        """
        Parse un fichier positionnel et retourne un DataFrame Polars.
        """
        if not filepath.exists():
            logger.error(f"Fichier non trouvé: {filepath}")
            return None

        # Détection automatique du type si non spécifié
        if file_type is None:
            file_type = self.detect_file_type(filepath)

        logger.info(f"Parsing {filepath.name} (Type: {file_type.name})")

        # Récupération de la configuration du format
        type_name = file_type.name.replace('_', '-')
        format_config = self.format_configs.get(type_name)

        if not format_config:
            # Fallback: essayer de lire comme CSV/TSV
            return self._read_as_delimited(filepath)

        # Parsing des lignes positionnelles
        records = []
        fields = format_config.get("fields", [])

        # Gestion des sous-types (ex: FICHCOMP avec ISOLEMENT, TRANSPORT)
        if "subtypes" in format_config:
            return self._parse_multi_type_file(filepath, format_config)

        try:
            with open(filepath, 'r', encoding=self.config.encoding, errors='replace') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    record = self._parse_positional_line(line, fields)
                    record["_line_number"] = line_num
                    record["_source_file"] = filepath.name
                    records.append(record)

                    if line_num % 10000 == 0:
                        logger.debug(f"  Traité: {line_num} lignes...")

        except Exception as e:
            logger.error(f"Erreur parsing {filepath.name}: {e}")
            self.stats["errors"].append({"file": str(filepath), "error": str(e)})
            return None

        if not records:
            logger.warning(f"Aucun enregistrement dans {filepath.name}")
            return None

        # Création du DataFrame Polars
        df = pl.DataFrame(records)

        # Conversion des dates
        date_fields = format_config.get("date_fields", [])
        for date_col in date_fields:
            if date_col in df.columns:
                df = df.with_columns([
                    pl.col(date_col).str.to_date("%Y%m%d", strict=False).alias(date_col)
                ])

        self.stats["files_processed"] += 1
        self.stats["lines_processed"] += len(records)

        logger.info(f"  ✓ Parsé: {len(df)} lignes, {len(df.columns)} colonnes")
        return df

    def _parse_multi_type_file(
        self,
        filepath: Path,
        format_config: dict
    ) -> Optional[pl.DataFrame]:
        """
        Parse un fichier avec plusieurs sous-types (ex: FICHCOMP).
        """
        subtypes = format_config.get("subtypes", {})
        all_records = []

        try:
            with open(filepath, 'r', encoding=self.config.encoding, errors='replace') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    # Détection du sous-type par analyse de la ligne
                    matched_subtype = None
                    for subtype_name, subtype_config in subtypes.items():
                        expected_length = subtype_config.get("line_length", 0)
                        if abs(len(line.strip()) - expected_length) < 20:
                            matched_subtype = subtype_name
                            break

                    if matched_subtype:
                        fields = subtypes[matched_subtype]["fields"]
                        record = self._parse_positional_line(line, fields)
                        record["_subtype"] = matched_subtype
                        record["_line_number"] = line_num
                        record["_source_file"] = filepath.name
                        all_records.append(record)

        except Exception as e:
            logger.error(f"Erreur parsing multi-type {filepath.name}: {e}")
            return None

        if not all_records:
            return None

        return pl.DataFrame(all_records)

    def _read_as_delimited(self, filepath: Path) -> Optional[pl.DataFrame]:
        """
        Fallback: lecture comme fichier délimité (CSV/TSV).
        """
        logger.info(f"  Tentative lecture délimitée pour {filepath.name}")

        try:
            # Détection du séparateur
            with open(filepath, 'r', encoding=self.config.encoding, errors='replace') as f:
                first_line = f.readline()

            if ';' in first_line:
                separator = ';'
            elif '\t' in first_line:
                separator = '\t'
            elif ',' in first_line:
                separator = ','
            else:
                separator = None  # Fichier positionnel sans config

            if separator:
                df = pl.read_csv(
                    filepath,
                    separator=separator,
                    ignore_errors=True,
                    truncate_ragged_lines=True,
                    encoding='utf8-lossy',
                    infer_schema_length=0  # Force toutes les colonnes en String
                )
                df = df.with_columns([
                    pl.lit(filepath.name).alias("_source_file")
                ])
                return df
            else:
                logger.warning(f"  Format non reconnu pour {filepath.name}")
                return None

        except Exception as e:
            logger.error(f"  Erreur lecture délimitée {filepath.name}: {e}")
            return None

    def export_to_csv(
        self,
        df: pl.DataFrame,
        output_name: str,
        include_metadata: bool = True
    ) -> Path:
        """
        Exporte un DataFrame vers CSV avec le séparateur configuré.
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        output_path = self.config.output_dir / f"{output_name}.csv"

        # Suppression des colonnes de métadonnées si demandé
        if not include_metadata:
            meta_cols = [c for c in df.columns if c.startswith('_')]
            df = df.drop(meta_cols)

        df.write_csv(output_path, separator=self.config.csv_separator)
        logger.info(f"  ✓ Exporté: {output_path}")

        return output_path

    def process_directory(
        self,
        priority_order: bool = True
    ) -> Dict[str, pl.DataFrame]:
        """
        Traite tous les fichiers du répertoire source.

        Args:
            priority_order: Si True, traite les fichiers par ordre de priorité
                           (RPS > RAA > VID-HOSP > autres)
        """
        if not self.config.source_dir.exists():
            logger.error(f"Répertoire source non trouvé: {self.config.source_dir}")
            return {}

        # Scan des fichiers
        data_extensions = {'.txt', '.csv', '.tsv', ''}
        files = [
            f for f in self.config.source_dir.iterdir()
            if f.is_file() and (f.suffix.lower() in data_extensions or not f.suffix)
        ]

        if not files:
            logger.warning(f"Aucun fichier trouvé dans {self.config.source_dir}")
            return {}

        logger.info(f"Trouvé {len(files)} fichiers à traiter")

        # Classification et tri par priorité
        classified_files: Dict[FileType, List[Path]] = {ft: [] for ft in FileType}

        for filepath in files:
            file_type = self.detect_file_type(filepath)
            classified_files[file_type].append(filepath)

        # Affichage du résumé
        logger.info("=" * 60)
        logger.info("CLASSIFICATION DES FICHIERS")
        for ft in FileType:
            if classified_files[ft]:
                logger.info(f"  {ft.name}: {len(classified_files[ft])} fichiers")
        logger.info("=" * 60)

        # Traitement par ordre de priorité
        results: Dict[str, pl.DataFrame] = {}

        processing_order = sorted(FileType, key=lambda x: x.value) if priority_order else FileType

        for file_type in processing_order:
            for filepath in classified_files[file_type]:
                df = self.parse_file(filepath, file_type)

                if df is not None:
                    key = f"{file_type.name}_{filepath.stem}"
                    results[key] = df

                    # Export automatique en CSV
                    self.export_to_csv(df, filepath.stem)

        # Rapport final
        logger.info("=" * 60)
        logger.info("TRAITEMENT TERMINÉ")
        logger.info(f"  Fichiers traités: {self.stats['files_processed']}")
        logger.info(f"  Lignes totales: {self.stats['lines_processed']}")
        logger.info(f"  Erreurs: {len(self.stats['errors'])}")
        logger.info("=" * 60)

        return results

    def get_schema(self, file_type: str) -> Optional[dict]:
        """
        Retourne le schéma d'un type de fichier.
        """
        return self.format_configs.get(file_type)


# =============================================================================
# UTILITAIRES
# =============================================================================

def compute_file_hash(filepath: Path) -> str:
    """Calcule le hash SHA-256 d'un fichier."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def merge_dataframes(
    dataframes: List[pl.DataFrame],
    on_columns: List[str] = None
) -> pl.DataFrame:
    """
    Fusionne plusieurs DataFrames.
    """
    if not dataframes:
        return pl.DataFrame()

    if len(dataframes) == 1:
        return dataframes[0]

    if on_columns:
        # Jointure sur colonnes spécifiques
        result = dataframes[0]
        for df in dataframes[1:]:
            result = result.join(df, on=on_columns, how="outer")
        return result
    else:
        # Concaténation verticale
        return pl.concat(dataframes, how="diagonal")


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main():
    """Point d'entrée principal."""
    import argparse

    parser = argparse.ArgumentParser(
        description="DIM ETL Processor - Moulinettes PMSI"
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        default="./data/pmsi",
        help="Répertoire source des fichiers PMSI"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="Répertoire de sortie pour les CSV"
    )
    parser.add_argument(
        "--year", "-y",
        type=str,
        default="2024",
        help="Année du format PMSI à utiliser"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Traiter un fichier spécifique uniquement"
    )
    parser.add_argument(
        "--list-formats",
        action="store_true",
        help="Afficher les formats disponibles"
    )

    args = parser.parse_args()

    # Configuration
    config = ETLConfig(
        source_dir=Path(args.source),
        output_dir=Path(args.output),
        year=args.year
    )

    # Initialisation du parser
    universal_parser = UniversalParser(config)

    # Liste des formats
    if args.list_formats:
        print("\nFormats PMSI disponibles:")
        for fmt_name, fmt_config in universal_parser.format_configs.items():
            print(f"  - {fmt_name}: {fmt_config.get('description', 'N/A')}")
        return

    # Traitement d'un fichier spécifique
    if args.file:
        filepath = Path(args.file)
        df = universal_parser.parse_file(filepath)
        if df is not None:
            universal_parser.export_to_csv(df, filepath.stem)
            print(f"\n✓ Traitement terminé: {len(df)} lignes")
        return

    # Traitement du répertoire complet
    results = universal_parser.process_directory(priority_order=True)

    print(f"\n✓ {len(results)} fichiers traités avec succès")


if __name__ == "__main__":
    main()
