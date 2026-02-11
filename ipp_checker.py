"""
===============================================================================
IPP_CHECKER.PY - Détection des IPP avec dates de naissance multiples
===============================================================================
DIM - Data Intelligence Médicale
Détecte les incohérences de dates de naissance pour un même patient (IPP)
dans les données PMSI fusionnées (problème Fondation Vallée / BI-Query).
===============================================================================

Contexte:
  Le support OSPI signale que les fichiers PMSI ne peuvent pas être intégrés
  dans BI-Query depuis PMSI-Pilot à cause de dates de naissance multiples
  pour certains patients. Ce module permet de détecter ces IPP problématiques.

Usage:
  python ipp_checker.py --source ./data/pmsi --years 2018 2019 2020 2021 2022
  python ipp_checker.py --source ./data/pmsi --years 2018-2022 --export csv
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import polars as pl
from loguru import logger


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class IPPCheckerConfig:
    """Configuration du vérificateur IPP."""
    source_dir: Path = field(default_factory=lambda: Path("./data/pmsi"))
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    configs_dir: Path = field(default_factory=lambda: Path("./configs"))
    csv_separator: str = ";"
    encoding: str = "utf-8"
    years: List[str] = field(default_factory=lambda: ["2018", "2019", "2020", "2021", "2022"])


# =============================================================================
# MAPPING DES COLONNES IPP / DATE DE NAISSANCE PAR TYPE DE FICHIER
# =============================================================================

# Chaque type de fichier PMSI a des noms de colonnes différents pour
# l'identifiant patient et la date de naissance.
COLUMN_MAPPINGS = {
    "RPS": {
        "patient_cols": ["NO_PATIENT", "NUM_PATIENT", "IPP", "NIP", "PATIENT_ID", "ID_PATIENT"],
        "birth_cols": ["DATE_NAISSANCE", "DATE_NAISS", "DDN"],
        "file_patterns": ["RPS", "rps", "RIMP", "rimp"],
    },
    "RAA": {
        "patient_cols": ["NO_PATIENT", "NUM_PATIENT", "IPP", "NIP", "PATIENT_ID", "ID_PATIENT"],
        "birth_cols": ["DATE_NAISSANCE", "DATE_NAISS", "DDN"],
        "file_patterns": ["RAA", "raa", "R3A", "r3a", "RPSA", "rpsa"],
    },
    "VID_HOSP": {
        "patient_cols": ["NO_ADMINISTRATIF", "NO_SEJOUR_PMSI", "NO_PATIENT", "IPP"],
        "birth_cols": ["DATE_NAISSANCE_REEL", "DATE_NAISSANCE", "DATE_NAISS"],
        "file_patterns": ["VIDHOSP", "vidhosp", "VID_HOSP", "VID-HOSP", "ANOHOSP", "anohosp"],
    },
    "VID_IPP": {
        "patient_cols": ["IPP_LOCAL", "IPP", "NO_PATIENT_ANO", "NO_PATIENT"],
        "birth_cols": ["DATE_NAISSANCE", "DATE_NAISS", "DDN"],
        "file_patterns": ["VIDIPP", "vidipp", "VID_IPP", "VID-IPP", "ipp_"],
    },
    "RSF_ACE": {
        "patient_cols": ["NO_PATIENT", "NUM_PATIENT", "IPP"],
        "birth_cols": ["DATE_NAISSANCE", "DATE_NAISS", "DDN"],
        "file_patterns": ["RSF", "rsf", "RSFACE", "rsface", "ACE", "ace"],
    },
    "RUM_RSS": {
        "patient_cols": ["NO_PATIENT", "NUM_PATIENT", "IPP"],
        "birth_cols": ["DATE_NAISSANCE", "DATE_NAISS", "DDN"],
        "file_patterns": ["RUM", "rum", "RSS", "rss"],
    },
}


# =============================================================================
# DÉTECTEUR D'IPP AVEC DATES DE NAISSANCE MULTIPLES
# =============================================================================

class IPPBirthDateChecker:
    """
    Détecte les IPP ayant plusieurs dates de naissance distinctes
    dans les données PMSI fusionnées.

    Problème: Quand un même patient (identifié par son IPP) a des dates
    de naissance différentes dans différents fichiers ou différentes années,
    cela empêche l'intégration dans BI-Query depuis PMSI-Pilot.
    """

    def __init__(self, config: IPPCheckerConfig = None):
        self.config = config or IPPCheckerConfig()
        self.format_configs: Dict[str, dict] = {}
        self._load_format_configs()

        # Résultats
        self.all_records: List[pl.DataFrame] = []
        self.merged_df: Optional[pl.DataFrame] = None
        self.duplicates_df: Optional[pl.DataFrame] = None
        self.stats: Dict[str, Any] = {
            "files_scanned": 0,
            "total_records": 0,
            "unique_patients": 0,
            "patients_multi_ddn": 0,
            "files_by_type": {},
            "years_covered": [],
        }

    def _load_format_configs(self) -> None:
        """Charge les configs de format pour le parsing positionnel."""
        if not self.config.configs_dir.exists():
            return

        for year in self.config.years:
            for config_file in self.config.configs_dir.glob(f"format_*_{year}.json"):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        fmt = json.load(f)
                        key = f"{fmt.get('type', 'UNKNOWN')}_{year}"
                        self.format_configs[key] = fmt
                except Exception as e:
                    logger.warning(f"Erreur chargement config {config_file}: {e}")

        # Fallback: charger les configs 2024 comme référence
        for config_file in self.config.configs_dir.glob("format_*_2024.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    fmt = json.load(f)
                    key = fmt.get("type", "UNKNOWN")
                    if key not in self.format_configs:
                        self.format_configs[key] = fmt
            except Exception:
                pass

    def _detect_file_type(self, filepath: Path) -> Optional[str]:
        """Détecte le type de fichier PMSI."""
        filename = filepath.name

        for file_type, mapping in COLUMN_MAPPINGS.items():
            for pattern in mapping["file_patterns"]:
                if pattern in filename:
                    return file_type

        return None

    def _detect_year(self, filepath: Path) -> Optional[str]:
        """Détecte l'année dans le nom du fichier."""
        filename = filepath.name
        for year in range(2015, 2030):
            if str(year) in filename:
                return str(year)
        return None

    def _find_column(self, df: pl.DataFrame, candidates: List[str]) -> Optional[str]:
        """Trouve la première colonne correspondante dans le DataFrame."""
        df_cols_upper = {col.upper(): col for col in df.columns}
        for candidate in candidates:
            if candidate.upper() in df_cols_upper:
                return df_cols_upper[candidate.upper()]
        return None

    def _read_file_flexible(self, filepath: Path) -> Optional[pl.DataFrame]:
        """
        Lit un fichier PMSI (CSV délimité ou positionnel) et retourne un DataFrame.
        """
        # 1. Essai en tant que fichier délimité (CSV/TSV)
        try:
            with open(filepath, 'r', encoding=self.config.encoding, errors='replace') as f:
                first_line = f.readline()

            separator = None
            if ';' in first_line:
                separator = ';'
            elif '\t' in first_line:
                separator = '\t'
            elif ',' in first_line:
                separator = ','

            if separator:
                df = pl.read_csv(
                    filepath,
                    separator=separator,
                    ignore_errors=True,
                    truncate_ragged_lines=True,
                    encoding='utf8-lossy',
                    infer_schema_length=0
                )
                if len(df.columns) > 2:
                    return df
        except Exception:
            pass

        # 2. Essai en tant que fichier positionnel
        try:
            return self._parse_positional(filepath)
        except Exception as e:
            logger.warning(f"Impossible de lire {filepath.name}: {e}")
            return None

    def _parse_positional(self, filepath: Path) -> Optional[pl.DataFrame]:
        """Parse un fichier positionnel avec les configs de format."""
        file_type = self._detect_file_type(filepath)
        if not file_type:
            return None

        # Chercher la config de format
        format_config = None
        for key, cfg in self.format_configs.items():
            cfg_type = cfg.get("type", "").replace("-", "_").upper()
            if cfg_type == file_type.replace("-", "_").upper():
                format_config = cfg
                break

        if not format_config:
            return None

        fields = format_config.get("fields", [])
        if not fields and "subtypes" in format_config:
            # Pour RUM/RSS, prendre le premier sous-type
            for subtype_name, subtype_cfg in format_config["subtypes"].items():
                fields = subtype_cfg.get("fields", [])
                if fields:
                    break

        if not fields:
            return None

        records = []
        with open(filepath, 'r', encoding=self.config.encoding, errors='replace') as f:
            for line in f:
                if not line.strip():
                    continue
                record = {}
                for field_def in fields:
                    name = field_def["name"]
                    start = field_def["start"] - 1
                    end = field_def["end"]
                    try:
                        record[name] = line[start:end].strip() if len(line) >= end else ""
                    except IndexError:
                        record[name] = ""
                records.append(record)

        if not records:
            return None

        return pl.DataFrame(records)

    def scan_files(self, source_dir: Path = None) -> List[Path]:
        """
        Scanner le dossier source pour trouver tous les fichiers PMSI
        correspondant aux années demandées.
        """
        source = source_dir or self.config.source_dir

        if not source.exists():
            logger.error(f"Dossier source non trouvé: {source}")
            return []

        data_extensions = {'.txt', '.csv', '.tsv', '.dat', ''}
        found_files = []

        # Recherche récursive
        for filepath in source.rglob("*"):
            if not filepath.is_file():
                continue
            if filepath.suffix.lower() not in data_extensions:
                continue

            # Vérifier que c'est un fichier PMSI reconnu
            file_type = self._detect_file_type(filepath)
            if file_type is None:
                continue

            found_files.append(filepath)

        logger.info(f"Fichiers PMSI trouvés: {len(found_files)}")
        return found_files

    def extract_patient_birth_data(
        self,
        filepath: Path
    ) -> Optional[pl.DataFrame]:
        """
        Extrait les paires (IPP, DATE_NAISSANCE) d'un fichier PMSI.
        Retourne un DataFrame avec les colonnes normalisées:
          - IPP: identifiant patient
          - DATE_NAISSANCE: date de naissance
          - SOURCE_FILE: nom du fichier source
          - FILE_TYPE: type de fichier PMSI
          - ANNEE: année détectée
        """
        file_type = self._detect_file_type(filepath)
        if not file_type:
            return None

        df = self._read_file_flexible(filepath)
        if df is None or df.is_empty():
            return None

        # Trouver les colonnes IPP et DDN
        mapping = COLUMN_MAPPINGS.get(file_type, {})
        patient_col = self._find_column(df, mapping.get("patient_cols", []))
        birth_col = self._find_column(df, mapping.get("birth_cols", []))

        if patient_col is None or birth_col is None:
            logger.warning(
                f"  {filepath.name}: colonnes manquantes "
                f"(patient={patient_col}, naissance={birth_col})"
            )
            return None

        # Extraction et normalisation
        year = self._detect_year(filepath) or "INCONNU"

        result = df.select([
            pl.col(patient_col).cast(pl.Utf8).str.strip_chars().alias("IPP"),
            pl.col(birth_col).cast(pl.Utf8).str.strip_chars().alias("DATE_NAISSANCE"),
        ]).with_columns([
            pl.lit(filepath.name).alias("SOURCE_FILE"),
            pl.lit(file_type).alias("FILE_TYPE"),
            pl.lit(year).alias("ANNEE"),
        ])

        # Filtrer les lignes vides
        result = result.filter(
            (pl.col("IPP") != "") &
            (pl.col("IPP").is_not_null()) &
            (pl.col("DATE_NAISSANCE") != "") &
            (pl.col("DATE_NAISSANCE").is_not_null()) &
            (pl.col("DATE_NAISSANCE") != "00000000")
        )

        logger.info(
            f"  ✓ {filepath.name} ({file_type}): "
            f"{len(result)} enregistrements patient/DDN extraits"
        )

        return result

    def run_check(
        self,
        source_dir: Path = None,
        files: List[Path] = None
    ) -> pl.DataFrame:
        """
        Exécute la vérification complète:
        1. Scanne tous les fichiers PMSI
        2. Extrait les paires IPP/DDN
        3. Fusionne et détecte les doublons

        Retourne un DataFrame avec les IPP problématiques.
        """
        logger.info("=" * 60)
        logger.info("VÉRIFICATION IPP - DATES DE NAISSANCE MULTIPLES")
        logger.info(f"  Années ciblées: {', '.join(self.config.years)}")
        logger.info("=" * 60)

        # 1. Scanner les fichiers
        if files is None:
            files = self.scan_files(source_dir)

        if not files:
            logger.warning("Aucun fichier PMSI trouvé")
            return pl.DataFrame()

        # 2. Extraire les données patient/DDN de chaque fichier
        all_extracts = []
        files_by_type: Dict[str, int] = {}

        for filepath in files:
            extract = self.extract_patient_birth_data(filepath)
            if extract is not None and not extract.is_empty():
                all_extracts.append(extract)
                file_type = self._detect_file_type(filepath) or "UNKNOWN"
                files_by_type[file_type] = files_by_type.get(file_type, 0) + 1
                self.stats["files_scanned"] += 1

        if not all_extracts:
            logger.warning("Aucune donnée patient/DDN extraite")
            return pl.DataFrame()

        # 3. Fusion de tous les extraits
        self.merged_df = pl.concat(all_extracts, how="diagonal")
        self.stats["total_records"] = len(self.merged_df)
        self.stats["files_by_type"] = files_by_type

        logger.info(f"\nTotal enregistrements fusionnés: {len(self.merged_df)}")

        # 4. Détection des IPP avec DDN multiples
        self.duplicates_df = self._detect_multi_birth_dates()

        return self.duplicates_df

    def _detect_multi_birth_dates(self) -> pl.DataFrame:
        """
        Détecte les IPP ayant plus d'une date de naissance distincte.

        Retourne un DataFrame avec:
          - IPP
          - NB_DDN_DISTINCTES: nombre de DDN différentes
          - DATES_NAISSANCE: liste des DDN trouvées
          - SOURCES: fichiers sources concernés
          - ANNEES: années concernées
          - NB_OCCURRENCES: nombre total d'enregistrements pour cet IPP
        """
        if self.merged_df is None or self.merged_df.is_empty():
            return pl.DataFrame()

        # Grouper par IPP, compter les DDN distinctes
        ipp_ddn = self.merged_df.group_by("IPP").agg([
            pl.col("DATE_NAISSANCE").n_unique().alias("NB_DDN_DISTINCTES"),
            pl.col("DATE_NAISSANCE").unique().cast(pl.List(pl.Utf8)).alias("DATES_NAISSANCE_LIST"),
            pl.col("SOURCE_FILE").unique().cast(pl.List(pl.Utf8)).alias("SOURCES_LIST"),
            pl.col("ANNEE").unique().cast(pl.List(pl.Utf8)).alias("ANNEES_LIST"),
            pl.col("FILE_TYPE").unique().cast(pl.List(pl.Utf8)).alias("TYPES_FICHIER_LIST"),
            pl.len().alias("NB_OCCURRENCES"),
        ])

        # Filtrer: garder uniquement les IPP avec > 1 DDN distincte
        multi_ddn = ipp_ddn.filter(pl.col("NB_DDN_DISTINCTES") > 1)

        # Convertir les listes en chaînes pour l'affichage
        multi_ddn = multi_ddn.with_columns([
            pl.col("DATES_NAISSANCE_LIST")
            .list.eval(pl.element().cast(pl.Utf8))
            .list.join(" | ")
            .alias("DATES_NAISSANCE"),

            pl.col("SOURCES_LIST")
            .list.eval(pl.element().cast(pl.Utf8))
            .list.join(" | ")
            .alias("SOURCES"),

            pl.col("ANNEES_LIST")
            .list.eval(pl.element().cast(pl.Utf8))
            .list.join(" | ")
            .alias("ANNEES"),

            pl.col("TYPES_FICHIER_LIST")
            .list.eval(pl.element().cast(pl.Utf8))
            .list.join(" | ")
            .alias("TYPES_FICHIER"),
        ])

        # Sélection finale et tri
        result = multi_ddn.select([
            "IPP",
            "NB_DDN_DISTINCTES",
            "DATES_NAISSANCE",
            "SOURCES",
            "ANNEES",
            "TYPES_FICHIER",
            "NB_OCCURRENCES",
        ]).sort("NB_DDN_DISTINCTES", descending=True)

        # Stats
        self.stats["unique_patients"] = ipp_ddn.height
        self.stats["patients_multi_ddn"] = result.height
        self.stats["years_covered"] = (
            self.merged_df.select(pl.col("ANNEE").unique())
            .to_series()
            .to_list()
        )

        logger.info("\n" + "=" * 60)
        logger.info("RÉSULTATS DE LA VÉRIFICATION")
        logger.info("=" * 60)
        logger.info(f"  Fichiers analysés:          {self.stats['files_scanned']}")
        logger.info(f"  Enregistrements totaux:     {self.stats['total_records']}")
        logger.info(f"  Patients uniques (IPP):     {self.stats['unique_patients']}")
        logger.info(f"  IPP avec DDN multiples:     {self.stats['patients_multi_ddn']}")
        logger.info(f"  Années couvertes:           {', '.join(self.stats['years_covered'])}")

        if result.height > 0:
            logger.warning(
                f"\n  ⚠ {result.height} IPP PROBLÉMATIQUES DÉTECTÉS ⚠"
            )
            logger.info("\n  Top 10 IPP les plus problématiques:")
            for row in result.head(10).iter_rows(named=True):
                logger.info(
                    f"    IPP {row['IPP']}: "
                    f"{row['NB_DDN_DISTINCTES']} DDN → {row['DATES_NAISSANCE']}"
                )
        else:
            logger.info("\n  ✓ Aucun IPP avec dates de naissance multiples détecté")

        return result

    def get_detail_for_ipp(self, ipp: str) -> pl.DataFrame:
        """
        Retourne le détail de tous les enregistrements pour un IPP donné.
        Utile pour investiguer un cas problématique.
        """
        if self.merged_df is None:
            return pl.DataFrame()

        return self.merged_df.filter(
            pl.col("IPP") == ipp
        ).sort(["ANNEE", "SOURCE_FILE"])

    def export_results(
        self,
        output_name: str = "ipp_ddn_multiples",
        include_detail: bool = True
    ) -> Dict[str, Path]:
        """
        Exporte les résultats vers des fichiers CSV.

        Retourne les chemins des fichiers créés.
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        exported = {}

        # 1. Export du résumé des IPP problématiques
        if self.duplicates_df is not None and not self.duplicates_df.is_empty():
            summary_path = self.config.output_dir / f"{output_name}_resume.csv"
            self.duplicates_df.write_csv(
                summary_path,
                separator=self.config.csv_separator
            )
            exported["resume"] = summary_path
            logger.info(f"  ✓ Résumé exporté: {summary_path}")

            # 2. Export du détail (tous les enregistrements des IPP problématiques)
            if include_detail and self.merged_df is not None:
                problematic_ipps = self.duplicates_df.select("IPP").to_series().to_list()
                detail_df = self.merged_df.filter(
                    pl.col("IPP").is_in(problematic_ipps)
                ).sort(["IPP", "ANNEE", "SOURCE_FILE"])

                detail_path = self.config.output_dir / f"{output_name}_detail.csv"
                detail_df.write_csv(
                    detail_path,
                    separator=self.config.csv_separator
                )
                exported["detail"] = detail_path
                logger.info(f"  ✓ Détail exporté: {detail_path}")

        # 3. Export du rapport JSON
        report = {
            "date_analyse": datetime.now().isoformat(),
            "configuration": {
                "source_dir": str(self.config.source_dir),
                "years": self.config.years,
            },
            "statistiques": self.stats,
            "ipp_problematiques": (
                self.duplicates_df.to_dicts()
                if self.duplicates_df is not None and not self.duplicates_df.is_empty()
                else []
            ),
        }

        report_path = self.config.output_dir / f"{output_name}_rapport.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        exported["rapport"] = report_path
        logger.info(f"  ✓ Rapport JSON exporté: {report_path}")

        return exported

    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé structuré pour l'API."""
        summary = {
            "stats": self.stats,
            "problematic_ipps": [],
        }

        if self.duplicates_df is not None and not self.duplicates_df.is_empty():
            summary["problematic_ipps"] = self.duplicates_df.to_dicts()

        return summary


# =============================================================================
# POINT D'ENTRÉE CLI
# =============================================================================

def main():
    """Point d'entrée en ligne de commande."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Détection des IPP avec dates de naissance multiples dans les données PMSI. "
            "Identifie les patients ayant des DDN incohérentes empêchant "
            "l'intégration BI-Query depuis PMSI-Pilot."
        )
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
        help="Répertoire de sortie"
    )
    parser.add_argument(
        "--years", "-y",
        type=str,
        nargs="+",
        default=["2018", "2019", "2020", "2021", "2022"],
        help="Années à analyser (ex: 2018 2019 2020 2021 2022)"
    )
    parser.add_argument(
        "--export",
        choices=["csv", "json", "all"],
        default="all",
        help="Format d'export des résultats"
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        default=True,
        help="Inclure le détail des enregistrements pour chaque IPP problématique"
    )

    args = parser.parse_args()

    # Gestion du format "2018-2022"
    years = []
    for y in args.years:
        if '-' in y:
            start, end = y.split('-')
            years.extend(str(yr) for yr in range(int(start), int(end) + 1))
        else:
            years.append(y)

    # Configuration
    config = IPPCheckerConfig(
        source_dir=Path(args.source),
        output_dir=Path(args.output),
        years=years
    )

    # Exécution
    checker = IPPBirthDateChecker(config)
    duplicates = checker.run_check()

    # Export
    if not duplicates.is_empty():
        exported = checker.export_results(include_detail=args.detail)
        print("\n✓ Fichiers exportés:")
        for name, path in exported.items():
            print(f"  - {name}: {path}")
    else:
        print("\n✓ Aucun IPP avec dates de naissance multiples détecté.")

    return checker


if __name__ == "__main__":
    main()
