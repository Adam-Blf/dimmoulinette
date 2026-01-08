"""
===============================================================================
PSY_LOGIC.PY - Logique Métier Épisodes Ambulatoires Psychiatrie
===============================================================================
DIM - Data Intelligence Médicale
Reconstruction des parcours patients et détection des anomalies organisationnelles
===============================================================================
"""

import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger


# =============================================================================
# CONFIGURATION
# =============================================================================

class AnomalieType(Enum):
    """Types d'anomalies détectables."""
    DUREE_EXCESSIVE = "DUREE_EXCESSIVE"          # Durée > 1 jour en ambulatoire
    CHEVAUCHEMENT = "CHEVAUCHEMENT"              # Dates qui se chevauchent
    SEQUENCE_INCOHERENTE = "SEQUENCE_INCOHERENTE"  # Ordre des dates incohérent
    PATIENT_MULTI_SITE = "PATIENT_MULTI_SITE"    # Patient sur plusieurs sites le même jour
    RUPTURE_PARCOURS = "RUPTURE_PARCOURS"        # Gap inexpliqué dans le parcours


@dataclass
class EpisodeConfig:
    """Configuration pour la reconstruction des épisodes."""
    max_gap_days: int = 1                # Écart max entre actes consécutifs d'un épisode
    seuil_duree_anomalie: int = 1        # Durée > X jours = anomalie
    group_by_columns: List[str] = field(
        default_factory=lambda: ["NO_PATIENT", "CODE_UM", "FINESS_PMSI"]
    )
    date_column: str = "DATE_ACTE"
    patient_column: str = "NO_PATIENT"
    um_column: str = "CODE_UM"
    site_column: str = "FINESS_PMSI"


# =============================================================================
# MOTEUR DE RECONSTRUCTION DES ÉPISODES
# =============================================================================

class EpisodeBuilder:
    """
    Constructeur d'épisodes pour la psychiatrie ambulatoire.

    Le format national RAA est insuffisant car il ne capture que des actes isolés.
    Cette classe reconstruit les "épisodes de soins" en regroupant les actes
    consécutifs pour un même patient/UM/site.
    """

    def __init__(self, config: EpisodeConfig = None):
        self.config = config or EpisodeConfig()
        self.episodes_df: Optional[pl.DataFrame] = None
        self.anomalies: List[Dict] = []

    def build_episodes(self, raa_df: pl.DataFrame) -> pl.DataFrame:
        """
        Reconstruit les épisodes à partir des données RAA.

        Algorithme:
        1. Trier par Patient + UM + Site + Date
        2. Détecter les ruptures (gap > max_gap_days)
        3. Attribuer un ID épisode à chaque groupe d'actes consécutifs
        4. Calculer la durée de chaque épisode
        5. Flaguer les anomalies
        """
        logger.info("Reconstruction des épisodes ambulatoires PSY...")

        # Validation des colonnes requises
        required_cols = [
            self.config.patient_column,
            self.config.date_column
        ]
        missing = [c for c in required_cols if c not in raa_df.columns]
        if missing:
            logger.error(f"Colonnes manquantes: {missing}")
            # Tentative de mapping alternatif
            raa_df = self._map_alternative_columns(raa_df)

        # S'assurer que la colonne date est au bon format
        if self.config.date_column in raa_df.columns:
            raa_df = raa_df.with_columns([
                pl.col(self.config.date_column).cast(pl.Date).alias(self.config.date_column)
            ])

        # Tri des données
        sort_cols = [
            c for c in self.config.group_by_columns + [self.config.date_column]
            if c in raa_df.columns
        ]
        raa_df = raa_df.sort(sort_cols)

        # Construction des épisodes
        episodes = self._assign_episode_ids(raa_df)

        # Calcul des durées
        episodes = self._calculate_durations(episodes)

        # Détection des anomalies
        episodes = self._detect_anomalies(episodes)

        self.episodes_df = episodes

        # Stats
        nb_episodes = episodes.select(pl.col("EPISODE_ID").n_unique()).item()
        nb_anomalies = episodes.filter(pl.col("FLAG_ANOMALIE")).height

        logger.info(f"  ✓ {nb_episodes} épisodes reconstruits")
        logger.info(f"  ⚠ {nb_anomalies} actes avec anomalie détectée")

        return episodes

    def _map_alternative_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Tente de mapper les colonnes alternatives si les colonnes standard sont absentes.
        """
        mappings = {
            "NO_PATIENT": ["patient", "PATIENT", "NUM_PATIENT", "IPP", "ID_PATIENT"],
            "CODE_UM": ["um", "UM", "UNITE_MED", "CODE_UNITE"],
            "FINESS_PMSI": ["site", "SITE", "FINESS", "ETABLISSEMENT"],
            "DATE_ACTE": ["date_debut", "DATE_DEBUT", "DATE", "DATE_SOINS"]
        }

        for standard_name, alternatives in mappings.items():
            if standard_name not in df.columns:
                for alt in alternatives:
                    if alt in df.columns:
                        df = df.rename({alt: standard_name})
                        logger.info(f"  Mapping colonne: {alt} -> {standard_name}")
                        break

        return df

    def _assign_episode_ids(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Attribue un ID épisode unique à chaque groupe d'actes consécutifs.
        """
        # Colonnes de groupement disponibles
        group_cols = [c for c in self.config.group_by_columns if c in df.columns]

        if not group_cols:
            group_cols = [self.config.patient_column]

        # Calcul du gap entre les dates consécutives
        df = df.with_columns([
            pl.col(self.config.date_column)
            .diff()
            .over(group_cols)
            .alias("_date_gap")
        ])

        # Détection des ruptures (nouvelle épisode si gap > seuil)
        df = df.with_columns([
            (
                (pl.col("_date_gap").is_null()) |  # Premier acte
                (pl.col("_date_gap") > pl.duration(days=self.config.max_gap_days))
            ).alias("_is_new_episode")
        ])

        # Numérotation des épisodes
        df = df.with_columns([
            pl.col("_is_new_episode")
            .cum_sum()
            .over(group_cols)
            .alias("_episode_num")
        ])

        # Création de l'ID épisode unique
        # Format: PATIENT_UM_SITE_NUM
        id_parts = []
        for col in group_cols:
            if col in df.columns:
                id_parts.append(pl.col(col).cast(pl.Utf8))

        id_parts.append(pl.col("_episode_num").cast(pl.Utf8))

        df = df.with_columns([
            pl.concat_str(id_parts, separator="_").alias("EPISODE_ID")
        ])

        return df

    def _calculate_durations(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calcule la durée de chaque épisode et les dates de début/fin.
        """
        date_col = self.config.date_column

        # Calcul des dates min/max par épisode
        episode_dates = df.group_by("EPISODE_ID").agg([
            pl.col(date_col).min().alias("DATE_DEBUT_EPISODE"),
            pl.col(date_col).max().alias("DATE_FIN_EPISODE"),
            pl.col(date_col).n_unique().alias("NB_JOURS_ACTES"),
            pl.len().alias("NB_ACTES_EPISODE")
        ])

        # Jointure pour enrichir le dataframe
        df = df.join(episode_dates, on="EPISODE_ID", how="left")

        # Calcul de la durée en jours
        df = df.with_columns([
            (
                (pl.col("DATE_FIN_EPISODE") - pl.col("DATE_DEBUT_EPISODE"))
                .dt.total_days()
            ).alias("DUREE_EPISODE_JOURS")
        ])

        return df

    def _detect_anomalies(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Détecte les anomalies organisationnelles selon les règles métier.

        Règle principale: Si durée > 1 jour en ambulatoire = anomalie
        (indicateur de problème d'aval ou d'organisation)
        """
        seuil = self.config.seuil_duree_anomalie

        # Flag principal: durée excessive
        df = df.with_columns([
            (pl.col("DUREE_EPISODE_JOURS") > seuil).alias("FLAG_ANOMALIE")
        ])

        # Type d'anomalie
        df = df.with_columns([
            pl.when(pl.col("DUREE_EPISODE_JOURS") > seuil)
            .then(pl.lit(AnomalieType.DUREE_EXCESSIVE.value))
            .otherwise(pl.lit(None))
            .alias("TYPE_ANOMALIE")
        ])

        # Message d'anomalie
        df = df.with_columns([
            pl.when(pl.col("FLAG_ANOMALIE"))
            .then(
                pl.concat_str([
                    pl.lit("Durée épisode: "),
                    pl.col("DUREE_EPISODE_JOURS").cast(pl.Utf8),
                    pl.lit(" jours (seuil: "),
                    pl.lit(str(seuil)),
                    pl.lit(" jour)")
                ])
            )
            .otherwise(pl.lit(None))
            .alias("MESSAGE_ANOMALIE")
        ])

        # Collecte des anomalies pour reporting
        anomalies = df.filter(pl.col("FLAG_ANOMALIE")).select([
            "EPISODE_ID",
            self.config.patient_column,
            "DATE_DEBUT_EPISODE",
            "DATE_FIN_EPISODE",
            "DUREE_EPISODE_JOURS",
            "TYPE_ANOMALIE"
        ]).unique()

        self.anomalies = anomalies.to_dicts()

        return df

    def get_anomaly_report(self) -> pl.DataFrame:
        """
        Génère un rapport détaillé des anomalies détectées.
        """
        if not self.anomalies:
            return pl.DataFrame()

        report_df = pl.DataFrame(self.anomalies)

        # Tri par durée décroissante (les plus longues en premier)
        report_df = report_df.sort("DUREE_EPISODE_JOURS", descending=True)

        return report_df

    def enrich_raa_with_episodes(self, raa_df: pl.DataFrame) -> pl.DataFrame:
        """
        Enrichit les données RAA originales avec les informations d'épisode.
        """
        if self.episodes_df is None:
            self.build_episodes(raa_df)

        # Sélection des colonnes à ajouter
        episode_cols = [
            "EPISODE_ID",
            "DATE_DEBUT_EPISODE",
            "DATE_FIN_EPISODE",
            "DUREE_EPISODE_JOURS",
            "NB_ACTES_EPISODE",
            "FLAG_ANOMALIE",
            "TYPE_ANOMALIE",
            "MESSAGE_ANOMALIE"
        ]

        available_cols = [c for c in episode_cols if c in self.episodes_df.columns]

        # On ne garde que les colonnes d'enrichissement uniques
        enrichment = self.episodes_df.select(
            ["_line_number", "_source_file"] + available_cols
        ).unique(subset=["_line_number", "_source_file"])

        # Jointure
        if "_line_number" in raa_df.columns:
            enriched = raa_df.join(
                enrichment,
                on=["_line_number", "_source_file"],
                how="left"
            )
        else:
            # Si pas de colonnes de jointure, retourner le df d'épisodes directement
            enriched = self.episodes_df

        return enriched


# =============================================================================
# ANALYSES COMPLÉMENTAIRES
# =============================================================================

class ParcoursAnalyzer:
    """
    Analyseur de parcours patients pour insights supplémentaires.
    """

    def __init__(self):
        self.stats = {}

    def analyze_parcours(self, episodes_df: pl.DataFrame) -> Dict:
        """
        Analyse les parcours patients à partir des épisodes.
        """
        stats = {}

        # Nombre total de patients
        if "NO_PATIENT" in episodes_df.columns:
            stats["nb_patients"] = episodes_df.select(
                pl.col("NO_PATIENT").n_unique()
            ).item()

        # Nombre total d'épisodes
        if "EPISODE_ID" in episodes_df.columns:
            stats["nb_episodes"] = episodes_df.select(
                pl.col("EPISODE_ID").n_unique()
            ).item()

        # Distribution des durées d'épisodes
        if "DUREE_EPISODE_JOURS" in episodes_df.columns:
            duree_stats = episodes_df.select([
                pl.col("DUREE_EPISODE_JOURS").mean().alias("duree_moyenne"),
                pl.col("DUREE_EPISODE_JOURS").median().alias("duree_mediane"),
                pl.col("DUREE_EPISODE_JOURS").max().alias("duree_max"),
            ]).to_dicts()[0]
            stats["durees"] = duree_stats

        # Taux d'anomalies
        if "FLAG_ANOMALIE" in episodes_df.columns:
            total = episodes_df.height
            anomalies = episodes_df.filter(pl.col("FLAG_ANOMALIE")).height
            stats["taux_anomalies"] = round(anomalies / total * 100, 2) if total > 0 else 0

        # Distribution par UM
        if "CODE_UM" in episodes_df.columns:
            um_dist = episodes_df.group_by("CODE_UM").agg([
                pl.len().alias("nb_actes"),
                pl.col("EPISODE_ID").n_unique().alias("nb_episodes")
            ]).sort("nb_actes", descending=True)
            stats["distribution_um"] = um_dist.to_dicts()

        self.stats = stats
        return stats

    def generate_summary_report(self, episodes_df: pl.DataFrame) -> str:
        """
        Génère un rapport textuel de synthèse.
        """
        stats = self.analyze_parcours(episodes_df)

        report = []
        report.append("=" * 60)
        report.append("RAPPORT DE SYNTHÈSE - ÉPISODES AMBULATOIRES PSY")
        report.append("=" * 60)
        report.append("")

        if "nb_patients" in stats:
            report.append(f"Patients uniques: {stats['nb_patients']}")
        if "nb_episodes" in stats:
            report.append(f"Épisodes reconstruits: {stats['nb_episodes']}")

        if "durees" in stats:
            report.append("")
            report.append("Durées des épisodes:")
            report.append(f"  - Moyenne: {stats['durees']['duree_moyenne']:.1f} jours")
            report.append(f"  - Médiane: {stats['durees']['duree_mediane']:.1f} jours")
            report.append(f"  - Maximum: {stats['durees']['duree_max']:.0f} jours")

        if "taux_anomalies" in stats:
            report.append("")
            report.append(f"Taux d'anomalies (durée > seuil): {stats['taux_anomalies']}%")

        if "distribution_um" in stats and len(stats["distribution_um"]) > 0:
            report.append("")
            report.append("Top 5 Unités Médicales:")
            for i, um in enumerate(stats["distribution_um"][:5], 1):
                report.append(f"  {i}. UM {um['CODE_UM']}: {um['nb_actes']} actes, {um['nb_episodes']} épisodes")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def process_raa_episodes(
    raa_filepath: Path,
    output_dir: Path = None,
    config: EpisodeConfig = None
) -> Tuple[pl.DataFrame, Dict]:
    """
    Fonction principale pour traiter les fichiers RAA et générer les épisodes.
    """
    from etl_processor import UniversalParser, ETLConfig

    output_dir = output_dir or Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parsing du fichier RAA
    etl_config = ETLConfig(output_dir=output_dir)
    parser = UniversalParser(etl_config)
    raa_df = parser.parse_file(raa_filepath)

    if raa_df is None:
        logger.error(f"Impossible de parser {raa_filepath}")
        return None, {}

    # Construction des épisodes
    episode_builder = EpisodeBuilder(config)
    episodes_df = episode_builder.build_episodes(raa_df)

    # Enrichissement des données RAA
    enriched_df = episode_builder.enrich_raa_with_episodes(raa_df)

    # Export
    output_path = output_dir / f"{raa_filepath.stem}_episodes.csv"
    enriched_df.write_csv(output_path, separator=";")
    logger.info(f"  ✓ Export: {output_path}")

    # Rapport d'anomalies
    anomaly_report = episode_builder.get_anomaly_report()
    if len(anomaly_report) > 0:
        anomaly_path = output_dir / f"{raa_filepath.stem}_anomalies.csv"
        anomaly_report.write_csv(anomaly_path, separator=";")
        logger.info(f"  ✓ Rapport anomalies: {anomaly_path}")

    # Analyse
    analyzer = ParcoursAnalyzer()
    stats = analyzer.analyze_parcours(episodes_df)

    # Rapport de synthèse
    summary = analyzer.generate_summary_report(episodes_df)
    summary_path = output_dir / f"{raa_filepath.stem}_synthese.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    logger.info(f"  ✓ Synthèse: {summary_path}")

    return enriched_df, stats


def main():
    """Point d'entrée principal."""
    import argparse

    parser = argparse.ArgumentParser(
        description="DIM PSY Logic - Reconstruction des épisodes ambulatoires"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Fichier RAA à traiter"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="Répertoire de sortie"
    )
    parser.add_argument(
        "--seuil", "-s",
        type=int,
        default=1,
        help="Seuil de durée pour anomalie (en jours)"
    )
    parser.add_argument(
        "--gap", "-g",
        type=int,
        default=1,
        help="Écart max entre actes d'un même épisode (en jours)"
    )

    args = parser.parse_args()

    config = EpisodeConfig(
        seuil_duree_anomalie=args.seuil,
        max_gap_days=args.gap
    )

    df, stats = process_raa_episodes(
        Path(args.input_file),
        Path(args.output),
        config
    )

    if df is not None:
        print(f"\n✓ Traitement terminé")
        print(f"  {stats.get('nb_episodes', 'N/A')} épisodes reconstruits")
        print(f"  {stats.get('taux_anomalies', 'N/A')}% d'anomalies")


if __name__ == "__main__":
    main()
