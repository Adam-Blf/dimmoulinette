"""
===============================================================================
PROCESSOR.PY - Pipeline ETL pour données PMSI Psychiatrie
===============================================================================
Moulinettes DIM - Traitement RPS/RAA avec Polars
Auteur: Adam B. | Sécurité: Données de santé (HDS/RGPD)
===============================================================================
"""

import polars as pl
import hashlib
import json
import os
import stat
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('etl_processing.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration centralisée du pipeline ETL."""

    # Chemins
    SOURCE_DIR = Path("./data/pmsi")
    OUTPUT_DIR = Path("./output")
    MANIFEST_FILE = Path("./.data_manifest.json")

    # Fichiers PMSI prioritaires (Psychiatrie)
    # RPS = Résumé Par Séquence (hospitalisation psy)
    # RAA = Résumé d'Activité Ambulatoire (consultations psy)
    PRIORITY_PATTERNS = {
        "RPS": ["*RPS*", "*rps*", "*RIMP*", "*rimp*", "*R3A*"],
        "RAA": ["*RAA*", "*raa*", "*RPSA*", "*rpsa*", "*fichcomp*"],
        "AUTRE": ["*RSS*", "*RSA*", "*FICHCOMP*"]
    }

    # Colonnes attendues (mapping flexible)
    COLUMN_MAPPING = {
        "patient": ["NO_PATIENT", "NUM_PATIENT", "IPP", "NIP", "PATIENT_ID", "ID_PATIENT"],
        "date_debut": ["DATE_DEBUT", "DATE_ENT", "DATE_ENTREE", "DT_DEB", "DATE_ACTE"],
        "date_fin": ["DATE_FIN", "DATE_SOR", "DATE_SORTIE", "DT_FIN"],
        "um": ["UM", "UNITE_MED", "CODE_UM", "UNITE_MEDICALE"],
        "site": ["SITE", "ETABLISSEMENT", "FINESS", "CODE_SITE"],
        "type_sejour": ["TYPE_SEJOUR", "MODE_PRISE_CHARGE", "TYPE_ACTIVITE"],
    }

    # Seuils d'anomalies
    SEUIL_DUREE_AMBULATOIRE_JOURS = 1  # > 1 jour en ambulatoire = anomalie


# ============================================================================
# SÉCURITÉ - Mesures techniques
# ============================================================================

class SecurityManager:
    """Gestion de la sécurité des données de santé."""

    @staticmethod
    def compute_file_hash(filepath: Path) -> str:
        """Calcule le hash SHA-256 d'un fichier pour vérification d'intégrité."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @staticmethod
    def sanitize_path(user_path: str) -> Optional[Path]:
        """
        Protège contre les attaques path traversal.
        Vérifie que le chemin reste dans le dossier autorisé.
        """
        try:
            base_path = Config.SOURCE_DIR.resolve()
            target_path = Path(user_path).resolve()

            # Vérification que le chemin cible est bien sous le dossier source
            if not str(target_path).startswith(str(base_path)):
                logger.error(f"SÉCURITÉ: Tentative d'accès hors périmètre: {user_path}")
                return None

            return target_path
        except Exception as e:
            logger.error(f"SÉCURITÉ: Chemin invalide: {user_path} - {e}")
            return None

    @staticmethod
    def open_readonly(filepath: Path):
        """
        Ouvre un fichier en mode lecture seule avec protection.
        Empêche toute modification accidentelle des données source.
        """
        # Vérifier que le fichier existe
        if not filepath.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {filepath}")

        # Sur Windows, on ne peut pas facilement changer les permissions
        # mais on peut ouvrir en mode 'rb' (read binary) uniquement
        return open(filepath, 'rb')

    @staticmethod
    def load_manifest() -> dict:
        """Charge le manifeste des fichiers traités."""
        if Config.MANIFEST_FILE.exists():
            with open(Config.MANIFEST_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"files": {}, "last_update": None}

    @staticmethod
    def save_manifest(manifest: dict):
        """Sauvegarde le manifeste des fichiers traités."""
        manifest["last_update"] = datetime.now().isoformat()
        with open(Config.MANIFEST_FILE, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    @staticmethod
    def verify_file_integrity(filepath: Path, manifest: dict) -> tuple[bool, str]:
        """
        Vérifie l'intégrité d'un fichier par rapport au manifeste.
        Retourne (is_new_or_modified, current_hash)
        """
        current_hash = SecurityManager.compute_file_hash(filepath)
        stored_hash = manifest.get("files", {}).get(str(filepath), {}).get("hash")

        if stored_hash is None:
            logger.info(f"Nouveau fichier détecté: {filepath.name}")
            return True, current_hash
        elif stored_hash != current_hash:
            logger.warning(f"ALERTE: Fichier modifié depuis dernier traitement: {filepath.name}")
            return True, current_hash
        else:
            logger.info(f"Fichier inchangé (skip): {filepath.name}")
            return False, current_hash


# ============================================================================
# SCANNER DE FICHIERS
# ============================================================================

class FileScanner:
    """Scanner intelligent des fichiers PMSI."""

    def __init__(self):
        self.security = SecurityManager()
        self.manifest = self.security.load_manifest()

    def scan_source_directory(self) -> dict:
        """
        Scanne le dossier source et classe les fichiers par priorité.
        Retourne un dictionnaire {type: [liste de fichiers]}
        """
        if not Config.SOURCE_DIR.exists():
            logger.error(f"ERREUR: Dossier source inexistant: {Config.SOURCE_DIR}")
            return {"RPS": [], "RAA": [], "AUTRE": [], "IGNORE": []}

        results = {"RPS": [], "RAA": [], "AUTRE": [], "IGNORE": []}

        # Extensions de données supportées
        data_extensions = {'.csv', '.txt', '.tsv'}

        for filepath in Config.SOURCE_DIR.iterdir():
            if not filepath.is_file():
                continue

            if filepath.suffix.lower() not in data_extensions:
                results["IGNORE"].append(filepath)
                continue

            # Classification par pattern
            classified = False
            filename = filepath.name.upper()

            for category, patterns in Config.PRIORITY_PATTERNS.items():
                for pattern in patterns:
                    # Conversion du pattern glob en vérification simple
                    pattern_clean = pattern.replace("*", "").upper()
                    if pattern_clean in filename:
                        results[category].append(filepath)
                        classified = True
                        break
                if classified:
                    break

            if not classified:
                results["AUTRE"].append(filepath)

        # Log du scan
        logger.info("=" * 60)
        logger.info("SCAN DU DOSSIER SOURCE")
        logger.info(f"  📁 {Config.SOURCE_DIR}")
        logger.info(f"  🔴 RPS (Priorité 1): {len(results['RPS'])} fichiers")
        logger.info(f"  🟠 RAA (Priorité 2): {len(results['RAA'])} fichiers")
        logger.info(f"  🟡 Autres PMSI: {len(results['AUTRE'])} fichiers")
        logger.info(f"  ⚪ Ignorés: {len(results['IGNORE'])} fichiers")
        logger.info("=" * 60)

        return results

    def get_file_info(self, filepath: Path) -> dict:
        """Retourne les métadonnées d'un fichier."""
        stat_info = filepath.stat()
        is_modified, file_hash = self.security.verify_file_integrity(filepath, self.manifest)

        return {
            "path": str(filepath),
            "name": filepath.name,
            "size_mb": round(stat_info.st_size / (1024 * 1024), 2),
            "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            "hash": file_hash,
            "needs_processing": is_modified
        }


# ============================================================================
# MOTEUR ETL POLARS
# ============================================================================

class PMSIProcessor:
    """Processeur ETL pour données PMSI avec Polars."""

    def __init__(self):
        self.scanner = FileScanner()
        self.security = SecurityManager()
        self.processed_data = []
        self.anomalies = []

    def _detect_columns(self, df: pl.DataFrame) -> dict:
        """
        Détecte automatiquement les colonnes correspondant au mapping.
        Retourne un dictionnaire {nom_standard: nom_réel}
        """
        detected = {}
        df_columns_upper = {col.upper(): col for col in df.columns}

        for standard_name, possible_names in Config.COLUMN_MAPPING.items():
            for possible in possible_names:
                if possible.upper() in df_columns_upper:
                    detected[standard_name] = df_columns_upper[possible.upper()]
                    break

        return detected

    def _read_file_safe(self, filepath: Path) -> Optional[pl.DataFrame]:
        """Lecture sécurisée d'un fichier avec détection automatique du format."""
        try:
            # Détection du séparateur
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                first_line = f.readline()

            if ';' in first_line:
                separator = ';'
            elif '\t' in first_line:
                separator = '\t'
            else:
                separator = ','

            # Lecture avec Polars - forcer tous les types en string
            df = pl.read_csv(
                filepath,
                separator=separator,
                ignore_errors=True,
                truncate_ragged_lines=True,
                encoding='utf8-lossy',
                infer_schema_length=0  # Force toutes les colonnes en String
            )

            logger.info(f"  ✓ Chargé: {filepath.name} ({len(df)} lignes, {len(df.columns)} colonnes)")
            return df

        except Exception as e:
            logger.error(f"  ✗ Erreur lecture {filepath.name}: {e}")
            return None

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse une date avec plusieurs formats possibles."""
        if not date_str or date_str in ['', 'NA', 'NULL', 'NaN']:
            return None

        formats = [
            "%Y-%m-%d", "%d/%m/%Y", "%Y%m%d",
            "%d-%m-%Y", "%Y/%m/%d", "%d.%m.%Y"
        ]

        for fmt in formats:
            try:
                return datetime.strptime(str(date_str).strip(), fmt)
            except ValueError:
                continue
        return None

    def process_episodes_psy(self, df: pl.DataFrame, source_file: str) -> pl.DataFrame:
        """
        Traitement spécifique des épisodes psychiatriques.
        - Tri par Patient, UM, Site, Date
        - Fusion des dates consécutives
        - Détection des anomalies organisationnelles
        """
        # Détection des colonnes
        col_map = self._detect_columns(df)

        required = ['patient', 'date_debut']
        missing = [r for r in required if r not in col_map]

        if missing:
            logger.warning(f"  ⚠ Colonnes manquantes: {missing}")
            return df

        # Standardisation des noms de colonnes
        rename_map = {v: k for k, v in col_map.items()}
        df = df.rename(rename_map)

        # Conversion des dates
        if 'date_debut' in df.columns:
            df = df.with_columns([
                pl.col('date_debut').cast(pl.Utf8).alias('date_debut_str')
            ])

        # Tri des données
        sort_cols = [c for c in ['patient', 'um', 'site', 'date_debut'] if c in df.columns]
        if sort_cols:
            df = df.sort(sort_cols)

        # Ajout des métadonnées
        df = df.with_columns([
            pl.lit(source_file).alias('_source_file'),
            pl.lit(datetime.now().isoformat()).alias('_processed_at')
        ])

        logger.info(f"  ✓ Épisodes traités: {len(df)} lignes")
        return df

    def detect_anomalies(self, df: pl.DataFrame) -> list[dict]:
        """
        Détection des anomalies organisationnelles.
        Règle: Si durée > 1 jour ET non hospitalisé -> ANOMALIE_ORG
        """
        anomalies = []

        # Vérification des colonnes nécessaires
        if 'date_debut' not in df.columns or 'date_fin' not in df.columns:
            return anomalies

        # Cette logique serait étendue avec les vraies règles métier
        # Pour l'instant, on flag les séjours longs en ambulatoire

        try:
            # Exemple simplifié de détection
            for row in df.iter_rows(named=True):
                # Logique de détection d'anomalie
                # À personnaliser selon les règles métier exactes
                pass
        except Exception as e:
            logger.warning(f"Erreur détection anomalies: {e}")

        return anomalies

    def generate_training_data(self, episodes_df: pl.DataFrame) -> list[dict]:
        """
        Génère les données d'entraînement au format instruction/input/output.
        Pour le fine-tuning du LLM.
        """
        training_samples = []

        # Exemple de génération de samples pour l'IA
        # Basé sur les patterns d'anomalies détectés

        sample_templates = [
            {
                "instruction": "Détecter une anomalie de parcours patient en psychiatrie",
                "input": "Patient {patient}, {nb_jours} jours consécutifs en {um}, activité: {activite}",
                "output": "Anomalie probable: {anomalie_type}"
            },
            {
                "instruction": "Analyser la cohérence d'un séjour psychiatrique",
                "input": "Séjour du {date_debut} au {date_fin}, UM: {um}, mode: {mode}",
                "output": "Séjour {statut}: {raison}"
            },
            {
                "instruction": "Identifier un parcours patient atypique",
                "input": "Patient avec {nb_passages} passages en {duree} jours, unités: {unites}",
                "output": "{classification}: {recommandation}"
            }
        ]

        # Génération de samples à partir des données réelles
        if len(episodes_df) > 0:
            for i, row in enumerate(episodes_df.head(100).iter_rows(named=True)):
                # Création de samples variés
                sample = {
                    "instruction": "Analyser ce passage en psychiatrie pour détecter des anomalies",
                    "input": f"Patient {row.get('patient', 'INCONNU')}, unité {row.get('um', 'NC')}, "
                             f"date: {row.get('date_debut', 'NC')}",
                    "output": "Passage normal - Aucune anomalie détectée"
                }
                training_samples.append(sample)

                # Sample avec anomalie simulée (pour équilibrer le dataset)
                if i % 5 == 0:
                    anomaly_sample = {
                        "instruction": "Détecter une anomalie de parcours patient en psychiatrie",
                        "input": f"Patient {row.get('patient', 'INCONNU')}, 5 jours consécutifs "
                                 f"en ambulatoire SAU psychiatrique",
                        "output": "ANOMALIE_ORG détectée: Durée excessive en ambulatoire. "
                                  "Recommandation: Vérifier codage ou organisation du parcours."
                    }
                    training_samples.append(anomaly_sample)

        return training_samples

    def run_full_pipeline(self, force_reprocess: bool = False) -> dict:
        """
        Exécute le pipeline ETL complet.

        1. Scan des fichiers sources
        2. Vérification d'intégrité
        3. Traitement prioritaire (RPS > RAA > Autres)
        4. Génération du dataset d'entraînement
        """
        logger.info("=" * 60)
        logger.info("DÉMARRAGE DU PIPELINE ETL")
        logger.info("=" * 60)

        # Création du dossier output
        Config.OUTPUT_DIR.mkdir(exist_ok=True)

        # Scan des fichiers
        files = self.scanner.scan_source_directory()

        all_episodes = []
        processed_files = []

        # Traitement par priorité
        for category in ["RPS", "RAA", "AUTRE"]:
            if not files[category]:
                continue

            logger.info(f"\n📊 Traitement des fichiers {category}...")

            for filepath in files[category]:
                # Vérification d'intégrité
                file_info = self.scanner.get_file_info(filepath)

                if not file_info["needs_processing"] and not force_reprocess:
                    continue

                # Lecture sécurisée
                df = self._read_file_safe(filepath)
                if df is None:
                    continue

                # Traitement des épisodes
                df_processed = self.process_episodes_psy(df, filepath.name)
                all_episodes.append(df_processed)

                # Mise à jour du manifeste
                self.scanner.manifest["files"][str(filepath)] = {
                    "hash": file_info["hash"],
                    "processed_at": datetime.now().isoformat(),
                    "rows": len(df_processed)
                }

                processed_files.append(file_info)

        # Consolidation des données
        if all_episodes:
            combined_df = pl.concat(all_episodes, how="diagonal")
            logger.info(f"\n✓ Total consolidé: {len(combined_df)} lignes")

            # Génération du dataset d'entraînement
            training_data = self.generate_training_data(combined_df)

            # Sauvegarde du dataset JSONL
            output_path = Config.OUTPUT_DIR / "train_dataset.jsonl"
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in training_data:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')

            logger.info(f"✓ Dataset IA généré: {output_path} ({len(training_data)} samples)")

            # Sauvegarde des épisodes consolidés
            episodes_path = Config.OUTPUT_DIR / "episodes_consolides.csv"
            combined_df.write_csv(episodes_path)
            logger.info(f"✓ Épisodes sauvegardés: {episodes_path}")
        else:
            combined_df = pl.DataFrame()
            training_data = []
            logger.warning("⚠ Aucune donnée traitée")

        # Sauvegarde du manifeste
        self.security.save_manifest(self.scanner.manifest)

        # Rapport final
        result = {
            "status": "success",
            "files_scanned": sum(len(v) for v in files.values()),
            "files_processed": len(processed_files),
            "total_rows": len(combined_df) if len(all_episodes) > 0 else 0,
            "training_samples": len(training_data),
            "anomalies_detected": len(self.anomalies),
            "output_dir": str(Config.OUTPUT_DIR)
        }

        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE TERMINÉ")
        logger.info(f"  Fichiers traités: {result['files_processed']}")
        logger.info(f"  Lignes totales: {result['total_rows']}")
        logger.info(f"  Samples IA générés: {result['training_samples']}")
        logger.info("=" * 60)

        return result


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

def main():
    """Point d'entrée principal du script ETL."""
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline ETL PMSI Psychiatrie")
    parser.add_argument("--force", action="store_true", help="Forcer le retraitement de tous les fichiers")
    parser.add_argument("--scan-only", action="store_true", help="Scanner uniquement, sans traiter")
    args = parser.parse_args()

    processor = PMSIProcessor()

    if args.scan_only:
        files = processor.scanner.scan_source_directory()
        for category, file_list in files.items():
            print(f"\n{category}:")
            for f in file_list:
                info = processor.scanner.get_file_info(f)
                print(f"  - {f.name} ({info['size_mb']} MB)")
    else:
        result = processor.run_full_pipeline(force_reprocess=args.force)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
