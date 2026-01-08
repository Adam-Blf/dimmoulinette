"""
===============================================================================
APP.PY - Interface Web FastAPI avec IA Integree
===============================================================================
DIM - Data Intelligence Medicale
Interface utilisateur pour les moulinettes PMSI + IA locale
===============================================================================
"""

import asyncio
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import polars as pl
from loguru import logger


# =============================================================================
# CONFIGURATION
# =============================================================================

class AppConfig:
    """Configuration de l'application."""
    SOURCE_DIR = Path(r"C:\Users\adamb\Downloads\frer")
    OUTPUT_DIR = Path("./output")
    TEMPLATES_DIR = Path("./templates")
    MODELS_DIR = Path("./models")
    CONFIGS_DIR = Path("./configs")
    UPLOAD_DIR = Path("./uploads")

    # Securite
    MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
    ALLOWED_EXTENSIONS = {'.txt', '.csv', '.tsv', '.xlsx', '.xls'}


# =============================================================================
# ETAT GLOBAL
# =============================================================================

class LogBuffer:
    """Buffer pour les logs en temps reel."""

    def __init__(self, max_lines: int = 500):
        self.logs: List[str] = []
        self.max_lines = max_lines

    def add(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        if len(self.logs) > self.max_lines:
            self.logs = self.logs[-self.max_lines:]

    def get_all(self) -> List[str]:
        return self.logs.copy()

    def clear(self):
        self.logs = []


class AppState:
    """Etat partage de l'application."""

    def __init__(self):
        self.processing_status: Dict[str, Any] = {}
        self.last_results: Dict[str, Any] = {}
        self.etl_running: bool = False
        self.training_running: bool = False
        self.episodes_running: bool = False
        self.ght_running: bool = False
        self.ai_manager = None


log_buffer = LogBuffer()
app_state = AppState()


# =============================================================================
# MODELES PYDANTIC
# =============================================================================

class ProcessRequest(BaseModel):
    """Requete de traitement de fichiers."""
    files: List[str] = []
    force_reprocess: bool = False


class EpisodeRequest(BaseModel):
    """Requete de generation d'episodes."""
    input_file: str
    seuil_duree: int = 1
    max_gap: int = 1


class GHTRequest(BaseModel):
    """Requete de visualisation GHT."""
    excel_file: str
    output_format: str = "png"


class TrainingRequest(BaseModel):
    """Requete de fine-tuning."""
    data_path: str = "./output/train_dataset.jsonl"
    epochs: int = 3


class AIModelRequest(BaseModel):
    """Requete de modele IA."""
    model: str = "mistral:7b-instruct"


class AIAnalyzeRequest(BaseModel):
    """Requete d'analyse IA."""
    file: str


class TrainRequest(BaseModel):
    """Requete d'entrainement."""
    epochs: int = 3
    data_file: str = None
    create_ollama: bool = True


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    # Startup
    logger.info("=" * 60)
    logger.info("DIM - Data Intelligence Medicale")
    logger.info("Demarrage de l'interface web...")
    logger.info("=" * 60)

    # Creation des dossiers
    AppConfig.OUTPUT_DIR.mkdir(exist_ok=True)
    AppConfig.TEMPLATES_DIR.mkdir(exist_ok=True)
    AppConfig.MODELS_DIR.mkdir(exist_ok=True)
    AppConfig.CONFIGS_DIR.mkdir(exist_ok=True)
    AppConfig.UPLOAD_DIR.mkdir(exist_ok=True)

    log_buffer.add("Application demarree")

    # Initialisation IA (lazy)
    try:
        from ai_manager import AIManager
        app_state.ai_manager = AIManager()
        log_buffer.add("Module IA charge")
    except Exception as e:
        logger.warning(f"Module IA non disponible: {e}")

    yield

    # Shutdown
    logger.info("Arret de l'application...")


# =============================================================================
# APPLICATION FASTAPI
# =============================================================================

app = FastAPI(
    title="DIM - Data Intelligence Medicale",
    description="Interface pour le traitement des donnees PMSI avec IA",
    version="2.0.0",
    lifespan=lifespan
)

# Configuration des templates
templates = Jinja2Templates(directory=str(AppConfig.TEMPLATES_DIR))


# =============================================================================
# ROUTES - PAGES HTML
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Page principale du dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})


# =============================================================================
# ROUTES - API IA
# =============================================================================

@app.get("/api/ai/status")
async def ai_status():
    """Retourne le statut de l'IA."""
    if app_state.ai_manager is None:
        return {
            "initialized": False,
            "available": False,
            "backend": None,
            "message": "AI manager not loaded"
        }

    try:
        app_state.ai_manager.initialize()
        status = app_state.ai_manager.get_status()
        return status
    except Exception as e:
        return {
            "initialized": False,
            "available": False,
            "error": str(e)
        }


@app.post("/api/ai/init")
async def init_ai():
    """Initialise et charge le modele IA."""
    if app_state.ai_manager is None:
        return {"status": "error", "message": "AI manager not available"}

    try:
        log_buffer.add("Initialisation de l'IA...")

        if app_state.ai_manager.initialize():
            if app_state.ai_manager.load_model():
                status = app_state.ai_manager.get_status()
                log_buffer.add(f"IA prete: {status.get('current_model', 'N/A')}")
                return {
                    "status": "success",
                    "model": status.get("current_model"),
                    "backend": status.get("backend")
                }
            else:
                return {"status": "error", "message": "Failed to load model"}
        else:
            return {"status": "error", "message": "No AI backend available. Install Ollama."}

    except Exception as e:
        logger.error(f"AI init error: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/ai/download")
async def download_model(request: AIModelRequest):
    """Telecharge un modele IA."""
    if app_state.ai_manager is None:
        return {"status": "error", "message": "AI manager not available"}

    try:
        log_buffer.add(f"Telechargement du modele: {request.model}")

        if app_state.ai_manager.initialize():
            if app_state.ai_manager.load_model(request.model):
                log_buffer.add(f"Modele {request.model} pret")
                return {"status": "success", "message": f"Model {request.model} ready"}
            else:
                return {"status": "error", "message": "Download failed"}
        else:
            return {"status": "error", "message": "No AI backend available"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/ai/training/check")
async def check_training_requirements():
    """Verifie les prerequis pour le fine-tuning."""
    try:
        from ai_manager import QLoRATrainer
        trainer = QLoRATrainer()
        reqs = trainer.check_requirements()
        return reqs
    except Exception as e:
        return {"ready": False, "error": str(e)}


@app.post("/api/ai/create-model")
async def create_ollama_model(background_tasks: BackgroundTasks):
    """Cree le modele Ollama personnalise dim-pmsi."""
    log_buffer.add("Creation du modele Ollama dim-pmsi...")

    async def create_model_task():
        try:
            from setup_llm import LLMSetup, SystemChecker

            # Verifier Ollama
            if not SystemChecker.check_ollama_installed():
                log_buffer.add("Erreur: Ollama n'est pas installe")
                app_state.last_results["create_model"] = {
                    "status": "error",
                    "message": "Ollama n'est pas installe. Telechargez-le sur https://ollama.ai"
                }
                return

            # Demarrer Ollama si necessaire
            if not SystemChecker.check_ollama_running():
                log_buffer.add("Demarrage d'Ollama...")
                if not SystemChecker.start_ollama():
                    log_buffer.add("Erreur: Impossible de demarrer Ollama")
                    app_state.last_results["create_model"] = {
                        "status": "error",
                        "message": "Impossible de demarrer Ollama"
                    }
                    return

            setup = LLMSetup()

            # Verifier si le modele de base est present
            available = SystemChecker.get_available_models()
            base_model = "mistral:7b-instruct"

            if not any("mistral" in m for m in available):
                log_buffer.add(f"Telechargement du modele de base: {base_model}")
                log_buffer.add("(Cela peut prendre plusieurs minutes...)")

                if not setup.manager.pull_base_model(base_model):
                    log_buffer.add("Erreur: Echec du telechargement du modele")
                    app_state.last_results["create_model"] = {
                        "status": "error",
                        "message": "Echec du telechargement du modele de base"
                    }
                    return

            # Creer le modele personnalise
            log_buffer.add("Creation du modele dim-pmsi...")

            if setup.manager.create_custom_model():
                log_buffer.add("Modele dim-pmsi cree avec succes!")
                app_state.last_results["create_model"] = {
                    "status": "success",
                    "model": "dim-pmsi",
                    "message": "Modele cree avec succes"
                }

                # Mettre a jour le modele dans ai_manager
                if app_state.ai_manager:
                    app_state.ai_manager.config.ollama_model = "dim-pmsi"
            else:
                log_buffer.add("Erreur: Echec de la creation du modele")
                app_state.last_results["create_model"] = {
                    "status": "error",
                    "message": "Echec de la creation du modele"
                }

        except Exception as e:
            logger.error(f"Create model error: {e}")
            log_buffer.add(f"Erreur: {str(e)}")
            app_state.last_results["create_model"] = {
                "status": "error",
                "message": str(e)
            }

    background_tasks.add_task(create_model_task)
    return {"status": "started", "message": "Creation du modele en cours..."}


@app.get("/api/ai/models")
async def list_ollama_models():
    """Liste les modeles Ollama disponibles."""
    try:
        from setup_llm import SystemChecker

        if not SystemChecker.check_ollama_running():
            return {"status": "error", "message": "Ollama n'est pas en cours d'execution", "models": []}

        models = SystemChecker.get_available_models()

        # Marquer le modele recommande
        model_list = []
        for m in models:
            model_list.append({
                "name": m,
                "is_dim": "dim-pmsi" in m.lower(),
                "recommended": "dim-pmsi" in m.lower() or "mistral" in m.lower()
            })

        return {
            "status": "success",
            "models": model_list,
            "has_dim_model": any(m["is_dim"] for m in model_list)
        }

    except Exception as e:
        return {"status": "error", "message": str(e), "models": []}


@app.post("/api/ai/training/start")
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """Lance le fine-tuning du modele."""
    if app_state.training_running:
        return {"status": "error", "message": "Un entrainement est deja en cours"}

    log_buffer.add(f"Demarrage du fine-tuning ({request.epochs} epochs)...")
    background_tasks.add_task(training_task, request.epochs, request.data_file, request.create_ollama)

    return {"status": "started", "message": "Fine-tuning demarre"}


async def training_task(epochs: int, data_file: str, create_ollama: bool):
    """Tache d'entrainement en arriere-plan."""
    app_state.training_running = True

    try:
        from ai_manager import QLoRATrainer, DatasetPreparer
        from psy_logic import EpisodeBuilder, EpisodeConfig
        from etl_processor import UniversalParser

        trainer = QLoRATrainer()

        # Verifier les prerequis
        reqs = trainer.check_requirements()

        if reqs["ready"]:
            # FULL QLoRA training
            log_buffer.add("GPU detecte - Fine-tuning QLoRA complet")

            # Preparer les donnees
            if data_file:
                preparer = DatasetPreparer()
                samples = preparer.load_from_jsonl(Path(data_file))
            else:
                # Chercher un fichier d'episodes
                episodes_files = list(AppConfig.OUTPUT_DIR.glob("*_episodes.csv"))
                if episodes_files:
                    log_buffer.add(f"Utilisation de {episodes_files[0].name}")
                    df = pl.read_csv(episodes_files[0], separator=";")
                    preparer = DatasetPreparer()
                    samples = preparer.prepare_from_episodes(df)
                else:
                    raise ValueError("Aucun fichier d'episodes trouve. Executez d'abord le pipeline ETL.")

            log_buffer.add(f"Dataset: {len(samples)} samples")

            # Entrainement
            if trainer.train(samples, epochs=epochs):
                log_buffer.add("Fine-tuning QLoRA termine!")
                app_state.last_results["training"] = {
                    "status": "completed",
                    "type": "qlora",
                    "epochs": epochs,
                    "samples": len(samples)
                }
            else:
                raise ValueError("Entrainement echoue")

        else:
            # Ollama Modelfile (alternative sans GPU)
            log_buffer.add("Pas de GPU - Creation d'un modele Ollama personnalise")

            if create_ollama and trainer.create_ollama_model("dim-pmsi"):
                log_buffer.add("Modele Ollama 'dim-pmsi' cree!")
                app_state.last_results["training"] = {
                    "status": "completed",
                    "type": "ollama_modelfile",
                    "model": "dim-pmsi"
                }
            else:
                # Juste creer le Modelfile
                modelfile_path = trainer.create_ollama_modelfile()
                log_buffer.add(f"Modelfile cree: {modelfile_path}")
                log_buffer.add("Executez: ollama create dim-pmsi -f ./models/DIM_PMSI_Modelfile")
                app_state.last_results["training"] = {
                    "status": "completed",
                    "type": "modelfile_only",
                    "path": str(modelfile_path)
                }

    except Exception as e:
        logger.error(f"Training error: {e}")
        log_buffer.add(f"Erreur entrainement: {str(e)}")
        app_state.last_results["training"] = {"status": "error", "message": str(e)}

    finally:
        app_state.training_running = False


@app.post("/api/ai/analyze-file")
async def analyze_file_with_ai(request: AIAnalyzeRequest):
    """Analyse un fichier avec l'IA."""
    if app_state.ai_manager is None or not app_state.ai_manager.is_loaded:
        return {"status": "error", "message": "AI not loaded. Initialize first."}

    try:
        filepath = AppConfig.SOURCE_DIR / request.file

        if not filepath.exists():
            return {"status": "error", "message": f"File not found: {request.file}"}

        log_buffer.add(f"Analyse IA de {request.file}...")

        # Parse file
        from etl_processor import UniversalParser
        parser = UniversalParser()
        df = parser.parse_file(filepath)

        if df is None:
            return {"status": "error", "message": "Failed to parse file"}

        # Build episodes if RAA
        from psy_logic import EpisodeBuilder, EpisodeConfig
        config = EpisodeConfig(seuil_duree_anomalie=1)
        builder = EpisodeBuilder(config)
        episodes_df = builder.build_episodes(df)

        # AI analysis (limit to first 20 for speed)
        episodes = []
        anomalies_count = 0

        for i, row in enumerate(episodes_df.head(20).iter_rows(named=True)):
            result = app_state.ai_manager.analyze_episode(row)
            episodes.append({
                "episode_id": row.get("EPISODE_ID", f"EP_{i}"),
                "duree": row.get("DUREE_EPISODE_JOURS", 0),
                "analysis": result.get("analysis", ""),
                "is_anomaly": result.get("is_anomaly", False)
            })
            if result.get("is_anomaly"):
                anomalies_count += 1

        log_buffer.add(f"Analyse terminee: {anomalies_count} anomalies sur {len(episodes)}")

        return {
            "status": "success",
            "episodes": episodes,
            "anomalies_count": anomalies_count,
            "total_analyzed": len(episodes),
            "summary": f"Analyse de {len(episodes)} episodes: {anomalies_count} anomalies detectees par l'IA"
        }

    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        return {"status": "error", "message": str(e)}


# =============================================================================
# ROUTES - UPLOAD & ANALYSE INTEGREE
# =============================================================================

@app.post("/api/upload/analyze")
async def upload_and_analyze(file: UploadFile = File(...)):
    """
    Pipeline integre: Upload -> ETL -> Episodes -> IA
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in AppConfig.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type not allowed: {ext}")

    try:
        # 1. Save uploaded file
        upload_path = AppConfig.UPLOAD_DIR / file.filename
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)

        log_buffer.add(f"Fichier recu: {file.filename}")

        # 2. ETL Processing
        from etl_processor import UniversalParser
        parser = UniversalParser()
        df = parser.parse_file(upload_path)

        if df is None:
            return {"status": "error", "message": "Failed to parse file"}

        rows_processed = len(df)
        log_buffer.add(f"ETL: {rows_processed} lignes parsees")

        # 3. Build episodes
        from psy_logic import EpisodeBuilder, EpisodeConfig
        config = EpisodeConfig(seuil_duree_anomalie=1)
        builder = EpisodeBuilder(config)
        episodes_df = builder.build_episodes(df)

        log_buffer.add(f"Episodes: {len(episodes_df)} episodes generes")

        # 4. AI Analysis
        ai_analysis = None

        if app_state.ai_manager and app_state.ai_manager.is_loaded:
            log_buffer.add("Analyse IA en cours...")

            episodes = []
            anomalies_count = 0

            # Analyze first 30 episodes
            for i, row in enumerate(episodes_df.head(30).iter_rows(named=True)):
                result = app_state.ai_manager.analyze_episode(row)
                episodes.append({
                    "episode_id": row.get("EPISODE_ID", f"EP_{i}"),
                    "duree": row.get("DUREE_EPISODE_JOURS", 0),
                    "analysis": result.get("analysis", "")[:200],  # Truncate
                    "is_anomaly": result.get("is_anomaly", False)
                })
                if result.get("is_anomaly"):
                    anomalies_count += 1

            ai_analysis = {
                "episodes": episodes,
                "anomalies_count": anomalies_count,
                "summary": f"{anomalies_count} anomalies detectees sur {len(episodes)} episodes analyses"
            }

            log_buffer.add(f"IA: {anomalies_count} anomalies detectees")

        else:
            # Rule-based analysis only
            anomalies = episodes_df.filter(pl.col("FLAG_ANOMALIE")).to_dicts() if "FLAG_ANOMALIE" in episodes_df.columns else []
            ai_analysis = {
                "episodes": [
                    {
                        "episode_id": a.get("EPISODE_ID", ""),
                        "duree": a.get("DUREE_EPISODE_JOURS", 0),
                        "analysis": "Anomalie detectee par regle metier (duree > seuil)",
                        "is_anomaly": True
                    }
                    for a in anomalies[:30]
                ],
                "anomalies_count": len(anomalies),
                "summary": f"{len(anomalies)} anomalies detectees par regles metier (IA non chargee)"
            }

        # 5. Export results
        output_path = AppConfig.OUTPUT_DIR / f"{Path(file.filename).stem}_analyzed.csv"
        episodes_df.write_csv(output_path, separator=";")

        return {
            "status": "success",
            "filename": file.filename,
            "rows_processed": rows_processed,
            "episodes_generated": len(episodes_df),
            "output_file": str(output_path),
            "ai_analysis": ai_analysis
        }

    except Exception as e:
        logger.error(f"Upload/analyze error: {e}")
        return {"status": "error", "message": str(e)}

    finally:
        # Cleanup upload
        if upload_path.exists():
            try:
                upload_path.unlink()
            except:
                pass


# =============================================================================
# ROUTES - API FICHIERS
# =============================================================================

@app.get("/api/files")
async def list_files():
    """Liste les fichiers disponibles dans le dossier source."""
    if not AppConfig.SOURCE_DIR.exists():
        return {
            "status": "error",
            "message": f"Dossier source non trouve: {AppConfig.SOURCE_DIR}",
            "files": []
        }

    try:
        from etl_processor import UniversalParser

        parser = UniversalParser()
        files = []

        for f in AppConfig.SOURCE_DIR.iterdir():
            if f.is_file():
                file_type = parser.detect_file_type(f)
                files.append({
                    "name": f.name,
                    "path": str(f),
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                    "type": file_type.name,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                })

        # Tri par type puis par nom
        files.sort(key=lambda x: (x["type"], x["name"]))

        # Comptage par categorie
        summary = {}
        for f in files:
            t = f["type"]
            summary[t] = summary.get(t, 0) + 1

        return {
            "status": "success",
            "source_dir": str(AppConfig.SOURCE_DIR),
            "files": files,
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Erreur listage fichiers: {e}")
        return {"status": "error", "message": str(e), "files": []}


@app.get("/api/system")
async def system_info():
    """Informations systeme."""
    gpu_info = {"available": False, "name": "N/A", "memory": "N/A"}

    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "memory": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            }
    except ImportError:
        pass

    return {
        "gpu": gpu_info,
        "etl_running": app_state.etl_running,
        "training_running": app_state.training_running,
        "episodes_running": app_state.episodes_running,
        "ght_running": app_state.ght_running,
        "ai_loaded": app_state.ai_manager.is_loaded if app_state.ai_manager else False,
        "source_dir": str(AppConfig.SOURCE_DIR),
        "output_dir": str(AppConfig.OUTPUT_DIR)
    }


# =============================================================================
# ROUTES - API ETL
# =============================================================================

@app.post("/api/etl/run")
async def run_etl(request: ProcessRequest, background_tasks: BackgroundTasks):
    """Lance le pipeline ETL."""
    if app_state.etl_running:
        return {"status": "error", "message": "Un traitement ETL est deja en cours"}

    log_buffer.add("Demarrage du pipeline ETL...")
    background_tasks.add_task(etl_task, request.files, request.force_reprocess)

    return {"status": "started", "message": "Pipeline ETL demarre"}


async def etl_task(files: List[str], force: bool):
    """Tache ETL en arriere-plan."""
    app_state.etl_running = True

    try:
        from etl_processor import UniversalParser, ETLConfig

        config = ETLConfig(
            source_dir=AppConfig.SOURCE_DIR,
            output_dir=AppConfig.OUTPUT_DIR
        )
        parser = UniversalParser(config)

        results = []

        if files:
            log_buffer.add(f"Traitement de {len(files)} fichiers selectionnes...")
            for filename in files:
                filepath = AppConfig.SOURCE_DIR / filename
                if filepath.exists():
                    df = parser.parse_file(filepath)
                    if df is not None:
                        parser.export_to_csv(df, filepath.stem)
                        results.append({
                            "file": filename,
                            "rows": len(df),
                            "status": "success"
                        })
                        log_buffer.add(f"  OK {filename}: {len(df)} lignes")
        else:
            log_buffer.add("Traitement de tous les fichiers du dossier...")
            all_results = parser.process_directory(priority_order=True)
            for key, df in all_results.items():
                results.append({
                    "file": key,
                    "rows": len(df),
                    "status": "success"
                })

        app_state.last_results["etl"] = {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "results": results,
            "stats": parser.stats
        }
        log_buffer.add(f"ETL termine: {len(results)} fichiers traites")

    except Exception as e:
        logger.error(f"Erreur ETL: {e}")
        log_buffer.add(f"Erreur ETL: {str(e)}")
        app_state.last_results["etl"] = {"status": "error", "message": str(e)}

    finally:
        app_state.etl_running = False


# =============================================================================
# ROUTES - API EPISODES
# =============================================================================

@app.post("/api/episodes/run")
async def run_episodes(request: EpisodeRequest, background_tasks: BackgroundTasks):
    """Lance la generation des episodes."""
    if app_state.episodes_running:
        return {"status": "error", "message": "Une generation d'episodes est deja en cours"}

    filepath = AppConfig.SOURCE_DIR / request.input_file
    if not filepath.exists():
        return {"status": "error", "message": f"Fichier non trouve: {request.input_file}"}

    log_buffer.add(f"Generation des episodes pour {request.input_file}...")
    background_tasks.add_task(episodes_task, filepath, request.seuil_duree, request.max_gap)

    return {"status": "started", "message": "Generation des episodes demarree"}


async def episodes_task(filepath: Path, seuil: int, gap: int):
    """Tache episodes en arriere-plan."""
    app_state.episodes_running = True

    try:
        from etl_processor import UniversalParser
        from psy_logic import EpisodeBuilder, EpisodeConfig, ParcoursAnalyzer

        # Parsing
        parser = UniversalParser()
        df = parser.parse_file(filepath)

        if df is None:
            raise ValueError("Impossible de parser le fichier")

        log_buffer.add(f"  Fichier parse: {len(df)} lignes")

        # Construction des episodes
        config = EpisodeConfig(
            seuil_duree_anomalie=seuil,
            max_gap_days=gap
        )
        builder = EpisodeBuilder(config)
        episodes_df = builder.build_episodes(df)

        # Export
        output_path = AppConfig.OUTPUT_DIR / f"{filepath.stem}_episodes.csv"
        episodes_df.write_csv(output_path, separator=";")
        log_buffer.add(f"  OK Export: {output_path.name}")

        # Analyse
        analyzer = ParcoursAnalyzer()
        stats = analyzer.analyze_parcours(episodes_df)

        # Anomalies
        anomalies = []
        if "FLAG_ANOMALIE" in episodes_df.columns:
            anomalies = episodes_df.filter(pl.col("FLAG_ANOMALIE")).to_dicts()

        app_state.last_results["episodes"] = {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "output_file": str(output_path),
            "stats": stats,
            "anomalies": anomalies[:100],
            "anomalies_count": len(anomalies)
        }
        log_buffer.add(f"Episodes generes: {stats.get('nb_episodes', 'N/A')} episodes, {len(anomalies)} anomalies")

    except Exception as e:
        logger.error(f"Erreur episodes: {e}")
        log_buffer.add(f"Erreur episodes: {str(e)}")
        app_state.last_results["episodes"] = {"status": "error", "message": str(e)}

    finally:
        app_state.episodes_running = False


# =============================================================================
# ROUTES - API GHT
# =============================================================================

@app.post("/api/ght/run")
async def run_ght(request: GHTRequest, background_tasks: BackgroundTasks):
    """Lance la generation du graphe GHT."""
    if app_state.ght_running:
        return {"status": "error", "message": "Une generation de graphe est deja en cours"}

    filepath = Path(request.excel_file)
    if not filepath.exists():
        filepath = AppConfig.SOURCE_DIR / request.excel_file

    if not filepath.exists():
        return {"status": "error", "message": f"Fichier Excel non trouve: {request.excel_file}"}

    log_buffer.add(f"Generation du graphe GHT pour {filepath.name}...")
    background_tasks.add_task(ght_task, filepath, request.output_format)

    return {"status": "started", "message": "Generation du graphe GHT demarree"}


async def ght_task(filepath: Path, output_format: str):
    """Tache GHT en arriere-plan."""
    app_state.ght_running = True

    try:
        from ficom_viz import GHTStructureParser, GHTGraphGenerator, FICOMConfig

        config = FICOMConfig(
            output_dir=AppConfig.OUTPUT_DIR,
            graph_format=output_format
        )

        # Parsing
        parser = GHTStructureParser(config)
        nodes = parser.parse_excel(filepath)

        if not nodes:
            raise ValueError("Aucune structure extraite du fichier")

        log_buffer.add(f"  Structure extraite: {len(nodes)} noeuds")

        # Export PMSI-Pilot
        pmsi_path = parser.export_to_pmsi_pilot()

        # Generation des graphes
        generator = GHTGraphGenerator(config)
        outputs = {"pmsi_pilot": str(pmsi_path)}

        # Graphviz
        gv_path = generator.generate_graphviz(nodes)
        if gv_path:
            outputs["graphviz"] = str(gv_path)
            log_buffer.add(f"  OK Graphe Graphviz: {gv_path.name}")

        # HTML (toujours)
        html_path = generator.generate_html_tree(nodes)
        outputs["html"] = str(html_path)
        log_buffer.add(f"  OK Arbre HTML: {html_path.name}")

        app_state.last_results["ght"] = {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "outputs": outputs,
            "nodes_count": len(nodes),
            "nodes": [{"id": n.id, "name": n.name, "type": n.node_type.value} for n in nodes]
        }
        log_buffer.add(f"Graphe GHT genere: {len(nodes)} noeuds")

    except Exception as e:
        logger.error(f"Erreur GHT: {e}")
        log_buffer.add(f"Erreur GHT: {str(e)}")
        app_state.last_results["ght"] = {"status": "error", "message": str(e)}

    finally:
        app_state.ght_running = False


# =============================================================================
# ROUTES - LOGS
# =============================================================================

@app.get("/api/logs")
async def get_logs():
    """Recupere les logs recents."""
    return {"logs": log_buffer.get_all()}


@app.get("/api/logs/stream")
async def stream_logs():
    """Stream des logs en temps reel (SSE)."""
    async def event_generator():
        last_count = 0
        while True:
            logs = log_buffer.get_all()
            if len(logs) > last_count:
                new_logs = logs[last_count:]
                for log in new_logs:
                    yield f"data: {json.dumps({'log': log})}\n\n"
                last_count = len(logs)
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@app.delete("/api/logs")
async def clear_logs():
    """Efface les logs."""
    log_buffer.clear()
    return {"status": "success", "message": "Logs effaces"}


# =============================================================================
# ROUTES - RESULTATS
# =============================================================================

@app.get("/api/results")
async def get_results():
    """Retourne tous les derniers resultats."""
    return app_state.last_results


@app.get("/api/results/{task_type}")
async def get_task_results(task_type: str):
    """Retourne les resultats d'une tache specifique."""
    if task_type not in app_state.last_results:
        return {"status": "no_data", "message": f"Aucun resultat pour {task_type}"}
    return app_state.last_results[task_type]


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Telecharge un fichier de sortie."""
    filepath = AppConfig.OUTPUT_DIR / filename

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Fichier non trouve")

    return FileResponse(
        filepath,
        filename=filename,
        media_type="application/octet-stream"
    )


# =============================================================================
# ROUTES - AUDIT
# =============================================================================

@app.get("/api/audit")
async def run_audit():
    """Execute un audit de securite."""
    try:
        from setup_env import SecurityAuditor
        auditor = SecurityAuditor()
        results = auditor.run_full_audit(fix=False)
        return results
    except Exception as e:
        return {"status": "error", "message": str(e)}


# =============================================================================
# POINT D'ENTREE
# =============================================================================

def main():
    """Lance le serveur FastAPI."""
    print("""
================================================================
       DIM - DATA INTELLIGENCE MEDICALE
       Moulinettes PMSI + IA Locale - Dashboard
================================================================
  Interface: http://localhost:8080
  Source:    C:\\Users\\adamb\\Downloads\\frer
  Output:    ./output
  IA:        Ollama / llama-cpp-python
================================================================
    """)

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8080,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
