@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo ╔══════════════════════════════════════════════════════════════════╗
echo ║     DIM - INSTALLATION ET ENTRAINEMENT AUTOMATIQUE               ║
echo ║     Data Intelligence Medicale - Fine-tuning LLM                 ║
echo ╚══════════════════════════════════════════════════════════════════╝
echo.

REM ============================================================
REM ETAPE 1: VERIFICATION PYTHON
REM ============================================================
echo [1/6] Verification de Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERREUR] Python n'est pas installe!
    echo.
    echo Telechargez Python 3.10+ depuis: https://www.python.org/downloads/
    echo Cochez "Add Python to PATH" lors de l'installation
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo        Python %PYVER% detecte

REM ============================================================
REM ETAPE 2: INSTALLATION DES DEPENDANCES DE BASE
REM ============================================================
echo.
echo [2/6] Installation des dependances de base...

REM Verifier pip
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo        Installation de pip...
    python -m ensurepip --upgrade
)

REM Installer les packages essentiels
echo        Installation des packages (cela peut prendre quelques minutes)...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet polars psutil

REM ============================================================
REM ETAPE 3: DETECTION DU MATERIEL
REM ============================================================
echo.
echo [3/6] Detection automatique du materiel...
echo.

REM Creer un script Python temporaire pour la detection
python -c "
import sys
import os

# Detection GPU
gpu_detected = False
gpu_name = 'Aucun'
vram_gb = 0

try:
    import torch
    if torch.cuda.is_available():
        gpu_detected = True
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
except:
    pass

# Si pas de PyTorch, essayer nvidia-smi
if not gpu_detected:
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 2:
                gpu_detected = True
                gpu_name = parts[0].strip()
                vram_gb = float(parts[1].strip()) / 1024
    except:
        pass

# Detection CPU/RAM
import platform
try:
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024**3)
except:
    ram_gb = 8

cpu_count = os.cpu_count() or 4

# Affichage
print('=' * 60)
print('  MATERIEL DETECTE')
print('=' * 60)
print(f'  CPU      : {cpu_count} coeurs')
print(f'  RAM      : {ram_gb:.1f} GB')
if gpu_detected:
    print(f'  GPU      : {gpu_name}')
    print(f'  VRAM     : {vram_gb:.1f} GB')
else:
    print(f'  GPU      : Non detecte (mode CPU)')
print('=' * 60)

# Ecrire la config pour le batch
with open('_gpu_detected.tmp', 'w') as f:
    f.write('1' if gpu_detected else '0')
with open('_vram_gb.tmp', 'w') as f:
    f.write(str(vram_gb))
"

REM Lire les resultats
set GPU_DETECTED=0
set VRAM_GB=0
if exist _gpu_detected.tmp (
    set /p GPU_DETECTED=<_gpu_detected.tmp
    del _gpu_detected.tmp
)
if exist _vram_gb.tmp (
    set /p VRAM_GB=<_vram_gb.tmp
    del _vram_gb.tmp
)

REM ============================================================
REM ETAPE 4: INSTALLATION PYTORCH (CPU ou CUDA)
REM ============================================================
echo.
echo [4/6] Installation de PyTorch...

if "%GPU_DETECTED%"=="1" (
    echo        GPU detecte - Installation PyTorch CUDA...
    python -m pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo        Erreur CUDA, tentative avec cu118...
        python -m pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    )
) else (
    echo        Pas de GPU - Installation PyTorch CPU...
    python -m pip install --quiet torch torchvision torchaudio
)

REM Verifier l'installation
python -c "import torch; print(f'        PyTorch {torch.__version__} installe')"
python -c "import torch; cuda=torch.cuda.is_available(); print(f'        CUDA disponible: {cuda}')"

REM ============================================================
REM ETAPE 5: INSTALLATION DES DEPENDANCES ML
REM ============================================================
echo.
echo [5/6] Installation des bibliotheques ML...

python -m pip install --quiet transformers>=4.36.0 datasets>=2.16.0 accelerate>=0.25.0
python -m pip install --quiet peft>=0.7.0 trl>=0.7.0

REM BitsAndBytes (seulement si GPU)
if "%GPU_DETECTED%"=="1" (
    echo        Installation bitsandbytes pour quantification 4-bit...
    python -m pip install --quiet bitsandbytes>=0.42.0
)

echo        Verification des installations...
python -c "import transformers; print(f'        Transformers {transformers.__version__}')"
python -c "import peft; print(f'        PEFT {peft.__version__}')"
python -c "import trl; print(f'        TRL {trl.__version__}')"

REM ============================================================
REM ETAPE 6: DETECTION ET CONFIGURATION OPTIMALE
REM ============================================================
echo.
echo [6/6] Configuration optimale...

REM Executer la detection complete
if exist hardware_detector.py (
    python hardware_detector.py --save training_config.json
) else (
    echo        hardware_detector.py non trouve, utilisation des valeurs par defaut
)

REM ============================================================
REM VERIFICATION DU DATASET
REM ============================================================
echo.
echo ════════════════════════════════════════════════════════════════════
echo   VERIFICATION DU DATASET
echo ════════════════════════════════════════════════════════════════════

if exist "output\train_dataset.jsonl" (
    echo   Dataset trouve: output\train_dataset.jsonl
    for %%A in ("output\train_dataset.jsonl") do echo   Taille: %%~zA bytes
) else (
    echo   [!] Dataset non trouve!
    echo.
    echo   Pour creer le dataset:
    echo     1. Lancez l'interface web: python app.py
    echo     2. Deposez vos fichiers PMSI
    echo     3. Lancez le traitement ETL
    echo   Ou:
    echo     python setup_llm.py --create-data
    echo.
    set /p CREATE_DATA="Voulez-vous creer un dataset de demonstration? (O/N): "
    if /i "!CREATE_DATA!"=="O" (
        if exist setup_llm.py (
            python setup_llm.py --create-data
        ) else (
            echo   setup_llm.py non trouve
        )
    )
)

REM ============================================================
REM LANCEMENT DE L'ENTRAINEMENT
REM ============================================================
echo.
echo ════════════════════════════════════════════════════════════════════
echo   PRET POUR L'ENTRAINEMENT
echo ════════════════════════════════════════════════════════════════════
echo.

if exist training_config.json (
    echo   Configuration chargee depuis training_config.json
    type training_config.json | findstr /C:"model_id" /C:"batch_size" /C:"profile_name"
)

echo.
echo   L'entrainement va utiliser la configuration optimale detectee.
echo   Duree estimee: 15-60 minutes selon le materiel et les donnees.
echo.

set /p CONFIRM="Lancer l'entrainement maintenant? (O/N): "
if /i not "%CONFIRM%"=="O" (
    echo.
    echo   Entrainement annule.
    echo   Pour lancer plus tard: python finetune_lora.py --auto-config
    echo.
    pause
    exit /b 0
)

echo.
echo ════════════════════════════════════════════════════════════════════
echo   LANCEMENT DU FINE-TUNING
echo ════════════════════════════════════════════════════════════════════
echo.

python finetune_lora.py --auto-config

echo.
if errorlevel 1 (
    echo ╔══════════════════════════════════════════════════════════════════╗
    echo ║                                                                  ║
    echo ║   ██╗  ██╗    ███████╗██████╗ ██████╗ ███████╗██╗   ██╗██████╗   ║
    echo ║   ╚██╗██╔╝    ██╔════╝██╔══██╗██╔══██╗██╔════╝██║   ██║██╔══██╗  ║
    echo ║    ╚███╔╝     █████╗  ██████╔╝██████╔╝█████╗  ██║   ██║██████╔╝  ║
    echo ║    ██╔██╗     ██╔══╝  ██╔══██╗██╔══██╗██╔══╝  ██║   ██║██╔══██╗  ║
    echo ║   ██╔╝ ██╗    ███████╗██║  ██║██║  ██║███████╗╚██████╔╝██║  ██║  ║
    echo ║   ╚═╝  ╚═╝    ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝  ║
    echo ║                                                                  ║
    echo ║                  ENTRAINEMENT ECHOUE                             ║
    echo ╚══════════════════════════════════════════════════════════════════╝
    echo.
    echo   Causes possibles:
    echo     - Memoire GPU insuffisante ^(reduisez batch_size^)
    echo     - Dataset manquant ou invalide
    echo     - Dependances manquantes
    echo.
    echo   Pour reduire l'utilisation memoire, editez training_config.json:
    echo     - batch_size: 1
    echo     - max_seq_length: 128
    echo.
) else (
    echo.
    echo ╔══════════════════════════════════════════════════════════════════╗
    echo ║                                                                  ║
    echo ║   ███████╗██╗   ██╗ ██████╗ ██████╗███████╗███████╗██╗██╗        ║
    echo ║   ██╔════╝██║   ██║██╔════╝██╔════╝██╔════╝██╔════╝██║██║        ║
    echo ║   ███████╗██║   ██║██║     ██║     █████╗  ███████╗██║██║        ║
    echo ║   ╚════██║██║   ██║██║     ██║     ██╔══╝  ╚════██║╚═╝╚═╝        ║
    echo ║   ███████║╚██████╔╝╚██████╗╚██████╗███████╗███████║██╗██╗        ║
    echo ║   ╚══════╝ ╚═════╝  ╚═════╝ ╚═════╝╚══════╝╚══════╝╚═╝╚═╝        ║
    echo ║                                                                  ║
    echo ║          ENTRAINEMENT TERMINE AVEC SUCCES !                      ║
    echo ║                                                                  ║
    echo ╠══════════════════════════════════════════════════════════════════╣
    echo ║                                                                  ║
    echo ║   Adaptateurs LoRA sauvegardes dans:                             ║
    echo ║     models\adapters\final_adapter\                               ║
    echo ║                                                                  ║
    echo ║   Pour tester le modele:                                         ║
    echo ║     python finetune_lora.py --predict "Patient: P001"            ║
    echo ║                                                                  ║
    echo ║   Prochaines etapes:                                             ║
    echo ║     1. Fusionner les adaptateurs avec le modele de base          ║
    echo ║     2. Convertir en GGUF pour Ollama                             ║
    echo ║     3. Creer un Modelfile personnalise                           ║
    echo ║                                                                  ║
    echo ╚══════════════════════════════════════════════════════════════════╝
    echo.
)
pause
