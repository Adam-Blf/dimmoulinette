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
REM ETAPE 0: VERIFICATION WINGET ET PREREQUIS
REM ============================================================
echo [0/7] Verification des prerequis systeme...

REM Verifier si winget est disponible
winget --version >nul 2>&1
if errorlevel 1 (
    echo        [!] Winget non disponible - installation manuelle requise
    echo            Installez App Installer depuis le Microsoft Store
    set WINGET_OK=0
) else (
    echo        Winget disponible
    set WINGET_OK=1
)

REM ============================================================
REM ETAPE 1: VERIFICATION ET INSTALLATION PYTHON
REM ============================================================
echo.
echo [1/7] Verification de Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo        Python non trouve!
    if "!WINGET_OK!"=="1" (
        echo        Installation de Python via winget...
        winget install Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
        if errorlevel 1 (
            echo        [ERREUR] Echec installation Python
            echo        Installez manuellement: https://www.python.org/downloads/
            pause
            exit /b 1
        )
        echo        Python installe! Redemarrez ce script.
        echo        ^(Le PATH sera mis a jour apres redemarrage du terminal^)
        pause
        exit /b 0
    ) else (
        echo        [ERREUR] Python non installe et winget non disponible
        echo        Installez Python 3.10+ depuis: https://www.python.org/downloads/
        pause
        exit /b 1
    )
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo        Python %PYVER% detecte

REM ============================================================
REM ETAPE 2: VERIFICATION ET INSTALLATION GIT
REM ============================================================
echo.
echo [2/7] Verification de Git...
git --version >nul 2>&1
if errorlevel 1 (
    echo        Git non trouve!
    if "!WINGET_OK!"=="1" (
        echo        Installation de Git via winget...
        winget install Git.Git --silent --accept-package-agreements --accept-source-agreements
        if errorlevel 1 (
            echo        [!] Echec installation Git ^(non bloquant^)
        ) else (
            echo        Git installe!
        )
    ) else (
        echo        [!] Git non installe ^(non bloquant pour l'entrainement^)
    )
) else (
    for /f "tokens=3" %%i in ('git --version 2^>^&1') do set GITVER=%%i
    echo        Git !GITVER! detecte
)

REM ============================================================
REM ETAPE 3: INSTALLATION DES DEPENDANCES DE BASE
REM ============================================================
echo.
echo [3/7] Installation des dependances de base...

REM Verifier pip
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo        Installation de pip...
    python -m ensurepip --upgrade
)

REM Nettoyer les installations corrompues
echo        Nettoyage des installations corrompues...
for /d %%d in ("%LOCALAPPDATA%\Programs\Python\Python*\Lib\site-packages\~*") do rd /s /q "%%d" 2>nul
for /d %%d in ("%LOCALAPPDATA%\Programs\Python\Python*\Lib\site-packages\-*") do rd /s /q "%%d" 2>nul

REM Installer les packages essentiels
echo        Installation des packages (cela peut prendre quelques minutes)...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet polars psutil

REM ============================================================
REM ETAPE 4: DETECTION DU MATERIEL
REM ============================================================
echo.
echo [4/7] Detection automatique du materiel...
echo.

REM Detection GPU via nvidia-smi
set GPU_DETECTED=0
nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    set GPU_DETECTED=1
    echo        GPU NVIDIA detecte
) else (
    echo        Pas de GPU NVIDIA detecte - Mode CPU
)

REM ============================================================
REM ETAPE 5: INSTALLATION PYTORCH (CPU ou CUDA)
REM ============================================================
echo.
echo [5/7] Installation de PyTorch...

if "!GPU_DETECTED!"=="1" (
    echo        GPU detecte - Installation PyTorch CUDA ~2.5GB...
    echo        Cela peut prendre plusieurs minutes, patientez...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo        Erreur CUDA 12.1, tentative avec CUDA 11.8...
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    )
) else (
    echo        Pas de GPU - Installation PyTorch CPU...
    python -m pip install torch torchvision torchaudio
)

REM Verifier l'installation
python -c "import torch; print(f'        PyTorch {torch.__version__} installe')"
python -c "import torch; cuda=torch.cuda.is_available(); print(f'        CUDA disponible: {cuda}')"

REM ============================================================
REM ETAPE 6: INSTALLATION DES DEPENDANCES ML
REM ============================================================
echo.
echo [6/7] Installation des bibliotheques ML...

python -m pip install --quiet "transformers>=4.36.0" "datasets>=2.16.0" "accelerate>=0.25.0"
python -m pip install --quiet "peft>=0.7.0" "trl>=0.7.0"

REM BitsAndBytes (seulement si GPU)
if "!GPU_DETECTED!"=="1" (
    echo        Installation bitsandbytes pour quantification 4-bit...
    python -m pip install --quiet "bitsandbytes>=0.42.0"
)

echo        Verification des installations...
python -c "import transformers; print(f'        Transformers {transformers.__version__}')"
python -c "import peft; print(f'        PEFT {peft.__version__}')"
python -c "import trl; print(f'        TRL {trl.__version__}')"

REM ============================================================
REM ETAPE 7: DETECTION ET CONFIGURATION OPTIMALE
REM ============================================================
echo.
echo [7/7] Configuration optimale...

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
