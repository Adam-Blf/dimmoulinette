@echo off
chcp 65001 >nul
echo ============================================================
echo    DIM - Fine-tuning LLM avec Detection Automatique
echo ============================================================
echo.

REM Verifier Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe ou pas dans le PATH
    pause
    exit /b 1
)

REM Detection automatique du materiel
echo [1/4] Detection automatique du materiel...
echo.
python hardware_detector.py
if errorlevel 1 (
    echo.
    echo [ATTENTION] Detection impossible, utilisation des valeurs par defaut
)

echo.
echo ============================================================
echo.

REM Verifier les dependances
echo [2/4] Verification des dependances...
python -c "import transformers, peft, datasets, trl" 2>&1
if errorlevel 1 (
    echo Installation des dependances manquantes...
    pip install transformers peft datasets trl accelerate
)

REM Verifier CUDA si disponible
python -c "import torch; cuda=torch.cuda.is_available(); print('CUDA:',cuda)" 2>&1
python -c "import torch; cuda=torch.cuda.is_available(); exit(0 if cuda else 1)" 2>&1
if errorlevel 1 (
    echo.
    echo [MODE CPU] Pas de GPU CUDA detecte - entrainement en mode CPU
    echo            L'entrainement sera plus lent mais fonctionnel
    echo.
) else (
    REM Verifier bitsandbytes pour GPU
    python -c "import bitsandbytes" 2>&1
    if errorlevel 1 (
        echo Installation de bitsandbytes pour quantification GPU...
        pip install bitsandbytes
    )
)

REM Creer le dataset si necessaire
echo.
echo [3/4] Verification du dataset...
if not exist "output\train_dataset.jsonl" (
    if exist "setup_llm.py" (
        echo Creation du dataset d'entrainement...
        python setup_llm.py --create-data
    ) else (
        echo.
        echo [ATTENTION] Dataset non trouve: output\train_dataset.jsonl
        echo Creez-le d'abord via l'interface web ou setup_llm.py
        echo.
    )
)

REM Confirmation
echo.
echo [4/4] Pret pour l'entrainement
echo.
echo La configuration optimale a ete detectee automatiquement.
echo Consultez training_config.json pour les details.
echo.
set /p confirm="Lancer l'entrainement avec ces parametres? (O/N): "
if /i not "%confirm%"=="O" (
    echo Annule.
    pause
    exit /b 0
)

REM Lancer l'entrainement avec auto-config
echo.
echo ============================================================
echo Lancement du fine-tuning avec configuration automatique...
echo ============================================================
echo.
python finetune_lora.py --auto-config

echo.
echo ============================================================
if errorlevel 1 (
    echo [ERREUR] L'entrainement a echoue
    echo.
    echo Causes possibles:
    echo   - Memoire insuffisante (reduisez batch_size dans training_config.json)
    echo   - Dataset manquant
    echo   - Dependances manquantes
) else (
    echo [OK] Entrainement termine avec succes!
    echo.
    echo Les adaptateurs sont sauvegardes dans: models\adapters\final_adapter\
    echo.
    echo Pour tester le modele:
    echo   python finetune_lora.py --predict "Patient: P001, UM: PSY-AMB, Duree: 5 jours"
)
echo ============================================================
pause
