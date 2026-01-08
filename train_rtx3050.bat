@echo off
echo ============================================================
echo    DIM - Fine-tuning LLM sur RTX 3050
echo ============================================================
echo.

REM Verifier Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe ou pas dans le PATH
    pause
    exit /b 1
)

REM Verifier CUDA
echo [1/5] Verification du GPU...
python -c "import torch; assert torch.cuda.is_available(), 'CUDA non disponible'; print('GPU:', torch.cuda.get_device_name(0)); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')" 2>&1
if errorlevel 1 (
    echo.
    echo [ERREUR] CUDA non disponible!
    echo.
    echo Installez PyTorch avec CUDA:
    echo   pip uninstall torch
    echo   pip install torch --index-url https://download.pytorch.org/whl/cu121
    echo.
    pause
    exit /b 1
)

REM Verifier les dependances
echo.
echo [2/5] Verification des dependances...
python -c "import transformers, peft, bitsandbytes, trl, datasets" 2>&1
if errorlevel 1 (
    echo Installation des dependances manquantes...
    pip install transformers peft bitsandbytes trl datasets accelerate
)

REM Creer le dataset si necessaire
echo.
echo [3/5] Verification du dataset...
if not exist "output\train_dataset.jsonl" (
    echo Creation du dataset d'entrainement...
    python setup_llm.py --create-data
)

REM Afficher la config
echo.
echo [4/5] Configuration:
echo   - Modele: microsoft/phi-2 (2.7B params)
echo   - Batch size: 1
echo   - Gradient accumulation: 8
echo   - Sequence length: 256
echo   - Epochs: 3
echo   - LoRA rank: 8
echo.
echo Utilisation VRAM estimee: ~3-4 GB
echo Duree estimee: 15-30 minutes (selon dataset)
echo.

REM Confirmation
set /p confirm="Lancer l'entrainement? (O/N): "
if /i not "%confirm%"=="O" (
    echo Annule.
    pause
    exit /b 0
)

REM Lancer l'entrainement
echo.
echo [5/5] Lancement du fine-tuning...
echo ============================================================
python finetune_lora.py

echo.
echo ============================================================
if errorlevel 1 (
    echo [ERREUR] L'entrainement a echoue
) else (
    echo [OK] Entrainement termine!
    echo.
    echo Les adaptateurs sont dans: models\adapters\final_adapter\
    echo.
    echo Pour tester:
    echo   python finetune_lora.py --predict "Patient: P001, UM: PSY-AMB, Duree: 5 jours"
)
echo ============================================================
pause
