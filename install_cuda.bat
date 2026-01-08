@echo off
echo ============================================================
echo    Installation PyTorch + CUDA pour RTX 3050
echo ============================================================
echo.

REM Desinstaller l'ancienne version
echo [1/3] Desinstallation de PyTorch CPU...
pip uninstall torch torchvision torchaudio -y

REM Installer PyTorch CUDA
echo.
echo [2/3] Installation de PyTorch avec CUDA 12.1...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Installer les dependances de fine-tuning
echo.
echo [3/3] Installation des dependances de fine-tuning...
pip install transformers datasets peft trl bitsandbytes accelerate
pip install polars loguru

REM Verification
echo.
echo ============================================================
echo Verification de l'installation:
echo ============================================================
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo ============================================================
if errorlevel 1 (
    echo [ERREUR] L'installation a echoue
    echo.
    echo Assurez-vous que:
    echo   1. Les drivers NVIDIA sont installes
    echo   2. CUDA Toolkit est installe (https://developer.nvidia.com/cuda-downloads)
) else (
    echo [OK] Installation terminee!
    echo.
    echo Lancez maintenant: train_rtx3050.bat
)
echo ============================================================
pause
