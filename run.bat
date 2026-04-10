@echo off
echo Starting AI Medical Diagnosis...
echo.

if not exist venv (
    echo [ERROR] Virtual environment not found!
    pause
    exit /b
)

if not exist medical_ai_model.h5 (
    echo [WARNING] Trained model medical_ai_model.h5 not found!
    echo Please run the training manually using: python train_model.py
    pause
)

echo.
echo Launching Web Interface...
.\venv\Scripts\python.exe app.py
pause
 