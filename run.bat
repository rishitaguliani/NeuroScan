@echo off
REM ═══════════════════════════════════════════════════════════
REM   NeuroScan Production Launcher - Windows
REM   One-click startup with comprehensive error handling
REM ═══════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

REM Change to script directory
cd /d "%~dp0"

cls
echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║         NEUROSCAN - ALZHEIMER'S MRI CLASSIFICATION         ║
echo ║              Production Launcher v2.0                       ║
echo ║           EfficientNet-B2 ^• 99%^+ Accuracy                   ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM Step 1: Check Python installation
echo [▶] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [✗] Python is not installed. Please install Python 3.9 or higher.
    echo.
    echo For support, please check the README.md file
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [✓] Python %PYTHON_VERSION% found
echo.

REM Step 2: Check if virtual environment exists
if not exist "venv\" (
    echo [▶] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [✗] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [✓] Virtual environment created
    echo.
)

REM Step 3: Activate virtual environment
echo [▶] Activating virtual environment...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [✓] Virtual environment activated
) else (
    echo [✗] Virtual environment not found. Please delete 'venv' folder and try again.
    pause
    exit /b 1
)
echo.

REM Step 4: Upgrade pip
echo [▶] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo [✓] pip upgraded
echo.

REM Step 5: Install dependencies
echo [▶] Checking dependencies...
if exist "requirements.txt" (
    pip install -q -r requirements.txt
    if %errorlevel% neq 0 (
        echo [✗] Failed to install dependencies
        pause
        exit /b 1
    )
    echo [✓] Dependencies installed
) else (
    echo [✗] requirements.txt not found!
    pause
    exit /b 1
)
echo.

REM Step 6: Check if model exists
echo [▶] Checking model file...
if not exist "model\best_model_b2.pth" (
    echo [✗] Model file not found at model\best_model_b2.pth
    pause
    exit /b 1
)
echo [✓] Model found
echo.

REM Step 7: Check if port 8000 is available
echo [▶] Checking port 8000 availability...
netstat -ano | findstr :8000 | findstr LISTENING >nul 2>&1
if %errorlevel% equ 0 (
    echo [!] Port 8000 is already in use
    set /p KILL_PROCESS="Do you want to kill the existing process and continue? (y/n): "
    if /i "!KILL_PROCESS!"=="y" (
        echo [▶] Killing existing process...
        for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
            taskkill /F /PID %%a >nul 2>&1
        )
        timeout /t 2 /nobreak >nul
        echo [✓] Process killed
    ) else (
        echo [✗] Port 8000 is occupied. Please free it and try again.
        pause
        exit /b 1
    )
)
echo.

REM Step 8: Detect device
echo [▶] Detecting acceleration device...
python -c "import torch; print(torch.backends.mps.is_available())" >nul 2>&1
if %errorlevel% equ 0 (
    for /f %%i in ('python -c "import torch; print(torch.backends.mps.is_available())"') do set MPS_CHECK=%%i
)
python -c "import torch; print(torch.cuda.is_available())" >nul 2>&1
if %errorlevel% equ 0 (
    for /f %%i in ('python -c "import torch; print(torch.cuda.is_available())"') do set CUDA_CHECK=%%i
)

set DEVICE_INFO=CPU
if "%MPS_CHECK%"=="True" (
    set DEVICE_INFO=MPS (Apple Silicon GPU)
    echo [✓] MPS acceleration detected - using Apple Silicon GPU
) else if "%CUDA_CHECK%"=="True" (
    set DEVICE_INFO=CUDA (NVIDIA GPU)
    echo [✓] CUDA acceleration detected - using NVIDIA GPU
) else (
    echo [!] No GPU detected - using CPU (slower performance)
)
echo.

REM Step 9: Start server
echo [▶] Starting NeuroScan server...
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   Server will start at: http://localhost:8000
echo   Device: %DEVICE_INFO%
echo   Press Ctrl+C to stop the server
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

REM Function to open browser
start /b cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8000"

REM Start server
python simple_api.py

if %errorlevel% neq 0 (
    echo.
    echo [✗] Server failed to start. Check error messages above.
    pause
    exit /b 1
)

pause