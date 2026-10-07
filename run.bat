@echo off
title Cocoon - Automated Data Cleaning Engine
color 0B

echo =================================================================
echo             COCOON DATA CLEANING SYSTEM STARTUP
echo =================================================================
echo.

:: Create a project-local virtual environment once.
if not exist "venv\Scripts\python.exe" (
    echo [INFO] Virtual environment not found. Creating one...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Ensure Python is installed.
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created.
)

:: Install requirements only when requirements.txt has changed.
set "REQ_HASH="
for /f "skip=1 tokens=1" %%H in ('certutil -hashfile requirements.txt SHA256 2^>nul') do if not defined REQ_HASH set "REQ_HASH=%%H"
set "OLD_HASH="
if exist "venv\requirements.sha256" set /p OLD_HASH=<"venv\requirements.sha256"

if not defined REQ_HASH (
    echo [WARNING] Could not calculate requirements hash. Verifying packages once...
    venv\Scripts\python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
) else if /I not "%REQ_HASH%"=="%OLD_HASH%" (
    echo [INFO] Installing/updating project requirements...
    venv\Scripts\python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
    >"venv\requirements.sha256" echo %REQ_HASH%
) else (
    echo [INFO] Requirements unchanged. Skipping package installation.
)

echo.
echo =================================================================
echo  Cocoon Server is starting...
echo  Open your web browser and navigate to:
echo
echo  --^>  http://127.0.0.1:8000
echo
echo  Press Ctrl+C in this terminal window to stop the server.
echo =================================================================
echo.

:: Start FastAPI server with Uvicorn
venv\Scripts\python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
if errorlevel 1 (
    echo.
    echo [ERROR] Server crashed or could not start. Ensure port 8000 is free.
    pause
)
