@echo off
setlocal EnableExtensions EnableDelayedExpansion
title ImageProcessing - Handwritten Digit Recognition

REM ─────────────────────────────────────────────────────────────
REM  One-click launcher for the FastAPI + HTML/CSS/JS app.
REM  Double-click this file, or run: run_app.bat
REM ─────────────────────────────────────────────────────────────

set "PROJECT_DIR=D:\3CS\ImageProcessing"
set "HOST=127.0.0.1"
set "PORT=8000"
set "APP_URL=http://%HOST%:%PORT%"
set "VENV_PY=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "MODEL=%PROJECT_DIR%\models\mnist_model.keras"

echo.
echo ============================================================
echo   Multi-Digit Handwritten Number Recognition
echo ============================================================
echo.

REM 1) Move to the project directory.
cd /d "%PROJECT_DIR%" 2>nul
if errorlevel 1 (
    echo [ERROR] Cannot change directory to:
    echo         %PROJECT_DIR%
    echo         Please check that the project folder exists.
    goto :fail
)

REM 2) Check the virtual environment.
if not exist "%VENV_PY%" (
    echo [ERROR] Python virtual environment not found:
    echo         %VENV_PY%
    echo.
    echo Create it and install dependencies first:
    echo         py -3.14 -m venv .venv
    echo         .venv\Scripts\python.exe -m pip install -r requirements.txt
    goto :fail
)

REM 3) Check the trained model.
if not exist "%MODEL%" (
    echo [ERROR] Model file not found:
    echo         %MODEL%
    echo.
    echo Train the model first:
    echo         .venv\Scripts\python.exe train_model.py
    goto :fail
)
for %%A in ("%MODEL%") do if %%~zA EQU 0 (
    echo [ERROR] Model file is empty:
    echo         %MODEL%
    echo.
    echo Train the model again:
    echo         .venv\Scripts\python.exe train_model.py
    goto :fail
)

REM 4) Check required Python packages.
echo [INFO] Checking dependencies...
"%VENV_PY%" -c "import fastapi, uvicorn, tensorflow, cv2, numpy, pandas, PIL, python_multipart" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Required Python packages are missing or broken.
    echo.
    echo Install them with:
    echo         .venv\Scripts\python.exe -m pip install -r requirements.txt
    goto :fail
)

REM 5) If the app is already running, do not start a second server
REM    and do not open another browser tab.
call :is_app_ready
if not errorlevel 1 (
    echo [INFO] The application is already running at:
    echo        %APP_URL%
    echo.
    echo [INFO] No duplicate server was started and no new browser tab was opened.
    echo        Open the URL above manually if needed.
    goto :success
)

REM 6) Start Uvicorn in its own console window. The /k switch keeps
REM    the window open if startup fails, so errors remain readable.
echo [INFO] Starting FastAPI server at %APP_URL% ...
start "ImageProcessing Server" /D "%PROJECT_DIR%" "%COMSPEC%" /k ""%VENV_PY%" -m uvicorn backend.main:app --host %HOST% --port %PORT%"

REM 7) Wait for the /health endpoint instead of relying on a fixed delay.
echo [INFO] Waiting for the API to become ready...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$url='%APP_URL%'; $deadline=(Get-Date).AddSeconds(120); while ((Get-Date) -lt $deadline) { try { $r = Invoke-RestMethod -Uri ($url + '/health') -TimeoutSec 2; if ($r.status -eq 'ok' -and $null -ne $r.model_loaded) { exit 0 } } catch { }; Start-Sleep -Milliseconds 700 }; exit 1"

if errorlevel 1 (
    echo.
    echo [ERROR] The API did not become ready within 120 seconds.
    echo         Check the "ImageProcessing Server" window for the error.
    echo         Common causes: port %PORT% is in use, or the model failed to load.
    goto :fail
)

REM 8) API is ready: open the frontend in the default browser.
echo [INFO] API is ready.
if not defined IMAGEPROCESSING_NO_BROWSER (
    start "" "%APP_URL%"
) else (
    echo [INFO] Browser opening skipped because IMAGEPROCESSING_NO_BROWSER is set.
)

echo.
echo [INFO] Application is running at %APP_URL%
echo [INFO] The server continues in the "ImageProcessing Server" window.
echo        Close that window to stop the application.

:success
if not defined IMAGEPROCESSING_NO_PAUSE pause >nul
exit /b 0

:fail
echo.
echo [ERROR] Startup failed. The message above explains what to fix.
if not defined IMAGEPROCESSING_NO_PAUSE pause >nul
exit /b 1

REM ─────────────────────────────────────────────────────────────
REM  Returns 0 when /health says the app and model are ready.
REM ─────────────────────────────────────────────────────────────
:is_app_ready
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-RestMethod -Uri '%APP_URL%/health' -TimeoutSec 3; if ($r.status -eq 'ok' -and $null -ne $r.model_loaded) { exit 0 } else { exit 1 } } catch { exit 1 }"
exit /b %errorlevel%
