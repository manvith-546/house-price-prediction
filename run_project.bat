@echo off
REM One-click setup for Windows: creates venv, installs packages, trains model, starts the web app
cd /d "%~dp0"
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat
echo Installing packages...
pip install -r requirements.txt
echo.
echo Training the model...
python src\train.py
if errorlevel 1 (
    echo Training failed. See the error above.
    pause
    exit /b 1
)
echo.
echo Starting the web app - open http://127.0.0.1:5000 in your browser
python app.py
pause
