#!/usr/bin/env bash
# One-step setup for Mac / Linux: creates venv, installs packages, trains the model, starts the web app
set -e
cd "$(dirname "$0")"
if [ ! -d venv ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi
source venv/bin/activate
echo "Installing packages..."
pip install -r requirements.txt
echo
echo "Training the model..."
python src/train.py
echo
echo "Starting the web app - open http://127.0.0.1:5000 in your browser"
python app.py
