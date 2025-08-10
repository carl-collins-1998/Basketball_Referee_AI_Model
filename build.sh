#!/usr/bin/env bash
# exit on error
set -o errexit

# Upgrade pip and install setuptools first
python -m pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt

# Create models directory
mkdir -p models

# Download model if URL is provided
if [ -n "$MODEL_URL" ]; then
    echo "Downloading model from $MODEL_URL"
    wget -O models/best.pt "$MODEL_URL" || echo "Model download failed - continuing anyway"
fi

echo "Build complete!"