#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Current Python version:"
python --version

# Install dependencies compatible with current Python
pip install --upgrade pip

# For Python 3.13, we need to use newer versions
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install python-multipart==0.0.6

# Install PyTorch with CPU support (latest version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install opencv-python-headless==4.8.1.78
pip install ultralytics
pip install numpy
pip install pillow
pip install pyyaml==6.0.1
pip install scikit-learn
pip install requests==2.31.0

# Create models directory
mkdir -p models

echo "Build complete!"
echo "Python version used:"
python --version