#!/usr/bin/env python
"""
AI Basketball Referee API Server
Run this script to start the API server
"""

import os
import sys
from pathlib import Path


def check_requirements():
    """Check if all required files exist"""
    required_files = [
        "main.py",
        "basketball_referee.py",
        "yolo_loader_fix.py"
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False

    return True


def check_model():
    """Check if model file exists"""
    model_path = r"C:/Users/carlc/Desktop/API  AI REFEREE MODEL/runs/detect/train3/weights/best.pt"

    if not os.path.exists(model_path):
        print(f"⚠️  Model file not found at: {model_path}")
        print("   The API will start but won't be able to process videos.")
        print("   You can train a new model using the /train_model/ endpoint.")
        return False

    print(f"✅ Model file found: {model_path}")
    return True


def main():
    print("=" * 60)
    print("AI BASKETBALL REFEREE API SERVER")
    print("=" * 60)

    # Check requirements
    if not check_requirements():
        print("\n❌ Cannot start server - missing required files")
        sys.exit(1)

    print("✅ All required files present")

    # Check model
    check_model()

    # Start server
    print("\n🚀 Starting API server...")
    print("   URL: http://127.0.0.1:8000")
    print("   Docs: http://127.0.0.1:8000/docs")
    print("   Press Ctrl+C to stop\n")

    # Import and run uvicorn
    try:
        import uvicorn
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    except ImportError:
        print("❌ uvicorn not installed. Run: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()