import os
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import shutil
from contextlib import asynccontextmanager
import urllib.request

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

# Import the fix BEFORE importing basketball_referee
try:
    import yolo_loader_fix
except ImportError:
    print("Warning: yolo_loader_fix not found")

import cv2
from basketball_referee import ImprovedFreeThrowScorer, CVATDatasetConverter, FreeThrowModelTrainer

# Global variables
MODEL_URL = os.getenv("MODEL_URL")
scorer_instance = None

def download_model():
    """Download model from URL if not present"""
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return True
    
    if not MODEL_URL:
        print("Model not found locally and MODEL_URL not set")
        print("Please set MODEL_URL environment variable in Render dashboard")
        return False
    
    try:
        print(f"Downloading model from {MODEL_URL}")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully")
        return True
    except Exception as e:
        print(f"Failed to download model: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    This is the modern way to handle application lifecycle in FastAPI.
    """
    # Startup logic
    global scorer_instance

    print("\n" + "=" * 60)
    print("AI BASKETBALL REFEREE API STARTING")
    print("=" * 60)
    print(f"Python file: {__file__}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model exists: {os.path.exists(MODEL_PATH)}")
    
    # Try to download model if on Render
    if "RENDER" in os.environ and not os.path.exists(MODEL_PATH):
        if download_model():
            print("✅ Model downloaded from URL")
        else:
            print("⚠️ Model download failed - API will run without scoring functionality")

	
    if os.path.exists(MODEL_PATH):
        try:
            print("Loading model...")
            scorer_instance = ImprovedFreeThrowScorer(MODEL_PATH)
            print("✅ Model loaded successfully!")
            print(f"Scorer type: {type(scorer_instance)}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Model file not found!")
        print("API will run but scoring functionality will be disabled.")

    print("=" * 60 + "\n")
    
    # This yield is where the application runs
    yield
    
    # Shutdown logic
    print("\n" + "=" * 60)
    print("AI BASKETBALL REFEREE API SHUTTING DOWN")
    print("=" * 60)
    if scorer_instance:
        print("Cleaning up resources...")
        # Add any cleanup code here if needed
    print("Goodbye!")


# Create FastAPI app with lifespan
print("Creating FastAPI app...")
app = FastAPI(
    title="AI Basketball Referee API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with status info."""
    return {
        "message": "AI Basketball Referee API",
        "status": "ready" if scorer_instance is not None else "model not loaded",
        "model_loaded": scorer_instance is not None,
        "endpoints": ["/", "/model_status", "/score_video/", "/train_model/", "/docs"],
        "deployment": "render" if "RENDER" in os.environ else "local",
        "instructions": "Set MODEL_URL environment variable in Render to enable scoring" if "RENDER" in os.environ and not scorer_instance else None
    }

@app.get("/model_status")
async def model_status():
    """Detailed model status."""
    return {
        "loaded": scorer_instance is not None,
        "path": MODEL_PATH,
        "exists": os.path.exists(MODEL_PATH),
        "size_mb": os.path.getsize(MODEL_PATH) / 1024 / 1024 if os.path.exists(MODEL_PATH) else 0,
        "scorer_type": str(type(scorer_instance)) if scorer_instance else None,
        "model_url": "Set" if MODEL_URL else "Not set",
        "environment": "render" if "RENDER" in os.environ else "local"
    }

@app.post("/score_video/")
async def score_video(video_file: UploadFile = File(...)) -> Dict[str, Any]:
    """Analyzes an uploaded video to detect and score free throws."""
    global scorer_instance

    print(f"\n=== Processing video: {video_file.filename} ===")

    if scorer_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please set MODEL_URL environment variable in Render dashboard."
        )

    # Save and process video
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = Path(temp_dir) / video_file.filename

        # Save video
        content = await video_file.read()
        with open(video_path, "wb") as f:
            f.write(content)
        print(f"Video saved: {len(content) / 1024 / 1024:.2f} MB")

        # Reset scorer
        scorer_instance.made_shots = 0
        scorer_instance.missed_shots = 0
        scorer_instance.shot_attempts = 0
        scorer_instance.shot_tracker.reset()

        # Process video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames: {total_frames}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run detection
            detections = scorer_instance.detect_objects(frame)
            hoop_info = scorer_instance.update_hoop_position(detections)
            ball_info = scorer_instance.find_ball(detections)
            player_bboxes = scorer_instance.find_players(detections)

            # Update shot tracking
            old_phase = scorer_instance.shot_tracker.shot_phase
            result = scorer_instance.shot_tracker.update(ball_info, hoop_info, player_bboxes, False)

            # Count attempts
            if old_phase == 'idle' and scorer_instance.shot_tracker.shot_phase == 'rising':
                scorer_instance.shot_attempts += 1
                print(f"Shot attempt #{scorer_instance.shot_attempts} at frame {frame_count}")

            # Count results
            if result == 'score':
                scorer_instance.made_shots += 1
                print(f"SCORE! Total: {scorer_instance.made_shots}")
                scorer_instance.shot_tracker.reset()
            elif result == 'miss':
                scorer_instance.missed_shots += 1
                print(f"MISS! Total: {scorer_instance.missed_shots}")
                scorer_instance.shot_tracker.reset()

            if frame_count % 100 == 0:
                print(f"Progress: {frame_count}/{total_frames} frames")

        cap.release()

        print(f"Processing complete. Frames: {frame_count}")

        return {
            "made_shots": scorer_instance.made_shots,
            "missed_shots": scorer_instance.missed_shots,
            "total_attempts": scorer_instance.shot_attempts,
            "frames_processed": frame_count
        }



@app.post("/train_model/")
async def train_model(
        cvat_zip_files: List[UploadFile] = File(..., description="CVAT YOLO 1.1 annotated datasets as ZIP files"),
        epochs: int = Form(150, description="Number of training epochs"),
        batch_size: int = Form(16, description="Training batch size"),
        model_size: str = Form("s", description="YOLO model size (n, s, m, l)"),
        device: str = Form("auto", description="Device to use for training (cpu, cuda, auto)")
) -> Dict[str, Any]:
    """
    Trains a new basketball referee model using uploaded CVAT annotated datasets.

    Args:
        cvat_zip_files: List of CVAT YOLO 1.1 annotated datasets (ZIP files)
        epochs: Number of training epochs (default: 150)
        batch_size: Training batch size (default: 16)
        model_size: YOLO model size - n/s/m/l (default: s)
        device: Training device - cpu/cuda/auto (default: auto)

    Returns:
        Training results including model performance metrics
    """
    if "RENDER" in os.environ:
        raise HTTPException(
            status_code=503,
            detail="Model training is disabled on Render due to resource limitations. Please train locally and upload the model."
        )
    global scorer_instance, MODEL_PATH

    print(f"\n=== Training New Model ===")
    print(f"Datasets: {len(cvat_zip_files)} files")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Model size: {model_size}, Device: {device}")

    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_cvat_dir, \
            tempfile.TemporaryDirectory() as temp_dataset_dir:

        # Step 1: Save uploaded CVAT files
        uploaded_paths = []
        for i, cvat_file in enumerate(cvat_zip_files):
            cvat_path = Path(temp_cvat_dir) / f"cvat_{i}_{cvat_file.filename}"

            try:
                content = await cvat_file.read()
                with open(cvat_path, "wb") as f:
                    f.write(content)
                uploaded_paths.append(str(cvat_path))
                print(f"Saved CVAT file {i + 1}: {cvat_file.filename} ({len(content) / 1024 / 1024:.2f} MB)")
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save CVAT file {cvat_file.filename}: {e}"
                )

        # Step 2: Convert CVAT to YOLO format
        try:
            print("\nConverting CVAT datasets to YOLO format...")
            converter = CVATDatasetConverter(uploaded_paths, str(temp_dataset_dir))
            converter.convert_multiple_cvat_to_yolo()
            print("✅ Dataset conversion complete")
        except Exception as e:
            print(f"❌ Dataset conversion failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Dataset conversion failed: {e}"
            )

        # Step 3: Train the model
        try:
            print("\nStarting model training...")
            trainer = FreeThrowModelTrainer(str(temp_dataset_dir), model_size=model_size)

            # Train
            training_results = trainer.train_model(
                epochs=epochs,
                batch_size=batch_size,
                device=device
            )

            # Validate
            print("\nValidating model...")
            validation_metrics = trainer.validate_model()

            # Get the best model path
            # The trainer saves to 'freethrow_training/freethrow_yolov8{model_size}'
            trained_model_dir = Path("freethrow_training") / f"freethrow_yolov8{model_size}"

            # Find the best.pt file
            best_model_paths = list(trained_model_dir.rglob("best.pt"))
            if not best_model_paths:
                raise ValueError("No best.pt file found after training")

            best_model_path = best_model_paths[0]  # Take the first one
            print(f"\nBest model saved at: {best_model_path}")

            # Step 4: Copy the new model to replace the current one
            new_model_dir = Path(MODEL_PATH).parent
            new_model_dir.mkdir(parents=True, exist_ok=True)

            # Backup current model if it exists
            if os.path.exists(MODEL_PATH):
                backup_path = MODEL_PATH + ".backup"
                shutil.copy2(MODEL_PATH, backup_path)
                print(f"Current model backed up to: {backup_path}")

            # Copy new model
            shutil.copy2(best_model_path, MODEL_PATH)
            print(f"New model copied to: {MODEL_PATH}")

            # Step 5: Reload the scorer with the new model
            try:
                scorer_instance = ImprovedFreeThrowScorer(MODEL_PATH)
                print("✅ New model loaded successfully!")
            except Exception as e:
                print(f"⚠️ Warning: Could not reload scorer with new model: {e}")

            # Prepare response
            return {
                "status": "success",
                "message": "Model training complete!",
                "model_path": str(MODEL_PATH),
                "model_size": model_size,
                "epochs_trained": epochs,
                "batch_size": batch_size,
                "device_used": device,
                "datasets_used": len(cvat_zip_files),
                "validation_metrics": {
                    "mAP50": float(validation_metrics.box.map50),
                    "mAP50-95": float(validation_metrics.box.map),
                },
                "class_metrics": {
                    "player": {
                        "AP50": float(validation_metrics.box.ap50[0]) if len(validation_metrics.box.ap50) > 0 else None,
                        "AP50-95": float(validation_metrics.box.ap[0]) if len(validation_metrics.box.ap) > 0 else None
                    },
                    "hoop": {
                        "AP50": float(validation_metrics.box.ap50[1]) if len(validation_metrics.box.ap50) > 1 else None,
                        "AP50-95": float(validation_metrics.box.ap[1]) if len(validation_metrics.box.ap) > 1 else None
                    },
                    "ball": {
                        "AP50": float(validation_metrics.box.ap50[2]) if len(validation_metrics.box.ap50) > 2 else None,
                        "AP50-95": float(validation_metrics.box.ap[2]) if len(validation_metrics.box.ap) > 2 else None
                    }
                }
            }

        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Model training failed: {str(e)}"
            )

@app.get("/health")
async def health():
    """Health check endpoint for Render"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    import socket

    # Configuration
    HOST = "0.0.0.0"  # Listen on all interfaces
    PORT = int(os.getenv("PORT", 10000))  # Use Render's PORT env var

    # For local development, check if port is available
    if "RENDER" not in os.environ:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((HOST, PORT))
                available_port = PORT
        except socket.error:
            print(f"⚠️  Port {PORT} is already in use. Finding an available port...")
            # Find available port logic here...
            available_port = 8001  # Fallback
    else:
        # On Render, use the PORT they provide
        available_port = PORT

    print(f"Starting server on:")
    print(f"  - http://127.0.0.1:{available_port}")
    if HOST == "0.0.0.0" and "RENDER" not in os.environ:
        print(f"  - http://10.0.0.164:{available_port}")  # Your local IP
        print(f"  - http://YOUR_NETWORK_IP:{available_port}")

    uvicorn.run(app, host=HOST, port=available_port)