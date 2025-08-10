import requests
import json
from pathlib import Path

# API base URL
BASE_URL = "http://127.0.0.1:8000"


def test_root():
    """Test the root endpoint"""
    print("\n=== Testing Root Endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_model_status():
    """Test the model status endpoint"""
    print("\n=== Testing Model Status ===")
    try:
        response = requests.get(f"{BASE_URL}/model_status")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_score_video(video_path):
    """Test the video scoring endpoint"""
    print(f"\n=== Testing Video Scoring with: {video_path} ===")

    if not Path(video_path).exists():
        print(f"Error: Video file not found at {video_path}")
        return None

    try:
        with open(video_path, 'rb') as f:
            files = {'video_file': (Path(video_path).name, f, 'video/mp4')}
            response = requests.post(f"{BASE_URL}/score_video/", files=files)

        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            print(f"\n🏀 RESULTS:")
            print(f"   Made Shots: {result['made_shots']}")
            print(f"   Missed Shots: {result['missed_shots']}")
            print(f"   Total Attempts: {result['total_attempts']}")
            print(f"   Frames Processed: {result['frames_processed']}")
        else:
            print(f"Error Response: {response.text}")

        return response.json() if response.status_code == 200 else None
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_docs():
    """Check if API documentation is available"""
    print("\n=== API Documentation ===")
    print(f"Interactive API docs available at: {BASE_URL}/docs")
    print(f"Alternative API docs available at: {BASE_URL}/redoc")


def main():
    """Run all tests"""
    print("=" * 60)
    print("AI BASKETBALL REFEREE API TESTER")
    print("=" * 60)

    # Test root endpoint
    root_result = test_root()

    # Test model status
    model_result = test_model_status()

    # Check if model is loaded
    if model_result and model_result.get('loaded'):
        print("\n✅ Model is loaded and ready!")

        # Test with a video if you have one
        # Uncomment and update the path to test video scoring
        # test_score_video("path/to/your/basketball_video.mp4")
    else:
        print("\n❌ Model is not loaded. Check server logs.")

    # Show docs info
    test_docs()

    print("\n" + "=" * 60)
    print("Testing complete!")


if __name__ == "__main__":
    main()