# 🏀 AI Basketball Referee

An intelligent computer vision system that automatically detects and scores basketball free throws using YOLOv8 and advanced shot tracking algorithms.

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-latest-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

- **Automatic Free Throw Detection**: Real-time detection of basketball shots
- **Accurate Scoring**: Advanced algorithms to determine made/missed shots
- **Video Analysis**: Process recorded basketball videos
- **Model Training**: Train custom models with your own annotated data
- **Web Interface**: User-friendly interface for video upload and analysis
- **REST API**: Full-featured API for integration with other applications
- **Multi-Platform**: Works on Windows, Mac, and Linux

## 🎯 How It Works

The system uses:
1. **YOLOv8** for object detection (player, ball, hoop)
2. **Custom shot tracking algorithm** that monitors ball trajectory
3. **Scoring logic** based on ball movement through the hoop's scoring plane

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- 4GB+ RAM recommended
- GPU optional but recommended for faster processing

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/basketball-referee.git
cd basketball-referee
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download or place your model**
Place your trained model at:
```
C:/Users/carlc/Desktop/API  AI REFEREE MODEL/runs/detect/train3/weights/best.pt
```
Or update the path in `main.py`

### Running the Application

#### Local Development
```bash
python main.py
```
Access at: http://127.0.0.1:8000

#### Quick Public Deployment
```bash
# Install ngrok first: https://ngrok.com/download
ngrok http 8000
```
Share the generated URL with anyone!

## 📖 Usage

### Web Interface

1. Open `web_interface.html` in your browser
2. The interface will auto-detect the API
3. Upload basketball videos to analyze
4. View real-time scoring results

### API Endpoints

- `GET /` - Landing page
- `GET /status` - API and model status
- `POST /score_video/` - Analyze a basketball video
- `POST /train_model/` - Train a new model
- `GET /docs` - Interactive API documentation

### Python Example
```python
import requests

# Analyze a video
with open("basketball_video.mp4", "rb") as f:
    files = {"video_file": f}
    response = requests.post("http://127.0.0.1:8000/score_video/", files=files)
    print(response.json())
```

### cURL Example
```bash
curl -X POST "http://127.0.0.1:8000/score_video/" \
     -H "accept: application/json" \
     -F "video_file=@basketball_video.mp4"
```

## 🎓 Training Custom Models

### Prepare Dataset

1. Annotate videos using CVAT with YOLO 1.1 format
2. Label three classes:
   - `player` (class 0)
   - `hoop` (class 1)
   - `ball` (class 2)

### Train Model

Using the API:
```python
files = [
    ('cvat_zip_files', open('dataset1.zip', 'rb')),
    ('cvat_zip_files', open('dataset2.zip', 'rb'))
]
data = {
    'epochs': 150,
    'batch_size': 16,
    'model_size': 's',
    'device': 'auto'
}
response = requests.post("http://127.0.0.1:8000/train_model/", files=files, data=data)
```

Or using the command line:
```bash
python basketball_referee.py --mode train --dataset_path ./dataset --epochs 150
```

## 🏗️ Architecture

```
basketball-referee/
├── main.py                 # FastAPI application
├── basketball_referee.py   # Core detection and scoring logic
├── yolo_loader_fix.py     # PyTorch compatibility fix
├── web_interface.html     # Frontend interface
├── requirements.txt       # Python dependencies
├── deploy_quick.bat      # Windows deployment script
└── models/               # Model storage directory
```

### Core Components

1. **ImprovedFreeThrowScorer**: Main class for video processing
2. **ShotTracker**: Tracks ball trajectory and determines scores
3. **CVATDatasetConverter**: Converts CVAT annotations to YOLO format
4. **FreeThrowModelTrainer**: Handles model training

## 🌐 Deployment

### Local Network
```bash
python main.py
# Access from other devices: http://YOUR_IP:8000
```

### Public Internet (Ngrok)
```bash
ngrok http 8000
# Share the HTTPS URL
```

### Cloud Deployment

#### Railway.app
```bash
railway login
railway init
railway up
```

#### Docker
```bash
docker build -t basketball-referee .
docker run -p 8000:8000 basketball-referee
```

## ⚙️ Configuration

### Environment Variables
- `MODEL_URL` - URL to download model from (for cloud deployment)
- `MAX_VIDEO_SIZE_MB` - Maximum video upload size (default: 100)
- `RATE_LIMIT_REQUESTS` - Requests per minute limit (default: 10)
- `PORT` - Server port (default: 8000)

### Model Configuration
Update these constants in `basketball_referee.py`:
```python
HOOP_CONFIDENCE_THRESHOLD = 0.2
BALL_CONFIDENCE_THRESHOLD = 0.1
MIN_FRAMES_BALL_THROUGH_PLANE = 2
```

## 📊 Performance

- **Processing Speed**: ~30 FPS on GPU, ~5-10 FPS on CPU
- **Accuracy**: 85-95% on good quality videos
- **Model Size**: ~50-100MB depending on architecture
- **RAM Usage**: 2-4GB during processing

## 🐛 Troubleshooting

### Model Not Loading
- Check model path in `main.py`
- Ensure model file exists and is not corrupted
- Verify PyTorch installation

### Poor Detection Accuracy
- Ensure good video quality and lighting
- Ball should be clearly visible
- Consider training with more data

### API Connection Issues
- Check firewall settings
- Verify port is not in use: `netstat -ano | findstr :8000`
- Try different port if needed

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- FastAPI framework
- OpenCV community
- CVAT annotation tool

## 📞 Support

- **Issues**: Open an issue on GitHub
- **Email**: your.email@example.com
- **Documentation**: http://127.0.0.1:8000/docs

## 🎥 Demo

[Watch Demo Video](https://your-demo-link.com)

---

**Made with ❤️ by [Your Name]**