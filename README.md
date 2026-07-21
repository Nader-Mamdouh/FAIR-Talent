# Tennis Match Analysis (FAIR)

Computer vision system for analyzing tennis match footage: player and ball tracking, court keypoint detection, mini-court visualization, speed metrics, and player scoring.

## Project structure

```
Tennis Model.Ai/
├── main.py                  # Core video processing pipeline
├── api.py                   # FastAPI service (upload video → analysis JSON)
├── app_rep.py               # Player scoring logic
├── analysis_point.py        # Report / stats helpers
├── requirements.txt
├── Dockerfile
├── gunicorn.conf.py
├── startup.txt
│
├── analysis/                # Notebooks and exploratory analysis
├── constants/               # Shared constants
├── court_line_detector/     # Court keypoint detection (ResNet)
├── mini_court/              # Mini-court overlay and coordinate mapping
├── trackers/                # YOLO player and ball trackers
├── utils/                   # Video I/O, drawing, and helper utilities
│
├── models/                  # Trained model weights (YOLO, keypoints)
├── input_videos/            # Place input match videos here (.mp4)
├── output_videos/           # Processed videos are written here
├── tracker_stubs/           # Cached detection results (.pkl)
│
├── data/
│   ├── dataset/             # Training / reference datasets
│   └── reports/             # Generated CSV reports
│
├── docs/
│   └── diagrams/            # ERD, use cases, UI mockups, GP documentation
│
├── backend/                 # .NET FAIR web API (auth, reports, chat)
│   └── FAIR.API/
│
└── .github/workflows/       # CI / deployment
```

## Features

- **Player detection** — YOLOv8 player tracking
- **Ball detection** — Fine-tuned YOLO ball tracking with interpolation
- **Speed analysis** — Player and ball speed from court coordinates
- **Mini court** — Top-down court view with live positions
- **Court keypoints** — ResNet-based court line / keypoint extraction
- **Player scoring** — Threshold-based performance scores from match stats

## Dataset

Tennis Ball Detection dataset (Roboflow):  
https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection

Place additional training data in `data/dataset/`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

Ensure model files are present under `models/`:

- `yolov8x.pt`
- `yolov5su.pt`
- `keypoints_model.pth`

## Usage

### Run the pipeline locally

1. Put a match video in `input_videos/` (e.g. `input_videos/match.mp4`).
2. Call `process_video` from Python, or run via the API (below).

The pipeline reads the video, runs detection and tracking, builds stats, and returns player scores as JSON.

### Run the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Endpoints**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/analyze-tennis/` | Upload a video (`.mp4`, `.avi`, `.mov`) and receive analysis JSON |

**Example**

```bash
curl -X POST "http://localhost:8000/analyze-tennis/" \
  -F "video=@input_videos/match.mp4"
```

### Docker

```bash
docker build -t tennis-analysis .
docker run -p 8000:8000 tennis-analysis
```

### Backend (.NET)

The FAIR platform API lives in `backend/FAIR.API/` (authentication, reports, chat). Open `FAIR.API.sln` in Visual Studio or run with the .NET CLI from that folder.

## Demo output

Sample processed video:  
https://github.com/Nader-Mamdouh/FAIR-Talent-Discovery/blob/main/Tennis%20Model.Ai/output_videos/output_video.avi

## Documentation

Project diagrams, ERD, sequence diagrams, and GP documents are in `docs/diagrams/`.

## Contributing

Issues and pull requests are welcome. For questions, open a GitHub issue.
