# Test Set Creator

Automatically creates compelling video compilations using ML-based content analysis.

## What It Does

- **Detects faces** and tracks recurring people (RetinaFace + FaceNet)
- **Analyzes audio** for speech and music (YAMNet)
- **Finds scene boundaries** automatically (PySceneDetect)
- **Scores chunks** and selects the best moments
- **Creates 5-minute compilation** from your videos

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Your Videos

Place your MP4 video files in the `input_videos/` folder:

```
TestSetCreator/
├── TestSetCreator_v3.py
├── input_videos/          ← Put your videos here
│   ├── video1.mp4
│   ├── video2.mp4
│   └── video3.mp4
├── requirements.txt
└── README.md
```

### 3. Run the Script

```bash
python TestSetCreator_v3.py
```

### 4. Get Your Results

The script creates an `output_YYYYMMDD_HHMMSS/` folder with:

- **`compilation.mp4`** - Your final 5-minute video!
- **`extracted_chunks/`** - Individual selected clips
- **`selection_report.txt`** - Detailed analysis and reasoning
- **`metadata/`** - JSON data for each processed video
- **`face_images/`** - Detected faces (for debugging)

## Configuration

Edit these constants in `TestSetCreator_v3.py`:

```python
MIN_CHUNK_DURATION = 8.0   # Minimum clip length (seconds)
MAX_CHUNK_DURATION = 20.0  # Maximum clip length (seconds)
TARGET_COMPILATION_DURATION = 300.0  # 5 minutes
```

## System Requirements

- **Python 3.8+**
- **~2GB RAM** for processing
- **GPU optional** (faster face detection with CUDA)

## Troubleshooting

**"No MP4 files found"**
→ Make sure videos are in `input_videos/` folder

**"RetinaFace not available"**
→ The script will auto-install on first run, then restart

**Slow processing?**
→ Normal for longer videos. ~1-2 min per minute of video.

## How It Works

1. **Scene Detection**: Finds natural cut points in your videos
2. **Face Recognition**: Identifies and tracks people across clips
3. **Audio Analysis**: Detects speech, music, and interesting sounds
4. **Scoring**: Ranks clips by:
   - Number of recurring faces (100 pts each)
   - Speech content (50 pts)
   - Music (30 pts)
   - Interesting audio (25 pts)
   - Duration (10 pts for 15s+ clips)
   - Video quality (5 pts)
5. **Selection**: Picks top clips until hitting 5-minute target
6. **Compilation**: Concatenates with proper encoding (no frozen frames!)

## Output Example

```
output_20251107_183045/
├── compilation.mp4                    ← YOUR VIDEO! 
├── selection_report.txt               ← Why each clip was selected
├── extracted_chunks/
│   ├── chunk_000_VideoName1_14.mp4
│   ├── chunk_001_VideoName1_16.mp4
│   └── chunk_002_VideoName2_4.mp4
├── metadata/
│   ├── video1_metadata.json
│   └── video2_metadata.json
└── face_images/
    ├── person1.jpg
    ├── person2.jpg
    └── person3.jpg
```

## License

MIT

---

**Built with:** RetinaFace · FaceNet · YAMNet · PySceneDetect · FFmpeg
