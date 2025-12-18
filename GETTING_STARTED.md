# Getting Started with Narrative Scene Understanding

This guide will help you set up and run the Narrative Scene Understanding system.

## Quick Start

### 1. Install Dependencies

#### Option A: Automatic Installation (Recommended)
```bash
# Install system dependencies (Ubuntu/Debian)
sudo bash scripts/install_dependencies.sh

# Install Python packages
pip install -e ".[full]"
```

#### Option B: Manual Installation
```bash
# Install system tools
sudo apt-get update
sudo apt-get install -y ffmpeg tesseract-ocr libsndfile1 libopencv-dev

# Install Python packages
pip install -e .

# Install optional dependencies for full functionality
pip install segment-anything transformers deep-sort-realtime insightface whisper easyocr
```

### 2. Download SOTA Models

The system uses several state-of-the-art models:

```bash
# Download all models (includes SAM, YOLOv5, Whisper, InsightFace)
python models/download_models.py

# Or download specific models
python models/download_models.py --sam_model vit_b --whisper_model base

# Download InsightFace separately
python models/download_insightface.py --model buffalo_l
```

**Model Options:**
- SAM: `vit_h` (2.4GB, highest quality), `vit_l` (1.2GB, balanced), `vit_b` (375MB, fastest)
- Whisper: `tiny`, `base`, `small`
- InsightFace: `buffalo_l` (330MB, high accuracy), `buffalo_s` (160MB, faster)

### 3. Verify Setup

```bash
# Run the verification script
python scripts/verify_setup.py
```

This will check:
- ✓ Python version (3.8+)
- ✓ Core dependencies
- ✓ Optional packages
- ✓ Downloaded models
- ✓ System tools
- ✓ Project modules

### 4. Run Your First Analysis

```bash
# Process a video
python narrative_scene_understanding.py path/to/video.mp4

# With a specific configuration
python narrative_scene_understanding.py path/to/video.mp4 --config configs/film_analysis.json

# Process and query
python narrative_scene_understanding.py path/to/video.mp4 --query "What is happening in this scene?"
```

## Architecture Overview

### SOTA Models Used

#### Visual Processing
- **SAM** (Segment Anything Model) - Object segmentation
- **YOLOv5** - Object detection (fallback)
- **BLIP** - Image captioning
- **DeepSORT** - Object tracking
- **InsightFace** - Face recognition

#### Audio Processing
- **Whisper** (OpenAI) - Speech transcription
- **Pyannote.audio** - Speaker diarization

#### Text Processing
- **EasyOCR/Tesseract** - Text detection and recognition
- **Llama/GPT** - Natural language query processing

### Pipeline Flow

```
1. Data Ingestion
   ├─ Frame extraction
   ├─ Scene boundary detection
   └─ Audio separation

2. Multi-Modal Processing
   ├─ Visual: SAM + BLIP + DeepSORT + InsightFace
   ├─ Audio: Whisper + Pyannote
   └─ OCR: EasyOCR + Tesseract

3. Knowledge Graph Construction
   ├─ Entity resolution
   ├─ Relationship inference
   └─ Causal/goal inference

4. Narrative Analysis
   ├─ Character analysis
   ├─ Causal chains
   └─ Theme extraction

5. Query Interface
   └─ LLM-powered Q&A

6. Output
   ├─ Knowledge graph (JSON)
   ├─ Scene summary (TXT)
   └─ Query responses
```

## Configuration

### Using Configurations

The system supports multiple configurations for different use cases:

```bash
# Film analysis (detailed character and theme analysis)
python narrative_scene_understanding.py video.mp4 --config configs/film_analysis.json

# Security footage (focus on actions and anomalies)
python narrative_scene_understanding.py video.mp4 --config configs/security.json

# High precision (maximum quality, slower)
python narrative_scene_understanding.py video.mp4 --config configs/high_precision.json
```

### Configuration Options

Edit `configs/default.json` to customize:

```json
{
  "device": "cuda",           // Use "cpu" if no GPU
  "frame_rate": 2.0,          // Frames per second to extract
  "max_frames": 500,          // Maximum frames to process
  "model_paths": {
    "sam": "models/sam_vit_h_4b8939.pth",
    "whisper": "base",
    "face_recognition": "models/insightface_model"
  },
  "vision": {
    "enable_sam": true,
    "enable_face_recognition": true
  }
}
```

## Advanced Usage

### Using Ollama for Local LLM

```python
from narrative_scene_understanding import process_video

config = {
    "query_engine": {
        "use_ollama": True,
        "ollama_model": "llama3",
        "ollama_url": "http://localhost:11434"
    }
}

results = process_video("video.mp4", config)
```

### Batch Processing

```bash
# Process multiple videos
python batch_process.py --directory videos/ --output results/
```

### Visualization

```bash
# Visualize knowledge graph
python visualize_graph.py output/narrative_graph.json

# Character emotional arcs
python visualize_character_arcs.py output/analysis_results.json
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```json
{
  "device": "cpu",
  "max_frames": 200,
  "model_paths": {
    "sam": "models/sam_vit_b_01ec64.pth"  // Use smaller model
  }
}
```

#### 2. Models Not Found
```bash
# Re-download models
python models/download_models.py --models_dir models/
```

#### 3. Import Errors
```bash
# Reinstall dependencies
pip install --force-reinstall -e ".[full]"
```

#### 4. FFmpeg Not Found
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

## Performance Tips

### 1. Use GPU Acceleration
- Install CUDA-enabled PyTorch
- Set `"device": "cuda"` in config

### 2. Reduce Processing Load
- Lower `frame_rate` (e.g., 1.0 fps)
- Use smaller SAM model (`vit_b`)
- Reduce `max_frames`

### 3. Use Model Caching
Models are automatically cached after first download in:
- `~/.cache/torch/hub/` (YOLOv5)
- `~/.cache/huggingface/` (BLIP)
- `~/.cache/whisper/` (Whisper)

## Development

### Running Tests
```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_vision.py -v

# With coverage
pytest tests/ --cov=modules --cov-report=html
```

### Code Quality
```bash
# Format code
black modules/ tests/

# Lint
flake8 modules/ tests/

# Type checking
mypy modules/
```

## CI/CD

### GitHub Actions Workflows

The project includes automated CI/CD:

1. **CI Pipeline** (`.github/workflows/ci.yml`)
   - Linting and code quality
   - Installation tests (Python 3.8-3.11)
   - Unit tests
   - Integration tests

2. **Model Download** (`.github/workflows/download-models.yml`)
   - Manual workflow to download and cache SOTA models
   - Supports different model variants

### Running CI Locally

```bash
# Install act (GitHub Actions local runner)
# https://github.com/nektos/act

# Run CI pipeline locally
act -j test-installation
```

## Support

### Resources
- [Full Documentation](docs/index.md)
- [API Reference](docs/api/index.md)
- [GitHub Issues](https://github.com/asadrizvi64/narrative-scene-understanding/issues)

### Getting Help

1. Check `scripts/verify_setup.py` output
2. Review logs in `output/` directory
3. Run tests: `pytest tests/test_installation.py`
4. Open an issue with:
   - Python version
   - Setup verification output
   - Error logs

## What's Next?

After setup:
1. ✅ Try processing a short video clip
2. ✅ Explore the knowledge graph output
3. ✅ Query the scene with natural language
4. ✅ Customize configurations for your use case
5. ✅ Integrate into your workflow

Happy analyzing! 🎬
