#!/usr/bin/env python3
"""
Verification script for Narrative Scene Understanding setup.
Checks if all required dependencies and models are properly installed.
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")

def check_status(success, name, details=""):
    """Print check status."""
    if success:
        print(f"{Colors.GREEN}✓{Colors.RESET} {name}")
        if details:
            print(f"  {details}")
    else:
        print(f"{Colors.RED}✗{Colors.RESET} {name}")
        if details:
            print(f"  {details}")
    return success

def check_python_version():
    """Check Python version."""
    print_header("Python Environment")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    is_valid = version.major == 3 and version.minor >= 8

    check_status(
        is_valid,
        f"Python version: {version_str}",
        "Required: Python 3.8 or higher" if not is_valid else ""
    )

    return is_valid

def check_core_packages():
    """Check if core packages are installed."""
    print_header("Core Dependencies")

    packages = {
        "numpy": "NumPy",
        "cv2": "OpenCV",
        "torch": "PyTorch",
        "PIL": "Pillow",
        "networkx": "NetworkX",
        "tqdm": "tqdm",
        "matplotlib": "Matplotlib",
    }

    all_present = True
    for module, name in packages.items():
        try:
            __import__(module)
            if module == "torch":
                import torch
                cuda_available = torch.cuda.is_available()
                device_info = f"CUDA available: {cuda_available}"
                if cuda_available:
                    device_info += f" (Device: {torch.cuda.get_device_name(0)})"
                check_status(True, name, device_info)
            else:
                check_status(True, name)
        except ImportError:
            check_status(False, name, "Not installed")
            all_present = False

    return all_present

def check_optional_packages():
    """Check if optional packages are installed."""
    print_header("Optional Dependencies (SOTA Models)")

    packages = {
        "segment_anything": "Segment Anything (SAM)",
        "transformers": "Transformers (BLIP)",
        "deep_sort_realtime": "DeepSORT Tracker",
        "insightface": "InsightFace",
        "whisper": "Whisper",
        "easyocr": "EasyOCR",
        "pytesseract": "Tesseract",
        "openai": "OpenAI API",
        "llama_cpp": "Llama-CPP-Python",
    }

    available = []
    missing = []

    for module, name in packages.items():
        try:
            __import__(module)
            check_status(True, name, "Installed")
            available.append(name)
        except ImportError:
            check_status(False, name, "Not installed (optional)")
            missing.append(name)

    print(f"\n{Colors.YELLOW}Note:{Colors.RESET} Optional packages enhance functionality but are not required.")
    print(f"Available: {len(available)}/{len(packages)}")

    return len(available) > 0

def check_models():
    """Check if required models are downloaded."""
    print_header("Model Files")

    models_dir = Path(__file__).parent.parent / "models"

    model_checks = {
        "SAM (ViT-H)": models_dir / "sam_vit_h_4b8939.pth",
        "SAM (ViT-L)": models_dir / "sam_vit_l_0b3195.pth",
        "SAM (ViT-B)": models_dir / "sam_vit_b_01ec64.pth",
        "InsightFace": models_dir / "insightface_model",
    }

    models_found = 0
    for name, path in model_checks.items():
        exists = path.exists()
        if exists:
            if path.is_dir():
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                check_status(True, name, f"Size: {size_mb:.1f} MB")
            else:
                size_mb = path.stat().st_size / (1024 * 1024)
                check_status(True, name, f"Size: {size_mb:.1f} MB")
            models_found += 1
        else:
            check_status(False, name, "Not found")

    if models_found == 0:
        print(f"\n{Colors.YELLOW}⚠ No models found!{Colors.RESET}")
        print(f"Run: python models/download_models.py")

    return models_found > 0

def check_system_tools():
    """Check if required system tools are installed."""
    print_header("System Tools")

    tools = {
        "ffmpeg": "FFmpeg (video processing)",
        "tesseract": "Tesseract (OCR)",
    }

    all_present = True
    for cmd, name in tools.items():
        import shutil
        exists = shutil.which(cmd) is not None
        check_status(exists, name, f"Command: {cmd}")
        if not exists:
            all_present = False

    return all_present

def check_modules():
    """Check if project modules can be imported."""
    print_header("Project Modules")

    modules = [
        "modules.ingestion",
        "modules.vision",
        "modules.audio",
        "modules.ocr",
        "modules.knowledge_graph",
        "modules.analysis",
        "modules.query",
        "modules.utils",
    ]

    all_imported = True
    for module in modules:
        try:
            __import__(module)
            check_status(True, module.replace("modules.", ""))
        except ImportError as e:
            check_status(False, module.replace("modules.", ""), f"Error: {e}")
            all_imported = False
        except Exception as e:
            check_status(False, module.replace("modules.", ""), f"Error during import: {e}")
            all_imported = False

    return all_imported

def check_config_files():
    """Check if configuration files exist."""
    print_header("Configuration Files")

    project_root = Path(__file__).parent.parent
    configs = {
        "Default": project_root / "configs" / "default.json",
        "Film Analysis": project_root / "configs" / "film_analysis.json",
        "High Precision": project_root / "configs" / "high_precision.json",
    }

    configs_found = 0
    for name, path in configs.items():
        exists = path.exists()
        check_status(exists, name, str(path.relative_to(project_root)))
        if exists:
            configs_found += 1

    return configs_found > 0

def print_summary(results):
    """Print overall summary."""
    print_header("Setup Summary")

    total = len(results)
    passed = sum(1 for r in results.values() if r)

    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All checks passed! ({passed}/{total}){Colors.RESET}")
        print(f"\nYour setup is ready to use!")
        print(f"\n{Colors.BOLD}Next steps:{Colors.RESET}")
        print(f"1. Process a video: python narrative_scene_understanding.py <video_path>")
        print(f"2. Run tests: pytest tests/")
        return True
    else:
        failed = total - passed
        print(f"{Colors.YELLOW}⚠ {failed} check(s) failed ({passed}/{total} passed){Colors.RESET}")
        print(f"\n{Colors.BOLD}Recommended actions:{Colors.RESET}")

        if not results.get("python"):
            print(f"- Upgrade Python to version 3.8 or higher")

        if not results.get("core_packages"):
            print(f"- Install core packages: pip install -e .")

        if not results.get("models"):
            print(f"- Download models: python models/download_models.py")

        if not results.get("system_tools"):
            print(f"- Install system dependencies: sudo bash scripts/install_dependencies.sh")

        print(f"\nFor optional packages: pip install -e \".[full]\"")
        return False

def main():
    """Run all checks."""
    print(f"\n{Colors.BOLD}Narrative Scene Understanding - Setup Verification{Colors.RESET}")
    print(f"This script will check if your environment is properly configured.\n")

    results = {
        "python": check_python_version(),
        "core_packages": check_core_packages(),
        "optional_packages": check_optional_packages(),
        "models": check_models(),
        "system_tools": check_system_tools(),
        "modules": check_modules(),
        "configs": check_config_files(),
    }

    success = print_summary(results)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
