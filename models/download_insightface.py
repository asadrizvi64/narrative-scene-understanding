#!/usr/bin/env python3
"""
Script to download InsightFace models for face recognition.
"""

import os
import sys
import argparse
import logging
import requests
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("insightface-downloader")

# Default output directory
DEFAULT_OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# InsightFace model URLs
INSIGHTFACE_MODELS = {
    "buffalo_l": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "description": "Buffalo_L - High accuracy face recognition model",
        "size": 330_000_000,  # ~330 MB
    },
    "buffalo_s": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
        "description": "Buffalo_S - Smaller, faster face recognition model",
        "size": 160_000_000,  # ~160 MB
    }
}

def download_file(url, output_path, expected_size=None):
    """
    Download a file with progress bar.

    Args:
        url: URL to download from
        output_path: Path to save the file
        expected_size: Expected file size in bytes (optional)

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading {url}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Start the download
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Get content length
        total_size = int(response.headers.get('content-length', 0))

        # Initialize progress bar
        progress_bar = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=os.path.basename(output_path)
        )

        # Download the file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        progress_bar.close()

        # Verify file size if expected size is provided
        if expected_size and os.path.getsize(output_path) < expected_size * 0.9:
            logger.warning(f"Downloaded file size seems incorrect: {output_path}")
            return False

        logger.info(f"Download completed: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """
    Extract a zip file.

    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Extracting {zip_path} to {extract_to}")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        logger.info(f"Extraction completed")
        return True

    except Exception as e:
        logger.error(f"Error extracting {zip_path}: {e}")
        return False

def download_insightface_model(model_name, output_dir):
    """
    Download and extract InsightFace model.

    Args:
        model_name: Name of the model to download
        output_dir: Directory to save the model

    Returns:
        True if successful, False otherwise
    """
    if model_name not in INSIGHTFACE_MODELS:
        logger.error(f"Unknown InsightFace model: {model_name}")
        logger.info(f"Available models: {', '.join(INSIGHTFACE_MODELS.keys())}")
        return False

    model_info = INSIGHTFACE_MODELS[model_name]

    # Create model directory
    model_dir = os.path.join(output_dir, "insightface_model")
    os.makedirs(model_dir, exist_ok=True)

    # Check if model already exists
    model_path = os.path.join(model_dir, model_name)
    if os.path.exists(model_path) and os.listdir(model_path):
        logger.info(f"Model already exists at {model_path}")

        # Check if it has the expected files
        expected_files = ["det_10g.onnx", "w600k_r50.onnx"]
        has_all_files = all(
            os.path.exists(os.path.join(model_path, f)) for f in expected_files
        )

        if has_all_files:
            logger.info("Model appears to be complete. Skipping download.")
            return True
        else:
            logger.warning("Model directory exists but appears incomplete. Re-downloading...")

    # Download the zip file
    zip_path = os.path.join(output_dir, f"{model_name}.zip")
    logger.info(f"Downloading {model_name}: {model_info['description']}")

    if not download_file(model_info["url"], zip_path, model_info["size"]):
        return False

    # Extract the zip file
    if not extract_zip(zip_path, model_dir):
        return False

    # Clean up zip file
    try:
        os.remove(zip_path)
        logger.info(f"Removed temporary file: {zip_path}")
    except Exception as e:
        logger.warning(f"Failed to remove temporary file {zip_path}: {e}")

    logger.info(f"InsightFace model {model_name} installed successfully at {model_dir}")
    return True

def install_insightface_package():
    """
    Install the insightface Python package.

    Returns:
        True if successful, False otherwise
    """
    try:
        import insightface
        logger.info(f"InsightFace package already installed (version: {insightface.__version__})")
        return True
    except ImportError:
        logger.info("InsightFace package not found. Installing...")

        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "insightface"])
            logger.info("InsightFace package installed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to install InsightFace package: {e}")
            logger.info("You can manually install it with: pip install insightface")
            return False

def main():
    parser = argparse.ArgumentParser(description="Download InsightFace models")
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the models"
    )
    parser.add_argument(
        "--model",
        choices=list(INSIGHTFACE_MODELS.keys()),
        default="buffalo_l",
        help="Model to download (default: buffalo_l)"
    )
    parser.add_argument(
        "--install_package",
        action="store_true",
        help="Install the insightface Python package"
    )

    args = parser.parse_args()

    logger.info("=== InsightFace Model Downloader ===")
    logger.info(f"Output directory: {args.output_dir}")

    # Install package if requested
    if args.install_package:
        if not install_insightface_package():
            logger.warning("Package installation failed, but continuing with model download")

    # Download the model
    success = download_insightface_model(args.model, args.output_dir)

    if success:
        logger.info("\n✅ InsightFace model download completed successfully!")
        logger.info(f"\nModel location: {os.path.join(args.output_dir, 'insightface_model', args.model)}")
        logger.info("\nNext steps:")
        logger.info("1. Ensure insightface package is installed: pip install insightface")
        logger.info("2. Update your config to point to the model directory")
        logger.info("3. Run your application")
    else:
        logger.error("\n⚠️ InsightFace model download failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
