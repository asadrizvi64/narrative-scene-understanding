#!/usr/bin/env python3
"""
Setup script for Narrative Scene Understanding package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="narrative-scene-understanding",
    version="0.1.0",
    author="Narrative Scene Understanding Team",
    author_email="example@example.com",
    description="A system for deep semantic understanding of visual narratives",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/narrative-scene-understanding",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video :: Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        # "opencv-python>=4.5.0",
        "opencv-python-headless>=4.5.0",
        "torch>=1.9.0",
        "networkx>=2.6.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.62.0",
        "ffmpeg-python>=0.2.0",
        "soundfile>=0.10.0",
        "pillow>=8.0.0",
    ],
    extras_require={
        "full": [
            "segment-anything>=1.0",
            "transformers>=4.15.0",
            "deep-sort-realtime>=1.3.1",
            "insightface>=0.6.0",
            "whisper>=0.1.0",
            "pyannote.audio>=2.0.0",
            "easyocr>=1.6.0",
            "pytesseract>=0.3.0",
            "llama-cpp-python>=0.1.0",
            "openai>=0.27.0",
            "plotly>=5.5.0",
            "dash>=2.0.0",
            "dash-bootstrap-components>=1.0.0",
            "pandas>=1.3.0",
            "pygraphviz>=1.7.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "black>=22.1.0",
            "flake8>=4.0.0",
            "mypy>=0.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "narrative-scene=narrative_scene_understanding:main",
            "narrative-batch=batch_process:main",
            "narrative-visualize=visualize_graph:main",
            "narrative-arcs=visualize_character_arcs:main",
            "narrative-summary=generate_summary_video:main",
            "narrative-explore=explore_results:main",
        ],
    },
)