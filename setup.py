from setuptools import setup, find_packages

setup(
    name="reactmotion",
    version="1.0.0",
    description="ReactMotionNet: Audio-to-Reactive-Motion Generation with Multi-Modal Ranking",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",
    ],
)
