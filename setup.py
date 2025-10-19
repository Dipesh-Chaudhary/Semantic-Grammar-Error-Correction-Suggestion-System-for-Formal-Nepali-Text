# setup.py
from setuptools import setup, find_packages

setup(
    name="nepali-semantic-gec",
    version="1.0.0",
    description="Semantic-Aware Nepali Grammar Error Correction System",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "peft",
        "accelerate",
        "bitsandbytes",
    ],
    python_requires=">=3.12",
)