from setuptools import setup, find_packages

setup(
    name="llm-project",
    version="0.1.0",
    description="A small-scale LLM implementation for learning",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "tiktoken",
        "datasets",
        "matplotlib>=3.5.0",
        "tqdm",
        "pyyaml",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov",
            "jupyter",
            "black",
            "flake8",
        ],
    },
)
