"""Setup script for Acceleray - Ray on Databricks ML toolkit."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="acceleray",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@company.com",
    description="Accelerate ML workloads with Ray on Databricks - distributed training, tuning, and inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/data-brickster/acceleray",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "isort>=5.13.0",
            "flake8>=7.0.0",
        ],
    },
)
