"""
Setup script for AXL package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="axl",
    version="0.1.0",
    author="AXL Contributors",
    author_email="your.email@example.com",
    description="Accelerated Linear Layers built on vector databases",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/axl",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "faiss": ["faiss-cpu>=1.7.0"],
        "faiss-gpu": ["faiss-gpu>=1.7.0"],
        "nanopq": ["nanopq>=0.2.0"],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "machine-learning",
        "deep-learning",
        "pytorch",
        "vector-database",
        "approximate-computation",
        "quantization",
        "faiss",
        "nanopq",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/axl/issues",
        "Source": "https://github.com/yourusername/axl",
        "Documentation": "https://axl.readthedocs.io/",
    },
) 