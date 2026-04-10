"""
SPLICE: Splicegraph Probabilistic Learning for Isoform Change Estimation

A comprehensive platform for discovery and analysis of differential splicing
events in RNA-seq data. Combines annotation-free junction discovery, multi-way
statistical testing, covariate regression, heterogeneity detection, and
functional annotation into a unified framework.
"""

from setuptools import setup, find_packages
from setuptools.command.install import install
from pathlib import Path


class PostInstallCommand(install):
    """Post-installation: attempt to build the Rust BAM reader extension."""

    def run(self):
        install.run(self)
        try:
            from splice.install_rust import try_build_rust_extension
            try_build_rust_extension()
        except Exception:
            # Swallow errors so pip install always succeeds
            pass


# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

setup(
    name="splice",
    version="1.0.0",
    description="SPLICE: Splicegraph Probabilistic Learning for Isoform Change Estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mariano Aguilar Vela",
    author_email="mariano.aguilarvela@hdr.qut.edu.au",
    license="MIT",
    url="https://github.com/mariano-aguilar-vela/splice",
    project_urls={
        "Bug Tracker": "https://github.com/mariano-aguilar-vela/splice/issues",
        "Documentation": "https://splice.readthedocs.io",
        "Source Code": "https://github.com/mariano-aguilar-vela/splice",
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords=[
        "bioinformatics",
        "RNA-seq",
        "alternative splicing",
        "differential splicing",
        "transcriptomics",
        "splicing analysis",
    ],
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples", "benchmark", "benchmark.*"]),
    install_requires=[
        "pysam>=0.22.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "statsmodels>=0.14.0",
        "click>=7.0.0",
        "tqdm>=4.65.0",
        "pyarrow>=14.0.0",
        "pyfastx>=2.0.0",
        "biopython>=1.81",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "numba>=0.58.0",
        "openpyxl>=3.1.0",
        "reportlab>=4.0",
        "matplotlib_venn>=1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "rust": [
            "maturin>=1.0",
        ],
    },
    cmdclass={
        "install": PostInstallCommand,
    },
    entry_points={
        "console_scripts": [
            "splice=splice.cli:main",
        ],
    },
    zip_safe=False,
    include_package_data=True,
)
