"""Setup script for tweedie-loss-cost-prediction."""

from setuptools import setup, find_packages

setup(
    name="tweedie-loss-cost-prediction",
    version="1.0.0",
    author="Juan Luo",
    author_email="juanluo2008@gmail.com",
    description="Tweedie loss implementations for zero-inflated cost prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/tweedie-loss-cost-prediction",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "ml": ["xgboost>=2.0.0", "lightgbm>=4.0.0"],
        "torch": ["torch>=2.1.0"],
        "spark": ["pyspark>=3.5.0"],
        "dask": ["dask[complete]>=2024.1.0", "dask-ml>=2023.3.24"],
        "full": [
            "xgboost>=2.0.0", "lightgbm>=4.0.0",
            "torch>=2.1.0", "pyspark>=3.5.0",
            "dask[complete]>=2024.1.0", "dask-ml>=2023.3.24",
        ],
        "viz": ["matplotlib>=3.7.0", "seaborn>=0.12.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
