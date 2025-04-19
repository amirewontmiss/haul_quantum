import os

from setuptools import find_packages, setup

# Read the long description from README.md
this_dir = os.path.abspath(os.path.dirname(__file__))
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="haul_quantum",
    version="0.1.1",
    description="Haul Quantum AI Framework: a next-gen hybrid quantum-classical ML library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Amire Ramazan",
    author_email="amireramazan0809@gmail.com",
    url="https://github.com/amirewontmiss/haul_quantum",
    packages=find_packages(where="."),
    install_requires=[
        "numpy>=1.20",
        "matplotlib>=3.0",
        "scikit-learn>=0.24",
        "torch>=1.8",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Documentation": "https://github.com/amirewontmiss/haul_quantum#readme",
        "Source": "https://github.com/amirewontmiss/haul_quantum",
        "Tracker": "https://github.com/amirewontmiss/haul_quantum/issues",
    },
)
