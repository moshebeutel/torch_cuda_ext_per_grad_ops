
import os
from pathlib import Path
from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


with open(Path(os.path.dirname(__file__)) / "requirements.txt") as f:
    required = f.readlines()

setup(
    name="cuda_per_sample_grads_manipulation",
    version="0.0.1",
    author="Moshe Beutel",
    author_email="moshebeutel@gmail.com",
    description="Preform per sample gradients operations using CUDA",
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/moshebeutel/torch_cuda_ext_per_grad_ops.git",
    packages=find_packages(exclude="tests"),
    python_requires=">=3.8",
    install_requires=required,
)