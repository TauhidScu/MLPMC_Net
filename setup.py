from setuptools import setup, find_packages

setup(
    name="mlpmc_net",
    version="1.0.0",
    description="Multi-Label Pathology Multi-Task Classification and Segmentation (MLPMC)",
    author="MLPMC Authors",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "nnunetv2>=2.2.1",
        "batchgenerators>=0.25",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "scipy>=1.10.0",
        "opencv-python>=4.7.0",
        "tqdm>=4.65.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
