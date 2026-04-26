from setuptools import setup, find_packages

setup(
    name="spatial-artifacts",
    version="0.1.0",
    description="Identification and Classification of Spatial Artifacts in Visium and VisiumHD Data",
    long_description=(
        "SpatialArtifacts provides a data-driven two-step workflow to identify, "
        "classify, and handle spatial artifacts in spatial transcriptomics data. "
        "The package combines median absolute deviation (MAD)-based outlier detection "
        "with morphological image processing (fill, outline, and star patterns) to "
        "detect edge and interior artifacts. It supports multiple platforms including "
        "10x Genomics Visium (standard and HD), allowing for consistent quality "
        "control across different spatial resolutions."
    ),
    author="Harriet Jiali He, Jacqueline R. Thompson, Michael Totty, Stephanie C. Hicks",
    author_email="jhe46@jh.edu, jthom338@jh.edu, mtotty2@jh.edu, shicks19@jhu.edu",
    url="https://github.com/CambridgeCat13/SpatialArtifacts-py",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "anndata",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)