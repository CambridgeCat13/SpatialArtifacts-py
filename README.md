# SpatialArtifacts-py

The goal of `SpatialArtifacts` is to detect interior and edge artifacts, such as dry spots caused by incomplete reagent coverage or tissue handling, in spatial transcriptomics data. This is the Python implementation of the SpatialArtifacts package, supporting the 10x Genomics `Visium` and `VisiumHD` platforms.

If you experience any issues or would like to make a suggestion, please open an issue on the [GitHub repository](https://github.com/CambridgeCat13/SpatialArtifacts-py/issues).

### Key Features

- **Multi-platform support**: Works on both standard **10x Visium** and high-resolution **VisiumHD**
- **Morphological detection**: Uses scipy-based focal transformations (fill, outline, star-pattern) to identify artifact clusters
- **Hierarchical classification**: Categorizes artifacts into actionable groups (e.g., *Large Edge Artifact*, *Small Interior Artifact*)
- **AnnData compatible**: Designed to work seamlessly with `AnnData` objects and the `scanpy` ecosystem

## Installation

```bash
pip install spatial-artifacts
```

## Example

```python
from spatial_artifacts import detect_edge_artifacts, classify_edge_artifacts

# 1. Detect artifacts
# Option A: Standard Visium (Hexagonal Grid)
adata = detect_edge_artifacts(adata, platform="visium", qc_metric="total_counts")

# Option B: VisiumHD (Square Grid)
# adata = detect_edge_artifacts(adata, platform="visiumhd", resolution="16um")

# 2. Classify results (platform independent)
# Note: For VisiumHD, remember to scale min_spots (e.g., min_spots=200)
adata = classify_edge_artifacts(adata, min_spots=20)

# 3. View classification
print(adata.obs["edge_artifact_classification"].value_counts())
```

## Tutorials

Detailed tutorials are available in the `example/` folder:
- [Standard Visium Tutorial](example/SpatialArtifacts_visium_tutorial.ipynb)
- [VisiumHD Tutorial](example/SpatialArtifacts_visiumhd_tutorial.ipynb)

## Contributors

- [Harriet Jiali He](https://www.linkedin.com/in/harriet-he-a5ba4b21b)
- [Jacqueline R. Thompson](https://www.linkedin.com/in/jacqueline-r-thompson-6a478a159)
- [Michael Totty](https://mictott.github.io)
- [Stephanie C. Hicks](https://stephaniehicks.com)