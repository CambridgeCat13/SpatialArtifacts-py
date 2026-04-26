import numpy as np


def classify_edge_artifacts(
    adata,
    qc_metric="total_counts",
    samples="sample_id",
    min_spots=20,
    name="edge_artifact",
    exclude_slides=None,
    verbose=True,
):
    """
    Classify detected artifacts into hierarchical categories.
    Equivalent to R classifyEdgeArtifacts().

    Must be run after detect_edge_artifacts_visium() or
    detect_edge_artifacts_visiumhd().

    Parameters
    ----------
    adata : AnnData
        AnnData object processed with detect_edge_artifacts_*().
    qc_metric : str
        QC metric column name (must exist, not used for logic) (default: "total_counts").
    samples : str
        Sample ID column name (default: "sample_id").
    min_spots : int
        Minimum number of spots/bins for an artifact to be classified as
        "large" (default: 20).
        Recommended values by platform:
        - Standard Visium (55um bins): 20-40
        - VisiumHD 16um bins: 100-200
        - VisiumHD 8um bins: 400-800
    name : str
        Prefix matching the name used in detect_edge_artifacts_*() (default: "edge_artifact").
    exclude_slides : list of str, optional
        Slide IDs whose edge detections should be forced to False.
        Requires a "slide" column in adata.obs (default: None).
    verbose : bool
        Whether to print progress messages (default: True).

    Returns
    -------
    AnnData
        Updated AnnData object with new columns in adata.obs:
        - {name}_true_edges : bool, edge flags after applying slide exclusions
        - {name}_classification : str, one of:
            "not_artifact"
            "large_edge_artifact"
            "small_edge_artifact"
            "large_interior_artifact"
            "small_interior_artifact"
    """
    required_cols = [
        f"{name}_edge",
        f"{name}_problem_id",
        f"{name}_problem_size",
    ]
    missing = [c for c in required_cols if c not in adata.obs.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Run detect_edge_artifacts_visium() or detect_edge_artifacts_visiumhd() first."
        )

    if verbose:
        print("Classifying artifact spots...")

    # Copy edge flags to true_edges column
    true_edges_col = f"{name}_true_edges"
    adata.obs[true_edges_col] = adata.obs[f"{name}_edge"].copy()

    # Exclude specified slides if provided
    if exclude_slides is not None and "slide" in adata.obs.columns:
        slide_mask = adata.obs["slide"].isin(exclude_slides)
        adata.obs.loc[slide_mask, true_edges_col] = False
        if verbose:
            print(f"Excluding edges from slides: {exclude_slides}")

    is_edge = adata.obs[true_edges_col].astype(bool)
    is_problem = adata.obs[f"{name}_problem_id"].notna()
    artifact_size = adata.obs[f"{name}_problem_size"].fillna(0).astype(int)

    # Initialize all as not_artifact
    classification_col = f"{name}_classification"
    adata.obs[classification_col] = "not_artifact"

    # Step 1: interior artifacts (problem area but NOT edge) - lower priority
    is_interior = is_problem & ~is_edge
    if is_interior.any():
        adata.obs.loc[is_interior & (artifact_size > min_spots), classification_col] = (
            "large_interior_artifact"
        )
        adata.obs.loc[is_interior & (artifact_size <= min_spots), classification_col] = (
            "small_interior_artifact"
        )

    # Step 2: edge artifacts - higher priority, overwrites interior
    if is_edge.any():
        adata.obs.loc[is_edge & (artifact_size > min_spots), classification_col] = (
            "large_edge_artifact"
        )
        adata.obs.loc[is_edge & (artifact_size <= min_spots), classification_col] = (
            "small_edge_artifact"
        )

    if verbose:
        print(f"Classification added: {classification_col}")
        print("\nClassification summary:")
        counts = adata.obs[classification_col].value_counts()
        for label, count in counts.items():
            print(f"  {label}: {count} spots")

    return adata