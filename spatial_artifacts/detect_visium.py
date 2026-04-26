import numpy as np
import pandas as pd

from .utils import compute_outlier_binary
from ._detection import clump_edges, problem_areas


def detect_edge_artifacts_visium(
    adata,
    qc_metric="n_genes_by_counts",
    samples="sample_id",
    mad_threshold=3,
    edge_threshold=0.75,
    min_cluster_size=40,
    shifted=False,
    batch_var="both",
    name="edge_artifact",
    in_tissue_col="in_tissue",
    verbose=True,
    keep_intermediate=False,
):
    """
    Detect edge artifacts and problem areas in Visium spatial transcriptomics data.
    Equivalent to R detectEdgeArtifacts_Visium().

    Parameters
    ----------
    adata : AnnData
        AnnData object with spatial transcriptomics data. Must have:
        - adata.obs[qc_metric]: QC metric column
        - adata.obs[samples]: sample ID column
        - adata.obs["array_row"]: array row coordinates
        - adata.obs["array_col"]: array column coordinates
        - adata.obs[in_tissue_col]: in-tissue indicator (optional)
    qc_metric : str
        QC metric column name in adata.obs (default: "n_genes_by_counts").
    samples : str
        Sample ID column name in adata.obs (default: "sample_id").
    mad_threshold : float
        Number of MADs below median to flag as outlier (default: 3).
    edge_threshold : float
        Minimum proportion of border coverage for edge detection (default: 0.75).
    min_cluster_size : int
        Minimum cluster size for morphological cleaning (default: 40).
    shifted : bool
        Whether to apply coordinate adjustment for hexagonal arrays (default: False).
    batch_var : str
        Batch variable for outlier detection: "slide", "sample_id", or "both"
        (default: "both").
    name : str
        Prefix for output column names (default: "edge_artifact").
    in_tissue_col : str
        Column name for in-tissue indicator (default: "in_tissue").
    verbose : bool
        Whether to print progress messages (default: True).
    keep_intermediate : bool
        Whether to keep intermediate outlier columns in adata.obs (default: False).

    Returns
    -------
    AnnData
        Updated AnnData object with new columns in adata.obs:
        - {name}_edge : bool, True if spot is an edge artifact
        - {name}_problem_id : str, cluster ID for problem area (NaN if none)
        - {name}_problem_size : int, size of the problem area cluster (0 if none)
    """
    if verbose:
        print(f"Computing outliers for metric: {qc_metric}")

    outlier_binary_col = f"{qc_metric}_{mad_threshold}MAD_outlier_binary"

    outlier_flags = compute_outlier_binary(
        adata,
        qc_metric=qc_metric,
        samples=samples,
        nmads=mad_threshold,
        batch_var=batch_var,
        in_tissue_col=in_tissue_col,
    )
    adata.obs[outlier_binary_col] = outlier_flags

    sample_list = adata.obs[samples].unique()

    # Initialize output columns
    adata.obs[f"{name}_edge"] = False
    adata.obs[f"{name}_problem_id"] = np.nan
    adata.obs[f"{name}_problem_size"] = 0

    if verbose:
        print("Detecting edges...")

    for sample_name in sample_list:
        sample_mask = adata.obs[samples] == sample_name
        sample_obs = adata.obs[sample_mask][
            ["array_row", "array_col", outlier_binary_col]
        ].copy()

        if in_tissue_col in adata.obs.columns:
            off_tissue = adata.obs.index[
                sample_mask & (~adata.obs[in_tissue_col].astype(bool))
            ].tolist()
        else:
            off_tissue = []

        edge_spots = clump_edges(
            sample_obs,
            off_tissue=off_tissue,
            outlier_col=outlier_binary_col,
            shifted=shifted,
            edge_threshold=edge_threshold,
            min_cluster_size=min_cluster_size,
        )

        if edge_spots:
            adata.obs.loc[edge_spots, f"{name}_edge"] = True
            if verbose:
                print(f"  Sample {sample_name}: {len(edge_spots)} edge spots detected")

    if verbose:
        print("Finding problem areas...")

    all_problem_dfs = []
    for sample_name in sample_list:
        sample_mask = adata.obs[samples] == sample_name
        sample_obs = adata.obs[sample_mask][
            ["array_row", "array_col", outlier_binary_col]
        ].copy()

        if in_tissue_col in adata.obs.columns:
            off_tissue = adata.obs.index[
                sample_mask & (~adata.obs[in_tissue_col].astype(bool))
            ].tolist()
        else:
            off_tissue = []

        prob_df = problem_areas(
            sample_obs,
            off_tissue=off_tissue,
            outlier_col=outlier_binary_col,
            unique_identifier=str(sample_name),
            shifted=shifted,
            min_cluster_size=min_cluster_size,
        )
        all_problem_dfs.append(prob_df)

    if all_problem_dfs:
        combined = pd.concat(all_problem_dfs, ignore_index=True)
        if len(combined) > 0:
            adata.obs.loc[combined["spotcode"], f"{name}_problem_id"] = (
                combined["clump_id"].values
            )
            adata.obs.loc[combined["spotcode"], f"{name}_problem_size"] = (
                combined["clump_size"].values
            )

    if not keep_intermediate:
        if outlier_binary_col in adata.obs.columns:
            del adata.obs[outlier_binary_col]
            if verbose:
                print(f"Removed intermediate column: {outlier_binary_col}")

    if verbose:
        total_edge = adata.obs[f"{name}_edge"].sum()
        total_problem = adata.obs[f"{name}_problem_id"].notna().sum()
        print(
            f"Edge artifact detection completed!\n"
            f"  Total edge spots: {total_edge}\n"
            f"  Total problem area spots: {total_problem}"
        )

    return adata