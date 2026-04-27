import numpy as np
import pandas as pd
from scipy.ndimage import label

from .utils import is_outlier
from ._morphology import focal_transformations


def _problem_areas_visiumhd(xyz_df, unique_identifier=None, min_cluster_size=5):
    """
    Detect interior problem areas in VisiumHD data using morphological processing.
    Equivalent to R problemAreas_WithMorphology_terra().

    Parameters
    ----------
    xyz_df : pd.DataFrame
        DataFrame with columns: x (array_col), y (array_row), outlier (binary).
        Index = bin barcodes.
    unique_identifier : str, optional
        Prefix for cluster IDs (default: "X").
    min_cluster_size : int
        Minimum cluster size in bins (default: 5).

    Returns
    -------
    pd.DataFrame with columns: spotcode, clump_id, clump_size
    """
    empty = pd.DataFrame(columns=["spotcode", "clump_id", "clump_size"])

    if unique_identifier is None:
        unique_identifier = "X"

    outlier_vals = xyz_df["outlier"].fillna(0).values
    if outlier_vals.sum() == 0:
        return empty

    xs = xyz_df["x"].values
    ys = xyz_df["y"].values

    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())

    n_rows = max_y - min_y + 1
    n_cols = max_x - min_x + 1

    grid = np.zeros((n_rows, n_cols), dtype=float)

    for i, (x, y, o) in enumerate(zip(xs, ys, outlier_vals)):
        ri = int(y) - min_y
        ci = int(x) - min_x
        grid[ri, ci] = float(o)

    processed = focal_transformations(grid, min_cluster_size=min_cluster_size)

    binary = np.where(np.isnan(processed), 0, processed).astype(int)
    labeled, num_features = label(binary, structure=np.ones((3, 3)))

    if num_features == 0:
        return empty

    # Build coordinate lookup: (y, x) -> barcode
    coord_to_barcode = {}
    for bc, x, y in zip(xyz_df.index, xs, ys):
        coord_to_barcode[(round(float(y), 6), round(float(x), 6))] = bc

    records = []
    for clump_id in range(1, num_features + 1):
        positions = np.argwhere(labeled == clump_id)
        clump_size = len(positions)

        if clump_size < min_cluster_size:
            continue

        for ri, ci in positions:
            y_coord = round(float(ri + min_y), 6)
            x_coord = round(float(ci + min_x), 6)
            bc = coord_to_barcode.get((y_coord, x_coord))
            if bc is not None:
                records.append({
                    "spotcode": bc,
                    "clump_id": f"{unique_identifier}_{clump_id}",
                    "clump_size": clump_size
                })

    if not records:
        return empty

    return pd.DataFrame(records)


def detect_edge_artifacts_visiumhd(
    adata,
    resolution,
    qc_metric="n_genes_by_counts",
    samples="sample_id",
    mad_threshold=3,
    buffer_width_um=80,
    min_cluster_area_um2=1280,
    batch_var="sample_id",
    col_x="array_col",
    col_y="array_row",
    name="edge_artifact",
    in_tissue_col="in_tissue",
    verbose=True,
    keep_intermediate=False,
):
    """
    Detect edge artifacts and interior problem areas in VisiumHD data.
    Equivalent to R detectEdgeArtifacts_VisiumHD().

    Parameters
    ----------
    adata : AnnData
        AnnData object with VisiumHD spatial transcriptomics data. Must have:
        - adata.obs[qc_metric]: QC metric column
        - adata.obs[samples]: sample ID column
        - adata.obs[col_x]: array column (bin index)
        - adata.obs[col_y]: array row (bin index)
    resolution : str
        VisiumHD resolution: "8um" or "16um" (REQUIRED).
    qc_metric : str
        QC metric column name in adata.obs (default: "n_genes_by_counts").
    samples : str
        Sample ID column name in adata.obs (default: "sample_id").
    mad_threshold : float
        Number of MADs below median to flag as outlier (default: 3).
    buffer_width_um : float
        Buffer zone width in micrometers for edge detection (default: 80).
        Converted to bins based on resolution.
    min_cluster_area_um2 : float
        Minimum cluster area in um^2 for interior problem areas (default: 1280).
        Converted to bins based on resolution.
    batch_var : str
        Batch variable for outlier detection: "slide", "sample_id", or "both"
        (default: "sample_id").
    col_x : str
        Column name for array column coordinates (default: "array_col").
    col_y : str
        Column name for array row coordinates (default: "array_row").
    name : str
        Prefix for output column names (default: "edge_artifact").
    in_tissue_col : str
        Column name for in-tissue indicator (default: "in_tissue").
    verbose : bool
        Whether to print progress messages (default: True).
    keep_intermediate : bool
        Whether to keep intermediate outlier columns (default: False).

    Returns
    -------
    AnnData
        Updated AnnData object with new columns in adata.obs:
        - {name}_edge : bool, True if bin is in buffer zone and is an outlier
        - {name}_problem_id : str, cluster ID for interior problem area (NaN if none)
        - {name}_problem_size : int, size of interior problem area cluster (0 if none)
    """
    if resolution not in ("8um", "16um"):
        raise ValueError(f"'resolution' must be '8um' or '16um'. Got: {resolution}")

    # Convert physical units to bins
    bin_size_um = 8 if resolution == "8um" else 16
    bin_area_um2 = bin_size_um ** 2
    buffer_width_bins = round(buffer_width_um / bin_size_um)
    min_cluster_size_bins = int(np.ceil(min_cluster_area_um2 / bin_area_um2))

    # Adjust min_cluster_size for 16um (4x fewer bins than 8um)
    if resolution == "16um":
        min_cluster_size_bins = max(2, min_cluster_size_bins // 4)

    obs = adata.obs
    xs = obs[col_x].values
    ys = obs[col_y].values

    coord_min_x = int(xs.min())
    coord_max_x = int(xs.max())
    coord_min_y = int(ys.min())
    coord_max_y = int(ys.max())

    if verbose:
        print("=" * 64)
        print("VisiumHD Edge Artifact Detection")
        print("=" * 64)
        print(f"Resolution: {resolution} (bin size = {bin_size_um}x{bin_size_um} um)")
        print(f"Coordinate Range: X[{coord_min_x}-{coord_max_x}], Y[{coord_min_y}-{coord_max_y}]")
        print(f"Buffer Width: {buffer_width_um} um -> {buffer_width_bins} bins")
        print(f"Min Cluster Area: {min_cluster_area_um2} um2 -> {min_cluster_size_bins} bins")

    # Step 1: Outlier detection
    if verbose:
        print("\n--- STEP 1: Outlier Detection ---")

    values = obs[qc_metric].values
    in_tissue = obs[in_tissue_col].values.astype(bool) if in_tissue_col in obs else None

    outlier_binary_col = f"{name}_outlier_binary"
    outlier_flags = is_outlier(
        values,
        subset=in_tissue,
        batch=obs[samples].values,
        nmads=mad_threshold,
        log=True,
    )
    if in_tissue is not None:
        outlier_flags[~in_tissue] = False

    adata.obs[outlier_binary_col] = outlier_flags

    if verbose:
        print(f"  Total outliers detected: {outlier_flags.sum()} bins")

    # Step 2: Edge artifact detection using buffer zone
    if verbose:
        print("\n--- STEP 2: Edge Artifact Detection (Buffer Zone) ---")

    adata.obs[f"{name}_edge"] = False

    sample_list = obs[samples].unique()
    for sample_name in sample_list:
        sample_mask = obs[samples] == sample_name
        sample_x = obs.loc[sample_mask, col_x].values
        sample_y = obs.loc[sample_mask, col_y].values
        sample_outlier = adata.obs.loc[sample_mask, outlier_binary_col].values

        in_buffer = (
            (sample_x <= coord_min_x + buffer_width_bins) |
            (sample_x >= coord_max_x - buffer_width_bins) |
            (sample_y <= coord_min_y + buffer_width_bins) |
            (sample_y >= coord_max_y - buffer_width_bins)
        )

        is_edge = in_buffer & sample_outlier
        adata.obs.loc[sample_mask, f"{name}_edge"] = is_edge

        if verbose:
            print(f"  {sample_name}: {is_edge.sum()} edge artifact bins")

    total_edge = adata.obs[f"{name}_edge"].sum()

    # Step 3: Interior problem area detection
    if verbose:
        print("\n--- STEP 3: Interior Problem Area Detection ---")

    adata.obs[f"{name}_problem_id"] = pd.NA
    adata.obs[f"{name}_problem_id"] = adata.obs[f"{name}_problem_id"].astype("object")
    adata.obs[f"{name}_problem_size"] = 0

    all_problem_dfs = []
    for sample_name in sample_list:
        sample_mask = obs[samples] == sample_name
        sample_obs = obs[sample_mask]

        xyz_df = pd.DataFrame({
            "x": sample_obs[col_x].values,
            "y": sample_obs[col_y].values,
            "outlier": adata.obs.loc[sample_mask, outlier_binary_col].values.astype(float),
        }, index=sample_obs.index)

        prob_df = _problem_areas_visiumhd(
            xyz_df,
            unique_identifier=str(sample_name),
            min_cluster_size=min_cluster_size_bins,
        )

        if len(prob_df) == 0:
            if verbose:
                print(f"  {sample_name}: No interior problem areas detected")
            continue

        # Keep only interior clusters (center not in buffer zone)
        interior_records = []
        for clump_id in prob_df["clump_id"].unique():
            cluster_spots = prob_df[prob_df["clump_id"] == clump_id]["spotcode"].values
            cluster_x = obs.loc[cluster_spots, col_x].values
            cluster_y = obs.loc[cluster_spots, col_y].values

            center_x = np.median(cluster_x)
            center_y = np.median(cluster_y)

            in_buffer = (
                (center_x <= coord_min_x + buffer_width_bins) or
                (center_x >= coord_max_x - buffer_width_bins) or
                (center_y <= coord_min_y + buffer_width_bins) or
                (center_y >= coord_max_y - buffer_width_bins)
            )

            if not in_buffer:
                interior_records.append(
                    prob_df[prob_df["clump_id"] == clump_id]
                )

        if interior_records:
            interior_df = pd.concat(interior_records, ignore_index=True)
            all_problem_dfs.append(interior_df)
            if verbose:
                n_clusters = len(interior_records)
                n_bins = len(interior_df)
                print(f"  {sample_name}: {n_bins} interior bins in {n_clusters} clusters")
        else:
            if verbose:
                print(f"  {sample_name}: No interior problem areas detected")

    if all_problem_dfs:
        combined = pd.concat(all_problem_dfs, ignore_index=True)
        adata.obs.loc[combined["spotcode"], f"{name}_problem_id"] = (
            combined["clump_id"].values
        )
        adata.obs.loc[combined["spotcode"], f"{name}_problem_size"] = (
            combined["clump_size"].values
        )

    total_interior = adata.obs[f"{name}_problem_id"].notna().sum()

    if not keep_intermediate:
        if outlier_binary_col in adata.obs.columns:
            del adata.obs[outlier_binary_col]

    if verbose:
        print("\n" + "=" * 64)
        print("Detection Complete!")
        print("=" * 64)
        total_outliers = outlier_flags.sum()
        print(f"Total outliers detected:      {total_outliers} bins")
        print(f"Edge artifacts (in buffer):   {total_edge} bins")
        print(f"Interior problem areas:       {total_interior} bins")
        if total_outliers > 0:
            rate = 100 * (total_edge + total_interior) / total_outliers
            print(f"Classification rate:          {rate:.1f}%")
        print("=" * 64)

    return adata