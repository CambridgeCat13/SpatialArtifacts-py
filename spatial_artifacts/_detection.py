import numpy as np
import pandas as pd
from scipy.ndimage import label

from ._morphology import focal_transformations


def _coords_to_grid(xyz_df, outlier_col):
    """
    Convert spot coordinates and outlier flags to a 2D numpy grid.
    Equivalent to terra::rast() + value assignment in R.

    Parameters
    ----------
    xyz_df : pd.DataFrame
        DataFrame with columns array_row, array_col, and outlier_col.
        Index = spot barcodes.
    outlier_col : str
        Column name for binary outlier indicator.

    Returns
    -------
    grid : np.ndarray
        2D array where 1 = outlier, 0 = normal, np.nan = empty cell.
    row_offset : int
    col_offset : int
    key_grid : np.ndarray
        2D array mapping grid positions to numeric spot indices (1-based).
    spot_index : list
        List of spot barcodes in order (1-based index into key_grid).
    """
    rows = xyz_df["array_row"].values
    cols = xyz_df["array_col"].values
    outliers = xyz_df[outlier_col].fillna(False).astype(float).values

    row_offset = int(rows.min())
    col_offset = int(cols.min())

    n_rows = int(rows.max()) - row_offset + 1
    n_cols = int(cols.max()) - col_offset + 1

    grid = np.full((n_rows, n_cols), np.nan)
    key_grid = np.full((n_rows, n_cols), np.nan)

    spot_index = list(xyz_df.index)

    for i, (r, c, o) in enumerate(zip(rows, cols, outliers)):
        ri = int(r) - row_offset
        ci = int(c) - col_offset
        grid[ri, ci] = o
        key_grid[ri, ci] = i + 1  # 1-based

    return grid, row_offset, col_offset, key_grid, spot_index


def _grid_indices_to_barcodes(indices_rc, key_grid, spot_index):
    """
    Map (row, col) grid indices back to spot barcodes.
    Equivalent to R lookupKey().

    Parameters
    ----------
    indices_rc : list of (row, col) tuples
    key_grid : np.ndarray
    spot_index : list of barcodes

    Returns
    -------
    list of barcodes
    """
    barcodes = []
    for r, c in indices_rc:
        if 0 <= r < key_grid.shape[0] and 0 <= c < key_grid.shape[1]:
            val = key_grid[r, c]
            if not np.isnan(val):
                barcodes.append(spot_index[int(val) - 1])
    return barcodes


def clump_edges(xyz_df, off_tissue, outlier_col,
                shifted=False, edge_threshold=0.75, min_cluster_size=40):
    """
    Detect edge artifact spots in a single sample.
    Equivalent to R clumpEdges().

    Parameters
    ----------
    xyz_df : pd.DataFrame
        DataFrame with columns array_row, array_col, and outlier_col.
        Index = spot barcodes.
    off_tissue : list
        Barcodes of off-tissue spots to exclude from results.
    outlier_col : str
        Column name for binary outlier indicator.
    shifted : bool
        Whether to apply coordinate adjustment for hexagonal arrays (default: False).
    edge_threshold : float
        Minimum proportion of border coverage for edge detection (default: 0.75).
    min_cluster_size : int
        Minimum cluster size for morphological cleaning (default: 40).

    Returns
    -------
    list of str
        Barcodes of spots classified as edge artifacts.
    """
    xyz_df = xyz_df.copy()
    xyz_df[outlier_col] = xyz_df[outlier_col].fillna(False)

    if xyz_df[outlier_col].sum() == 0:
        return []

    if shifted:
        odd_cols = xyz_df["array_col"] % 2 == 1
        xyz_df.loc[odd_cols, "array_col"] -= 1

    grid, row_offset, col_offset, key_grid, spot_index = _coords_to_grid(xyz_df, outlier_col)

    processed = focal_transformations(grid, min_cluster_size=min_cluster_size)

    binary = np.where(np.isnan(processed), 0, processed).astype(int)
    labeled, num_features = label(binary, structure=np.ones((3, 3)))

    if num_features == 0:
        return []

    # Replace 0 with nan for coverage calculation
    rast = np.where(grid == 0, np.nan, grid)
    clumps = labeled.astype(float)
    clumps[labeled == 0] = np.nan

    edge_clumps = set()

    # Method 1: coverage-based (large continuous edges)
    for border_slice, rast_slice in [
        (clumps[0, :],        rast[0, :]),         # north
        (clumps[-1, :],       rast[-1, :]),         # south
        (clumps[:, 0],        rast[:, 0]),          # west
        (clumps[:, -1],       rast[:, -1]),         # east
    ]:
        total = np.sum(~np.isnan(rast_slice))
        if total > 0:
            coverage = np.sum(~np.isnan(border_slice)) / total
            if coverage >= edge_threshold:
                edge_clumps.update(
                    int(v) for v in border_slice[~np.isnan(border_slice)]
                )

    # Method 2: any cluster touching edge
    for border_slice in [
        clumps[0, :], clumps[-1, :], clumps[:, 0], clumps[:, -1]
    ]:
        edge_clumps.update(
            int(v) for v in border_slice[~np.isnan(border_slice)]
        )

    if not edge_clumps:
        return []

    indices_rc = []
    for clump_id in edge_clumps:
        positions = np.argwhere(labeled == clump_id)
        indices_rc.extend(map(tuple, positions))

    barcodes = _grid_indices_to_barcodes(indices_rc, key_grid, spot_index)
    result = [b for b in barcodes if b not in off_tissue]

    return result


def problem_areas(xyz_df, off_tissue, outlier_col,
                  unique_identifier=None, shifted=False, min_cluster_size=40):
    """
    Identify all connected clusters of outlier spots in a single sample.
    Equivalent to R problemAreas().

    Parameters
    ----------
    xyz_df : pd.DataFrame
        DataFrame with columns array_row, array_col, and outlier_col.
        Index = spot barcodes.
    off_tissue : list
        Barcodes of off-tissue spots to exclude from results.
    outlier_col : str
        Column name for binary outlier indicator.
    unique_identifier : str, optional
        Prefix for cluster IDs (default: "X").
    shifted : bool
        Whether to apply coordinate adjustment for hexagonal arrays (default: False).
    min_cluster_size : int
        Minimum cluster size for morphological cleaning (default: 40).

    Returns
    -------
    pd.DataFrame with columns:
        spotcode : str
        clump_id : str
        clump_size : int
    """
    empty = pd.DataFrame(columns=["spotcode", "clump_id", "clump_size"])

    xyz_df = xyz_df.copy()
    xyz_df[outlier_col] = xyz_df[outlier_col].fillna(False)

    if xyz_df[outlier_col].sum() == 0:
        return empty

    if shifted:
        odd_cols = xyz_df["array_col"] % 2 == 1
        xyz_df.loc[odd_cols, "array_col"] -= 1

    if unique_identifier is None:
        unique_identifier = "X"

    grid, row_offset, col_offset, key_grid, spot_index = _coords_to_grid(xyz_df, outlier_col)

    processed = focal_transformations(grid, min_cluster_size=min_cluster_size)

    binary = np.where(np.isnan(processed), 0, processed).astype(int)
    labeled, num_features = label(binary, structure=np.ones((3, 3)))

    if num_features == 0:
        return empty

    records = []
    for clump_id in range(1, num_features + 1):
        positions = np.argwhere(labeled == clump_id)
        clump_size = len(positions)
        barcodes = _grid_indices_to_barcodes(
            map(tuple, positions), key_grid, spot_index
        )
        for bc in barcodes:
            if bc not in off_tissue:
                records.append({
                    "spotcode": bc,
                    "clump_id": f"{unique_identifier}_{clump_id}",
                    "clump_size": clump_size
                })

    if not records:
        return empty

    return pd.DataFrame(records)