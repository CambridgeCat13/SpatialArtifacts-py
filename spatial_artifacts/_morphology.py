import numpy as np
from scipy.ndimage import generic_filter, label


def _my_fill(x):
    """
    Fill center pixel if completely surrounded by outliers in a 3x3 window.
    Equivalent to R my_fill().
    x is a flat array of length 9; x[4] is the center.
    NaN represents off-tissue (NA in R).
    """
    center = x[4]
    if np.isnan(center):
        return np.nan
    if center == 1:
        return 1.0
    neighbors = np.copy(x)
    neighbors[4] = np.nan
    neighbors = np.where(np.isnan(neighbors), 1.0, neighbors)
    if np.sum(neighbors) == 8:
        return 1.0
    return 0.0


def _my_fill_star(x):
    """
    Fill center pixel if all four cardinal neighbors are outliers.
    Equivalent to R my_fill_star().
    x is a flat array of length 9; x[4] is the center.
    Star pattern: only N/S/E/W (indices 1, 3, 5, 7) are considered.
    """
    center = x[4]
    if np.isnan(center):
        return np.nan
    if center == 1:
        return 1.0
    cardinal = [x[1], x[3], x[5], x[7]]
    if np.nansum(cardinal) == 4:
        return 1.0
    return 0.0


def _my_outline(x):
    """
    Fill center pixel if completely outlined by outliers in a 5x5 window perimeter.
    Equivalent to R my_outline().
    x is a flat array of length 25; x[12] is the center.
    Border indices (0-indexed): top row 0-4, bottom row 20-24,
    left col 5,10,15, right col 9,14,19.
    """
    center = x[12]
    if np.isnan(center):
        return np.nan
    if center == 1:
        return 1.0
    border_idx = list(range(0, 5)) + list(range(20, 25)) + [5, 10, 15] + [9, 14, 19]
    border = np.copy(x[border_idx])
    border = np.where(np.isnan(border), 1.0, border)
    if np.sum(border) == 16:
        return 1.0
    return 0.0


def focal_transformations(grid, min_cluster_size=40):
    """
    Apply sequential morphological operations to connect and clean outlier regions.
    Equivalent to R focal_transformations().

    Parameters
    ----------
    grid : np.ndarray
        2D binary array where 1 = outlier, 0 = normal, np.nan = off-tissue.
    min_cluster_size : int
        Minimum size for isolated normal-region clusters. Clusters smaller
        than this threshold will be filled in as outliers (default: 40).

    Returns
    -------
    np.ndarray
        Processed 2D array with cleaned and connected outlier regions.
    """
    # Step 1: 3x3 fill
    r2 = generic_filter(grid, _my_fill, size=3, mode="constant", cval=1.0)

    # Step 2: 5x5 outline
    r3 = generic_filter(r2, _my_outline, size=5, mode="constant", cval=1.0)

    # Step 3: star pattern (3x3 but only cardinal directions)
    r3_s = generic_filter(r3, _my_fill_star, size=3, mode="constant", cval=1.0)

    # Step 4: remove small holes in the outlier region
    # Invert: find connected components of non-outlier areas
    rev = np.where(r3_s == 1, 0.0, 1.0)
    rev = np.where(np.isnan(r3_s), np.nan, rev)

    rev_binary = np.where(np.isnan(rev), 0, rev).astype(int)
    labeled, num_features = label(rev_binary, structure=np.ones((3, 3)))

    if num_features > 0:
        flip_clumps = []
        for clump_id in range(1, num_features + 1):
            clump_size = np.sum(labeled == clump_id)
            if clump_size < min_cluster_size:
                flip_clumps.append(clump_id)

        if flip_clumps:
            r4 = np.copy(r3_s)
            for clump_id in flip_clumps:
                r4[labeled == clump_id] = 1.0
            return r4

    return r3_s