import numpy as np
import pandas as pd


def is_outlier(values, subset=None, batch=None, nmads=3, log=True):
    """
    Detect lower-tail outliers using MAD-based thresholding.
    Equivalent to scuttle::isOutlier(type="lower") in R.

    Parameters
    ----------
    values : array-like
        Numeric QC metric values (e.g. total counts, detected genes).
    subset : array-like of bool, optional
        Boolean mask indicating which observations to use for threshold
        calculation (e.g. in_tissue == True). All observations are scored
        but thresholds are computed only from the subset (default: all).
    batch : array-like, optional
        Batch labels for per-batch threshold calculation (e.g. sample_id
        or slide). If None, a single global threshold is used (default: None).
    nmads : float
        Number of MADs below the median to use as the lower threshold
        (default: 3).
    log : bool
        Whether to apply log10(x + 1) transformation before computing
        thresholds (default: True).

    Returns
    -------
    np.ndarray of bool
        True where a value is a lower-tail outlier.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)

    if subset is None:
        subset = np.ones(n, dtype=bool)
    else:
        subset = np.asarray(subset, dtype=bool)

    transformed = np.log10(values + 1) if log else values.copy()
    outlier_flags = np.zeros(n, dtype=bool)

    if batch is None:
        batch = np.zeros(n, dtype=int)
    else:
        batch = np.asarray(batch)

    for batch_val in np.unique(batch):
        batch_mask = batch == batch_val
        use_mask = batch_mask & subset

        if use_mask.sum() == 0:
            continue

        batch_vals = transformed[use_mask]
        median_val = np.nanmedian(batch_vals)
        mad_val = np.nanmedian(np.abs(batch_vals - median_val)) * 1.4826

        threshold = median_val - nmads * mad_val
        outlier_flags[batch_mask] = transformed[batch_mask] < threshold

    return outlier_flags


def compute_outlier_binary(adata, qc_metric, samples, nmads=3,
                           batch_var="both", in_tissue_col="in_tissue"):
    """
    Compute binary outlier flags per spot, optionally using slide and/or
    sample as batch variables.
    Equivalent to the outlier detection block in R detectEdgeArtifacts_Visium().

    Parameters
    ----------
    adata : AnnData
        AnnData object with QC metrics in adata.obs.
    qc_metric : str
        Column name of the QC metric in adata.obs.
    samples : str
        Column name of the sample ID in adata.obs.
    nmads : float
        MAD threshold (default: 3).
    batch_var : str
        One of "slide", "sample_id", or "both" (default: "both").
    in_tissue_col : str
        Column name for in-tissue indicator (default: "in_tissue").

    Returns
    -------
    np.ndarray of bool
        Binary outlier flags for each spot.
    """
    obs = adata.obs
    values = obs[qc_metric].values
    in_tissue = obs[in_tissue_col].values.astype(bool) if in_tissue_col in obs else None

    outlier_slide = np.zeros(len(obs), dtype=bool)
    outlier_sample = np.zeros(len(obs), dtype=bool)

    if batch_var in ("slide", "both"):
        if "slide" in obs.columns:
            outlier_slide = is_outlier(
                values,
                subset=in_tissue,
                batch=obs["slide"].values,
                nmads=nmads,
                log=True
            )

    if batch_var in ("sample_id", "both"):
        outlier_sample = is_outlier(
            values,
            subset=in_tissue,
            batch=obs[samples].values,
            nmads=nmads,
            log=True
        )

    if batch_var == "both":
        outlier_binary = outlier_slide | outlier_sample
    elif batch_var == "slide":
        outlier_binary = outlier_slide
    else:
        outlier_binary = outlier_sample

    if in_tissue is not None:
        outlier_binary[~in_tissue] = False

    return outlier_binary