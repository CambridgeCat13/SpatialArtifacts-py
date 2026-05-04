from .detect_visium import detect_edge_artifacts_visium
from .detect_visiumhd import detect_edge_artifacts_visiumhd
from .classify import classify_edge_artifacts

def detect_edge_artifacts(adata, platform, **kwargs):
    platform = platform.lower()
    if platform == "visium":
        return detect_edge_artifacts_visium(adata, **kwargs)
    elif platform == "visiumhd":
        return detect_edge_artifacts_visiumhd(adata, **kwargs)
    else:
        raise ValueError(f"Unknown platform: '{platform}'. Choose 'visium' or 'visiumhd'.")

__all__ = [
    "detect_edge_artifacts",
    "detect_edge_artifacts_visium",
    "detect_edge_artifacts_visiumhd",
    "classify_edge_artifacts",
]