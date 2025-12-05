"""
Color utilities for visualization.
"""

import numpy as np
from config.constants import COLOR_PALETTE


def get_color_map(labels: np.ndarray) -> dict:
    """
    Create a color map for cluster labels.
    
    Assigns colors from palette to each unique label.
    Noise label (-1 from DBSCAN) is mapped to gray.
    
    Args:
        labels: Array of cluster labels
        
    Returns:
        Dictionary mapping labels to hex colors
    """
    unique_labels = list(np.unique(labels))
    color_map = {}
    
    for i, lbl in enumerate(unique_labels):
        if lbl == -1:
            color_map[lbl] = "#777777"  # Gray for noise
        else:
            color_map[lbl] = COLOR_PALETTE[i % len(COLOR_PALETTE)]
    
    return color_map
