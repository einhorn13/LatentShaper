# __init__.py

from .nodes_pipeline import (
    LS_Loader, 
    LS_EQ, 
    LS_Filters,
    LS_Dynamics, 
    LS_Eraser, 
    LS_Metadata,
    LS_Analyzer,
    LS_Save,
    LS_Apply
)
from .nodes_merging import (
    LS_Merger
)

NODE_CLASS_MAPPINGS = {
    # Pipeline
    "LS_Loader": LS_Loader,
    "LS_EQ": LS_EQ,
    "LS_Filters": LS_Filters,
    "LS_Dynamics": LS_Dynamics,
    "LS_Eraser": LS_Eraser,
    "LS_Metadata": LS_Metadata,
    "LS_Analyzer": LS_Analyzer,
    "LS_Save": LS_Save,
    "LS_Apply": LS_Apply,
    # Merging
    "LS_Merger": LS_Merger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Pipeline
    "LS_Loader": "LS Loader (Raw)",
    "LS_EQ": "LS EQ (Structure)",
    "LS_Filters": "LS Filters (Signal)",
    "LS_Dynamics": "LS Dynamics (Rank)",
    "LS_Eraser": "LS Eraser (Block/Concept)",
    "LS_Metadata": "LS Metadata Editor",
    "LS_Analyzer": "LS Analyzer (Visual)",
    "LS_Save": "LS Save to Disk",
    "LS_Apply": "LS Apply to Model",
    # Merging
    "LS_Merger": "LS Merger (6-Input)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']