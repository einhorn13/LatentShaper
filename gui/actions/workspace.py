# gui/actions/workspace.py

import gradio as gr
import os
from core.resources import ResourceMonitor
from ..context import workspace, config
from .common import validate_and_fix_filename

def refresh_workspace_ui():
    models = []
    for name in workspace.list_models():
        m = workspace.get_model(name)
        rank = m.info.get("rank", "?")
        size = ResourceMonitor.format_bytes(m.size_bytes)
        models.append([name, rank, size])
    return models

def load_files_to_workspace(files):
    if not files: return gr.update(), None
    if isinstance(files, str): files = [files]
    
    paths = []
    for f in files:
        if isinstance(f, str): paths.append(f)
        elif hasattr(f, 'name'): paths.append(f.name)
    
    for f in paths:
        try:
            workspace.load_from_disk(f)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            gr.Warning(f"Failed to load {os.path.basename(f)}: {e}")
    
    return refresh_workspace_ui(), None

def load_from_server_path(path):
    """Loads a model directly from a server path (Disk Dropdown)."""
    if not path: return gr.update()
    try:
        workspace.load_from_disk(path)
        gr.Info(f"Loaded {os.path.basename(path)} into RAM")
    except Exception as e:
        gr.Error(f"Load failed: {e}")
    return refresh_workspace_ui()

def save_workspace_model(name):
    if not name or not workspace.exists(name):
        return gr.Warning("Select a model first.")
    
    try:
        base_name = name
        if base_name.lower().endswith(".safetensors"):
            base_name = base_name[:-12]
            
        final_name = validate_and_fix_filename(base_name, is_workspace=False)
        path = os.path.join(config.output_dir, final_name)
        
        workspace.save_to_disk(name, path)
        gr.Info(f"Saved to {path}")
    except ValueError as ve:
        gr.Warning(str(ve))
    except Exception as e:
        gr.Error(f"Save failed: {e}")

def delete_workspace_model(name):
    if not name: return gr.update()
    workspace.delete_model(name)
    return refresh_workspace_ui()

def handle_sidebar_select(evt: gr.SelectData):
    """
    Handles sidebar selection to auto-fill all main tabs.
    Returns 16 outputs to sync UI state.
    """
    row_idx = evt.index[0]
    all_models = workspace.list_models()
    
    selected_name = ""
    if 0 <= row_idx < len(all_models):
        selected_name = all_models[row_idx]
    
    # Default empty update
    if not selected_name:
        # 5 tabs with (drop, upload) + 2 single drops + 4 output names
        return [gr.update(), None] * 5 + [gr.update(), gr.update()] + [gr.update()] * 4

    base_clean = selected_name
    if base_clean.lower().endswith(".safetensors"):
        base_clean = base_clean[:-12]

    name_extracted = f"{base_clean}_extracted"
    name_resized = f"{base_clean}_resized"
    name_morphed = f"{base_clean}_morphed"
    name_utils = f"{base_clean}_opt"

    return [
        # 1. Analyze
        gr.update(value=selected_name), None,
        # 2. Extract Base
        gr.update(value=selected_name), None,
        # 3. Resize
        gr.update(value=[selected_name]), None,
        # 4. Morph
        gr.update(value=[selected_name]), None,
        # 5. Utils (NEW)
        gr.update(value=[selected_name]), None,
        # 6. Merge
        gr.update(value=selected_name),
        # 7. Metadata
        gr.update(value=selected_name),
        
        # Output Names
        gr.update(value=name_extracted),
        gr.update(value=name_resized),
        gr.update(value=name_morphed),
        gr.update(value=name_utils) # (NEW)
    ]

def load_settings(): return config.get("output_dir", "output")
def save_settings(output_dir):
    if not output_dir.strip(): return "Error: Path empty."
    config.set("output_dir", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return "Settings Saved."