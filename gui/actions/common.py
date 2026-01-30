# gui/actions/common.py

import gradio as gr
import os
import re
from ..context import workspace

def _make_desc(action: str, files: list, extra: str = "") -> str:
    if not files: return action
    # Handle both file objects and strings
    first = files[0]
    name = first.name if hasattr(first, 'name') else str(first)
    base = os.path.basename(name)
    
    count = len(files)
    suffix = f" (+{count-1})" if count > 1 else ""
    return f"{action}: {base}{suffix} {extra}"

def resolve_path_priority(ws_input, disk_input, upload_input) -> list[str]:
    """
    Resolves inputs from 3 sources into a single list of valid paths/names.
    Priority: Workspace > Disk > Upload.
    Handles Gradio 'filepath' type (strings) and 'file' type (objects).
    """
    results = []
    
    # 1. Workspace
    if ws_input:
        if isinstance(ws_input, list): results.extend(ws_input)
        else: results.append(ws_input)
        
    # 2. Disk
    if disk_input:
        if isinstance(disk_input, list): results.extend(disk_input)
        else: results.append(disk_input)
        
    # 3. Upload
    if upload_input:
        # Normalize to list
        u_files = upload_input if isinstance(upload_input, list) else [upload_input]
        
        for f in u_files:
            if isinstance(f, str):
                results.append(f)
            elif hasattr(f, 'name'):
                results.append(f.name)
            else:
                results.append(str(f))
            
    # Filter duplicates and empty strings
    unique_results = []
    seen = set()
    for p in results:
        if p and p not in seen:
            unique_results.append(p)
            seen.add(p)
            
    return unique_results

def smart_resolve_inputs(file_input, dropdown_input):
    # Deprecated wrapper
    return resolve_path_priority(dropdown_input, None, file_input)

def suggest_output_name(ws_in, disk_in, up_in, suffix: str):
    inputs = resolve_path_priority(ws_in, disk_in, up_in)
    if not inputs: return gr.update()
    
    first_input = inputs[0]
    base_name = os.path.basename(first_input)
    name, _ = os.path.splitext(base_name)
    
    if name.lower().endswith(".safetensors"): name = name[:-12]
    return f"{name}{suffix}"

def get_workspace_choices(): return workspace.list_models()
def toggle_output_input(is_workspace_save): return gr.update(label="Result Name (Workspace)" if is_workspace_save else "Output Filename (Disk)")

def reset_morph_ui():
    return [
        1.0, 1.0, 1.0, 1.0, False, 1.0, 1.0, 1.0, True, 
        False, 0.1, False, False, False, 0.1, 0, 0, 
        False, 0.01, "", False, 0.3, 0.6
    ]

def ui_toggle_enabled(is_checked): return gr.update(interactive=is_checked)
def ui_toggle_disabled(is_checked): return gr.update(interactive=not is_checked)
def ui_toggle_ties(algorithm):
    is_ties = "TIES" in algorithm if algorithm else False
    return gr.update(visible=is_ties)

def sanitize_filename(name: str) -> str:
    if not name: return ""
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    return name.strip()

def validate_and_fix_filename(name: str, is_workspace: bool) -> str:
    clean = sanitize_filename(name)
    if not clean: raise ValueError("Filename cannot be empty or contain only special characters.")
    if is_workspace: return clean
    else:
        if not clean.lower().endswith(".safetensors"): clean += ".safetensors"
        return clean