
import gradio as gr
import os
from core.resources import ResourceMonitor
from ..context import workspace, config
from .common import validate_and_fix_filename

def refresh_workspace_ui():
    models =[]
    for name in workspace.list_models():
        m = workspace.get_model(name)
        rank = m.info.get("rank", "?")
        arch = m.info.get("arch", "Unknown")
        
        # Компактное отображение архитектуры для экономии места в таблице
        if "FLUX" in arch: short_arch = "FLUX"
        elif "S3-DiT" in arch: short_arch = "S3-DiT"
        elif "WAN" in arch: short_arch = "WAN"
        elif "LTX" in arch: short_arch = "LTX"
        elif "SDXL" in arch: short_arch = "SDXL"
        else: short_arch = arch[:8] + ".." if len(arch) > 10 else arch
            
        size = ResourceMonitor.format_bytes(m.size_bytes)
        models.append([name, rank, short_arch, size])
    return models

def load_files_to_workspace(files):
    if not files: return gr.update(), None
    if isinstance(files, str): files = [files]
    paths =[]
    for f in files:
        if isinstance(f, str): paths.append(f)
        elif hasattr(f, 'name'): paths.append(f.name)
    
    for f in paths:
        try: workspace.load_from_disk(f)
        except Exception as e: gr.Warning(f"Failed to load {os.path.basename(f)}: {e}")
    return refresh_workspace_ui(), None

def load_from_server_path(path):
    if not path: return gr.update()
    try:
        workspace.load_from_disk(path)
        gr.Info(f"Loaded {os.path.basename(path)} into RAM")
    except Exception as e:
        gr.Error(f"Load failed: {e}")
    return refresh_workspace_ui()

def save_workspace_model(name):
    if not name or not workspace.exists(name): return gr.Warning("Select a model first.")
    try:
        base_name = name
        if base_name.lower().endswith(".safetensors"): base_name = base_name[:-12]
        final_name = validate_and_fix_filename(base_name, is_workspace=False)
        path = os.path.join(config.output_dir, final_name)
        workspace.save_to_disk(name, path)
        gr.Info(f"Saved to {path}")
    except ValueError as ve: gr.Warning(str(ve))
    except Exception as e: gr.Error(f"Save failed: {e}")

def delete_workspace_model(name):
    if not name: return gr.update()
    workspace.delete_model(name)
    return refresh_workspace_ui()

def handle_sidebar_select(evt: gr.SelectData):
    row_idx = evt.index[0]
    all_models = workspace.list_models()
    
    selected_name = ""
    if 0 <= row_idx < len(all_models):
        selected_name = all_models[row_idx]
    
    if not selected_name:
        return (
            [gr.update(), None] * 5 + [gr.update(), gr.update()] + 
            [gr.update()] * 4 + ["⚪ **Architecture:** None Selected"] + 
            [gr.update(visible=True)] * 3 + [gr.update(visible=False)] * 2
        )

    base_clean = selected_name
    if base_clean.lower().endswith(".safetensors"):
        base_clean = base_clean[:-12]

    name_extracted = f"{base_clean}_extracted"
    name_resized = f"{base_clean}_resized"
    name_morphed = f"{base_clean}_morphed"
    name_utils = f"{base_clean}_opt"

    model = workspace.get_model(selected_name)
    if model:
        from core.model_specs import ModelRegistry
        from core.architectures.base import UnknownArchitecture
        spec = ModelRegistry.get_spec(list(model.assembly.modules.keys()))
        
        if isinstance(spec, UnknownArchitecture):
            arch_text = f"🔴 **Architecture:** {spec.name} (Unsupported)"
        else:
            arch_text = f"🟢 **Architecture:** {spec.name}"
        regions = spec.get_regions()
    else:
        arch_text = "⚪ **Architecture:** Unknown"
        regions = ["IN", "MID", "OUT"]

    return[
        gr.update(value=selected_name), None,
        gr.update(value=selected_name), None,
        gr.update(value=[selected_name]), None,
        gr.update(value=[selected_name]), None,
        gr.update(value=[selected_name]), None,
        gr.update(value=selected_name),
        gr.update(value=selected_name),
        
        gr.update(value=name_extracted),
        gr.update(value=name_resized),
        gr.update(value=name_morphed),
        gr.update(value=name_utils),
        
        arch_text,
        gr.update(visible="IN" in regions),
        gr.update(visible="MID" in regions),
        gr.update(visible="OUT" in regions),
        gr.update(visible="ADAPTER" in regions),
        gr.update(visible="OTHER" in regions)
    ]

def load_settings(): return config.get("output_dir", "output")
def save_settings(output_dir):
    if not output_dir.strip(): return "Error: Path empty."
    config.set("output_dir", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return "Settings Saved."