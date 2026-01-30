# gui/components.py

import gradio as gr
from core.config import ConfigManager
from core.io_manager import SafeStreamer

def create_source_selector(
    label_prefix: str = "Model", 
    show_ws: bool = True, 
    show_disk: bool = True, 
    show_upload: bool = True,
    multiselect: bool = False,
    disk_dir_type: str = "loras" # 'loras' or 'checkpoints'
):
    """
    Creates a unified input source selector (Workspace + Disk + Upload).
    Returns a dict of components to be wired up.
    """
    config = ConfigManager()
    
    # Determine initial disk path
    disk_path = config.loras_dir if disk_dir_type == "loras" else config.checkpoints_dir
    disk_files = SafeStreamer.scan_directory(disk_path) if disk_path else []
    
    components = {}
    
    with gr.Group():
        gr.Markdown(f"#### {label_prefix} Source")
        
        with gr.Row():
            if show_ws:
                components["ws"] = gr.Dropdown(
                    label="Workspace (RAM)", 
                    choices=[], 
                    multiselect=multiselect, 
                    scale=3,
                    elem_id=f"{label_prefix}_ws"
                )
            else:
                components["ws"] = None

            if show_disk:
                components["disk"] = gr.Dropdown(
                    label=f"Disk ({disk_path or 'Not Set'})", 
                    choices=disk_files, 
                    multiselect=multiselect,
                    scale=3,
                    allow_custom_value=True,
                    elem_id=f"{label_prefix}_disk"
                )
                components["refresh"] = gr.Button("🔄", size="sm", scale=0)
            else:
                components["disk"] = None
                components["refresh"] = None

        if show_upload:
            with gr.Accordion("Upload from Computer", open=False):
                components["upload"] = gr.File(
                    label="Upload File", 
                    file_count="multiple" if multiselect else "single",
                    type="filepath"
                )
        else:
            components["upload"] = None

    # Internal wiring for Refresh
    if show_disk and components["refresh"]:
        def _refresh():
            # Re-read config in case it changed
            current_path = config.loras_dir if disk_dir_type == "loras" else config.checkpoints_dir
            files = SafeStreamer.scan_directory(current_path, force_refresh=True)
            return gr.update(label=f"Disk ({current_path or 'Not Set'})", choices=files)
        
        components["refresh"].click(_refresh, outputs=[components["disk"]])

    return components