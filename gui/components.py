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
    disk_dir_type: str = "loras"
):
    """
    Creates a unified input source selector with Refresh and Clear capabilities.
    """
    config = ConfigManager()
    disk_path = config.loras_dir if disk_dir_type == "loras" else config.checkpoints_dir
    disk_files = SafeStreamer.scan_directory(disk_path) if disk_path else []
    
    components = {}
    
    with gr.Group():
        with gr.Row():
            gr.Markdown(f"#### {label_prefix}")
            # Action buttons for the whole group
            with gr.Row():
                components["refresh"] = gr.Button("🔄", size="sm", variant="secondary", scale=0)
                components["clear"] = gr.Button("🗑️", size="sm", variant="secondary", scale=0)
        
        with gr.Row():
            if show_ws:
                components["ws"] = gr.Dropdown(
                    label="Workspace (RAM)", 
                    choices=[], 
                    multiselect=multiselect, 
                    scale=3
                )
            else:
                components["ws"] = None

            if show_disk:
                components["disk"] = gr.Dropdown(
                    label=f"Disk ({disk_path or 'Not Set'})", 
                    choices=disk_files, 
                    multiselect=multiselect,
                    scale=3,
                    allow_custom_value=True
                )
            else:
                components["disk"] = None

        if show_upload:
            with gr.Accordion("Upload", open=False):
                components["upload"] = gr.File(
                    label="Upload File", 
                    file_count="multiple" if multiselect else "single",
                    type="filepath"
                )
        else:
            components["upload"] = None

    # --- Internal Wiring ---
    
    # Refresh Logic
    if show_disk and components["refresh"]:
        def _refresh():
            current_path = config.loras_dir if disk_dir_type == "loras" else config.checkpoints_dir
            files = SafeStreamer.scan_directory(current_path, force_refresh=True)
            return gr.update(choices=files)
        components["refresh"].click(_refresh, outputs=[components["disk"]])

    # Clear Logic
    if components["clear"]:
        def _clear():
            updates = []
            if show_ws: updates.append(gr.update(value=[] if multiselect else None))
            if show_disk: updates.append(gr.update(value=[] if multiselect else None))
            if show_upload: updates.append(gr.update(value=None))
            return updates

        clear_outputs = []
        if show_ws: clear_outputs.append(components["ws"])
        if show_disk: clear_outputs.append(components["disk"])
        if show_upload: clear_outputs.append(components["upload"])
        
        components["clear"].click(_clear, outputs=clear_outputs)

    return components