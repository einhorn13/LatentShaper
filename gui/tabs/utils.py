# gui/tabs/utils.py

import gradio as gr
from gui.actions import toggle_output_input
from gui.actions.submission import submit_utils
from gui.actions.common import suggest_output_name
from gui.components import create_source_selector

def create_utils_tab():
    with gr.TabItem("Utils", id="Utils"):
        gr.Markdown("### 🛠️ LoRA Maintenance Tools")
        gr.Markdown("Convert formats, standardize keys, and rescale Alpha values.")
        
        sel = create_source_selector("Input Model(s)", multiselect=True, disk_dir_type="loras")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 📦 Converter")
                ut_prec = gr.Radio(["Keep", "FP32", "FP16", "BF16"], value="Keep", label="Target Precision", 
                                  info="BF16 is recommended for S3-DiT/Turbo.")
                ut_norm = gr.Checkbox(label="Fix Key Names (ComfyUI format)", value=False, 
                                     info="Fixes broken layers (e.g. layers_0 -> layers.0) and attention blocks.")
            
            with gr.Column():
                gr.Markdown("#### 📐 Alpha Rescaler")
                ut_alpha_mode = gr.Radio(["Keep", "Set Custom", "Sync with Rank"], value="Keep", label="Alpha Mode")
                ut_alpha_val = gr.Number(label="Custom Alpha", value=1.0, precision=1, visible=False)
                
                def toggle_alpha(mode):
                    return gr.update(visible=(mode == "Set Custom"))
                
                ut_alpha_mode.change(toggle_alpha, ut_alpha_mode, ut_alpha_val)

        with gr.Row():
            ut_save_ws = gr.Checkbox(label="Save to Workspace", value=True)
            ut_out = gr.Textbox(label="Output Name")
        
        ut_save_ws.change(toggle_output_input, ut_save_ws, ut_out)
        
        def _update_name(ws, disk, up): return suggest_output_name(ws, disk, up, "_opt")
        triggers = [t for t in [sel["ws"], sel["disk"], sel["upload"]] if t is not None]
        for t in triggers:
            t.change(_update_name, [sel["ws"], sel["disk"], sel["upload"]], ut_out)
            
        gr.Button("Run Utilities", variant="primary").click(
            submit_utils,
            [
                sel["ws"], sel["disk"], sel["upload"], ut_out,
                ut_prec, ut_norm, ut_alpha_mode, ut_alpha_val, ut_save_ws
            ],
            None
        )

    return {
        "ws": sel["ws"], "disk": sel["disk"], "upload": sel["upload"], "out_name": ut_out
    }