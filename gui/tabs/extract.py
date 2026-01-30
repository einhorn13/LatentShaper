# gui/tabs/extract.py

import gradio as gr
from gui.actions import toggle_output_input, submit_extract
from gui.actions.common import suggest_output_name
from gui.components import create_source_selector

def create_extract_tab():
    with gr.TabItem("Extract", id="Extract"):
        gr.Markdown("### ⛏️ LoRA Extraction")
        gr.Markdown("**Formula:** `Fine-Tuned` - `Base` = `Delta` (SVD Approximation)")
        
        # 1. Base Model Selector (Checkpoints Dir)
        base_sel = create_source_selector("Base Model", multiselect=False, disk_dir_type="checkpoints", show_upload=False)

        # 2. Tuned Model Selector (LoRAs Dir)
        tuned_sel = create_source_selector("Fine-Tuned Model(s)", multiselect=True, disk_dir_type="loras", show_upload=True)

        # --- SETTINGS ---
        with gr.Row():
            with gr.Column():
                ex_rank = gr.Slider(4, 256, 64, step=4, label="Rank", info="Higher = more detail.")
                ex_thresh = gr.Slider(0.0, 0.01, 0.0001, step=0.0001, label="Noise Gate", 
                                     info="Ignores small weight changes (prevents BF16 noise). Try 1e-4.")
            
            with gr.Column():
                ex_save_ws = gr.Checkbox(label="Save to Workspace", value=True)
                ex_out = gr.Textbox(label="Output Name", placeholder="extracted_lora")

        ex_save_ws.change(toggle_output_input, ex_save_ws, ex_out)
        
        # Auto-naming logic
        def _update_name(ws, disk, up):
            return suggest_output_name(ws, disk, up, "_extracted")

        # FIX: Filter None triggers
        triggers = [t for t in [tuned_sel["ws"], tuned_sel["disk"], tuned_sel["upload"]] if t is not None]
        for t in triggers:
            t.change(_update_name, [tuned_sel["ws"], tuned_sel["disk"], tuned_sel["upload"]], ex_out)

        gr.Button("Extract LoRA", variant="primary").click(
            submit_extract, 
            [
                base_sel["ws"], base_sel["disk"], 
                tuned_sel["ws"], tuned_sel["disk"], tuned_sel["upload"], 
                ex_rank, ex_thresh, ex_out, ex_save_ws
            ], 
            None
        )

    return {
        "base_ws": base_sel["ws"], "base_disk": base_sel["disk"],
        "tuned_ws": tuned_sel["ws"], "tuned_disk": tuned_sel["disk"], "tuned_upload": tuned_sel["upload"],
        "out_name": ex_out
    }