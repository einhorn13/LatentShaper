# gui/tabs/extract.py

import gradio as gr
import os
from gui.actions import toggle_output_input, submit_extract
from gui.actions.common import suggest_output_name
from gui.components import create_source_selector

def create_extract_tab():
    with gr.TabItem("Extract", id="Extract"):
        gr.Markdown("### ⛏️ LoRA Extraction")
        gr.Markdown("**Formula:** `(Tuned - Base) * Scale`")
        
        # 1. Selectors
        # Base Model (Single Checkpoint)
        base_sel = create_source_selector("Base Model", multiselect=False, disk_dir_type="checkpoints", show_upload=False)
        
        # Tuned Models (Multiple Checkpoints) - FIXED: disk_dir_type="checkpoints"
        tuned_sel = create_source_selector("Fine-Tuned Model(s)", multiselect=True, disk_dir_type="checkpoints", show_upload=True)

        # --- SETTINGS ---
        with gr.Row():
            with gr.Column(scale=1):
                ex_rank = gr.Slider(4, 256, 64, step=4, label="Rank", info="Capacity.")
                ex_thresh = gr.Slider(0.0, 0.01, 0.0001, step=0.0001, label="Noise Gate", info="Filter small changes.")
            
            with gr.Column(scale=1):
                ex_scale = gr.Slider(0.01, 2.0, 1.0, step=0.01, label="Extract Scale", 
                                    info="Bake this multiplier into weights. Use 0.05-0.1 for Turbo/Distilled!")
                ex_alpha = gr.Number(label="Force Alpha (Optional)", value=0, 
                                    info="If 0, Alpha = Rank. Try 1.0 for Turbo.")

        with gr.Row():
            ex_save_ws = gr.Checkbox(label="Save to Workspace", value=True)
            ex_out = gr.Textbox(label="Output Name", placeholder="extracted_lora")

        ex_save_ws.change(toggle_output_input, ex_save_ws, ex_out)
        
        # Auto-naming
        def _update_name(ws, disk, up): return suggest_output_name(ws, disk, up, "_extracted")
        triggers = [t for t in [tuned_sel["ws"], tuned_sel["disk"], tuned_sel["upload"]] if t is not None]
        for t in triggers:
            t.change(_update_name, [tuned_sel["ws"], tuned_sel["disk"], tuned_sel["upload"]], ex_out)

        gr.Button("Extract LoRA", variant="primary").click(
            submit_extract, 
            [
                base_sel["ws"], base_sel["disk"], 
                tuned_sel["ws"], tuned_sel["disk"], tuned_sel["upload"], 
                ex_rank, ex_thresh, ex_scale, ex_alpha, ex_out, ex_save_ws
            ], 
            None
        )

    return {
        "base_ws": base_sel["ws"], "base_disk": base_sel["disk"],
        "tuned_ws": tuned_sel["ws"], "tuned_disk": tuned_sel["disk"], "tuned_upload": tuned_sel["upload"],
        "out_name": ex_out
    }