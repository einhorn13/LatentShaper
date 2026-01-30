# gui/tabs/bridge.py

import gradio as gr
from gui.actions.submission import submit_bridge
from gui.components import create_source_selector

def create_bridge_tab():
    with gr.TabItem("Bridge", id="Bridge"):
        gr.Markdown("### 🌉 Neural Architecture Bridge")
        gr.Markdown("Convert **SDXL LoRAs** to **Z-Image Turbo** using *Orthogonal Basis Expansion*.")
        
        sel = create_source_selector("Source SDXL LoRA", multiselect=False, disk_dir_type="loras")
        
        with gr.Row():
            with gr.Column(scale=2):
                br_out = gr.Textbox(label="Result Name", placeholder="my_migrated_concept")
            
            with gr.Column(scale=1):
                br_strength = gr.Slider(0.01, 1.5, 0.5, label="Concept Strength", 
                                      info="Turbo models are sensitive. 0.5 is recommended.")
                br_save_ws = gr.Checkbox(label="Save to Workspace (RAM)", value=True)
                br_btn = gr.Button("🚀 Start Migration", variant="primary", size="lg")
        
        br_btn.click(
            submit_bridge,
            [sel["ws"], sel["disk"], sel["upload"], br_out, br_strength, br_save_ws],
            None
        )
        
    return {"ws": sel["ws"], "disk": sel["disk"], "upload": sel["upload"], "out": br_out}