# gui/tabs/checkpoint_merge.py

import gradio as gr
import numpy as np
import pandas as pd
from gui.components import create_source_selector
from gui.actions.checkpoint_merge import submit_checkpoint_merge

def create_checkpoint_merge_tab():
    default_w = [0.5] * 31
    default_str = ", ".join([f"{x:.2f}" for x in default_w])
    default_df = pd.DataFrame({"Block": range(31), "Alpha": default_w})

    with gr.TabItem("Checkpoint Merge", id="CkptMerge"):
        gr.Markdown("### 🏗️ Checkpoint Merger & Baker")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 1. Models")
                sel_a = create_source_selector("Model A (Primary)", show_upload=False, disk_dir_type="checkpoints")
                sel_b = create_source_selector("Model B (Secondary)", show_upload=False, disk_dir_type="checkpoints")
                sel_c = create_source_selector("Model C (Base/Diff)", show_upload=False, disk_dir_type="checkpoints")
                sel_lora = create_source_selector("LoRA to Bake", disk_dir_type="loras")
            
            with gr.Column():
                gr.Markdown("#### 2. Settings")
                mode = gr.Dropdown(["Weighted Sum", "Add Difference", "SLERP", "TIES"], value="Weighted Sum", label="Merge Algorithm")
                with gr.Row():
                    te_policy = gr.Radio(["Copy A", "Copy B", "Merge"], value="Copy A", label="Text Encoder")
                    vae_policy = gr.Radio(["Copy A", "Copy B"], value="Copy A", label="VAE")
                
                precision = gr.Radio(["BF16", "FP16", "FP32"], value="BF16", label="Output Precision")
                lora_strength = gr.Slider(0.0, 2.0, 1.0, label="LoRA Baking Strength")
                
                out_name = gr.Textbox(label="Output Filename", placeholder="my_custom_model.safetensors")
                btn = gr.Button("🚀 Start Merge", variant="primary", size="lg")

        gr.Markdown("---")
        gr.Markdown("#### 3. Block Weights (MBW)")
        
        with gr.Row():
            with gr.Column(scale=1):
                preset = gr.Dropdown(
                    ["Flat (0.5)", "Linear A->B", "Linear B->A", "Smooth Step", "Keep Structure (IN)", "Keep Style (OUT)"],
                    label="Presets"
                )
                weight_str = gr.Textbox(
                    label="Manual Weights (31 values)", 
                    value=default_str,
                    placeholder="1.0, 0.5, 0.5...",
                    info="Index 0 is Base, 1-30 are Blocks."
                )
            with gr.Column(scale=2):
                weight_plot = gr.BarPlot(
                    value=default_df,
                    x="Block", y="Alpha", 
                    title="Merge Profile", 
                    y_lim=[0, 1],
                    height=250
                )

        def apply_preset(p_name):
            if p_name == "Flat (0.5)": w = [0.5] * 31
            elif p_name == "Linear A->B": w = np.linspace(0, 1, 31).tolist()
            elif p_name == "Linear B->A": w = np.linspace(1, 0, 31).tolist()
            elif p_name == "Smooth Step":
                x = np.linspace(0, 1, 31)
                w = (3 * x**2 - 2 * x**3).tolist()
            elif p_name == "Keep Structure (IN)": w = [1.0]*11 + [0.5]*10 + [0.0]*10
            elif p_name == "Keep Style (OUT)": w = [0.0]*11 + [0.5]*10 + [1.0]*10
            else: w = [0.5] * 31
            
            s = ", ".join([f"{x:.2f}" for x in w])
            df = pd.DataFrame({"Block": range(len(w)), "Alpha": w})
            return s, df

        preset.change(apply_preset, preset, [weight_str, weight_plot])

        btn.click(
            submit_checkpoint_merge,
            [
                sel_a["disk"], sel_b["disk"], sel_c["disk"], sel_lora["disk"], sel_lora["ws"], sel_lora["upload"],
                weight_str, out_name, mode, te_policy, vae_policy, precision, lora_strength
            ],
            None
        )

    return {
        "sel_a": sel_a, "sel_b": sel_b, "sel_c": sel_c, "sel_lora": sel_lora, "out_name": out_name
    }