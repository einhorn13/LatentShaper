# gui/tabs/resize.py

import gradio as gr
from gui.actions import toggle_output_input, ui_toggle_disabled, submit_resize
from gui.actions.common import suggest_output_name
from gui.components import create_source_selector

def create_resize_tab():
    with gr.TabItem("Resize", id="Resize"):
        gr.Markdown("### 📉 Rank Compression (SVD)")
        gr.Markdown("Reduce LoRA size by discarding low-energy singular values.")
        
        sel = create_source_selector("Input Model(s)", multiselect=True, disk_dir_type="loras")

        with gr.Row():
            with gr.Column():
                re_rank = gr.Slider(1, 128, 32, step=4, label="Target Rank", info="Hard limit for the output dimension.")
                with gr.Group():
                    re_auto_rank = gr.Checkbox(label="Auto-Rank (Dynamic)", info="Calculate optimal rank for each block independently.")
                    re_auto_thr = gr.Slider(0.8, 0.99, 0.95, label="Info Retention", 
                                           info="0.95 = Keep 95% of information, discard 5% noise.")
            with gr.Column():
                re_save_ws = gr.Checkbox(label="Save to Workspace", value=True)
                re_out = gr.Textbox(label="Output Name")
        
        re_save_ws.change(toggle_output_input, re_save_ws, re_out)
        re_auto_rank.change(ui_toggle_disabled, inputs=re_auto_rank, outputs=re_rank)
        
        def _update_name(ws, disk, up): return suggest_output_name(ws, disk, up, "_resized")
        
        # FIX: Filter None triggers
        triggers = [t for t in [sel["ws"], sel["disk"], sel["upload"]] if t is not None]
        for t in triggers:
            t.change(_update_name, [sel["ws"], sel["disk"], sel["upload"]], re_out)
        
        gr.Button("Resize LoRA", variant="primary").click(
            submit_resize, 
            [sel["ws"], sel["disk"], sel["upload"], re_rank, re_out, re_auto_rank, re_auto_thr, re_save_ws], 
            None
        )

    return {"ws": sel["ws"], "disk": sel["disk"], "upload": sel["upload"], "out_name": re_out}