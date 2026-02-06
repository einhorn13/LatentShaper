# gui/tabs/morph.py

import gradio as gr
from gui.actions import toggle_output_input, submit_morph, reset_morph_ui
from gui.actions.common import suggest_output_name
from gui.components import create_source_selector

def create_morph_tab():
    with gr.TabItem("Morph", id="Morph"):
        gr.Markdown("### 🧬 Morphing Studio")
        gr.Markdown("Fine-tune LoRA weights using region-based equalizers and mathematical filters.")
        
        sel = create_source_selector("Input Model(s)", multiselect=True, disk_dir_type="loras")

        with gr.Tabs():
            with gr.Tab("Weight Control"):
                with gr.Row(): 
                    eq_global = gr.Slider(0.1, 3.0, 1.0, label="Global Scale")
                    eq_interpolate = gr.Checkbox(label="Linear Interpolation", value=False)
                
                with gr.Row():
                    eq_in = gr.Slider(0.0, 2.0, 1.0, label="IN (Structure)")
                    eq_mid = gr.Slider(0.0, 2.0, 1.0, label="MID (Concepts)")
                    eq_out = gr.Slider(0.0, 2.0, 1.0, label="OUT (Style)")

            with gr.Tab("Advanced Filters"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### ✂️ Spectral Gate (SVD)")
                        filter_chk = gr.Checkbox(label="Enable Spectral Gate")
                        filter_adaptive = gr.Checkbox(label="Adaptive (SNR) Mode")
                        filter_thr = gr.Slider(0.0, 5.0, 0.1, label="Threshold")
                        filter_inv = gr.Checkbox(label="Invert Filter")
                    
                    with gr.Column():
                        gr.Markdown("#### 🎲 DARE (Random Drop)")
                        dare_chk = gr.Checkbox(label="Enable DARE")
                        dare_rate = gr.Slider(0.0, 0.99, 0.1, label="Drop Rate")
                
                gr.Markdown("---")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 🚫 Band-Stop Filter (Frequency)")
                        bs_chk = gr.Checkbox(label="Enable Band-Stop")
                        with gr.Row():
                            bs_start = gr.Slider(0.0, 1.0, 0.3, label="Start Freq")
                            bs_end = gr.Slider(0.0, 1.0, 0.6, label="End Freq")

            with gr.Tab("Dynamics & Eraser"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 🎨 Dynamics")
                        temp = gr.Slider(0.5, 2.0, 1.0, label="Temperature")
                        fft = gr.Slider(0.1, 1.0, 1.0, label="FFT Smoothing")
                        clamp = gr.Slider(0.9, 1.0, 1.0, label="Quantile Clamp")
                        fix_alpha = gr.Checkbox(label="Sync Alpha=Rank", value=True)
                        
                        with gr.Group():
                            homeostatic = gr.Checkbox(label="🛡️ Safety (Homeostatic)", value=False)
                            homeostatic_thr = gr.Number(label="Target Mag.", value=0.01, step=0.001)

                    with gr.Column():
                        gr.Markdown("#### 🗑️ Eraser Tools")
                        gr.Markdown("**Vector Eraser (SVD)**")
                        with gr.Row():
                            eraser_start = gr.Number(label="Start Idx", value=0, precision=0)
                            eraser_end = gr.Number(label="End Idx", value=0, precision=0)
                        gr.Markdown("**Block Eraser**")
                        erase_blocks = gr.Textbox(label="Remove Blocks", placeholder="e.g. 4, 9-11, 28")

        with gr.Row():
            mo_save_ws = gr.Checkbox(label="Save to Workspace", value=True)
            mo_out = gr.Textbox(label="Output Name", placeholder="morphed_model_name")
            mo_btn = gr.Button("Add to Queue", variant="primary")
            mo_reset = gr.Button("Reset Defaults", variant="secondary")

        mo_save_ws.change(toggle_output_input, mo_save_ws, mo_out)
        
        def _update_name(ws, disk, up): return suggest_output_name(ws, disk, up, "_morphed")
        
        triggers = [t for t in [sel["ws"], sel["disk"], sel["upload"]] if t is not None]
        for t in triggers:
            t.change(_update_name, [sel["ws"], sel["disk"], sel["upload"]], mo_out)
        
        mo_ui_list = [
            eq_global, eq_in, eq_mid, eq_out, eq_interpolate, 
            temp, fft, clamp, fix_alpha, 
            filter_chk, filter_thr, filter_inv, filter_adaptive, 
            dare_chk, dare_rate, eraser_start, eraser_end,
            homeostatic, homeostatic_thr, erase_blocks,
            bs_chk, bs_start, bs_end
        ]
        
        mo_reset.click(reset_morph_ui, outputs=mo_ui_list)
        
        mo_btn.click(
            submit_morph, 
            [
                sel["ws"], sel["disk"], sel["upload"], mo_out, 
                eq_global, eq_in, eq_mid, eq_out, eq_interpolate,
                temp, fft, clamp, fix_alpha,
                filter_chk, filter_thr, filter_inv, filter_adaptive, 
                dare_chk, dare_rate, 
                eraser_start, eraser_end, 
                mo_save_ws,
                homeostatic, homeostatic_thr,
                erase_blocks,
                bs_chk, bs_start, bs_end
            ], 
            None
        )
    
    return {
        "ws": sel["ws"], "disk": sel["disk"], "upload": sel["upload"], "out_name": mo_out, 
        "eq_global": eq_global, "eq_in": eq_in, "eq_mid": eq_mid, "eq_out": eq_out, 
        "eq_interpolate": eq_interpolate, "temp": temp, "fft": fft, "clamp": clamp, 
        "fix_alpha": fix_alpha, "filter_chk": filter_chk, "filter_thr": filter_thr, 
        "filter_inv": filter_inv, "filter_adaptive": filter_adaptive,
        "dare_chk": dare_chk, "dare_rate": dare_rate,
        "eraser_start": eraser_start, "eraser_end": eraser_end,
        "homeostatic": homeostatic, "homeostatic_thr": homeostatic_thr,
        "erase_blocks": erase_blocks, "bs_chk": bs_chk,
        "bs_start": bs_start, "bs_end": bs_end
    }