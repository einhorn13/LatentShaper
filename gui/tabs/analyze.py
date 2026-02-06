# gui/tabs/analyze.py

import gradio as gr
from gui.actions import run_analysis
from gui.components import create_source_selector

def create_analyze_tab():
    with gr.TabItem("Analyze", id="Analyze"):
        gr.Markdown("### 🔍 Advanced Diagnostics")
        gr.Markdown("Deep spectral analysis to identify overtraining, noise, and information density.")
        
        sel = create_source_selector("Target Model", multiselect=False, disk_dir_type="loras")
        
        analyze_btn = gr.Button("Run Deep Analysis", variant="primary")
        
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Spectrum"): 
                        plot_spectrum = gr.LinePlot(x="Rank", y="LogValue", title="SVD Info Density", height=300, 
                                                   tooltip=["Rank", "Value"])
                        gr.Markdown("<small>Shows how much unique information is in each rank.</small>")
                    with gr.Tab("Region Energy"): 
                        plot_energy = gr.BarPlot(x="Region", y="Energy", title="Block Energy Distribution", height=300)
                        gr.Markdown("<small>IN: Structure | MID: Concepts | OUT: Style.</small>")
                    with gr.Tab("Layer Heatmap"): 
                        plot_heatmap = gr.Image(label="Attention/MLP Activity Matrix", interactive=False)
                        gr.Markdown("<small>Vertical: Blocks. Horizontal: Components. Bright spots = High activity.</small>")
            
            with gr.Column(scale=2):
                analysis_report = gr.HTML("Results will appear here.")
                rec_state = gr.State([])
                with gr.Group():
                    gr.Markdown("#### 🛠️ Smart Advisor")
                    rec_drop = gr.Dropdown(label="Recommendations", choices=[], info="Automated fixes.")
                    fix_btn = gr.Button("Apply Fix & Open Morph", variant="secondary")
    
    return {
        "ws": sel["ws"], "disk": sel["disk"], "upload": sel["upload"], 
        "btn": analyze_btn, 
        "plot_s": plot_spectrum, "plot_e": plot_energy, "plot_h": plot_heatmap, 
        "report": analysis_report, "rec_state": rec_state, "rec_drop": rec_drop, "fix_btn": fix_btn
    }