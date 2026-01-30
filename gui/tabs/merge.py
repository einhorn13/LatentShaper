# gui/tabs/merge.py

import gradio as gr
from gui.actions import (
    toggle_output_input, 
    add_files_to_merge, 
    add_workspace_to_merge, 
    clear_merge_list, 
    normalize_weights_ui, 
    submit_merge
)
# Import new actions
from gui.actions.submission import distribute_weights_ui, invert_weights_ui

def create_merge_tab():
    with gr.TabItem("Merge", id="Merge"):
        gr.Markdown("### 🔗 Multi-LoRA Merging")
        gr.Markdown("Combine multiple models using advanced algorithms. Supports **Negative Weights** for subtraction.")
        
        merge_state = gr.State({})
        
        with gr.Row():
            with gr.Column(scale=1):
                me_files = gr.File(label="Add Disk Files", file_count="multiple")
                me_ws_drop = gr.Dropdown(label="Add Workspace Model", choices=[])
                
                gr.Markdown("#### Weight Tools")
                with gr.Row():
                    dist_btn = gr.Button("⚖️ Even", size="sm")
                    norm_btn = gr.Button("📊 Norm", size="sm")
                with gr.Row():
                    inv_btn = gr.Button("🔄 Invert", size="sm")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="sm")

            with gr.Column(scale=3):
                me_table = gr.Dataframe(
                    headers=["Filename", "Rank", "Mix Ratio"], 
                    column_count=(3, "fixed"), 
                    interactive=True, 
                    datatype=["str", "str", "number"],
                    label="Merge Composition"
                )
        
        # Data Loading
        me_files.upload(add_files_to_merge, [me_files, me_table, merge_state], [me_table, merge_state, me_files])
        me_ws_drop.change(add_workspace_to_merge, [me_ws_drop, me_table], [me_table, me_ws_drop])
        
        # Tool Actions
        clear_btn.click(clear_merge_list, outputs=[me_table, merge_state, me_files])
        norm_btn.click(normalize_weights_ui, me_table, me_table)
        dist_btn.click(distribute_weights_ui, me_table, me_table)
        inv_btn.click(invert_weights_ui, me_table, me_table)
        
        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=1):
                me_algo = gr.Dropdown(
                    ["SVD (Smart Blend)", "Concat (Lossless)", "Orthogonal (Unique)", "TIES (Conflict Fix)", "SLERP (Geodesic 2-Model)"], 
                    value="SVD (Smart Blend)", label="Algorithm",
                    info="SVD is best for Turbo. TIES handles sign conflicts well."
                )
            with gr.Column(scale=1):
                me_rank = gr.Slider(1, 256, 64, step=4, label="Target Rank")
            with gr.Column(scale=1):
                me_str = gr.Slider(0.1, 5.0, 1.0, label="Global Strength", info="Master multiplier.")
        
        # Hidden defaults for advanced merge params
        with gr.Row(visible=False):
            me_auto_rank = gr.Checkbox(value=False)
            me_auto_thr = gr.Number(value=0.0)
            me_prune = gr.Checkbox(value=False)
            me_prune_thr = gr.Number(value=0.0)
            me_ties_dens = gr.Number(value=0.3)

        with gr.Row():
            me_save_ws = gr.Checkbox(label="Save to Workspace", value=True)
            me_out = gr.Textbox(label="Result Name", placeholder="merged_model")
        
        me_save_ws.change(toggle_output_input, me_save_ws, me_out)
        
        gr.Button("Start Merge", variant="primary", size="lg").click(
            submit_merge, 
            [me_table, merge_state, me_rank, me_out, me_algo, me_str, me_auto_rank, me_auto_thr, me_prune, me_prune_thr, me_ties_dens, me_save_ws], 
            None
        )

    return {"ws_drop": me_ws_drop}