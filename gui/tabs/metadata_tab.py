# gui/tabs/metadata_tab.py

import gradio as gr
import json
from ..context import workspace

def create_metadata_tab():
    with gr.TabItem("Metadata", id="Metadata"):
        gr.Markdown("### 📝 Metadata & Triggers Editor")
        gr.Markdown("View and edit the internal JSON header of the `.safetensors` file.")
        
        with gr.Row():
            with gr.Column(scale=2):
                md_drop = gr.Dropdown(label="Select Workspace Model", choices=[], info="Only models loaded in RAM can be edited.")
                md_refresh = gr.Button("Load Metadata")
                md_json = gr.Code(label="Header JSON", language="json", interactive=True)
                md_save = gr.Button("Update RAM Model", variant="primary")
            
            with gr.Column(scale=1):
                gr.Markdown("#### Quick Edit")
                q_title = gr.Textbox(label="Model Title", info="ss_output_name: Display name in WebUIs.")
                q_tags = gr.Textbox(label="Trigger Words", info="Tags found in ss_tag_frequency.")
                q_desc = gr.Textbox(label="Description", lines=3, info="Internal notes for the model.")
        
        def on_load(name):
            model = workspace.get_model(name)
            if not model: return "{}", "", "", ""
            m = model.metadata
            return json.dumps(m, indent=4), m.get("ss_output_name", ""), m.get("ss_tag_frequency", ""), m.get("modelspec.description", "")
        
        def on_save(name, json_str):
            model = workspace.get_model(name)
            if not model: return gr.Warning("Select model")
            try:
                model.metadata = json.loads(json_str)
                return gr.Info("Metadata updated in Workspace.")
            except Exception as e: return gr.Error(f"JSON Error: {e}")
            
        md_refresh.click(on_load, [md_drop], [md_json, q_title, q_tags, q_desc])
        md_save.click(on_save, [md_drop, md_json], None)
    return {"drop": md_drop}