# gui_workspace.py

import gradio as gr
from gui import actions
from core.workspace import WorkspaceManager
from core.config import ConfigManager
from core.io_manager import SafeStreamer

workspace = WorkspaceManager()
config = ConfigManager()

def create_sidebar():
    with gr.Column(elem_id="sidebar_col"):
        # --- 1. RESOURCE MONITOR ---
        with gr.Group():
            gr.Markdown("### 🖥️ Resources")
            resource_html = gr.HTML(value=actions.format_resource_html())
            res_timer = gr.Timer(2.0)
            res_timer.tick(actions.update_resources, outputs=[resource_html])

        # --- 2. WORKSPACE MANAGER ---
        with gr.Group():
            gr.Markdown("### 🧠 Workspace (RAM)")
            
            with gr.Tabs():
                with gr.Tab("Upload"):
                    file_uploader = gr.File(
                        label="Drop File",
                        file_count="multiple",
                        file_types=[".safetensors"],
                        type="filepath",
                        height=80,
                        min_width=50
                    )
                with gr.Tab("Server"):
                    # Scan loras dir for quick load
                    server_files = SafeStreamer.scan_directory(config.loras_dir)
                    server_drop = gr.Dropdown(
                        label="Load from Disk", 
                        choices=server_files, 
                        allow_custom_value=True
                    )
                    server_load_btn = gr.Button("Load to RAM", size="sm")

            # Model List
            workspace_list = gr.Dataframe(
                headers=["Name", "Rank", "Size"],
                datatype=["str", "number", "str"],
                column_count=(3, "fixed"),
                interactive=False, 
                label="Loaded Models"
            )
            
            # Actions Toolbar
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
                save_btn = gr.Button("💾 Save", size="sm")
                del_btn = gr.Button("🗑️ Del", size="sm")

            selected_model_name = gr.State("")

            # --- EVENTS ---
            
            file_uploader.upload(
                actions.load_files_to_workspace, 
                inputs=[file_uploader], 
                outputs=[workspace_list, file_uploader]
            )
            
            server_load_btn.click(
                actions.load_from_server_path,
                inputs=[server_drop],
                outputs=[workspace_list]
            )

            refresh_btn.click(actions.refresh_workspace_ui, outputs=[workspace_list])
            
            def on_select_internal(evt: gr.SelectData):
                row_idx = evt.index[0]
                all_models = workspace.list_models()
                if 0 <= row_idx < len(all_models):
                    return all_models[row_idx]
                return ""
                
            workspace_list.select(on_select_internal, outputs=[selected_model_name])

            save_btn.click(actions.save_workspace_model, inputs=[selected_model_name], outputs=[])
            del_btn.click(actions.delete_workspace_model, inputs=[selected_model_name], outputs=[workspace_list])

        # --- 3. MINI QUEUE ---
        with gr.Group():
            gr.Markdown("### ⏳ Active Task")
            mini_status = gr.Label(value="Idle", show_label=False)
            mini_progress = gr.HTML()
            
            q_timer = gr.Timer(1.0)
            q_timer.tick(actions.update_mini_queue, outputs=[mini_status, mini_progress])

    return workspace_list