
import gradio as gr
from gui import actions
from core.workspace import WorkspaceManager
from core.config import ConfigManager
from core.io_manager import SafeStreamer

# Инициализация синглтонов
workspace = WorkspaceManager()
config = ConfigManager()

def create_sidebar():
    """
    Создает левую панель управления: Ресурсы, Workspace (RAM) и Активные задачи.
    """
    with gr.Column(elem_id="sidebar_col"):
        
        # --- БЛОК 1: МОНИТОРИНГ РЕСУРСОВ ---
        with gr.Group():
            gr.Markdown("### 🖥️ Resources")
            resource_html = gr.HTML(value=actions.format_resource_html())
            
            # Таймер обновления ресурсов (каждые 2 секунды)
            res_timer = gr.Timer(2.0)
            res_timer.tick(actions.update_resources, outputs=[resource_html])

        # --- БЛОК 2: WORKSPACE (RAM) ---
        with gr.Group():
            gr.Markdown("### 🧠 Workspace (RAM)")
            
            # Динамический индикатор архитектуры выбранной модели
            arch_indicator = gr.Markdown("⚪ **Architecture:** None Selected")
            
            with gr.Tabs():
                with gr.Tab("Upload"):
                    file_uploader = gr.File(
                        label="Drop File", 
                        file_count="multiple", 
                        file_types=[".safetensors"],
                        type="filepath", 
                        height=100
                    )
                with gr.Tab("Server"):
                    # Сканирование директории лор, указанной в конфиге
                    server_files = SafeStreamer.scan_directory(config.loras_dir)
                    server_drop = gr.Dropdown(
                        label="Load from Disk", 
                        choices=server_files, 
                        allow_custom_value=True
                    )
                    server_load_btn = gr.Button("🚀 Load to RAM", size="sm", variant="secondary")

            # Основная таблица моделей в памяти
            workspace_list = gr.Dataframe(
                headers=["Name", "Rank", "Arch", "Size"], 
                datatype=["str", "number", "str", "str"],
                column_count=(4, "fixed"), 
                interactive=False, 
                label="Loaded Models",
                elem_id="workspace_table",
                wrap=True
            )
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
                save_btn = gr.Button("💾 Save", size="sm")
                del_btn = gr.Button("🗑️ Del", size="sm")

            # Скрытое состояние для хранения имени выбранной модели
            selected_model_name = gr.State("")

            # Логика загрузки и управления списком
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
            refresh_btn.click(
                actions.refresh_workspace_ui, 
                outputs=[workspace_list]
            )
            
            # Обработка выбора строки в таблице
            def on_select_internal(evt: gr.SelectData):
                row_idx = evt.index[0]
                all_models = workspace.list_models()
                if 0 <= row_idx < len(all_models):
                    return all_models[row_idx]
                return ""
                
            workspace_list.select(on_select_internal, outputs=[selected_model_name])
            
            save_btn.click(
                actions.save_workspace_model, 
                inputs=[selected_model_name], 
                outputs=[]
            )
            del_btn.click(
                actions.delete_workspace_model, 
                inputs=[selected_model_name], 
                outputs=[workspace_list]
            )

        # --- БЛОК 3: УПРАВЛЕНИЕ ТЕКУЩЕЙ ЗАДАЧЕЙ ---
        with gr.Group():
            gr.Markdown("### ⏳ Active Task")
            
            # Состояние для ID текущей задачи (нужно для отмены)
            active_job_id = gr.State("")
            
            # HTML контейнер для прогресса и описания задачи
            mini_progress_html = gr.HTML(
                "<div style='color:#888;font-size:11px;'>Queue idle</div>"
            )
            
            # Кнопка немедленной остановки (видима только во время выполнения)
            btn_abort = gr.Button(
                "🛑 Stop Current", 
                size="sm", 
                variant="stop", 
                visible=False
            )
            
            # Таймер мониторинга очереди (каждые 1.5 сек)
            q_timer = gr.Timer(1.5)
            q_timer.tick(
                actions.update_mini_queue, 
                outputs=[mini_progress_html, btn_abort, active_job_id]
            )
            
            # Экстренная отмена задачи по её ID
            btn_abort.click(
                actions.cancel_task, 
                inputs=[active_job_id], 
                outputs=[]
            )

    return workspace_list, arch_indicator