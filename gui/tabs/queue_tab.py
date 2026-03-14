
import gradio as gr
from gui.actions.resources import (
    refresh_queue_table, 
    handle_history_interaction, 
    bulk_cancel_pending, 
    bulk_clear_finished,
    cancel_task
)

def create_queue_tab():
    with gr.TabItem("Queue & History", id="Queue"):
        gr.Markdown("### 📋 Task Management")
        
        with gr.Row():
            with gr.Column(scale=4):
                q_table = gr.Dataframe(
                    headers=["Sel", "Time", "Task", "Status", "Duration", "Progress", "Message", "ID"],
                    datatype=["bool", "str", "str", "str", "str", "str", "str", "str"],
                    interactive=True,
                    label="Job History (Select a row to see Traceback)"
                )
                
                job_log_viewer = gr.Code(
                    label="Logs / Traceback",
                    language="python",
                    visible=False,
                    lines=10
                )
                
            with gr.Column(scale=1):
                gr.Markdown("#### Global Actions")
                btn_cancel_all = gr.Button("🛑 Cancel All Pending", variant="stop")
                btn_clear_all = gr.Button("🧹 Clear Finished", variant="secondary")
                
                gr.Markdown("#### Selection Actions")
                selected_id = gr.Textbox(label="Selected Job ID", interactive=False)
                btn_recall = gr.Button("🔄 Recall Parameters", variant="primary")
                btn_cancel_sel = gr.Button("Stop Selected")
                
                sync_status = gr.HTML("<small style='color:#888'>Auto-sync active</small>")

        # Авто-обновление
        auto_timer = gr.Timer(2.0)
        auto_timer.tick(refresh_queue_table, outputs=[q_table, sync_status])

        # Взаимодействие
        q_table.select(
            handle_history_interaction, 
            inputs=[q_table], 
            outputs=[job_log_viewer, selected_id]
        )
        
        btn_cancel_all.click(bulk_cancel_pending, outputs=[q_table, sync_status])
        btn_clear_all.click(bulk_clear_finished, outputs=[q_table, sync_status])
        btn_cancel_sel.click(cancel_task, inputs=[selected_id], outputs=[q_table, sync_status])

    return {
        "table": q_table,
        "recall_btn": btn_recall,
        "selected_id": selected_id
    }