# gui/tabs/queue_tab.py

import gradio as gr
from gui.actions import refresh_queue_table

def create_queue_tab():
    with gr.TabItem("Queue History", id="Queue"):
         gr.Markdown("See Sidebar for active tasks. This is history.")
         q_table = gr.Dataframe(headers=["Time", "Task", "Status", "Progress", "Message"])
         q_refresh = gr.Button("Refresh")
         
         q_refresh.click(refresh_queue_table, outputs=[q_table, gr.Label(visible=False)])
         
    return q_table