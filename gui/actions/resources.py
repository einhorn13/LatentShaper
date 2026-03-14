
import gradio as gr
import os
from datetime import datetime
from core.resources import ResourceMonitor
from core.structs import JobStatus
from gui import context

_last_rendered_hash = ""

def format_duration(seconds: float) -> str:
    if seconds <= 0: return "0s"
    if seconds < 1: return f"{int(seconds*1000)}ms"
    m, s = divmod(int(seconds), 60)
    if m > 0: return f"{m}m {s}s"
    return f"{s}s"

def format_resource_html():
    """Отрисовка панели ресурсов в сайдбаре."""
    stats = ResourceMonitor.get_status()
    def _bar(percent, color="blue"):
        return f'<div style="background:#eee;border-radius:4px;height:8px;width:100%;margin-top:2px;"><div style="background:{color};width:{percent}%;height:100%;border-radius:4px;"></div></div>'
    
    return f"""
    <div style="font-size: 12px; padding: 2px;">
        <div style="margin-bottom: 4px;"><b>RAM:</b> {ResourceMonitor.format_bytes(stats['ram_used'])} {_bar(stats['ram_percent'], "#4caf50")}</div>
        <div><b>VRAM:</b> {ResourceMonitor.format_bytes(stats['vram_used'])} {_bar(stats['vram_percent'], "#2196f3")}</div>
    </div>
    """

def update_resources():
    return format_resource_html()

def update_mini_queue():
    """Обновление виджета текущей задачи в сайдбаре."""
    job = context.queue_manager.current_job
    if job:
        prog = int(job.progress * 100)
        btn_visibility = gr.update(visible=True)
        html = f"""
        <div style="font-size: 11px;"><b>{job.description}</b> ({prog}%)</div>
        <div style="background:#eee;height:4px;width:100%;margin:4px 0;"><div style="background:#ff9800;width:{prog}%;height:100%;"></div></div>
        <div style="font-size: 10px; color: #666;">{job.message}</div>
        """
        return html, btn_visibility, job.id
    return "<div style='color:#888;font-size:11px;'>Queue idle</div>", gr.update(visible=False), ""

def refresh_queue_table(force=False):
    """Умное обновление таблицы истории с поддержкой Duration."""
    global _last_rendered_hash
    current_hash = context.queue_manager.get_state_hash()
    
    if current_hash == _last_rendered_hash and not force:
        return gr.update(), gr.update()

    _last_rendered_hash = current_hash
    jobs = context.queue_manager.get_all_jobs()
    data = []
    for j in jobs:
        ts = datetime.fromtimestamp(j.created_at).strftime('%H:%M:%S')
        icon = {"PENDING": "⏳", "RUNNING": "🔄", "COMPLETED": "✅", "FAILED": "❌", "CANCELLED": "🚫"}.get(j.status.name, "❓")
        
        data.append([
            False, 
            ts, 
            j.description, 
            f"{icon} {j.status.name}", 
            format_duration(j.duration),
            f"{int(j.progress * 100)}%", 
            j.message, 
            j.id
        ])
    
    now = datetime.now().strftime('%H:%M:%S')
    return data, f"<small style='color:#888'>Last sync: {now}</small>"

def handle_history_interaction(evt: gr.SelectData, data):
    """Просмотр логов/ошибок при клике на строку."""
    if not evt.index or evt.index[0] < 0: return gr.update(visible=False), ""
    try:
        job_id = data.iloc[evt.index[0], 7]
        job = context.queue_manager.jobs.get(job_id)
        if not job: return gr.update(visible=False), ""
        
        log_content = job.logs if job.logs else f"Task: {job.description}\nStatus: {job.status.name}\nNo errors captured."
        return gr.update(value=log_content, visible=True, label=f"Details for {job_id}"), job_id
    except:
        return gr.update(visible=False), ""

def cancel_task(job_id):
    if not job_id: return refresh_queue_table(force=True)
    context.queue_manager.cancel_job(job_id)
    return refresh_queue_table(force=True)

def bulk_cancel_pending():
    context.queue_manager.cancel_all_pending()
    return refresh_queue_table(force=True)

def bulk_clear_finished():
    context.queue_manager.clear_finished()
    return refresh_queue_table(force=True)

def recall_parameters(job_id):
    """Восстановление параметров в UI."""
    if not job_id: return gr.Warning("Select a job first!")
    job = context.queue_manager.jobs.get(job_id)
    if not job or not job.config_data: return gr.Warning("No parameters found.")
    # Логика маппинга будет расширена в gui_layout.py
    return job.config_data

def clear_queue_history():
    """Совместимость со старыми вызовами."""
    return bulk_clear_finished()