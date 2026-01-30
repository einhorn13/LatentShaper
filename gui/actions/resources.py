# gui/actions/resources.py

from datetime import datetime
from core.resources import ResourceMonitor
from core.structs import JobStatus
# Import the module itself to access mutable globals correctly
from .. import context

def format_resource_html():
    stats = ResourceMonitor.get_status()
    
    def _bar(percent, color="blue"):
        return f"""
        <div style="background-color: #eee; border-radius: 4px; height: 10px; width: 100%; margin-top: 2px;">
            <div style="background-color: {color}; width: {percent}%; height: 100%; border-radius: 4px;"></div>
        </div>
        """

    html = f"""
    <div style="font-size: 12px;">
        <div style="margin-bottom: 5px;">
            <b>RAM:</b> {ResourceMonitor.format_bytes(stats['ram_used'])} / {ResourceMonitor.format_bytes(stats['ram_total'])}
            {_bar(stats['ram_percent'], "#4caf50")}
        </div>
        <div>
            <b>VRAM ({stats['gpu_name']}):</b> {ResourceMonitor.format_bytes(stats['vram_used'])} / {ResourceMonitor.format_bytes(stats['vram_total'])}
            {_bar(stats['vram_percent'], "#2196f3")}
        </div>
    </div>
    """
    return html

def update_resources():
    return format_resource_html()

def update_mini_queue():
    job = context.queue_manager.current_job
    if job:
        prog = int(job.progress * 100)
        html = f"""
        <div style="font-size: 11px; margin-bottom: 2px;">{job.message}</div>
        <div style="background-color: #eee; height: 6px; width: 100%;">
            <div style="background-color: #ff9800; width: {prog}%; height: 100%;"></div>
        </div>
        """
        return f"Running: {job.description}", html
    else:
        return "Idle", ""

def refresh_queue_table(filter_mode="All"):
    jobs = context.queue_manager.get_all_jobs()
    data = []
    for j in jobs:
        if filter_mode == "Active" and j.status not in [JobStatus.PENDING, JobStatus.RUNNING]: continue
        if filter_mode == "Completed" and j.status != JobStatus.COMPLETED: continue
        if filter_mode == "Failed" and j.status != JobStatus.FAILED: continue
        ts = datetime.fromtimestamp(j.created_at).strftime('%H:%M:%S')
        icon = {"RUNNING": "🔄", "COMPLETED": "✅", "FAILED": "❌", "CANCELLED": "🚫", "PENDING": "⏳"}.get(j.status.name, "?")
        data.append([ts, j.description, f"{icon} {j.status.name}", f"{int(j.progress * 100)}%", j.message])
    
    is_working_now = context.queue_manager.is_working()
    notification = ""
    
    # Correctly accessing global state through module reference
    if context.was_working and not is_working_now: notification = "✅ All tasks completed!"
    elif is_working_now: notification = "🔄 Processing queue..."
    else: notification = "💤 Idle"
    
    context.was_working = is_working_now
    return data, notification

def clear_queue_history():
    context.queue_manager.clear_history()
    return refresh_queue_table("All")