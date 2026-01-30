# gui/tabs/settings.py

import gradio as gr
from core.config import ConfigManager
from core.io_manager import SafeStreamer

def create_settings_tab():
    config = ConfigManager()
    
    with gr.TabItem("Settings", id="Settings"):
        gr.Markdown("### ⚙️ Application Settings")
        
        with gr.Group():
            gr.Markdown("#### 📂 Directory Paths")
            gr.Markdown("Set these to your ComfyUI `models` folders to enable direct disk access.")
            
            ckpt_input = gr.Textbox(
                label="Checkpoints Directory (Base Models)", 
                value=config.checkpoints_dir,
                placeholder="C:/ComfyUI/models/checkpoints"
            )
            
            lora_input = gr.Textbox(
                label="LoRAs Directory (Tuned Models)", 
                value=config.loras_dir,
                placeholder="C:/ComfyUI/models/loras"
            )
            
            save_btn = gr.Button("Save Paths & Rescan", variant="primary")
            status_msg = gr.Label(visible=False)

        def save_settings(ckpt_path, lora_path):
            config.set("checkpoints_dir", ckpt_path)
            config.set("loras_dir", lora_path)
            
            # Force rescan caches
            c_count = len(SafeStreamer.scan_directory(ckpt_path, force_refresh=True))
            l_count = len(SafeStreamer.scan_directory(lora_path, force_refresh=True))
            
            return gr.Label(value=f"✅ Saved! Found {c_count} Checkpoints, {l_count} LoRAs.", visible=True)

        save_btn.click(save_settings, [ckpt_input, lora_input], [status_msg])