# gui_launcher.py
# Renamed from gui.py to avoid conflict with 'gui' package folder

import sys
import os

# Ensure we can import core modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gui_layout
from core.logger import Logger

def launch():
    Logger.info("Initializing GUI...")
    app = gui_layout.create_ui()
    
    # Launch settings
    app.queue().launch(
        inbrowser=True, 
        server_name="127.0.0.1", 
        server_port=7860,
        favicon_path=None
    )

if __name__ == "__main__":
    try:
        launch()
    except KeyboardInterrupt:
        Logger.info("Shutting down...")
    except Exception as e:
        Logger.error(f"GUI Crash: {e}")