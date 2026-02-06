# gui_layout.py

import gradio as gr
import gui_workspace
from core.version import __version__
from gui.tabs.analyze import create_analyze_tab
from gui.tabs.extract import create_extract_tab
from gui.tabs.resize import create_resize_tab
from gui.tabs.morph import create_morph_tab
from gui.tabs.merge import create_merge_tab
from gui.tabs.metadata_tab import create_metadata_tab
from gui.tabs.queue_tab import create_queue_tab
from gui.tabs.bridge import create_bridge_tab
from gui.tabs.settings import create_settings_tab
from gui.tabs.utils import create_utils_tab
from gui.tabs.checkpoint_merge import create_checkpoint_merge_tab
from gui.actions import get_workspace_choices, run_analysis, apply_recommendation, handle_sidebar_select

def create_ui():
    with gr.Blocks(title=f"Latent Shaper v{__version__}") as app:
        gr.Markdown(f"# ⚡ Latent Shaper <small>v{__version__}</small>")
        
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                workspace_list = gui_workspace.create_sidebar()

            with gr.Column(scale=4):
                with gr.Tabs() as main_tabs:
                    an = create_analyze_tab()
                    ex = create_extract_tab()
                    re = create_resize_tab()
                    mo = create_morph_tab()
                    ut = create_utils_tab()
                    br = create_bridge_tab()
                    me = create_merge_tab()
                    ck = create_checkpoint_merge_tab()
                    md = create_metadata_tab()
                    create_queue_tab()
                    create_settings_tab()

        # --- EVENT WIRING ---

        def sync_drops():
            c = get_workspace_choices()
            return [
                gr.update(choices=c), # Analyze
                gr.update(choices=c), # Extract Base
                gr.update(choices=c), # Extract Tuned
                gr.update(choices=c), # Resize
                gr.update(choices=c), # Morph
                gr.update(choices=c), # Utils
                gr.update(choices=c), # Bridge
                gr.update(choices=c), # Merge
                gr.update(choices=c), # Ckpt Merge A
                gr.update(choices=c), # Ckpt Merge B
                gr.update(choices=c), # Ckpt Merge C
                gr.update(choices=c), # Ckpt Merge LoRA
                gr.update(choices=c)  # Metadata
            ]

        all_ws_drops = [
            an["ws"], ex["base_ws"], ex["tuned_ws"], re["ws"], mo["ws"], ut["ws"], br["ws"], me["ws_drop"], 
            ck["sel_a"]["ws"], ck["sel_b"]["ws"], ck["sel_c"]["ws"], ck["sel_lora"]["ws"], md["drop"]
        ]
        
        workspace_list.change(sync_drops, None, all_ws_drops)

        workspace_list.select(
            handle_sidebar_select,
            None,
            [
                an["ws"], an["upload"],
                ex["base_ws"], ex["base_disk"],
                re["ws"], re["upload"],
                mo["ws"], mo["upload"],
                ut["ws"], ut["upload"],
                me["ws_drop"],
                md["drop"],
                ex["out_name"], re["out_name"], mo["out_name"], ut["out_name"]
            ]
        )

        an["btn"].click(
            run_analysis,
            [an["ws"], an["disk"], an["upload"]],
            [an["plot_s"], an["plot_e"], an["plot_h"], an["report"], an["rec_state"], an["rec_drop"]]
        )
        
        apply_outputs = [
            main_tabs, mo["upload"], mo["ws"], mo["out_name"],
            mo["eq_global"], mo["eq_in"], mo["eq_mid"], mo["eq_out"], mo["eq_interpolate"],
            mo["temp"], mo["fft"], mo["clamp"], mo["fix_alpha"],
            mo["filter_chk"], mo["filter_thr"], mo["filter_inv"], mo["filter_adaptive"],
            mo["dare_chk"], mo["dare_rate"],
            mo["eraser_start"], mo["eraser_end"],
            mo["homeostatic"], mo["homeostatic_thr"]
        ]
        
        an["fix_btn"].click(
            apply_recommendation,
            [an["rec_state"], an["rec_drop"], an["upload"], an["ws"]],
            apply_outputs
        )

    return app