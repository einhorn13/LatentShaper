
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
from gui.tabs.settings import create_settings_tab
from gui.tabs.utils import create_utils_tab
from gui.tabs.checkpoint_merge import create_checkpoint_merge_tab
from gui.actions import (
    get_workspace_choices, 
    run_analysis, 
    apply_recommendation, 
    handle_sidebar_select,
    recall_parameters
)

# Custom CSS for UI optimization and scroll prevention
custom_css = """
#workspace_table { font-size: 12px !important; }
#workspace_table td { 
    padding: 4px 4px !important; 
    white-space: normal !important; 
    word-break: break-word !important;
    line-height: 1.1 !important;
}
.sidebar_col { border-right: 1px solid #ddd; }
"""

def create_ui():
    """
    Constructs the full Gradio Blocks interface.
    """
    with gr.Blocks(title=f"Latent Shaper v{__version__}", fill_width=True) as app:
        # Inject custom styles
        gr.HTML(f"<style>{custom_css}</style>", visible=False)
        
        gr.Markdown(f"# ⚡ Latent Shaper <small>v{__version__}</small>")
        
        with gr.Row():
            # --- Sidebar (30% width) ---
            with gr.Column(scale=3, min_width=400, elem_id="sidebar_col"):
                workspace_list, arch_indicator = gui_workspace.create_sidebar()

            # --- Main Content (70% width) ---
            with gr.Column(scale=7):
                with gr.Tabs() as main_tabs:
                    an = create_analyze_tab()
                    ex = create_extract_tab()
                    re = create_resize_tab()
                    mo = create_morph_tab()
                    ut = create_utils_tab()
                    me = create_merge_tab()
                    ck = create_checkpoint_merge_tab()
                    md = create_metadata_tab()
                    # Queue returns data for Recall wiring
                    qd = create_queue_tab()
                    create_settings_tab()

        # --- Global Event Handlers ---

        def sync_all_dropdowns():
            """Refreshes all workspace dropdowns across all tabs."""
            choices = get_workspace_choices()
            return [gr.update(choices=choices) for _ in range(12)]

        # Map all dropdowns that need synchronization
        all_ws_dropdowns = [
            an["ws"], ex["base_ws"], ex["tuned_ws"], re["ws"], 
            mo["ws"], ut["ws"], me["ws_drop"], 
            ck["sel_a"]["ws"], ck["sel_b"]["ws"], ck["sel_c"]["ws"], 
            ck["sel_lora"]["ws"], md["drop"]
        ]
        
        # Trigger sync on workspace list change
        workspace_list.change(sync_all_dropdowns, None, all_ws_dropdowns)

        # Sidebar selection logic: Auto-fills tab inputs and updates architecture info
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
                ex["out_name"], re["out_name"], mo["out_name"], ut["out_name"],
                arch_indicator,
                mo["eq_in"], mo["eq_mid"], mo["eq_out"], mo["eq_adapter"], mo["eq_other"]
            ]
        )

        # Analysis execution
        an["btn"].click(
            run_analysis,
            [an["ws"], an["disk"], an["upload"]],
            [an["plot_s"], an["plot_e"], an["plot_h"], an["report"], an["rec_state"], an["rec_drop"]]
        )
        
        # Advisor Fix Logic: Applies recommended params to Morph tab
        advisor_apply_outputs = [
            main_tabs, mo["upload"], mo["ws"], mo["out_name"],
            mo["eq_global"], mo["eq_in"], mo["eq_mid"], mo["eq_out"], mo["eq_adapter"], mo["eq_other"], mo["eq_interpolate"],
            mo["temp"], mo["fft"], mo["clamp"], mo["fix_alpha"],
            mo["filter_chk"], mo["filter_thr"], mo["filter_inv"], mo["filter_adaptive"],
            mo["dare_chk"], mo["dare_rate"],
            mo["eraser_start"], mo["eraser_end"],
            mo["homeostatic"], mo["homeostatic_thr"]
        ]
        
        an["fix_btn"].click(
            apply_recommendation,
            [an["rec_state"], an["rec_drop"], an["upload"], an["ws"]],
            advisor_apply_outputs
        )

        # --- Recall Engine Mapping ---

        def do_recall_parameters(job_id):
            """
            Fetches job configuration and maps it to UI components.
            Only Morph jobs are fully mapped in this implementation.
            """
            params = recall_parameters(job_id)
            if not params or not isinstance(params, dict):
                return [gr.update()] * 25
            
            # Map for Morph Tab
            if "eq_in" in params:
                return [
                    gr.Tabs(selected="Morph"),
                    params.get("eq_global", 1.0), 
                    params.get("eq_in", 1.0), 
                    params.get("eq_mid", 1.0), 
                    params.get("eq_out", 1.0),
                    params.get("eq_adapter", 1.0), 
                    params.get("eq_other", 1.0), 
                    params.get("eq_interpolate", False),
                    params.get("temperature", 1.0), 
                    params.get("fft_cutoff", 1.0), 
                    params.get("clamp_quantile", 1.0),
                    params.get("fix_alpha", True), 
                    params.get("spectral_enabled", False), 
                    params.get("spectral_threshold", 0.1),
                    params.get("spectral_remove_structure", False),
                    params.get("spectral_adaptive", False),
                    params.get("dare_enabled", False), 
                    params.get("dare_rate", 0.1),
                    params.get("eraser_start", 0), 
                    params.get("eraser_end", 0),
                    params.get("homeostatic", False),
                    params.get("homeostatic_thr", 0.01),
                    params.get("erase_blocks", ""),
                    params.get("band_stop_enabled", False),
                    gr.update() # Placeholder for additional output
                ]
            
            # Fallback for unmapped job types
            return [gr.update()] * 25

        # Wire the Recall button from Queue tab
        qd["recall_btn"].click(
            do_recall_parameters,
            inputs=[qd["selected_id"]],
            outputs=[
                main_tabs, 
                mo["eq_global"], mo["eq_in"], mo["eq_mid"], mo["eq_out"],
                mo["eq_adapter"], mo["eq_other"], mo["eq_interpolate"],
                mo["temp"], mo["fft"], mo["clamp"], mo["fix_alpha"],
                mo["filter_chk"], mo["filter_thr"], mo["filter_inv"], mo["filter_adaptive"],
                mo["dare_chk"], mo["dare_rate"], mo["eraser_start"], mo["eraser_end"],
                mo["homeostatic"], mo["homeostatic_thr"], mo["erase_blocks"], mo["bs_chk"],
                qd["selected_id"]
            ]
        )

    return app