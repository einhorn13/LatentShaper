# gui/actions/analysis.py

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
from .common import resolve_path_priority
from ..context import pipeline, advisor

def run_analysis(ws_in, disk_in, up_in, progress=gr.Progress()):
    inputs = resolve_path_priority(ws_in, disk_in, up_in)
    if not inputs: 
        return None, None, None, "<div style='color:red'>❌ Error: No model selected.</div>", [], gr.update(choices=[], value=None)
    
    target = inputs[0]
    data = None
    for p, m, res in pipeline.analyze_spectrum_gen(target):
        progress(p, desc=m)
        if res: data = res
    
    if not data or not data.get("spectrum"): 
        return None, None, None, "<div style='color:red'>❌ Analysis failed.</div>", [], gr.update(choices=[], value=None)

    # 1. Plots
    clean_spectrum = [float(x) for x in data["spectrum"]]
    df_spectrum = pd.DataFrame({"Rank": range(1, len(clean_spectrum)+1), "Value": clean_spectrum, "LogValue": np.log10(np.array(clean_spectrum)+1e-9).tolist()})
    df_energy = pd.DataFrame({"Region": list(data["block_energy"].keys()), "Energy": list(data["block_energy"].values())})

    heatmap_img = None
    if data.get("heatmap"):
        plt.figure(figsize=(10, 12))
        sns.heatmap(np.array(data["heatmap"]), cmap="viridis")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        heatmap_img = Image.open(buf)
        plt.close()

    # 2. Evaluate Recommendations
    metrics = {
        "kurtosis": data["kurtosis"], 
        "block_energy": data["block_energy"], 
        "current_rank": data["avg_rank"], 
        "current_alpha": data["avg_alpha"], 
        "magnitude": data["magnitude"], 
        "knee_rank": data["knee_rank"],
        "intrinsic_rank": data.get("intrinsic_rank", data["knee_rank"])
    }
    recommendations = advisor.evaluate(metrics)
    
    active_profile_html = ""
    for rec in recommendations:
        if rec.id.startswith("psp_"):
            color = "#ff9800" if rec.severity == "warning" else "#2196f3"
            active_profile_html = f"<span style='background:{color}; color:white; padding:2px 8px; border-radius:4px; font-weight:bold;'>{rec.title}</span>"
            break

    report_html = f"""
    <div style='line-height:1.6'>
        <b>Architecture:</b> {data['model_name']}<br>
        <b>Profile:</b> {active_profile_html or 'Neutral'}<br>
        <b>Avg Rank:</b> {data['avg_rank']} | <b>Intrinsic Rank (95%):</b> {data.get('intrinsic_rank', 'N/A')}<br>
        <b>Avg Alpha:</b> {data['avg_alpha']}<br>
        <b>Mid Dominance:</b> {data.get('mid_dominance', 1.0):.2f}<br>
        <b>Kurtosis:</b> {data['kurtosis']:.2f} | <b>Magnitude:</b> {data['magnitude']:.5f}
    </div>
    """
    
    rec_choices = [r.title for r in recommendations]
    dropdown_update = gr.update(choices=rec_choices, value=rec_choices[0] if rec_choices else None, interactive=bool(rec_choices))

    return df_spectrum, df_energy, heatmap_img, report_html, recommendations, dropdown_update

def apply_recommendation(recommendations, selected_title, current_file, current_drop):
    # Note: This function signature might need update if we want to auto-fill the new selectors
    # For now, we return updates for the Morph tab parameters only.
    if not recommendations or not selected_title: return [gr.update()] * 23
    target_rec = next((r for r in recommendations if r.title == selected_title), None)
    if not target_rec: return [gr.update()] * 23

    p = target_rec.params
    def val(key, default=gr.update()): return p[key] if key in p else default
    
    # We don't update file inputs here to avoid complexity, user manually switches tab
    return [
        gr.Tabs(selected="Morph"), 
        # Skip file inputs updates for now
        gr.update(), gr.update(), gr.update(),
        val("eq_global"), val("eq_in"), val("eq_mid"), val("eq_out"), False,
        val("temperature"), val("fft_cutoff"), val("clamp_quantile"), val("fix_alpha_chk", True),
        val("spectral_enabled", False), val("spectral_threshold", 0.1), val("spectral_remove_structure", False), 
        val("spectral_adaptive", False), val("dare_enabled", False), val("dare_rate", 0.1), 
        val("eraser_start", 0), val("eraser_end", 0),
        val("homeostatic", False), val("homeostatic_thr", 0.01)
    ]