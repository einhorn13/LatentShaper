# gui/actions/submission.py

import gradio as gr
import os
from core.structs import Job, JobType
from .common import _make_desc, resolve_path_priority, validate_and_fix_filename
from ..context import queue_manager, pipeline, workspace

def submit_extract(base_ws, base_disk, tuned_ws, tuned_disk, tuned_upload, rank, threshold, out, save_ws):
    # Base Model (Single)
    base_paths = resolve_path_priority(base_ws, base_disk, None)
    if not base_paths: return gr.Info("Error: Select Base Model.")
    base_path = base_paths[0]

    # Tuned Models (Multiple)
    tuned_paths = resolve_path_priority(tuned_ws, tuned_disk, tuned_upload)
    if not tuned_paths: return gr.Info("Error: Select Tuned Model.")
        
    try: final_out = validate_and_fix_filename(out, save_ws)
    except ValueError as e: return gr.Warning(str(e))
    
    desc = _make_desc("Extract", tuned_paths, f"R{rank}/T{threshold}")
    
    params = {
        'base_path': base_path, 
        'rank': int(rank), 
        'threshold': float(threshold),
        'save_to_workspace': save_ws
    }
    
    job = Job(JobType.EXTRACT, tuned_paths, final_out, params, description=desc)
    queue_manager.submit_job(job)
    return gr.Info(f"Queued: {desc}")

def submit_resize(ws_in, disk_in, up_in, rank, out, auto_rank_chk, auto_rank_thr, save_ws):
    inputs = resolve_path_priority(ws_in, disk_in, up_in)
    if not inputs: return gr.Info("Error: No files selected.")
    try: final_out = validate_and_fix_filename(out, save_ws)
    except ValueError as e: return gr.Warning(str(e))
    
    mode = " (Auto)" if auto_rank_chk else ""
    desc = _make_desc("Resize", inputs, f"-> {rank}{mode}")
    job = Job(JobType.RESIZE, inputs, final_out, {'rank': int(rank), 'auto_rank_threshold': float(auto_rank_thr) if auto_rank_chk else 0.0, 'save_to_workspace': save_ws}, description=desc)
    queue_manager.submit_job(job)
    return gr.Info(f"Queued: {desc}")

def submit_morph(
    ws_in, disk_in, up_in, out_name, 
    eq_global, eq_in, eq_mid, eq_out, eq_interpolate,
    temp_val, fft_cutoff, clamp_val, fix_alpha_chk,
    filter_chk, filter_thr, filter_inv, filter_adaptive,
    dare_chk, dare_rate, 
    eraser_start, eraser_end,
    save_ws,
    homeostatic_chk, homeostatic_thr,
    erase_blocks_str,
    bs_chk, bs_start, bs_end
):
    inputs = resolve_path_priority(ws_in, disk_in, up_in)
    if not inputs: return gr.Info("Error: No files selected.")
    try: final_out = validate_and_fix_filename(out_name, save_ws)
    except ValueError as e: return gr.Warning(str(e))
    
    params = {
        "eq_global": float(eq_global), "eq_in": float(eq_in), "eq_mid": float(eq_mid), "eq_out": float(eq_out),
        "eq_interpolate": eq_interpolate,
        "temperature": float(temp_val), "fft_cutoff": float(fft_cutoff), 
        "clamp_quantile": float(clamp_val), "fix_alpha": fix_alpha_chk,
        "spectral_enabled": filter_chk, "spectral_threshold": float(filter_thr), 
        "spectral_remove_structure": filter_inv, "spectral_adaptive": filter_adaptive,
        "dare_enabled": dare_chk, "dare_rate": float(dare_rate),
        "eraser_start": int(eraser_start), "eraser_end": int(eraser_end),
        "save_to_workspace": save_ws,
        "homeostatic": homeostatic_chk, "homeostatic_thr": float(homeostatic_thr),
        "erase_blocks": erase_blocks_str,
        "band_stop_enabled": bs_chk, "band_stop_start": float(bs_start), "band_stop_end": float(bs_end)
    }
    
    mode_str = "Interpolated" if eq_interpolate else "Regions"
    desc = _make_desc("Morph", inputs, f"[{mode_str}]")
    job = Job(JobType.MORPH, inputs, final_out, params, description=desc)
    queue_manager.submit_job(job)
    return gr.Info(f"Queued: {desc}")

def submit_bridge(ws_in, disk_in, up_in, out_name, strength, save_ws):
    inputs = resolve_path_priority(ws_in, disk_in, up_in)
    if not inputs: return gr.Info("Error: Please select a source SDXL LoRA.")
    try: final_out = validate_and_fix_filename(out_name, save_ws)
    except ValueError as e: return gr.Warning(str(e))
    
    params = {"strength": float(strength), "save_to_workspace": save_ws, "is_bridge": True}
    fname = os.path.basename(inputs[0])
    job = Job(JobType.MORPH, [inputs[0]], final_out, params, description=f"Bridge: {fname} -> Turbo")
    queue_manager.submit_job(job)
    return gr.Info("Migration Queued")

def submit_utils(ws_in, disk_in, up_in, out, prec, norm, alpha_mode, alpha_val, save_ws):
    inputs = resolve_path_priority(ws_in, disk_in, up_in)
    if not inputs: return gr.Info("Error: No files selected.")
    try: final_out = validate_and_fix_filename(out, save_ws)
    except ValueError as e: return gr.Warning(str(e))
    
    params = {
        "precision": prec,
        "normalize_keys": norm,
        "save_to_workspace": save_ws
    }
    
    if alpha_mode == "Sync with Rank":
        params["alpha_equals_rank"] = True
    elif alpha_mode == "Set Custom":
        params["target_alpha"] = float(alpha_val)
        
    desc = _make_desc("Utils", inputs, f"[{prec}]")
    job = Job(JobType.UTILS, inputs, final_out, params, description=desc)
    queue_manager.submit_job(job)
    return gr.Info(f"Queued: {desc}")

def submit_merge(w_data, path_map, rank, out, algo_raw, global_str, auto_rank_chk, auto_rank_thr, prune_chk, prune_thr, ties_density, save_ws):
    if w_data is None: return gr.Info("Error: Merge list is empty.")
    if hasattr(w_data, "values"): w_data = w_data.values.tolist()
    try: final_out = validate_and_fix_filename(out, save_ws)
    except ValueError as e: return gr.Warning(str(e))
    
    algo = algo_raw.split(" ")[0] if algo_raw else "SVD"
    files, ratios = [], []
    for row in w_data:
        if not row or len(row) < 3: continue
        fname = row[0]
        if path_map and fname in path_map: files.append(path_map[fname])
        elif workspace.exists(fname): files.append(fname)
        else: continue
        ratios.append(float(row[2]))
            
    if not files: return gr.Info("Error: No valid models to merge.")
    
    params = {
        'ratios': ratios, 'rank': int(rank), 'algorithm': algo, 'global_strength': float(global_str),
        'auto_rank_threshold': float(auto_rank_thr) if auto_rank_chk else 0.0,
        'pruning_threshold': float(prune_thr) if prune_chk else 0.0,
        'ties_density': float(ties_density), 'save_to_workspace': save_ws
    }
    job = Job(JobType.MERGE, files, final_out, params, description=f"Merge {len(files)} via {algo}")
    queue_manager.submit_job(job)
    return gr.Info("Merge Queued")

def add_files_to_merge(new_files, current_data, current_map):
    if not new_files: return current_data, current_map, None
    updated_map = current_map.copy() if current_map else {}
    updated_data = current_data.values.tolist() if hasattr(current_data, "values") else (current_data or [])
    existing = {row[0] for row in updated_data}
    for f in new_files:
        # Handle both file objects and strings (paths)
        path = f.name if hasattr(f, 'name') else str(f)
        fname = os.path.basename(path)
        if fname not in existing:
            updated_data.append([fname, str(pipeline.get_lora_info(path)), 1.0])
            updated_map[fname] = path
    return updated_data, updated_map, None

def add_workspace_to_merge(selected_model, current_data):
    if not selected_model: return current_data, gr.update()
    updated_data = current_data.values.tolist() if hasattr(current_data, "values") else (current_data or [])
    if selected_model not in {row[0] for row in updated_data}:
        m = workspace.get_model(selected_model)
        if m: updated_data.append([selected_model, str(m.info.get("rank", "?")), 1.0])
    return updated_data, None

def clear_merge_list(): return [], {}, None
def normalize_weights_ui(data):
    if data is None: return []
    rows = data.values.tolist() if hasattr(data, "values") else data
    total_abs = sum(abs(float(r[2])) for r in rows)
    if total_abs == 0: return data
    return [[r[0], r[1], round(float(r[2])/total_abs, 4)] for r in rows]
def distribute_weights_ui(data):
    if data is None: return []
    rows = data.values.tolist() if hasattr(data, "values") else data
    count = len(rows)
    if count == 0: return data
    val = round(1.0 / count, 4)
    return [[r[0], r[1], val] for r in rows]
def invert_weights_ui(data):
    if data is None: return []
    rows = data.values.tolist() if hasattr(data, "values") else data
    return [[r[0], r[1], round(float(r[2]) * -1.0, 4)] for r in rows]