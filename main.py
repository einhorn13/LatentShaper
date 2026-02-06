# main.py

import argparse
import sys
import os
from core.pipeline import ZPipeline
from core.config import ConfigManager
from core.logger import Logger
from core.version import __version__

def run_cli(args):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = ZPipeline(device=device)

    if args.command == "extract":
        Logger.info(f"Extracting to {args.output}...")
        for progress, msg in pipeline.extract_lora_gen(args.base, args.tuned, args.output, args.rank):
            print(f"[{int(progress*100)}%] {msg}", end="\r")
        Logger.info("Extraction Done.")
        
    elif args.command == "resize":
        Logger.info(f"Resizing to {args.output}...")
        for progress, msg in pipeline.resize_lora_gen(args.lora, args.output, args.rank):
            print(f"[{int(progress*100)}%] {msg}", end="\r")
        Logger.info("Resize Done.")
        
    elif args.command == "morph":
        Logger.info(f"Morphing to {args.output}...")
        params = {
            "eq_in": args.eq_in,
            "eq_mid": args.eq_mid,
            "eq_out": args.eq_out,
            "spectral_enabled": args.spectral,
            "spectral_threshold": args.spectral_thr,
            "spectral_remove_structure": args.spectral_inv,
            "dare_enabled": args.dare,
            "dare_rate": args.dare_rate
        }
        for progress, msg in pipeline.morph_lora_gen(args.lora, args.output, params):
            print(f"[{int(progress*100)}%] {msg}", end="\r")
        Logger.info("Morph Done.")

def main():
    Logger.info(f"Starting Latent Shaper v{__version__}...")
    
    config = ConfigManager()
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir, exist_ok=True)
        Logger.info(f"Output directory: {config.output_dir}")

    parser = argparse.ArgumentParser(description=f"Latent Shaper v{__version__} - Advanced LoRA & Checkpoint Toolkit")
    parser.add_argument("--version", action="version", version=f"Latent Shaper {__version__}")
    parser.add_argument("--gui", action="store_true", help="Launch Graphical User Interface")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: Extract
    extract_parser = subparsers.add_parser("extract", help="Extract LoRA")
    extract_parser.add_argument("--base", type=str, required=True, help="Base model path")
    extract_parser.add_argument("--tuned", type=str, required=True, help="Tuned model path")
    extract_parser.add_argument("--output", type=str, default="extracted.safetensors", help="Output LoRA path")
    extract_parser.add_argument("--rank", type=int, default=64, help="Rank")

    # Command: Resize
    resize_parser = subparsers.add_parser("resize", help="Resize LoRA")
    resize_parser.add_argument("--lora", type=str, required=True, help="Input LoRA path")
    resize_parser.add_argument("--output", type=str, default="resized.safetensors", help="Output LoRA path")
    resize_parser.add_argument("--rank", type=int, required=True, help="Target rank")

    # Command: Morph
    morph_parser = subparsers.add_parser("morph", help="Morph LoRA (EQ, DARE, Filter)")
    morph_parser.add_argument("--lora", type=str, required=True, help="Input LoRA path")
    morph_parser.add_argument("--output", type=str, default="morphed.safetensors", help="Output LoRA path")
    morph_parser.add_argument("--eq-in", type=float, default=1.0, help="EQ Input Blocks")
    morph_parser.add_argument("--eq-mid", type=float, default=1.0, help="EQ Mid Blocks")
    morph_parser.add_argument("--eq-out", type=float, default=1.0, help="EQ Output Blocks")
    morph_parser.add_argument("--dare", action="store_true", help="Enable DARE")
    morph_parser.add_argument("--dare-rate", type=float, default=0.1, help="DARE Drop Rate")
    morph_parser.add_argument("--spectral", action="store_true", help="Enable Spectral Filtering")
    morph_parser.add_argument("--spectral-thr", type=float, default=0.1, help="Spectral Threshold")
    morph_parser.add_argument("--spectral-inv", action="store_true", help="Invert Spectral Filter (Remove Structure)")

    args = parser.parse_args()

    if args.gui or len(sys.argv) == 1:
        Logger.info("Launching GUI...")
        import gui_launcher
        gui_launcher.launch()
    else:
        run_cli(args)

if __name__ == "__main__":
    main()