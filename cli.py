import argparse
import os
import sys
import logging
from app.pipeline import process_3d_pipeline

def main():
    parser = argparse.ArgumentParser(description="StyleForge 3D CLI")
    parser.add_argument("input_file", help="Path to input 3D model (OBJ/GLTF/GLB)")
    parser.add_argument("style_id", help="ID of the style (e.g. 1) or Name")
    parser.add_argument("--output", "-o", help="Output directory (default: current_dir/output)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)
        
    output_dir = args.output
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), "output")
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure logging to show info in CLI
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    try:
        print(f"Starting processing for {args.input_file} with style {args.style_id}...")
        final_path = process_3d_pipeline(args.input_file, args.style_id, output_dir)
        print(f"Success! Result saved to: {final_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
