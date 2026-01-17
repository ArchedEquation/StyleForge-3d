import os
import shutil
import subprocess
import uuid
import logging
import time
import sys
from app.style_transfer import run_style_transfer

from app.config import get_blender_exec

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STYLES_DIR = os.path.join(BASE_DIR, "styles")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

logger = logging.getLogger("StyleForgePipeline")

def process_3d_pipeline(input_path: str, style_id: str, output_dir: str):
    """
    Executes the full 3D style transfer pipeline.
    
    Args:
        input_path (str): Path to the input 3D file (.obj, .glb, etc).
        style_id (str): ID of the style to apply (e.g. "1").
        output_dir (str): Directory to save intermediate and final results.
        
    Returns:
        str: Path to the final stylized GLB file.
    """
    
    # 1. Validate Style
    style_filename = None
    if os.path.exists(STYLES_DIR):
        for f in os.listdir(STYLES_DIR):
            if f.startswith(f"style-{style_id}."):
                style_filename = f
                break
    
    if not style_filename:
        # Fallback for CLI if full path provided or loose check
        if os.path.exists(style_id):
            style_filename = style_id # Treat as path
        else:
            raise ValueError(f"Style {style_id} not found in {STYLES_DIR}")

    style_path = os.path.join(STYLES_DIR, style_filename) if style_filename in os.listdir(STYLES_DIR) else style_filename

    # 2. Step 1: Geometry Processing (Unwrap & Bake) (No Blender)
    logger.info("Step 1: Geometry Processing (Unwrap & Bake - Python)...")
    cmd = [
        sys.executable,  
        os.path.join(SCRIPTS_DIR, "process_mesh_python.py"),
        input_path, output_dir, "processed"
    ]
    
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"Blender output: {res.stdout}")
        logger.error(f"Blender error: {res.stderr}")
        raise RuntimeError(f"Geometry processing failed: {res.stderr}")
    logger.info(f"Geometry processed in {time.time() - t0:.2f}s")

    # Check for baked content
    baked_content = os.path.join(output_dir, "processed_content.png")
    if not os.path.exists(baked_content):
         raise RuntimeError("Failed to generate content map")

    # 3. Step 2: Style Transfer
    logger.info("Step 2: Neural Style Transfer...")
    stylized_texture = os.path.join(output_dir, "processed_style.png")
    
    # Run NST
    # Adjust steps based on need? Defaulting to 150 as per previous code.
    run_style_transfer(baked_content, style_path, stylized_texture, num_steps=150)
    
    logger.info("Step 3: Packing Final Model...")
    processed_glb = os.path.join(output_dir, "processed.glb")
    final_glb = os.path.join(output_dir, "styleforge_result.glb")
    
    pack_cmd = [
        sys.executable, 
        os.path.join(SCRIPTS_DIR, "pack_mesh_python.py"),
        processed_glb, stylized_texture, final_glb
    ]
    res_pack = subprocess.run(pack_cmd, capture_output=True, text=True)
    if res_pack.returncode != 0:
        raise RuntimeError(f"Packing failed: {res_pack.stderr}")
        
    logger.info(f"Pipeline Completed. Output: {final_glb}")
    return final_glb
