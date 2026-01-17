import trimesh
import sys
import os
from PIL import Image
import numpy as np

def pack_mesh_python(input_glb, texture_path, output_glb):
    print(f"Packing {input_glb} with {texture_path}")
    
    mesh = trimesh.load(input_glb, force='mesh')
    
    # Load texture
    try:
        pil_image = Image.open(texture_path).convert("RGB")
    except Exception as e:
        print(f"Error loading texture: {e}")
        sys.exit(1)

    # Create a material with this image
    # Trimesh visuals
    material = trimesh.visual.material.PBRMaterial(
        name='StyledMaterial',
        baseColorTexture=pil_image
    )
    
    # Create new mesh with this visual
    # If the input was a scene, we might have issues, but our process script output a joined mesh.
    if isinstance(mesh, trimesh.Scene):
         # If scene, try to find the geometry
         if len(mesh.geometry) == 0:
             sys.exit(1)
         # Assume single geometry for MVP
         geom = list(mesh.geometry.values())[0]
    else:
        geom = mesh
        
    # Assign material
    geom.visual.material = material
    
    # Export
    # GLB export in Trimesh sometimes has issues with textures?
    # It usually works if Pillow is installed.
    geom.export(output_glb)
    print(f"Exported packed GLB to {output_glb}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python pack_mesh_python.py <input> <texture> <output>")
        sys.exit(1)
    pack_mesh_python(sys.argv[1], sys.argv[2], sys.argv[3])
