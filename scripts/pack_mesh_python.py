import trimesh
import sys
import os
from PIL import Image
import numpy as np

def pack_mesh_python(input_glb, texture_path, output_glb):
    print(f"Packing {input_glb} with {texture_path}")
    
    mesh = trimesh.load(input_glb, force='mesh')
    
    # Load texture
    # Load texture
    if not os.path.exists(texture_path):
        print(f"Error: Texture file not found at {texture_path}")
        sys.exit(1)

    try:
        pil_image = Image.open(texture_path).convert("RGB")
        # Step 1: Verify NST Output Integrity (Basic Check)
        if pil_image.size == (0, 0):
             raise ValueError("Empty image")
    except Exception as e:
        print(f"Error loading texture: {e}")
        sys.exit(1)

    # 3. Create PBR Material
    # Ensure we use 'baseColorTexture' equivalent in Trimesh
    material = trimesh.visual.material.PBRMaterial(
        name='StyledMaterial',
        baseColorTexture=pil_image,
        metallicFactor=0.0,
        roughnessFactor=1.0, 
        alphaMode='OPAQUE'
    )
    
    # 4. Handle Scene/Mesh
    if isinstance(mesh, trimesh.Scene):
         if len(mesh.geometry) == 0:
             print("Error: Empty geometry in GLB")
             sys.exit(1)
         # For simplicity, we grab the first geometry or join them
         # Ideally, we should iterate, but our pipeline produces single mesh
         geom = list(mesh.geometry.values())[0]
         # Just in case, if user provided a scene with multiple parts, warning.
         if len(mesh.geometry) > 1:
             print("Warning: Multiple geometries found. Applying texture to first one only.")
    else:
        geom = mesh

    # 5. Validate UV Mapping (Step 4)
    if not hasattr(geom.visual, 'uv') or geom.visual.uv is None or len(geom.visual.uv) == 0:
        print("Error: Mesh has no UV coordinates. Cannot apply texture.")
        # Try to retrieve from vertex attributes if missing in visual?
        if hasattr(geom, 'vertices') and len(geom.vertices) > 0:
             # Check if we can generate or find them? 
             # No, pipeline upstream MUST provide UVs.
             sys.exit(1)
        sys.exit(1)
        
    # Check UV bounds roughly?
    uvs = geom.visual.uv
    if uvs.min() < -2.0 or uvs.max() > 2.0:
        print("Warning: UVs outside typical 0-1 range (wrapped?). This is allowed but worth noting.")

    # 6. Apply Texture via TextureVisuals (Step 3)
    # Force TextureVisuals to ensure GLB uses texture
    from trimesh.visual import TextureVisuals
    
    # We construct a new visual object
    new_visual = TextureVisuals(
        uv=geom.visual.uv,
        image=pil_image,
        material=material
    )
    geom.visual = new_visual
    
    # 7. Final Validation & Export (Step 8)
    # Trimesh export logic for GLB should pick up the PBR material
    geom.export(output_glb)
    print(f"Exported packed GLB to {output_glb}")
    
    # Optional: Validity Check of the exported file directly? 
    # That requires reloading it and checking materials.
    if os.path.exists(output_glb):
        print("Validation: GLB created successfully.")
    else:
        print("Error: GLB export failed.")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python pack_mesh_python.py <input> <texture> <output>")
        sys.exit(1)
    pack_mesh_python(sys.argv[1], sys.argv[2], sys.argv[3])
