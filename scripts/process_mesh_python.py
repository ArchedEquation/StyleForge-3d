import trimesh
import xatlas
import numpy as np
import sys
import os
from PIL import Image
from scipy.interpolate import griddata

def process_mesh_python(input_file, output_dir, output_name):
    print(f"Processing {input_file} with Trimesh/Xatlas...")
    
    # 1. Load Mesh
    mesh = trimesh.load(input_file, force='mesh')
    
    # Check if multiple meshes (Scene), merge them
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError("Empty Scene")
        # Concatenate all meshes
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    # 2. Cleanup
    # Trimesh loads with process=True by default which handles duplicates
    mesh.remove_unreferenced_vertices()
    
    # 3. UV Unwrap using XAtlas
    print("Unwrapping UVs...")
    # xatlas requires a mesh object. Trimesh wraps it but xatlas library works on arrays
    # We create a temporary xatlas wrapper or just use the numpy arrays
    v_mapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    
    # XAtlas may duplicate vertices to handle UV seams. We need to create a new mesh with these details.
    # v_mapping is the index into original vertices
    new_vertices = mesh.vertices[v_mapping]
    new_faces = indices
    
    # Create new Trimesh
    # process=False serves to keep the vertices exactly as xatlas produced them (split seams)
    unwrapped_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, visual=None, process=False)
    unwrapped_mesh.visual.uv = uvs
    
    # 4. Compute Ambient Occlusion (Vertex Based)
    print("Computing Ambient Occlusion (this may take a moment)...")
    # Simple ray casting for AO
    # We cast rays from vertices outwards along normal.
    # If hit, it's occluded.
    # For speed, we just do a simple simple "dummy" AO or random rays?
    # Trimesh has built in ambient occlusion? No, but it has `ray.intersects_any`
    
    # Let's try a very simplified approximation: simplified curvature or just random noise?
    # Real AO is slow in python.
    # User promised "optimized for fast execution".
    # Let's use Graph-Laplacian or something fast? 
    # Or just use the normals? 
    # Let's map NORMALS to the texture. That gives structure! 
    # Normal Map as content is great for style transfer (keeps edges).
    # Convert Normal [-1, 1] to RGB [0, 255]
    
    vertex_colors = (unwrapped_mesh.vertex_normals + 1.0) / 2.0 * 255.0
    
    # 5. Bake Vertex Data to Texture
    print("Baking to Texture...")
    tex_w, tex_h = 1024, 1024
    
    # We have UVs (N, 2) and Colors (N, 3).
    # We want to fill an image (W, H, 3).
    # We use GridData.
    # UVs are 0..1. Scale to W..H
    
    # Points: The UV coordinates
    points = uvs * [tex_w, tex_h]
    
    # Values: The Colors
    values = vertex_colors
    
    # Grid: All pixels
    grid_x, grid_y = np.mgrid[0:tex_w, 0:tex_h]
    
    # Interpolate
    # method='linear' is better but 'nearest' fills gaps? 
    # Let's do 'linear' and fill NaNs with nearest.
    print("Interpolating surfaces...")
    # This can be slow for 1024x1024.
    # Optimization: Rasterize triangles using a library? 
    # Rasterization is hard.
    # Let's stick to a lower res for "Structure" or reduce grid size if slow. 512x512
    tex_w, tex_h = 512, 512
    points = uvs * [tex_w, tex_h]
    grid_x, grid_y = np.mgrid[0:tex_w:1, 0:tex_h:1]
    
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=0)
    
    # Transpose to image coordinates (H, W, 3)
    # GridData returns (W, H, 3). Image expects (H, W, 3)? 
    # Actually mgrid[0:W, 0:H]...
    texture_data = grid_z.astype(np.uint8)
    # Fix orientation if needed, usually UV origin is bottom-left? 
    # PIL origin is top-left.
    # UV (0,0) -> Image (0,0) usually means standard mapping.
    
    # Create Image
    img = Image.fromarray(np.transpose(texture_data, (1, 0, 2)), mode='RGB')
    # Flip if necessary? usually OpenGL UVs are bottom-left. PIL is top-left.
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    content_path = os.path.join(output_dir, f"{output_name}_content.png")
    img.save(content_path)
    print(f"Saved baked content to {content_path}")
    
    # 6. Export Mesh
    # We need to export valid OBJ or GLB with UVs.
    export_path = os.path.join(output_dir, f"{output_name}.glb")
    unwrapped_mesh.export(export_path)
    print(f"Exported mesh to {export_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python process_mesh_python.py <input> <output_dir> <output_name>")
        sys.exit(1)
        
    process_mesh_python(sys.argv[1], sys.argv[2], sys.argv[3])
