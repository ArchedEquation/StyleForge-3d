import bpy
import sys
import os

# Usage: blender --background --python pack_mesh.py -- <input_glb> <texture_path> <output_glb>

argv = sys.argv
if "--" not in argv:
    argv = []
else:
    argv = argv[argv.index("--") + 1:]

if len(argv) < 3:
    print("Usage: blender --background --python pack_mesh.py -- <input_glb> <texture_path> <output_glb>")
    sys.exit(1)

input_glb = argv[0]
texture_path = argv[1]
output_glb = argv[2]

print(f"Packing: {input_glb} with {texture_path}")

# 1. Clear/Reset
bpy.ops.wm.read_factory_settings(use_empty=True)

# 2. Import GLB
bpy.ops.import_scene.gltf(filepath=input_glb)

# 3. Find Material and Replace Texture
# We assume there is one material or we update all that use the bake texture.
# The previous script created a material named "BakeMat".
mat = bpy.data.materials.get("BakeMat")
if not mat:
    # Logic if material name changed or multiple materials
    # Just try to find the first material on the first mesh
    mesh_objs = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    if mesh_objs and mesh_objs[0].data.materials:
        mat = mesh_objs[0].data.materials[0]

if mat and mat.use_nodes:
    nodes = mat.node_tree.nodes
    # Find the Image Texture node.
    # We look for a node with an image.
    tex_node = None
    for n in nodes:
        if n.type == 'TEX_IMAGE':
            tex_node = n
            break
    
    if tex_node:
        # Load the newly styled image
        try:
            new_img = bpy.data.images.load(texture_path)
            tex_node.image = new_img
            print("Replaced texture image.")
        except Exception as e:
            print(f"Failed to load texture: {e}")
            sys.exit(1)
            
        # Ensure it is connected to Base Color for the export
        # Previous script connected it to nothing (just bake target), 
        # or we need to ensure it's connected to BSDF Base Color.
        bsdf = None
        for n in nodes:
            if n.type == 'BSDF_PRINCIPLED':
                bsdf = n
                break
        
        if bsdf:
            mat.node_tree.links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
            print("Linked texture to Base Color.")

# 4. Export
bpy.ops.export_scene.gltf(filepath=output_glb, export_format='GLB', export_uv=True)
print(f"Exported packed GLB to {output_glb}")
