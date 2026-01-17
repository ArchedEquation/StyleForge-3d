import bpy
import sys
import os

# Argument parsing manually since bpy consumes argv
# Usage: blender --background --python process_mesh.py -- <input_file> <output_dir> <output_name>

argv = sys.argv
if "--" not in argv:
    argv = []  # as if no args are passed
else:
    argv = argv[argv.index("--") + 1:]

if len(argv) < 3:
    print("Usage: blender --background --python process_mesh.py -- <input_file> <output_dir> <output_name>")
    sys.exit(1)

input_file = argv[0]
output_dir = argv[1]
output_name = argv[2] # e.g. "model" without extension

print(f"Processing: {input_file}")

# 1. Clear existing scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# 2. Import File
ext = os.path.splitext(input_file)[1].lower()
if ext == ".obj":
    bpy.ops.import_scene.obj(filepath=input_file)
elif ext in [".gltf", ".glb"]:
    bpy.ops.import_scene.gltf(filepath=input_file)
elif ext == ".fbx":
    bpy.ops.import_scene.fbx(filepath=input_file)
else:
    print(f"Unsupported extension: {ext}")
    sys.exit(1)

# 3. Select the mesh object
mesh_objs = [obj for obj in bpy.data.objects if obj.type == 'MESH']
if not mesh_objs:
    print("No mesh found.")
    sys.exit(1)

# Join meshes if multiple (simplification)
ctx = bpy.context.copy()
ctx['active_object'] = mesh_objs[0]
ctx['selected_editable_objects'] = mesh_objs
bpy.ops.object.join(ctx)
target_obj = ctx['active_object']

# 4. Cleanup and UV Unwrap
bpy.ops.object.select_all(action='DESELECT')
target_obj.select_set(True)
bpy.context.view_layer.objects.active = target_obj
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles()
bpy.ops.uv.smart_project(island_margin=0.02)
bpy.ops.object.mode_set(mode='OBJECT')

# 5. Setup Material and Texture for Baking
mat = bpy.data.materials.new(name="BakeMat")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

# Add Image Texture node
tex_node = nodes.new('ShaderNodeTexImage')
img = bpy.data.images.new("ContentMap", width=1024, height=1024)
tex_node.image = img

# Add Principled BSDF (just to have a material output, though baking specific channels ignores this mostly)
bsdf = nodes.new('ShaderNodeBsdfPrincipled')
out = nodes.new('ShaderNodeOutputMaterial')
links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

# Assign material
if target_obj.data.materials:
    target_obj.data.materials[0] = mat
else:
    target_obj.data.materials.append(mat)

# 6. Bake Ambient Occlusion (AO)
# We need to use Cycles for baking
bpy.context.scene.render.engine = 'CYCLES'
# Set low samples for speed
bpy.context.scene.cycles.samples = 16
bpy.context.scene.cycles.device = 'CPU' # Safest for headless
# Make sure the texture node is selected for baking target
nodes.active = tex_node
tex_node.select = True

print("Baking AO...")
bpy.ops.object.bake(type='AO')

# Save the Baked Image
baked_image_path = os.path.join(output_dir, f"{output_name}_content.png")
img.filepath_raw = baked_image_path
img.file_format = 'PNG'
img.save()
print(f"Saved content map to {baked_image_path}")

# 7. Export the Mesh (GLB)
# Ideally, we want the GLB to reference the TEXTURE we will eventually generate.
# However, for now, we export the mesh with UVs.
# We will manipulate the texture file post-generation.
export_path = os.path.join(output_dir, f"{output_name}.glb")
bpy.ops.export_scene.gltf(filepath=export_path, export_format='GLB', export_uv=True)
print(f"Exported mesh to {export_path}")
