#Write the below command to the Blender script
# import bpy
# import os
# bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
# bpy.ops.object.select_all(action='SELECT')
# bpy.ops.object.delete(use_global=False, confirm=False)
# filename = os.path.join(os.path.dirname(bpy.data.filepath), r"C:\Users\denta\Downloads\gmsh-4.11.1-Windows64\pythonscript.py")
# exec(compile(open(filename).read(), filename, 'exec'))
#Then:
# Go to window > toggle system console

# Below Script is just a starting point. Modify to suit


# Script to generate model in blender
# Still havent done adding addtion loop, extrude manifold to generate corrosion issue causing cross section reduction
import bpy
import numpy as np
import os

def createPanel(x,y,z, lx,ly,lz, name="Panel"):
    bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=(lx, ly, lz), scale=(1, 1, 1))
    bpy.ops.transform.resize(value=(x,y,z), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.context.object.name = name

COLUMN_RADIUS = 0.14
PANEL_WIDTH = COLUMN_RADIUS*2+20*0.001
PANEL_LENGTH = 3
PANEL_DEPTH = 2
COLUMN_HEIGHT = 9
BASEPLATE_THICKNESS = 0.025
BASEPLATE_RADIUS = 0.3

PATH_TO_GMSH = r"C:\Users\denta\Downloads\gmsh-4.11.1-Windows64\gmsh.exe"
USERNAME = "denta"


bpy.ops.mesh.primitive_elbow_joint_add(align='WORLD', location=(0, 0, COLUMN_HEIGHT/2+BASEPLATE_THICKNESS), rotation=(0, 0, 0), change=False, radius=COLUMN_RADIUS, div=32, angle=0.00, startLength=COLUMN_HEIGHT/2, endLength=COLUMN_HEIGHT/2)
bpy.ops.object.modifier_add(type='SOLIDIFY')
bpy.context.object.modifiers["Solidify"].thickness = 9.5/1000
bpy.ops.object.apply_all_modifiers()
bpy.context.object.name = "Column"


createPanel(x=PANEL_WIDTH, y=PANEL_LENGTH, z=PANEL_DEPTH, lx=0, ly=PANEL_LENGTH/2-COLUMN_RADIUS-10/1000, lz=COLUMN_HEIGHT+BASEPLATE_THICKNESS-PANEL_DEPTH/2, name="Panel1")
bpy.ops.object.modifier_add(type='BOOLEAN')
bpy.context.object.modifiers["Boolean"].object = bpy.data.objects["Column"]
bpy.ops.object.apply_all_modifiers()
bpy.ops.mesh.primitive_cylinder_add(radius=BASEPLATE_RADIUS, depth=BASEPLATE_THICKNESS, enter_editmode=False, align='WORLD', location=(0, 0, BASEPLATE_THICKNESS/2), scale=(1, 1, 1))


# Script to convert each object in Blender into an '.inp' file to load to PrePoMax
for obj in bpy.data.objects:
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    name = obj.name
    bpy.ops.export_mesh.stl(filepath=rf"C:\Users\{USERNAME}\OneDrive\Documents\{name}.stl", check_existing=True, filter_glob="*.stl", use_selection=True, global_scale=1, ascii=False, use_mesh_modifiers=True, batch_mode='OFF', axis_forward='-Z', axis_up='Y')
    with open(rf"C:\Users\{USERNAME}\OneDrive\Documents\volume.geo",'w') as script:
        built_in = '"Built-in"'
        script.write(rf"Merge 'C:\Users\{USERNAME}\OneDrive\Documents\{name}.stl';\n//+\nSetFactory({built_in});\n//+\nSurface Loop(1) = {1};\n//+\nVolume(1) = {1};")
    os.system(rf"{PATH_TO_GMSH} C:\Users\{USERNAME}\OneDrive\Documents\volume.geo -3 -o C:\Users\{USERNAME}\OneDrive\Documents\{name}.inp")
