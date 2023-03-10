import bpy
import math

# Just an isolated example to create a corrosion area inside the cylinder causing reduction in cross sectional area
Corrosion_Depth = 0.2
column_thickness = 9.5/1000
Column_Internal_Radius = 0.14
column_depth = 0.75 #xample
corrosion_thickness = 4/1000
corrosion_length = 0.8
corrosion_start_vertical_location = 0.25 #start at 0.5m. Not real but easier to visualise
BasePlateDepth = 25/1000

#corrosion_start  #need to figure out the math of rotating the vetical plane to pizza slice the corrosion area. Good enough for now
angle1 = 0
#corrosion_end_sweep
angle2 = 1.7


bpy.ops.mesh.primitive_plane_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 1.5708, angle1), scale=(0.4, 0.4, 0.4))
bpy.ops.mesh.primitive_plane_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), 
rotation=(0,1.5708,angle2), scale=(1, 1, 1))

bpy.ops.mesh.primitive_cylinder_add(end_fill_type='NOTHING', vertices=32, radius=Column_Internal_Radius, depth=Corrosion_Depth, enter_editmode=False, align='WORLD', location=(0, 0, Corrosion_Depth/2+BasePlateDepth+corrosion_start_vertical_location), scale=(1, 1, 1))
bpy.context.object.name = "corrosion"
bpy.ops.object.modifier_add(type='SOLIDIFY')
bpy.context.object.modifiers["Solidify"].solidify_mode = 'NON_MANIFOLD'     #This one took me agessss. Not intuitive (its not extruding)
bpy.context.object.modifiers["Solidify"].thickness = corrosion_thickness
bpy.context.object.modifiers["Solidify"].offset = 1
bpy.ops.object.modifier_add(type='BOOLEAN')
bpy.context.object.modifiers["Boolean"].object = bpy.data.objects["Plane"]
bpy.ops.object.modifier_add(type='BOOLEAN')
bpy.context.object.modifiers["Boolean.001"].object = bpy.data.objects["Plane.001"]
bpy.ops.object.apply_all_modifiers()
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_pattern(pattern="Plane*")
bpy.ops.object.delete(use_global=False)
bpy.ops.object.select_pattern(pattern="corrosion")

bpy.ops.mesh.primitive_cylinder_add(end_fill_type='NOTHING',vertices=32, radius=Column_Internal_Radius, depth=column_depth, enter_editmode=False, align='WORLD', location=(0, 0, column_depth/2+BasePlateDepth), scale=(1, 1, 1))

bpy.ops.object.modifier_add(type='SOLIDIFY')
bpy.context.object.modifiers["Solidify"].offset = 1
bpy.context.object.modifiers["Solidify"].thickness = column_thickness


bpy.ops.object.modifier_add(type='BOOLEAN')
bpy.context.object.modifiers["Boolean"].object = bpy.data.objects["corrosion"]

bpy.ops.object.apply_all_modifiers()
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_pattern(pattern="corrosion*")
bpy.ops.object.delete(use_global=False)

bpy.context.space_data.shading.show_xray = True
bpy.context.space_data.shading.show_shadows = True
bpy.context.space_data.shading.color_type = 'TEXTURE'
