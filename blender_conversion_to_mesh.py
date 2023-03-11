import bpy
import os
import numpy as np


def select_object(object_name):
    bpy.ops.object.select_all(action='DESELECT')
    obj = bpy.context.scene.objects[object_name]
    bpy.data.objects[object_name].select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def delete_object(object_name):
    select_object(object_name)
    bpy.ops.object.delete(use_global=False, confirm=False)


def slice_object(object_name_1, object_name_2, self_select=False, scale=False):
    if scale == True:
        select_object(object_name_2)
        bpy.ops.transform.resize(value=(3, 3, 3), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                                 orient_matrix_type='GLOBAL', mirror=False, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1,
                                 use_proportional_connected=False, use_proportional_projected=False, snap=False,
                                 snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST',
                                 use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True,
                                 use_snap_selectable=False)

    select_object(object_name_1)
    bpy.ops.object.modifier_add(type='BOOLEAN')
    bpy.context.object.modifiers["Boolean"].operation = 'INTERSECT'

    bpy.context.object.modifiers["Boolean"].use_self = self_select

    bpy.context.object.modifiers["Boolean"].object = bpy.data.objects[object_name_2]
    bpy.ops.object.apply_all_modifiers()


def difference(object_name_1, object_name_2):
    select_object(object_name_1)
    bpy.ops.object.modifier_add(type='BOOLEAN')
    bpy.context.object.modifiers["Boolean"].operation = 'DIFFERENCE'
    bpy.context.object.modifiers["Boolean"].use_self = True
    bpy.context.object.modifiers["Boolean"].object = bpy.data.objects[object_name_2]
    bpy.ops.object.apply_all_modifiers()


def rename_object(object_name, new_name):
    select_object(object_name)
    bpy.context.object.name = new_name


def duplicate_object(object_name, new_name):
    select_object(object_name)
    bpy.ops.object.duplicate_move(OBJECT_OT_duplicate={"linked": False, "mode": 'TRANSLATION'},
                                  TRANSFORM_OT_translate={"value": (0, 0, 0), "orient_axis_ortho": 'X',
                                                          "orient_type": 'GLOBAL',
                                                          "orient_matrix": ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
                                                          "orient_matrix_type": 'GLOBAL',
                                                          "constraint_axis": (False, False, False), "mirror": False,
                                                          "use_proportional_edit": False,
                                                          "proportional_edit_falloff": 'SMOOTH', "proportional_size": 1,
                                                          "use_proportional_connected": False,
                                                          "use_proportional_projected": False, "snap": False,
                                                          "snap_elements": {'INCREMENT'}, "use_snap_project": False,
                                                          "snap_target": 'CLOSEST', "use_snap_self": True,
                                                          "use_snap_edit": True, "use_snap_nonedit": True,
                                                          "use_snap_selectable": False, "snap_point": (0, 0, 0),
                                                          "snap_align": False, "snap_normal": (0, 0, 0),
                                                          "gpencil_strokes": False, "cursor_transform": False,
                                                          "texture_space": False, "remove_on_cancel": False,
                                                          "view2d_edge_pan": False, "release_confirm": False,
                                                          "use_accurate": False, "use_automerge_and_split": False})
    bpy.context.object.name = new_name
def rotate_object(object_name, angle1, angle2):
    angle = np.deg2rad(angle2 - angle1)
    select_object(object_name)
    bpy.ops.transform.rotate(value=angle, orient_axis='Z', orient_type='GLOBAL',
                             orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             constraint_axis=(False, False, True), mirror=False, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                             use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'},
                             use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True,
                             use_snap_nonedit=True, use_snap_selectable=False)


def createKnife(angle1):
    bpy.ops.mesh.primitive_plane_add(size=20, enter_editmode=False,
                                     align='WORLD',
                                     location=(0, 0, 0),
                                     rotation=(0, np.deg2rad(90), np.deg2rad(angle1)),
                                     scale=(1, 1, 1))
    bpy.ops.object.select_all(action='DESELECT')


def create_corrosion_pizza(cylinder_radius, corrosion_height, corrosion_thickness):
    bpy.ops.mesh.primitive_cylinder_add(end_fill_type='NOTHING', radius=cylinder_radius, depth=corrosion_height,
                                        enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.ops.object.modifier_add(type='SOLIDIFY')
    bpy.context.object.modifiers["Solidify"].solidify_mode = 'NON_MANIFOLD'
    bpy.context.object.modifiers["Solidify"].offset = 1
    bpy.context.object.modifiers["Solidify"].thickness = corrosion_thickness
    bpy.ops.object.apply_all_modifiers()
    bpy.ops.object.select_all(action='DESELECT')


def pizza_slice(cylinder_outer_radius, cylinder_thickness, corrosion_thickness, corrosion_height, angle1, angle2,
                object_name):
    createKnife(angle1)
    create_corrosion_pizza(cylinder_outer_radius - cylinder_thickness, corrosion_height, corrosion_thickness)
    rename_object("Cylinder", object_name)
    slice_object(object_name, "Plane", self_select=True)
    duplicate_object(object_name, "Tempt")
    rotate_object("Plane", angle1, angle2)
    slice_object("Tempt", "Plane", self_select=True, scale=True)
    difference(object_name, "Tempt")
    delete_object("Tempt")
    difference(object_name, "Plane")
    delete_object("Plane")


def deleteall():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)


def calcualteAngle(length, radius):
    circumference = 2 * np.pi * radius
    if length > circumference:
        print("Length is greater than circumference")
        return 0
    else:
        angle = length / circumference * 360
    return angle


def createAngle(length, cylinder_outer_radius):
    angle2 = calcualteAngle(length, cylinder_outer_radius)
    angle1 = 90 - angle2 / 2
    angle2 = angle1 + angle2
    return angle1, angle2


def moveObject(object_name, x, y, z):
    select_object(object_name)
    bpy.ops.transform.translate(value=(x, y, z), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                                orient_matrix_type='GLOBAL', constraint_axis=(False, False, False), mirror=False,
                                use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1,
                                use_proportional_connected=False, use_proportional_projected=False, snap=False,
                                snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST',
                                use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True,
                                use_snap_selectable=False)


def createCorrosion(length, cylinder_outer_radius, cylinder_thickness, corrosion_thickness, corrosion_height,
                    baseplate_thickness):
    angle1, angle2 = createAngle(length, cylinder_outer_radius)
    pizza_slice(cylinder_outer_radius, cylinder_thickness, corrosion_thickness, corrosion_height, angle1, angle2,
                "column")
    moveObject("column", 0, 0, corrosion_height / 2 + baseplate_thickness)


def createColumn(radius,height,thickness,z,name='Column'):
    bpy.ops.mesh.primitive_elbow_joint_add(align='WORLD',
            location=(0, 0,z),
            rotation=(0, 0, 0),
            change=False,
            radius=radius,
            div=32, angle=0.00,
            startLength=height/2,
            endLength=height/2)

    bpy.context.object.name = name
    bpy.ops.object.modifier_add(type='SOLIDIFY')
    bpy.context.object.modifiers["Solidify"].thickness =thickness
    bpy.ops.object.apply_all_modifiers()


def createbaseplate(baseplate_radius, baseplate_thickness, name='Baseplate'):
    bpy.ops.mesh.primitive_cylinder_add(radius=baseplate_radius,
                                        depth=baseplate_thickness,
                                        enter_editmode=False, align='WORLD',
                                        location=(0, 0, baseplate_thickness/2),
                                        scale=(1, 1, 1))
    bpy.context.object.name = name

def createPanel(x,y,z, lx,ly,lz, name="Panel"):
    bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=(lx, ly, lz), scale=(1, 1, 1))
    bpy.ops.transform.resize(value=(x,y,z), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.context.object.name = name

# The Panel
class ToolBoxPanel(bpy.types.Panel):
    bl_label = 'Gantry Design'
    bl_idname = 'Gantry_design_tool'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Gantry Design'

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.operator('wm.myop')  # Button 1
        row1 = layout.row()
        row1.operator('wm.myexp')  # Button 2




# This Class is for Button 1
class GantryDesign(bpy.types.Operator):
    '''Open the Gantry Dialog box'''
    bl_label = 'Gantry Dimension'
    bl_idname = 'wm.myop'  # Link this to Button 1

    text = bpy.props.StringProperty(name="texy", default="texy")

    COLUMN_RADIUS: bpy.props.FloatProperty(name="COLUMN_RADIUS", default=0.14)
    COLUMN_HEIGHT: bpy.props.FloatProperty(name="COLUMN_HEIGHT", default=6)
    COLUMN_THICKNESS: bpy.props.FloatProperty(name="COLUMN_THICKNESS", default=9.5 / 1000)
    BASEPLATE_THICKNESS: bpy.props.FloatProperty(name="BASEPLATE_THICKNESS", default=25 / 1000)
    BASEPLATE_RADIUS: bpy.props.FloatProperty(name="BASEPLATE_RADIUS", default=0.25)
    PANEL_DEPTH: bpy.props.FloatProperty(name="PANEL_DEPTH", default=1.25)
    PANEL_LENGTH: bpy.props.FloatProperty(name="PANEL_LENGTH", default=2.25)
    CORROSION_LENGTH: bpy.props.FloatProperty(name='CORROSION_LENGTH', default=0.2)
    CORROSION_THICKNESS: bpy.props.FloatProperty(name='CORROSION_THICKNESS', default=5/1000)
    CORROSION_HEIGHT: bpy.props.FloatProperty(name='CORROSION_HEIGHT', default=0.25)

    def execute(self, context):
        deleteall()
        PANEL_WIDTH = self.COLUMN_RADIUS * 2 + 20 * 0.001
        z = self.COLUMN_HEIGHT / 2 + self.BASEPLATE_THICKNESS
        createPanel(x=PANEL_WIDTH,
                    y=self.PANEL_LENGTH,
                    z=self.PANEL_DEPTH,
                    lx=0,
                    ly=self.PANEL_LENGTH / 2 - self.COLUMN_RADIUS - 10 / 1000,
                    lz=self.COLUMN_HEIGHT + self.BASEPLATE_THICKNESS - self.PANEL_DEPTH / 2, name="Panel")
        createColumn(self.COLUMN_RADIUS, self.COLUMN_HEIGHT, self.COLUMN_THICKNESS, z)
        createbaseplate(self.BASEPLATE_RADIUS, self.BASEPLATE_THICKNESS)
        createCorrosion(self.CORROSION_LENGTH, self.COLUMN_RADIUS, self.COLUMN_THICKNESS, self.CORROSION_THICKNESS,
                        self.CORROSION_HEIGHT, self.BASEPLATE_THICKNESS)

        difference('Panel', 'Column')
        rotate_object('column', 0, -90)
        difference('Column', 'column')

        bpy.ops.object.select_all(action='DESELECT')

        bpy.ops.mesh.primitive_plane_add(size=20, enter_editmode=False,
                                         align='WORLD',
                                         location=(0, 0, self.CORROSION_HEIGHT+0.1+self.BASEPLATE_THICKNESS),
                                         rotation=(0, 0, 0),
                                         scale=(1, 1, 1))
        slice_object('Column', 'Plane')
        delete_object('Plane')
        delete_object('column')

        

        createColumn(self.COLUMN_RADIUS, self.COLUMN_HEIGHT-self.CORROSION_HEIGHT-0.1,self.COLUMN_RADIUS,(self.COLUMN_HEIGHT-self.CORROSION_HEIGHT-0.1)/2 + self.BASEPLATE_THICKNESS + self.CORROSION_HEIGHT + 0.1,name='UpperColumn')  

        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


# This Class is for Button 2
class ExportGantry(bpy.types.Operator):
    '''Open the Gantry Dialog box'''
    bl_label = 'Export Components'
    bl_idname = 'wm.myexp'  # Link this to Button 1

    USERNAME: bpy.props.StringProperty(name='USERNAME', default='denta')
    PATH_TO_GMSH: bpy.props.StringProperty(name='PATH_TO_GMSH',
                                           default=r"C:\Users\denta\Downloads\gmsh-4.11.1-Windows64\gmsh.exe")

    def execute(self, context):
        USERNAME = self.USERNAME
        PATH_TO_GMSH = self.PATH_TO_GMSH
        for obj in bpy.data.objects:
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            name = obj.name
            bpy.ops.export_mesh.stl(filepath=rf"C:\Users\{USERNAME}\OneDrive\Documents\{name}.stl", check_existing=True,
                                    filter_glob="*.stl", use_selection=True, global_scale=1, ascii=False,
                                    use_mesh_modifiers=True, batch_mode='OFF', axis_forward='-Z', axis_up='Y')
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


# Register All Classes
def register():
    bpy.utils.register_class(GantryDesign)
    bpy.utils.register_class(ToolBoxPanel)
    bpy.utils.register_class(ExportGantry)


def unregister():
    bpy.utils.unregister_class(GantryDesign)
    bpy.utils.unregister_class(ToolBoxPanel)
    bpy.utils.unregister_class(ExportGantry)


if __name__ == "__main__":
    register()
    
    
