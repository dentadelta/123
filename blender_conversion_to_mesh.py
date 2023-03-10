#once run, press N in the main view to bring out the side panel, then click on the Gantry Design Tool
#still need to create a quick tool to create a lot of loops around the column 
#still need to create a quick tool to automatically create the corrosion hole using the extrude "manifold" tool  (need to figure out how to select the right vertex, faces...)



import bpy
import os

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
        row1.operator('wm.myexp')# Button 2

            
# This Class is for Button 1
class GantryDesign(bpy.types.Operator):
    '''Open the Gantry Dialog box'''
    bl_label = 'Gantry Dimension'
    bl_idname = 'wm.myop'  #Link this to Button 1
    
    text = bpy.props.StringProperty(name="texy", default="texy")

    COLUMN_RADIUS : bpy.props.FloatProperty(name="COLUMN_RADIUS", default=0.14)
    COLUMN_HEIGHT : bpy.props.FloatProperty(name="COLUMN_HEIGHT", default=6)
    COLUMN_THICKNESS : bpy.props.FloatProperty(name="COLUMN_THICKNESS",default=9.5/1000)
    BASEPLATE_THICKNESS : bpy.props.FloatProperty(name="BASEPLATE_THICKNESS",default=25/1000)
    BASEPLATE_RADIUS : bpy.props.FloatProperty(name="BASEPLATE_RADIUS", default=0.25)
    PANEL_DEPTH: bpy.props.FloatProperty(name="PANEL_DEPTH",default=1.25)
    PANEL_LENGTH: bpy.props.FloatProperty(name="PANEL_LENGTH",default=2.25)

    def createPanel(self,x,y,z, lx,ly,lz, name="Panel"):
        bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=(lx, ly, lz), scale=(1, 1, 1))
        bpy.ops.transform.resize(value=(x,y,z), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        bpy.context.object.name = name

    def execute(self, context):
        PANEL_WIDTH =self.COLUMN_RADIUS*2+20*0.001
        z = self.COLUMN_HEIGHT/2+self.BASEPLATE_THICKNESS
        bpy.ops.mesh.primitive_elbow_joint_add(align='WORLD', 
        location=(0, 0,z), 
        rotation=(0, 0, 0), 
        change=False, 
        radius=self.COLUMN_RADIUS, 
        div=32, angle=0.00, 
        startLength=self.COLUMN_HEIGHT/2,
        endLength=self.COLUMN_HEIGHT/2)
        
        bpy.context.object.name = "Column"
        bpy.ops.object.modifier_add(type='SOLIDIFY')
        bpy.context.object.modifiers["Solidify"].thickness =self.COLUMN_THICKNESS
        bpy.ops.object.apply_all_modifiers()

        bpy.ops.mesh.primitive_cylinder_add(radius=self.BASEPLATE_RADIUS, depth=self.BASEPLATE_THICKNESS, enter_editmode=False, align='WORLD', location=(0, 0, self.BASEPLATE_THICKNESS/2), scale=(1, 1, 1))

        self.createPanel(x=PANEL_WIDTH, y=self.PANEL_LENGTH, z=self.PANEL_DEPTH, lx=0, ly=self.PANEL_LENGTH/2-self.COLUMN_RADIUS-10/1000, lz=self.COLUMN_HEIGHT+self.BASEPLATE_THICKNESS-self.PANEL_DEPTH/2, name="Panel")
        bpy.ops.object.modifier_add(type='BOOLEAN')
        bpy.context.object.modifiers["Boolean"].object = bpy.data.objects["Column"]
        bpy.ops.object.apply_all_modifiers()

        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)




# This Class is for Button 2
class ExportGantry(bpy.types.Operator):
    '''Open the Gantry Dialog box'''
    bl_label = 'Export Components'
    bl_idname = 'wm.myexp' #Link this to Button 1

    USERNAME: bpy.props.StringProperty(name='USERNAME',default='denta')
    PATH_TO_GMSH: bpy.props.StringProperty(name='PATH_TO_GMSH',default=r"C:\Users\denta\Downloads\gmsh-4.11.1-Windows64\gmsh.exe")

    def execute(self, context):
        USERNAME = self.USERNAME
        PATH_TO_GMSH = self.PATH_TO_GMSH
        for obj in bpy.data.objects:
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            name = obj.name
            bpy.ops.export_mesh.stl(filepath=rf"C:\Users\{USERNAME}\OneDrive\Documents\{name}.stl", check_existing=True, filter_glob="*.stl", use_selection=True, global_scale=1, ascii=False, use_mesh_modifiers=True, batch_mode='OFF', axis_forward='-Z', axis_up='Y')
            with open(rf"C:\Users\{USERNAME}\OneDrive\Documents\volume.geo",'w') as script:
                built_in = '"Built-in"'
                script.write(rf"Merge 'C:\Users\{USERNAME}\OneDrive\Documents\{name}.stl';\n//+\nSetFactory({built_in});\n//+\nSurface Loop(1) = {1};\n//+\nVolume(1) = {1};")
            os.system(rf"{PATH_TO_GMSH} C:\Users\{USERNAME}\OneDrive\Documents\volume.geo -3 -o C:\Users\{USERNAME}\OneDrive\Documents\{name}.inp")
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



