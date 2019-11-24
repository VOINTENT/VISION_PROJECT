import bpy
import os

import RenderBurst

#create cameras
bpy.ops.object.camera_add(location=[0,10,20], rotation=[0.7,0,3.1415927410125732])
bpy.ops.object.camera_add(location=[30,5,20], rotation=[0.436,0,3.1415927410125732])
bpy.ops.object.camera_add(location=[40,20,60], rotation=[0.436,0.7,8.1415927410125732])

#delete cube
bpy.ops.object.select_all(action='DESELECT')
bpy.data.objects['Cube'].select_set(True)
bpy.ops.object.delete() 

#import model
full_path_to_file = "C:/Users/rockg/OneDrive/Документы/Учеба/neural_networks/hand_textures_example.obj"
bpy.ops.import_scene.obj(filepath=full_path_to_file)



