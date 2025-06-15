import bpy
import math
import mathutils
import csv
import os

MODE = 'export'#export or display
GRID_WIDTH = 1280
GRID_HEIGHT = 800
HFOV_DEG = 127.0
VFOV_DEG = 79.5
RES_X = 1280
RES_Y = 800

IOR_AIR = 1.000
IOR_POLYCARB = 1.586
IOR_WATER = 1.333

OUTPUT_CSV = "ray_mapping_raw.csv"

def refract_ray(incident, normal, n1, n2):
    incident = incident.normalized()
    normal = normal.normalized()
    cos_i = -normal.dot(incident)
    sin_t2 = (n1 / n2)**2 * (1 - cos_i**2)
    if sin_t2 > 1:
        return None
    cos_t = math.sqrt(1 - sin_t2)
    return (n1 / n2) * incident + (n1 / n2 * cos_i - cos_t) * normal

def ray_plane_intersection(ray_origin, ray_dir, plane_point, plane_normal):
    denom = plane_normal.dot(ray_dir)
    if abs(denom) < 1e-6:
        return None
    t = (plane_point - ray_origin).dot(plane_normal) / denom
    if t < 0:
        return None
    return ray_origin + t * ray_dir

def get_camera_ray(cam_obj, u, v):
    sensor_width = 2 * math.tan(math.radians(HFOV_DEG / 2))
    sensor_height = 2 * math.tan(math.radians(VFOV_DEG / 2))

    ndc_x = (u + 0.5) / GRID_WIDTH
    ndc_y = (v + 0.5) / GRID_HEIGHT

    pixel_cam_z = (ndc_x - 0.5) * sensor_width
    pixel_cam_x = (0.5 - ndc_y) * sensor_height
    pixel_cam_y = -1.0

    local_dir = mathutils.Vector((pixel_cam_x, pixel_cam_z, pixel_cam_y)).normalized()
    world_dir = cam_obj.matrix_world.to_quaternion() @ local_dir
    world_origin = cam_obj.matrix_world.translation
    return world_origin, world_dir.normalized()

def create_line(start, end, name):
    mesh = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata([start, end], [(0, 1)], [])
    mesh.update()
    return obj

def main():
    print("Starting ray tracing script...")

    cam = bpy.data.objects.get('Camera_001')
    tube = bpy.data.objects.get('tube')
    plane = bpy.data.objects.get('hit_plane')

    if not cam or not tube or not plane:
        print("[ERROR] One or more scene objects not found.")
        return

    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_eval = tube.evaluated_get(depsgraph)
    plane_eval = plane.evaluated_get(depsgraph)

    plane_point = plane_eval.matrix_world.translation
    plane_normal = plane_eval.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, 1))

    results = []
    scene = bpy.context.scene

    total_valid = 0

    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            cam_origin, ray_dir = get_camera_ray(cam, j, i)

            hit1, loc1, normal1, _, _, _ = scene.ray_cast(depsgraph, cam_origin, ray_dir)
            if not hit1:
                create_line(cam_origin, cam_origin + ray_dir * 0.1, f"miss_air_{i}_{j}")
                continue

            refracted1 = refract_ray(ray_dir, normal1, IOR_AIR, IOR_POLYCARB)
            if refracted1 is None:
                create_line(cam_origin, loc1, f"tir_air_{i}_{j}")
                continue

            offset1 = loc1 + refracted1 * 0.001
            hit2, loc2, normal2, _, _, _ = scene.ray_cast(depsgraph, offset1, refracted1)
            if not hit2:
                create_line(cam_origin, loc1, f"ray_air_{i}_{j}")
                create_line(loc1, loc1 + refracted1 * 0.05, f"miss_tube_{i}_{j}")
                continue

            refracted2 = refract_ray(refracted1, -normal2, IOR_POLYCARB, IOR_WATER)
            if refracted2 is None:
                create_line(cam_origin, loc1, f"ray_air_{i}_{j}")
                create_line(loc1, loc2, f"ray_tube_{i}_{j}")
                create_line(loc2, loc2 + refracted1 * 0.05, f"tir_tube_{i}_{j}")
                continue

            offset2 = loc2 + refracted2 * 0.001
            final_hit = ray_plane_intersection(offset2, refracted2, plane_point, plane_normal)

            if final_hit is None:
                flipped_refracted2 = -refracted2
                final_hit = ray_plane_intersection(offset2, flipped_refracted2, plane_point, plane_normal)
                if final_hit is not None:
                    refracted2 = flipped_refracted2

            if final_hit is None:
                create_line(cam_origin, loc1, f"ray_air_{i}_{j}")
                create_line(loc1, loc2, f"ray_tube_{i}_{j}")
                create_line(loc2, loc2 + refracted2 * 0.05, f"miss_plane_{i}_{j}")
                continue

            total_valid += 1

            if MODE == 'display':
                create_line(cam_origin, loc1, f"ray_air_{i}_{j}")
                create_line(loc1, loc2, f"ray_tube_{i}_{j}")
                create_line(loc2, final_hit, f"ray_water_{i}_{j}")

            results.append((j, i, final_hit.y, final_hit.z))

    print(f"Ray tracing complete. Total samples: {GRID_WIDTH * GRID_HEIGHT}")
    print(f"Valid final hits: {total_valid}")

    if MODE == 'export':
        output_path = os.path.join(bpy.path.abspath("//"), OUTPUT_CSV)
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['camera_x', 'camera_y', 'distorted_x', 'distorted_y'])
            for row in results:
                writer.writerow(row)
        print(f"Export complete: {OUTPUT_CSV}")

main()
