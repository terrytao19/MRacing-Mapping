import cv2
import numpy as np
import os
import json
import math
import open3d as o3d


def convert_json_xyz(cone_points_json):
    xyz = []
    pcd = o3d.geometry.PointCloud()
    with open(cone_points_json) as jsonFile:
        cone_data = json.load(jsonFile)
        for cone in cone_data:
            cone_point = cone.get("point")
            xyz.append([cone_point[0], cone_point[1], 0])
    # print(np.asarray(xyz), "\n")
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz))
    return pcd


trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, -1.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])


def find_transform(cone_points_json, previous_points_json):
    xyz = convert_json_xyz(cone_points_json)
    previous_xyz = convert_json_xyz(previous_points_json)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        xyz, previous_xyz, 50, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p.transformation[:-1, :-1].reshape((3, 3)), "\n")
    return [reg_p2p.transformation[0][3], reg_p2p.transformation[1][3]]


class ICP:
    def __init__(self, video_width=1280, video_height=720, window_width=2560, window_height=1600):
        self.video_width = video_width
        self.video_height = video_height
        self.window_height = window_height
        self.window_width = window_width
        self.height_ratio = window_height / video_height
        self.width_ratio = window_width / video_width
        self.average_point = [0, 0]
        self.out_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\velocity_vectors'
        self.cone_points_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\cone_points'

    def find_all_transforms(self):
        for frame in range(len(os.listdir(self.cone_points_dir))):
            print("velocity vector progress | ", frame)
            if frame != 0:
                velocity_vector = find_transform(os.path.join(self.cone_points_dir, str(frame) + '.json'),
                                                 os.path.join(self.cone_points_dir, str(frame - 1) + '.json'))
                velocity_vector = {
                    "velocity_vector": velocity_vector
                }
                json_object = json.dumps(velocity_vector, indent=4)
                jsonfile = open(os.path.join(self.out_dir, str(frame) + '.json'), 'w')
                jsonfile.write(json_object)
                jsonfile.close()

        with open(os.path.join(self.out_dir, "1.json")) as jsonFile:
            velocity_vector = json.load(jsonFile)
            jsonfile = open(os.path.join(self.out_dir, '0.json'), 'w')
            json_object = json.dumps(velocity_vector, indent=4)
            jsonfile.write(json_object)
            jsonfile.close()


