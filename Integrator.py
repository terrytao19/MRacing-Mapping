import cv2
import numpy as np
import os
import json
import math
from slam.utils import ICP
from scipy.io import savemat



def euclidean_distance(point1, point2):
    return math.sqrt(((point2[0] - point1[0]) ** 2) + ((point2[1] - point1[1]) ** 2))


def get_affine(translation, rotation):
    return np.asarray([[math.cos(rotation), -math.sin(rotation), translation[0]],
                       [math.sin(rotation), math.cos(rotation), translation[1]],
                       [0, 0, 1]])


def apply_affine(affine, point):
    if len(point) == 2:
        point.append(1)
    affine = np.asarray(affine)
    point = np.asarray(point)
    x = np.matmul(affine, point)
    return [x[0], x[1]]


class Integrator:
    def __init__(self, video_width=1280, video_height=720, window_width=2560, window_height=1600, video=True,
                 relative=True,
                 frame_rate=30):
        self.video_width = video_width
        self.video_height = video_height
        self.window_height = window_height
        self.window_width = window_width
        self.height_ratio = window_height / video_height
        self.width_ratio = window_width / video_width
        self.video = video
        self.relative = relative
        self.frame_rate = frame_rate
        self.absolute_scale = .05
        self.car_xy = [-8000, 4000]
        self.car_angle = 0
        self.prev_rotation = 0
        # self.integrated_points = []
        # self.car_pose = get_affine(self.integration_start_point,
        #                            self.integration_start_angle)  # transformation such that when applied to a set of points, they move into the car's frame
        self.bbox_storage_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\bbox_storage'
        self.cone_points_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\cone_points'
        self.velocity_vectors_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\velocity_vectors'
        self.velocity_integrals_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\integrated_velocities'
        self.boundaries_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\boundaries'
        self.integrated_points_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\integrated_points'
        self.mat_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\cone_points.mat'

    def integrate(self, velocity, rotation):
        self.car_angle = rotation
        vel_factor = 1
        self.car_xy[0] -= velocity * math.cos(self.car_angle) * vel_factor
        self.car_xy[1] += velocity * math.sin(self.car_angle) * vel_factor
        self.car_angle -= math.pi / 2
        self.car_angle *= -1
        self.prev_rotation = rotation

    def get_absolute_point_cloud(self, frame, velocity, rotation):
        self.integrate(velocity, rotation)
        affine = get_affine(self.car_xy, self.car_angle)
        with open(os.path.join(self.cone_points_dir, str(frame) + '.json')) as jsonFile:
            data = json.load(jsonFile)
            point_cloud = []
            for cone in data:
                point = cone.get("point")
                point = apply_affine(affine, point)
                point[0] *= self.absolute_scale
                point[1] *= self.absolute_scale
                point_cloud.append(point)
        return [point_cloud, affine]

        # with open(os.path.join(self.cone_points_dir, '0' + '.json')) as jsonFile:
        #     data = json.load(jsonFile)
        #     for cone in data:
        #         point = cone.get("point")
        #         point = apply_affine(self.car_pose, point)
        #         self.integrated_points.append([point[0] * self.absolute_scale, point[1] * self.absolute_scale])

    # def modify_car_pose(self, delta_affine):
    #     # where delta_affine is absolute
    #     delta_rotation = math.acos(delta_affine[0][0])
    #     # print(self.car_pose)
    #     rotation = math.acos(self.car_pose[0][0]) + delta_rotation
    #     c = math.cos(rotation)
    #     s = math.sin(rotation)
    #     self.car_pose[0][0] = c
    #     self.car_pose[0][1] = -s
    #     self.car_pose[0][2] += delta_affine[0][2]
    #
    #     self.car_pose[1][0] = s
    #     self.car_pose[1][1] = c
    #     self.car_pose[1][2] += delta_affine[1][2]
    #
    # def integrate(self, deltas):
    #     delta_rotation, delta_dist = deltas[0], deltas[1]
    #     self.car_angle += delta_rotation
    #     delta_xy = [math.cos(self.car_angle) * delta_dist, math.sin(self.car_angle) * delta_dist]
    #     self.car_xy[0] += delta_xy[0]
    #     self.car_xy[1] += delta_xy[1]
    #     return [self.car_xy, self.car_angle]

    def generate_points_on_boundaries(self, frame):
        num_points_per_boundary = 5
        generated_points = []
        with open(os.path.join(self.boundaries_dir, str(frame) + '.json')) as jsonFile:
            data = json.load(jsonFile)
            for boundary in data:
                p1 = boundary.get("p1")
                p2 = boundary.get("p2")
                v = [(p2[0] - p1[0]) / num_points_per_boundary, (p2[1] - p1[1]) / num_points_per_boundary]
                for i in range(num_points_per_boundary):
                    generated_points.append([p1[0] + (i * v[0]), p1[1] + (i * v[1])])
        return generated_points

    def save_pnp_json_as_mat(self):
        frames = np.asarray(os.listdir(self.cone_points_dir)).shape
        mat_out = np.empty(frames, dtype=object)
        frame = 0
        for i in range(frames[0]):
            points_list = self.generate_points_on_boundaries(i)
            ranges = []
            angles = []
            cartesian = []
            count = 0
            for xy in points_list:
                xy[0] *= .002
                xy[1] *= .002
                cartesian.append(xy)
                angles.append(math.atan2(xy[1], xy[0]))
                ranges.append(euclidean_distance(xy, [0, 0]))
                count += 1
            mat_dict = {
                "Ranges": ranges,
                "Angles": angles,
                "Cartesian": cartesian,
                "Count": count
            }
            mat_out[frame] = mat_dict
            print(mat_out[frame])
            frame += 1
        k = 0
        for mat in mat_out:
            if mat is not None:
                k += 1
        print("done", k)
        savemat(self.mat_dir, {"cone_scans": mat_out})

    # def get_scaled_frame_points(self, frame):
    #     generated_points = self.generate_points_on_boundaries(frame)
    #     scaled_points = []
    #     for point in generated_points:
    #         scaled_points.append([point[0] * self.absolute_scale, point[1] * self.absolute_scale])
    #     return scaled_points

    # def do_icp(self, frame):
    #     if frame > 0:
    #         frame_points = self.get_scaled_frame_points(frame)
    #         transformed_points = []
    #         for point in frame_points:
    #             transformed_points.append(apply_affine(self.car_pose, point))
    #         delta_affine = ICP.icp(np.asarray(self.integrated_points), np.asarray(transformed_points), verbose=False)
    #         # print(transformed_points)
    #         # print(self.integrated_points)
    #         self.modify_car_pose(delta_affine)
    #
    #         new_integrated_points = []
    #         for point in frame_points:
    #             transformed_point = apply_affine(self.car_pose, point)
    #             self.integrated_points.append(transformed_point)
    #             new_integrated_points.append(transformed_point)
    #         # print(new_integrated_points)
    #         return self.car_pose
    #     else:
    #         frame_points = self.get_scaled_frame_points(frame)
    #         new_integrated_points = []
    #         for point in frame_points:
    #             self.integrated_points.append(apply_affine(self.car_pose, point))
    #             new_integrated_points.append(apply_affine(self.car_pose, point))
    #         return self.car_pose
