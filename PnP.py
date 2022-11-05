import cv2
import numpy as np
import os
import json
import math
from scipy.io import savemat
import random


def euclidean_distance(point1, point2):
    return math.sqrt(((point2[0] - point1[0]) ** 2) + ((point2[1] - point1[1]) ** 2))


class PnP:
    def __init__(self, video_width=1280, video_height=720, window_width=2560, window_height=1600):
        self.video_width = video_width
        self.video_height = video_height
        self.window_width = window_width
        self.window_height = window_height
        self.height_ratio = window_height / video_height
        self.width_ratio = window_width / video_width

        self.cone_id = 0
        self.point_correlation_threshold = 70  # px
        self.integration_frames = 0

        self.figure_points_3D = [[0.0, -45.0, 94.0],  # Normal bbox
                                 [0.0, 45.0, 94.0],
                                 [0.0, -45.0, 0.0],
                                 [0.0, 45.0, 0.0]]

        self.distortion_coeffs = np.zeros((4, 1))
        self.focal_length = video_width
        self.center = (video_width / 2, video_height / 2)
        self.matrix_camera = np.array(
            [[self.focal_length, 0, self.center[0]],
             [0, self.focal_length, self.center[1]],
             [0, 0, 1]], dtype="double"
        )
        self.bbbox_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\bbox_storage'
        self.out_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\cone_points'
        self.mat_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\cone_points.mat'

    def scale_figure_points_3D(self, scale):
        for r in range(len(self.figure_points_3D)):
            for c in range(r):
                self.figure_points_3D[r][c] *= scale

    def get_frame_points(self, frame):
        xy_list = []
        with open(os.path.join(self.out_dir, str(frame) + '.json')) as jsonFile:
            data = json.load(jsonFile)
            for cone in data:
                xy = cone.get("point")
                xy[0] += self.video_width / 2
                xy[1] = self.video_height - xy[1]
                xy[0] *= self.width_ratio
                xy[1] *= self.height_ratio
                if 0 < xy[0] < self.window_width and 0 < xy[1] < self.window_height:
                    xy_list.append([cone.get("point")[0], cone.get("point")[1]])
        return xy_list

    def plotter_frame_transform(self, point):
        point[0] += self.video_width / 2
        point[1] = self.video_height - point[1]
        point[0] *= self.width_ratio
        point[1] *= self.height_ratio
        return [int(point[0]), int(point[1])]

    def save_pnp_json_as_mat(self):
        frames = np.asarray(os.listdir(self.out_dir)).shape
        mat_out = np.empty(frames, dtype=object)
        frame = 0
        for i in range(frames[0]):
            directory = os.path.join(self.out_dir, str(i) + '.json')
            with open(os.path.join(self.out_dir, directory)) as jsonFile:
                data = json.load(jsonFile)
                ranges = []
                angles = []
                cartesian = []
                count = 0
                for cone in data:
                    xy = cone.get("point")
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

    def do_pnp(self, frame):
        path = os.path.join(self.bbbox_dir, str(frame) + '.json')
        print("PnP Progress | ", frame)
        with open(os.path.join(self.bbbox_dir, path)) as jsonFile:
            cone_data = json.load(jsonFile)
            out_cone_list = []
            for cone in cone_data:
                cone_box = cone["point"]
                cone_color = cone["color"]
                cone_conf = cone["conf"]

                cone_box[1] -= 300
                cone_box[3] -= 300

                keypoints = np.array(
                    [[cone_box[0], cone_box[1]], [cone_box[2], cone_box[1]], [cone_box[0], cone_box[3]],
                     [cone_box[2], cone_box[3]]])

                success, vector_rotation, vector_translation = cv2.solvePnP(np.array(self.figure_points_3D),
                                                                            keypoints,
                                                                            self.matrix_camera,
                                                                            self.distortion_coeffs, flags=0)

                cone_point = [vector_translation.tolist()[0][0], self.video_height - vector_translation.tolist()[1][0]]

                cone_point = [cone_point[0] * .196875, cone_point[1] * .35]
                # cone_point = [cone_point[0] * .2, cone_point[1] * .5]

                p1 = cone_point.copy()
                p1 = self.plotter_frame_transform(p1)
                if not (0 < p1[0] < self.window_width and 0 < p1[1] < self.window_height):
                    continue

                cone_id = self.cone_id

                if frame == 0:
                    self.cone_id += 1

                else:
                    with open(os.path.join(self.out_dir, str(frame - 1) + '.json')) as prevFile:
                        prev_data = json.load(prevFile)
                        min_dist = self.window_width + self.window_height
                        for prev_cone in prev_data:
                            prev_cone_point = prev_cone.get("point")
                            dist = euclidean_distance(cone_point, prev_cone_point)
                            if dist < min_dist and prev_cone.get("color") == cone_color:
                                min_dist = dist
                                cone_id = prev_cone.get("id")

                        if min_dist > self.point_correlation_threshold:
                            self.cone_id += 1
                            cone_id = self.cone_id

                if frame > self.integration_frames:
                    integration_point = cone_point
                    for prev_frame in range(self.integration_frames):
                        with open(os.path.join(self.out_dir, str(frame - prev_frame) + '.json')) as prevFile:
                            prev_data = json.load(prevFile)
                            integrate = False
                            for prev_cone in prev_data:
                                prev_cone_id = prev_cone.get("id")
                                if prev_cone_id == cone_id:
                                    integrate = True
                                    prev_cone_point = prev_cone.get("point")
                                    integration_point[0] += prev_cone_point[0]
                                    integration_point[1] += prev_cone_point[1]
                                    break
                            if integrate:
                                cone_point[0] = integration_point[0] / (self.integration_frames + 1)
                                cone_point[1] = integration_point[1] / (self.integration_frames + 1)

                cone = {
                    "point": cone_point,
                    "color": cone_color,
                    "conf": cone_conf,
                    "id": cone_id
                }
                out_cone_list.append(cone)
            json_object = json.dumps(out_cone_list, indent=4)
            jsonfile = open(
                os.path.join(self.out_dir, str(frame) + '.json'), 'w')
            jsonfile.write(json_object)
            jsonfile.close()

    def do_pnp_all(self):
        for i in range(len(os.listdir(self.bbbox_dir))):
            self.do_pnp(i)
