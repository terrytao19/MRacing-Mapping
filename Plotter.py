import cv2
import numpy as np
import os
import json
import math


def transform_2d(vector, transformation_matrix):
    transformed_vector = vector
    transformed_vector[0] = (transformation_matrix[0][0] * vector[0]) + (transformation_matrix[0][1] * vector[1])
    transformed_vector[1] = (transformation_matrix[1][0] * vector[0]) + (transformation_matrix[1][1] * vector[1])
    transformed_vector[0] += transformation_matrix[0][2]
    transformed_vector[1] += transformation_matrix[1][2]
    return transformed_vector


class Plotter:
    def __init__(self, video_width=1280, video_height=720, window_width=2560, window_height=1600, video=True, relative = True,
                 frame_rate=30, out_directory='C:\\Users\\terry\\MICHIGAN\\MRacing\\videos\\out\\export.mp4'):
        self.out_directory = out_directory
        self.video_width = video_width
        self.video_height = video_height
        self.window_height = window_height
        self.window_width = window_width
        self.height_ratio = window_height / video_height
        self.width_ratio = window_width / video_width
        self.video = video
        self.relative = relative
        self.frame_rate = frame_rate
        self.absolute_scale = .01
        if video:
            self.base_img = np.zeros((self.window_height, self.window_width, 3), dtype="uint8")
            if relative:
                cv2.line(self.base_img, (int(self.window_width / 2 - 50), int(self.window_height)),
                         (int(self.window_width / 2), int(self.window_height - 50)), (255, 255, 255), 10)
                cv2.line(self.base_img, (int(self.window_width / 2 + 50), int(self.window_height)),
                         (int(self.window_width / 2), int(self.window_height - 50)), (255, 255, 255), 10)
            self.frame_img = self.base_img.copy()
        self.vid_writer = cv2.VideoWriter(self.out_directory, cv2.VideoWriter_fourcc(*'mp4v'), self.frame_rate,
                                          (self.window_width, self.window_height))

    def write_frame(self):
        self.vid_writer.write(self.frame_img)
        self.frame_img = self.base_img.copy()

    def plot_text(self, text):
        cv2.putText(self.frame_img, text, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)

    def plot_circle(self, xyr):
        cv2.circle(self.frame_img, [int(xyr[0]), int(xyr[1])], int(xyr[2]), (0, 255, 0), 3)

    def plot_angle(self, angle):
        length = 100
        center = [int(self.window_width / 2), int(self.window_height - 500)]
        x = int(length * math.cos(angle) + center[0])
        y = int(length * math.sin(angle) + center[1])
        cv2.arrowedLine(self.frame_img, center, [x, y], (0, 255, 0), 3)

    def plot_boundaries(self, boundaries_json):
        with open(boundaries_json) as jsonFile:
            data = json.load(jsonFile)
            for boundary in data:
                p1 = boundary.get("p1")
                p1[0] += self.video_width / 2
                p1[1] = self.video_height - p1[1]
                p1[0] *= self.width_ratio
                p1[1] *= self.height_ratio
                if 0 < p1[0] < self.window_width and 0 < p1[1] < self.window_height:
                    p1 = [int(p1[0]), int(p1[1])]
                    p2 = boundary.get("p2")
                    p2[0] += self.video_width / 2
                    p2[1] = self.video_height - p2[1]
                    p2[0] *= self.width_ratio
                    p2[1] *= self.height_ratio
                    if 0 < p2[0] < self.window_width and 0 < p2[1] < self.window_height:
                        p2 = [int(p2[0]), int(p2[1])]
                        color = boundary.get("color")
                        cv2.line(self.frame_img, p2, p1, color, 3)

    def plot_line(self, slope):
        origin = [int(self.window_width / 2), int(self.window_height - 500)]
        end_point = [int(-self.window_width * slope / abs(slope)), int(-self.window_width * slope ** 2 / abs(slope))]
        cv2.line(self.frame_img, origin, end_point, (0, 255, 0), 5)

    def plot_car(self, transformation_matrix_json):
        with open(transformation_matrix_json) as jsonFile:
            data = json.load(jsonFile)
            transformation_matrix = data.get("transformation_matrix")
            car_origin_transform = [[1, 0, self.window_width / 2], [0, 1, self.window_height / 2], [0, 0, 1]]

            transformation_matrix[0][2] *= self.absolute_scale
            transformation_matrix[1][2] *= self.absolute_scale

            origin = [0, 0]
            front_point = [0, -70]
            left_point = [-20, 0]
            right_point = [20, 0]
            origin = transform_2d(origin, transformation_matrix)
            front_point = transform_2d(front_point, transformation_matrix)
            left_point = transform_2d(left_point, transformation_matrix)
            right_point = transform_2d(right_point, transformation_matrix)

            origin = transform_2d(origin, car_origin_transform)
            front_point = transform_2d(front_point, car_origin_transform)
            left_point = transform_2d(left_point, car_origin_transform)
            right_point = transform_2d(right_point, car_origin_transform)

            origin = [int(origin[0]), int(origin[1])]
            front_point = [int(front_point[0]), int(front_point[1])]
            left_point = [int(left_point[0]), int(left_point[1])]
            right_point = [int(right_point[0]), int(right_point[1])]

            cv2.arrowedLine(self.frame_img, origin, front_point, (20, 20, 255), 7)
            cv2.arrowedLine(self.frame_img, origin, left_point, (20, 255, 20), 7)
            cv2.arrowedLine(self.frame_img, origin, right_point, (20, 255, 20), 7)

    def plot_mean_point(self, mean_point_json):
        with open(mean_point_json) as jsonFile:
            mean_point_data = json.load(jsonFile)
            average_point = mean_point_data.get("average_point")
            color = mean_point_data.get("color")
            if -self.window_width / 2 + 1 <= average_point[0] <= self.window_width / 2 - 1 and \
                    1 <= average_point[1] <= self.window_height - 1:
                average_point = (int(average_point[0]), int(average_point[1]))
                cv2.circle(self.frame_img, average_point, 10, color, -1)

    def plot_vector(self, vector_json):
        with open(vector_json) as jsonFile:
            vector_data = json.load(jsonFile)
            vector = vector_data.get("velocity_vector")
            scale = 2
            origin = [int(self.window_width / 2), int(self.window_height - 500)]
            end_point = [int(origin[0] + (vector[0] * scale)), int(origin[1] + (vector[1] * scale))]
            if 1 <= end_point[0] <= self.window_width - 1 and 1 <= end_point[1] <= self.window_height - 1:
                cv2.arrowedLine(self.frame_img, origin, end_point, (0, 255, 0), 10)

    def plot_frame(self, points_json):
        with open(points_json) as jsonFile:
            cone_data = json.load(jsonFile)
            for cone in cone_data:
                cone_color = cone.get("color")
                cone_point = cone.get("point")
                # points stored in json with origin on bottom center of frame, need to convert into cv2 point
                if -self.window_width / 2 + 1 <= cone_point[0] <= self.window_width / 2 - 1 and \
                        1 <= cone_point[1] <= self.window_height - 1:
                    cone_point[0] += self.video_width / 2
                    cone_point[1] = self.video_height - cone_point[1]
                    cone_point[0] *= self.width_ratio
                    cone_point[1] *= self.height_ratio
                    cone_point = (int(cone_point[0]), int(cone_point[1]))
                    cv2.circle(self.frame_img, cone_point, 10, cone_color, -1)

    def plot_frame_force_color(self, points_json, color):
        with open(points_json) as jsonFile:
            cone_data = json.load(jsonFile)
            for cone in cone_data:
                cone_color = cone.get("color")
                cone_point = cone.get("point")
                # points stored in json with origin on bottom center of frame, need to convert into cv2 point
                if -self.window_width / 2 + 1 <= cone_point[0] <= self.window_width / 2 - 1 and \
                        1 <= cone_point[1] <= self.window_height - 1:
                    cone_point[0] += self.video_width / 2
                    cone_point[1] = self.video_height - cone_point[1]
                    cone_point[0] *= self.width_ratio
                    cone_point[1] *= self.height_ratio
                    cone_point = (int(cone_point[0]), int(cone_point[1]))
                    cv2.circle(self.frame_img, cone_point, 10, color, -1)

    def plot_frame_bbox(self, bbox_json):
        with open(bbox_json) as jsonFile:
            cone_data = json.load(jsonFile)
            for cone in cone_data:
                cone_color = cone.get("color")
                cone_point = cone.get("point")
                # cone_point[0] -= self.video_width / 2
                # cone_point[2] -= self.video_width / 2
                # cone_point[1] = self.video_height - cone_point[1]
                # cone_point[3] = self.video_height - cone_point[3]
                cone_point[0] *= self.height_ratio
                cone_point[1] *= self.width_ratio
                cone_point[2] *= self.height_ratio
                cone_point[3] *= self.width_ratio
                cv2.rectangle(self.frame_img, (int(cone_point[0]), int(cone_point[1])),
                              (int(cone_point[2]), int(cone_point[3])), cone_color, 2)
