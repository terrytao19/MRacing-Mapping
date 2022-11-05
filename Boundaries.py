import cv2
import numpy as np
import os
import json
import math


def euclidean_distance(point1, point2):
    return math.sqrt(((point2[0] - point1[0]) ** 2) + ((point2[1] - point1[1]) ** 2))


def get_point(id, data):
    for cone in data:
        cone_id = cone.get("id")
        if cone_id == id:
            return cone.get("point")
    return [0, 0]


def get_color(id, data):
    for cone in data:
        cone_id = cone.get("id")
        if cone_id == id:
            return cone.get("color")
    return [0, 0, 0]


class Boundaries:
    def __init__(self, video_width=1280, video_height=720, window_width=2560, window_height=1600):
        self.video_width = video_width
        self.video_height = video_height
        self.window_width = window_width
        self.window_height = window_height
        self.height_ratio = window_height / video_height
        self.width_ratio = window_width / video_width
        self.integration_frames = 3  # point id correlations from previous frames
        self.max_horizontal_correlation = 200
        self.correlation_threshold = 320

        self.cone_points_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\cone_points'
        self.boundaries_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\boundaries'

    def find_boundaries(self, frame):
        print("Boundaries Progress | ", frame)

        with open(os.path.join(self.cone_points_dir, str(frame) + '.json')) as jsonFile:
            data = json.load(jsonFile)
            boundary_list = []

            for cone in range(len(data)):
                cone = data[cone]
                cone_id = cone.get("id")
                p1 = cone.get("point")
                p2 = p1
                min_dist = self.window_width + self.window_height
                for other_cone in range(len(data)):
                    other_cone = data[other_cone]
                    other_cone_point = other_cone.get("point")
                    other_cone_id = other_cone.get("id")
                    if other_cone_id != cone_id:
                        if other_cone_point[1] < p1[1] and cone.get("color") == other_cone.get("color"):
                            dist = euclidean_distance(p1, other_cone_point)
                            if dist < min_dist:
                                min_dist = dist
                                p2 = other_cone_point

                if min_dist < self.correlation_threshold:
                    boundary = {
                        "p1": p1,
                        "p2": p2,
                        "color": cone.get("color"),
                        "ids": [cone_id, other_cone_id]
                    }
                    boundary_list.append(boundary)

            if frame > self.integration_frames:
                for prev_frame in range(self.integration_frames):
                    with open(
                            os.path.join(self.boundaries_dir, str(frame - prev_frame - 1) + '.json')) as prev_jsonFile:
                        prev_data = json.load(prev_jsonFile)
                        for prev_boundary in prev_data:
                            prev_ids = prev_boundary.get("ids")
                            record_boundary = True
                            for boundary_pair in boundary_list:
                                if boundary_pair.get("ids") == prev_ids:
                                    record_boundary = False
                            if record_boundary:
                                p1 = get_point(prev_ids[0], data)
                                p2 = get_point(prev_ids[1], data)
                                if p1 != [0, 0] and p2 != [0, 0] and euclidean_distance(p1,
                                                                                        p2) < self.correlation_threshold:
                                    c1 = get_color(prev_ids[0], data)
                                    c2 = get_color(prev_ids[1], data)
                                    if c1 == c2:


                                        p1_ = p1
                                        p2_ = p2

                                        p1[0] += self.video_width / 2
                                        p1[1] = self.video_height - p1[1]
                                        p1[0] *= self.width_ratio
                                        p1[1] *= self.height_ratio
                                        # if 0 < p1[0] < self.window_width and 0 < p1[1] < self.window_height:
                                        p1 = [int(p1[0]), int(p1[1])]
                                        p2[0] += self.video_width / 2
                                        p2[1] = self.video_height - p2[1]
                                        p2[0] *= self.width_ratio
                                        p2[1] *= self.height_ratio
                                        # if 0 < p2[0] < self.window_width and 0 < p2[1] < self.window_height:
                                        p2 = [int(p2[0]), int(p2[1])]
                                        boundary = {
                                            "p1": p1_,
                                            "p2": p2_,
                                            "color": prev_boundary.get("color"),
                                            "ids": [prev_ids[0], prev_ids[1]]
                                        }
                                        boundary_list.append(boundary)

            #  Tree search all boundaries for longest chain starting from the lowest point on screen

            # lowest_blue = 0
            # lowest_yellow = 0
            # start_blue_id = None
            # start_yellow_id = None
            # blue_sequence = []
            # yellow_sequence = []
            # stop_blue_search = False
            # stop_yellow_search = False
            # for boundary in boundary_list:
            #     if boundary.get("color") == [255, 0, 0]:
            #         if boundary.get("p2")[1] > lowest_blue:
            #             lowest_blue = boundary.get("p2")[1]
            #             start_blue_id = boundary.get("ids")[1]
            #
            #     elif boundary.get("color") == [0, 210, 255]:
            #         if boundary.get("p2")[1] > lowest_yellow:
            #             lowest_yellow = boundary.get("p2")[1]
            #             start_yellow_id = boundary.get("ids")[1]
            #
            # blue_sequence.append([start_blue_id])
            # yellow_sequence.append([start_yellow_id])
            #
            # while not stop_blue_search:
            #     new_blue_sequence = []
            #     continue_search = False
            #     for chain in blue_sequence:
            #         start_id = chain[-1]
            #         new_chain = chain
            #         for boundary in boundary_list:
            #             if boundary.get("ids")[1] == start_id:
            #                 continue_search = True
            #                 new_chain.append(boundary.get("ids")[0])
            #                 new_blue_sequence.append(new_chain)
            #                 new_chain = chain
            #     if new_blue_sequence:
            #         blue_sequence = new_blue_sequence
            #     if not continue_search:
            #         stop_blue_search = True
            #
            # while not stop_yellow_search:
            #     print("yellow")
            #     new_yellow_sequence = []
            #     continue_search = False
            #     for chain in yellow_sequence:
            #         start_id = chain[-1]
            #         new_chain = chain
            #         for boundary in boundary_list:
            #             if boundary.get("ids")[1] == start_id:
            #                 continue_search = True
            #                 new_chain.append(boundary.get("ids")[0])
            #                 new_yellow_sequence.append(new_chain)
            #                 new_chain = chain
            #     if new_yellow_sequence:
            #         yellow_sequence = new_yellow_sequence
            #     if not continue_search:
            #         stop_yellow_search = True
            #
            # longest_blue_chain = 0
            # longest_yellow_chain = 0
            # longest_blue_chain_idx = 0
            # longest_yellow_chain_idx = 0
            # for idx, chain in enumerate(blue_sequence):
            #     if len(chain) > longest_blue_chain:
            #         longest_blue_chain = len(chain)
            #         longest_blue_chain_idx = idx
            #
            # for idx, chain in enumerate(yellow_sequence):
            #     if len(chain) > longest_yellow_chain:
            #         longest_yellow_chain = len(chain)
            #         longest_yellow_chain_idx = idx
            #
            # if len(blue_sequence) > 0:
            #     true_blue_chain = blue_sequence[longest_blue_chain_idx]
            #     true_yellow_chain = yellow_sequence[longest_yellow_chain_idx]
            # else:
            #     true_blue_chain = []
            #     true_yellow_chain = []
            # print(true_blue_chain)

            # for id_pair in true_blue_chain:

            json_object = json.dumps(boundary_list, indent=4)
            jsonfile = open(
                os.path.join(self.boundaries_dir, str(frame) + '.json'), 'w')
            jsonfile.write(json_object)
            jsonfile.close()

    def find_all_boundaries(self):
        for frame in range(len(os.listdir(self.cone_points_dir))):
            self.find_boundaries(frame)
