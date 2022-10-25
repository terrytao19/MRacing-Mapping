import cv2
import numpy as np
import os
import json
import math


def get_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


class PerpendicularBBox:
    def __init__(self, video_width=1280, video_height=720, window_width=2560, window_height=1600):
        self.video_width = video_width
        self.video_height = video_height
        self.window_width = window_width
        self.window_height = window_height

        self.bbbox_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\bbox_storage'
        self.out_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\perpendicular-bboxes'

    def find_pbboxes(self, i):
        path = os.path.join(self.bbbox_dir, str(i) + '.json')
        print(path)
        with open(os.path.join(self.bbbox_dir, path)) as jsonFile:
            cone_data = json.load(jsonFile)
            size_threshold = .04

            for cone in range(len(cone_data)):
                bbox = cone_data[cone].get("point")
                bbox_area = get_area(bbox)
                for other_cone in range(cone, len(cone_data)):
                    other_bbox = cone_data[other_cone].get("point")
                    other_bbox_area = get_area(other_bbox)
                    if cone != other_cone:
                        if abs((other_bbox_area - bbox_area) / bbox_area) < size_threshold:
                            closest_bbox = other_bbox




                cone = {
                    "point": cone_point,
                    "color": cone_color,
                    "conf": cone_conf
                }
                out_cone_list.append(cone)
            json_object = json.dumps(out_cone_list, indent=4)
            jsonfile = open(
                os.path.join(self.out_dir, str(i) + '.json'), 'w')
            jsonfile.write(json_object)
            jsonfile.close()

    def do_pnp_all(self):
        for i in range(len(os.listdir(self.bbbox_dir))):
            self.do_pnp(i)
