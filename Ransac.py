import cv2
import numpy as np
import os
import json
import math
from sklearn import linear_model

class Ransac:
    def __init__(self, video_width=1280, video_height=720, window_width=2560, window_height=1600):
        self.video_width = video_width
        self.video_height = video_height
        self.window_width = window_width
        self.window_height = window_height
        self.height_ratio = window_height / video_height
        self.width_ratio = window_width / video_width

    def do_ransac(self, cone_points_json):
        with open(cone_points_json) as jsonFile:
            data = json.load(jsonFile)
            x, y = [], []
            for cone in data:
                point = cone.get("point")
                if -self.window_width / 2 + 1 <= point[0] <= self.window_width / 2 - 1 and \
                        1 <= point[1] <= self.window_height - 1:
                    point[0] += self.video_width / 2
                    point[1] = self.video_height - point[1]
                    point[0] *= self.width_ratio
                    point[1] *= self.height_ratio
                    x.append(point[0])
                    y.append(point[1])
        a, b = np.polyfit(x, y, 1)
        return(a)