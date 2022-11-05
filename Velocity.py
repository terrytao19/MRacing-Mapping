import cv2
import numpy as np
import os
import json
import math
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors
from math import sqrt, pi


def calc_R(x, y, xc, yc):
    """
    calculate the distance of each 2D points from the center (xc, yc)
    """
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)


def f(c, x, y):
    """
    calculate the algebraic distance between the data points
    and the mean circle centered at c=(xc, yc)
    """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()


def sigma(coords, x, y, r):
    """Computes Sigma for circle fit."""
    dx, dy, sum_ = 0., 0., 0.

    for i in range(len(coords)):
        dx = coords[i][1] - x
        dy = coords[i][0] - y
        sum_ += (sqrt(dx * dx + dy * dy) - r) ** 2
    return sqrt(sum_ / len(coords))


def hyper_fit(coords, IterMax=99, verbose=False):
    """
    Fits coords to circle using hyperfit algorithm.
    Inputs:
        - coords, list or numpy array with len>2 of the form:
        [
    [x_coord, y_coord],
    ...,
    [x_coord, y_coord]
    ]
        or numpy array of shape (n, 2)
    Outputs:
        - xc : x-coordinate of solution center (float)
        - yc : y-coordinate of solution center (float)
        - R : Radius of solution (float)
        - residu : s, sigma - variance of data wrt solution (float)
    """
    X, X = None, None
    if isinstance(coords, np.ndarray):
        X = coords[:, 0]
        Y = coords[:, 1]
    elif isinstance(coords, list):
        X = np.array([x[0] for x in coords])
        Y = np.array([x[1] for x in coords])
    else:
        raise Exception("Parameter 'coords' is an unsupported type: " + str(type(coords)))

    n = X.shape[0]

    Xi = X - X.mean()
    Yi = Y - Y.mean()
    Zi = Xi * Xi + Yi * Yi

    # compute moments
    Mxy = (Xi * Yi).sum() / n
    Mxx = (Xi * Xi).sum() / n
    Myy = (Yi * Yi).sum() / n
    Mxz = (Xi * Zi).sum() / n
    Myz = (Yi * Zi).sum() / n
    Mzz = (Zi * Zi).sum() / n

    # computing the coefficients of characteristic polynomial
    Mz = Mxx + Myy
    Cov_xy = Mxx * Myy - Mxy * Mxy
    Var_z = Mzz - Mz * Mz

    A2 = 4 * Cov_xy - 3 * Mz * Mz - Mzz
    A1 = Var_z * Mz + 4. * Cov_xy * Mz - Mxz * Mxz - Myz * Myz
    A0 = Mxz * (Mxz * Myy - Myz * Mxy) + Myz * (Myz * Mxx - Mxz * Mxy) - Var_z * Cov_xy
    A22 = A2 + A2

    # finding the root of the characteristic polynomial
    y = A0
    x = 0.
    for i in range(IterMax):
        Dy = A1 + x * (A22 + 16. * x * x)
        xnew = x - y / Dy
        if xnew == x or not np.isfinite(xnew):
            break
        ynew = A0 + xnew * (A1 + xnew * (A2 + 4. * xnew * xnew))
        if abs(ynew) >= abs(y):
            break
        x, y = xnew, ynew

    det = x * x - x * Mz + Cov_xy
    Xcenter = (Mxz * (Myy - x) - Myz * Mxy) / det / 2.
    Ycenter = (Myz * (Mxx - x) - Mxz * Mxy) / det / 2.

    x = Xcenter + X.mean()
    y = Ycenter + Y.mean()
    r = sqrt(abs(Xcenter ** 2 + Ycenter ** 2 + Mz))
    s = sigma(coords, x, y, r)
    iter_ = i
    if verbose:
        print('Regression complete in {} iterations.'.format(iter_))
        print('Sigma computed: ', s)
    return x, y, r, s


def transform_2d(vector, transformation_matrix):
    transformed_vector = vector
    transformed_vector[0] = (transformation_matrix[0][0] * vector[0]) + (transformation_matrix[0][1] * vector[1])
    transformed_vector[1] = (transformation_matrix[1][0] * vector[0]) + (transformation_matrix[1][1] * vector[1])
    transformed_vector[0] += transformation_matrix[0][2]
    transformed_vector[1] += transformation_matrix[1][2]
    return transformed_vector


def euclidean_distance(point1, point2):
    return math.sqrt(((point2[0] - point1[0]) ** 2) + ((point2[1] - point1[1]) ** 2))


#  b1 and b2 in form: [p1, p2] = [[p1x, p1y], [p2x, p2y]]
def delta_angle(b1, b2):
    v1 = [b1[1][0] - b1[0][0], b1[1][1] - b1[0][1]]
    v2 = [b2[1][0] - b2[0][0], b2[1][1] - b2[0][1]]

    t1 = math.atan2(v1[1], v1[0])
    t2 = math.atan2(v2[1], v2[0])

    dt = t2 - t1
    return -dt


class Velocity:
    def __init__(self, video_width=1280, video_height=720, window_width=2560, window_height=1600):
        self.video_width = video_width
        self.video_height = video_height
        self.window_width = window_width
        self.window_height = window_height
        self.height_ratio = window_height / video_height
        self.width_ratio = window_width / video_width
        self.correlation_threshold = 50  # Measured in px
        self.integrated_angle = math.pi / 2
        self.transformation_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.turning_ratio = 4500  # px per radian
        self.out_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\velocity_vectors'
        self.integral_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\integrated_velocities'
        self.cone_points_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\cone_points'
        self.boundaries_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\boundaries'

    def fit_circle(self, frame):
        if frame == 0:
            return [0, 0, 0]

        with open(os.path.join(self.cone_points_dir, str(frame) + '.json')) as jsonFile:
            data = json.load(jsonFile)
            xy_list_blue = []
            xy_list_yellow = []
            virtual_points = 100
            for i in range(int(virtual_points / 2)):
                xy_list_blue.append((int(self.window_width / 2 - 200), int(self.window_height - 0)))
            for i in range(int(virtual_points / 2)):
                xy_list_yellow.append((int(self.window_width / 2 + 200), int(self.window_height - 0)))
            for i in range(int(virtual_points / 2)):
                xy_list_blue.append((int(self.window_width / 2 - 200), int(self.window_height - 250)))
            for i in range(int(virtual_points / 2)):
                xy_list_yellow.append((int(self.window_width / 2 + 200), int(self.window_height - 250)))
            for cone in data:
                xy = cone.get("point")
                xy[0] += self.video_width / 2
                xy[1] = self.video_height - xy[1]
                xy[0] *= self.width_ratio
                xy[1] *= self.height_ratio
                if 1000 < xy[0] < self.window_width - 1000 and 500 < xy[1] < self.window_height:
                    if cone.get("color") == [255, 0, 0]:
                        xy_list_blue.append([cone.get("point")[0], cone.get("point")[1]])
                    elif cone.get("color") == [0, 210, 255]:
                        xy_list_yellow.append([cone.get("point")[0], cone.get("point")[1]])
        if frame > 0:
            with open(os.path.join(self.cone_points_dir, str(frame - 1) + '.json')) as jsonFile:
                data = json.load(jsonFile)
                for cone in data:
                    xy = cone.get("point")
                    xy[0] += self.video_width / 2
                    xy[1] = self.video_height - xy[1]
                    xy[0] *= self.width_ratio
                    xy[1] *= self.height_ratio
                    if 0 < xy[0] < self.window_width and 0 < xy[1] < self.window_height:
                        if cone.get("color") == [255, 0, 0]:
                            xy_list_blue.append([cone.get("point")[0], cone.get("point")[1]])
                        elif cone.get("color") == [0, 210, 255]:
                            xy_list_yellow.append([cone.get("point")[0], cone.get("point")[1]])
        xb, yb, rb, xy, yy, ry = 0, 0, 0, 0, 0, 0
        avg = 1
        disp = 0
        if len(xy_list_blue) > virtual_points + 2:
            xb, yb, rb, sb = hyper_fit(xy_list_blue)
            avg = 1
            disp = 200
            if len(xy_list_yellow) > virtual_points + 2:
                xy, yy, ry, sy = hyper_fit(xy_list_yellow)
                avg = 2
                disp = 0
        elif len(xy_list_yellow) > virtual_points + 2:
            xy, yy, ry, sy = hyper_fit(xy_list_yellow)
            avg = 1
            disp = -200
            if len(xy_list_blue) > virtual_points + 2:
                xb, yb, rb, sb = hyper_fit(xy_list_blue)
                avg = 2
                disp = 0
        else:
            return self.fit_circle(frame - 1)
        xyr = [(xb + xy) / avg, (yb + yy) / avg, (rb + ry) / avg]
        xyr[0] += disp
        for e in xyr:
            if np.isnan(e) or np.isinf(e):
                return self.fit_circle(frame - 1)

        if xyr[2] > 8000:
            return [0, 0, 0]
        return xyr

    def get_delta(self, frame):
        delta_dist = 1
        xyr = self.fit_circle(frame)
        if xyr == [0, 0, 0]:
            return [0, delta_dist]
        else:
            radius = xyr[2]
            circum = 2 * radius * math.pi
            d_theta = delta_dist / circum
            return [d_theta, delta_dist]

    def integrate_angle(self, frame, velocity):
        velocity *= 7
        xyr = self.fit_circle(frame)
        radius = xyr[2]
        circum = 2 * radius * math.pi
        if circum > 1e-6:
            d_theta = velocity / circum
        else:
            return self.integrated_angle
        if xyr[0] < self.window_width / 2:
            d_theta *= -1
        if abs(d_theta) > .02:
            self.integrated_angle += d_theta
        with open(os.path.join(self.boundaries_dir, str(frame) + '.json')) as jsonFile:
            data = json.load(jsonFile)
            for boundary in data:
                if boundary.get("color") == [0, 100, 100] and boundary.get("p1")[1] < 300:
                    self.integrated_angle = math.pi / 2
        return self.integrated_angle

    def integrate(self, frame):
        with open(os.path.join(self.out_dir, str(frame) + '.json')) as jsonFile:
            data = json.load(jsonFile)
            vel_vector = data.get("velocity_vector")
            self.integrated_angle += vel_vector[0] / self.turning_ratio
            # self.transformation_matrix[0][0] = math.cos(self.integrated_angle)
            # self.transformation_matrix[0][1] = -math.sin(self.integrated_angle)
            # self.transformation_matrix[1][0] = math.sin(self.integrated_angle)
            # self.transformation_matrix[1][1] = math.cos(self.integrated_angle)
            # self.transformation_matrix[0][2] += vel_vector[0]
            self.transformation_matrix[1][2] += vel_vector[1]
            transformation_matrix = {
                "transformation_matrix": self.transformation_matrix
            }
            print("velocity integration progress | ", frame)
            json_object = json.dumps(transformation_matrix, indent=4)
            jsonfile = open(os.path.join(self.integral_dir, str(frame) + '.json'), 'w')
            jsonfile.write(json_object)
            jsonfile.close()

    def integrate_all(self):
        for i in range(len(os.listdir(self.out_dir))):
            self.integrate(i)

    def find_velocity_vector(self, cones_json, prev_cones_json):
        vel_vector = [0, 0]
        with open(cones_json) as jsonFile:
            cone_data = json.load(jsonFile)
            with open(prev_cones_json) as prevJsonFile:
                prev_cone_data = json.load(prevJsonFile)
                for cone in cone_data:
                    point = cone.get("point")
                    closest_point = None
                    min_distance = self.window_width + self.window_height
                    for prev_cone in prev_cone_data:
                        prev_point = prev_cone.get("point")
                        distance = euclidean_distance(point, prev_point)
                        if distance < min_distance and distance < self.correlation_threshold:
                            closest_point = prev_point
                    if closest_point is not None:
                        vel_vector[0] += closest_point[0] - point[0]
                        vel_vector[1] += point[1] - closest_point[1]
        return [vel_vector[0] / len(vel_vector), vel_vector[1] / len(vel_vector)]

    def find_dumb_velocity(self, frame):
        if frame > 0:
            return euclidean_distance(
                self.find_velocity_vector(os.path.join(self.cone_points_dir, str(frame) + '.json'),
                                          os.path.join(self.cone_points_dir, str(frame - 1) + '.json')), (0, 0))
        else:
            return 0

    def find_all_velocity_vectors(self):
        for frame in range(len(os.listdir(self.cone_points_dir))):
            print("velocity vector progress | ", frame)
            if frame == 0:
                velocity_vector = {
                    "velocity_vector": [0.0, 0.0]
                }
                json_object = json.dumps(velocity_vector, indent=4)
                jsonfile = open(os.path.join(self.out_dir, str(0) + '.json'), 'w')
                jsonfile.write(json_object)
                jsonfile.close()

            if frame != 0:
                velocity_vector = self.find_velocity_vector(os.path.join(self.cone_points_dir, str(frame) + '.json'),
                                                            os.path.join(self.cone_points_dir,
                                                                         str(frame - 1) + '.json'))
                moving_average = 3
                if frame > moving_average:
                    for i in range(1, moving_average):
                        with open(os.path.join(self.out_dir, str(frame - i) + '.json')) as jsonFile:
                            data = json.load(jsonFile)
                            v_vector = data.get("velocity_vector")
                            velocity_vector[0] += v_vector[0]
                            velocity_vector[1] += v_vector[1]

                    velocity_vector[0] /= moving_average
                    velocity_vector[1] /= moving_average

                velocity_vector = {
                    "velocity_vector": velocity_vector
                }

        with open(os.path.join(self.out_dir, "1.json")) as jsonFile:
            velocity_vector = json.load(jsonFile)
            jsonfile = open(os.path.join(self.out_dir, '0.json'), 'w')
            json_object = json.dumps(velocity_vector, indent=4)
            jsonfile.write(json_object)
            jsonfile.close()
