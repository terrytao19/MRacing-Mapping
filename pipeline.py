import cv2
import numpy as np
import os
import json

from Plotter import Plotter
from PnP import PnP
from ICP import ICP
from Velocity import Velocity
from Ransac import Ransac
from Boundaries import Boundaries

if __name__ == '__main__':
    pnp = PnP()
    # pnp.do_pnp_all()
    plotter = Plotter()
    boundaries = Boundaries()
    # ransac = Ransac()

    # boundaries.find_all_boundaries()

    # plotter_absolute = Plotter(out_directory='C:\\Users\\terry\\MICHIGAN\\MRacing\\videos\\out\\export-absolute.mp4')

    # icp = ICP()
    # icp.find_all_transforms()

    velocity = Velocity()
    # velocity.find_all_velocity_vectors()
    # velocity.integrate_all()

    bbox_storage_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\bbox_storage'
    cone_points_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\cone_points'
    velocity_vectors_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\velocity_vectors'
    velocity_integrals_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\integrated_velocities'
    boundaries_dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\boundaries'

    for frame_number in range(len(os.listdir(cone_points_dir))):
        plotter.plot_frame(os.path.join(cone_points_dir, str(frame_number) + '.json'))
        # plotter.plot_frame_bbox(os.path.join(bbox_storage_dir, str(frame_number) + '.json'))
        # plotter_absolute.plot_vector(os.path.join(velocity_vectors_dir, str(frame_number) + '.json'))
        # plotter_absolute.plot_car(os.path.join(velocity_integrals_dir, str(frame_number) + '.json'))
        # plotter.plot_line(ransac.do_ransac(os.path.join(cone_points_dir, str(frame_number) + '.json')))
        plotter.plot_boundaries(os.path.join(boundaries_dir, str(frame_number) + '.json'))
        # plotter.plot_angle(velocity.integrate_angle(frame_number))
        # plotter.plot_frame(os.path.join(cone_points_dir, str(frame_number - 8) + '.json'))
        plotter.plot_circle(velocity.fit_circle(frame_number))
        plotter.plot_text(str(len(pnp.get_frame_points(frame_number))))

        plotter.write_frame()
        # plotter_absolute.write_frame()
        print("Plotter Progress | ", frame_number)

###

    # plotter = Plotter()
    # plotter.__int__()
    # dir = 'C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\runs\\bbox_storage'
    # for frame_number in range(len(os.listdir(dir))):
    #     plotter.plot_frame_bbox(os.path.join(dir, str(frame_number) + '.json'))
    #     print(frame_number)

