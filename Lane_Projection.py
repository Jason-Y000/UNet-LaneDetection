import cv2
import numpy as np

def project_2D_to_3D_camera(x_array, y_array, depth_to_y_map, f_x, f_y, o_x, o_y):
    ''' Given the pixel image coordinates and a constant mapping of depth to y-values,
        calculate the resulting projection of points in the camera coordinate frame.
        Requires the focal lengths in x and y and the optical centers
    '''
    x_cam_coords = []
    y_cam_coords = []
    z_cam_coords = []

    for x,y in zip(x_array,y_array):
        x_project = (x - o_x)/f_x*depth_to_y_map[y]
        y_project = (y - o_y)/f_y*depth_to_y_map[y]
        z_project = depth_to_y_map[y]

        x_cam_coords.append(x_project)
        y_cam_coords.append(y_project)
        z_cam_coords.append(z_project)

    return x_cam_coords, y_cam_coords, z_cam_coords

def gen_depth_to_y_map(min_z, max_z, num_y_pixels):
    ''' Create the constant depth to y map '''

    depth_to_y_map = {}
    delta_z = (max_z-min_z)/num_y_pixels

    for n,y in enumerate(range(num_y_pixels-1,-1,-1),1):
        depth_to_y_map[y] = delta_z*n

    return depth_to_y_map
