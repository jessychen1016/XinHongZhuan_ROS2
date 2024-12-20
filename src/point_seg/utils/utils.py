## The code below is "ported" from 
# https://github.com/ros/common_msgs/tree/noetic-devel/sensor_msgs/src/sensor_msgs
# I'll make an official port and PR to this repo later: 
# https://github.com/ros2/common_interfaces
import sys
from collections import namedtuple
import ctypes
import math
import struct
from sensor_msgs.msg import PointCloud2, PointField
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def scanclustering(pcd):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Info) as cm:labels = np.array(
        pcd.cluster_dbscan(eps=0.1, min_points=5, print_progress=False))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    # o3d.visualization.draw_geometries([pcd],
    #                                 zoom=0.455,
    #                                 front=[-0.4999, -0.1659, -0.8499],
    #                                 lookat=[2.1813, 2.0619, 2.0999],
    #                                 up=[0.1204, -0.9852, 0.1215])
    return pcd

def assign_z_to_2d_points(points_2d, points_3d):
    """
    Assigns each 2D point a Z value based on the average Z of points in the 3D point cloud
    that share the same X, Y coordinates.

    :param points_2d: [n, 2] array of 2D points (x, y).
    :param points_3d: [m, 3] array of 3D points (x, y, z).
    :return: [n, 3] array with 2D points extended to 3D (x, y, z).
    """

    # Create a pandas DataFrame from the 3D points to group by (x, y) and calculate mean z
    df_3d = pd.DataFrame(points_3d, columns=["x", "y", "z"])

    # Group by 'x' and 'y' and calculate the mean of 'z' for each (x, y)
    avg_z_df = df_3d.groupby(['x', 'y'], as_index=False)['z'].mean()

    # Convert points_2d into a pandas DataFrame to merge
    df_2d = pd.DataFrame(points_2d, columns=["x", "y"])

    # Merge the 2D points with the average z values from 3D point cloud
    result_df = pd.merge(df_2d, avg_z_df, on=["x", "y"], how="left")

    # Return the result as a numpy array
    return result_df.to_numpy()