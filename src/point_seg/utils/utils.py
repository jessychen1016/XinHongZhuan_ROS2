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
from math import radians, degrees, sin, cos, atan2, sqrt, asin

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 637100  # Radius of earth in meters
    # print(c*r)
    return c * r

def bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing between two points
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    initial_bearing = atan2(x, y)
    initial_bearing = degrees(initial_bearing)
    return (initial_bearing + 360) % 360

def project_point_to_line(lat1, lon1, lat2, lon2, latp, lonp):
    """
    Project point (latp, lonp) onto the great circle line defined by (lat1,lon1) and (lat2,lon2)
    Returns projected point (lat, lon)
    """
    # Convert all points to radians
    lat1, lon1, lat2, lon2, latp, lonp = map(radians, [lat1, lon1, lat2, lon2, latp, lonp])
    
    # Calculate bearing and distance from point to line start
    brng = bearing(lat1, lon1, lat2, lon2)
    d = haversine_distance(lat1, lon1, latp, lonp)
    
    # Calculate projected point
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    d_rad = d / 6371.0  # angular distance in radians
    
    lat = asin(sin(lat1_rad)*cos(d_rad) + cos(lat1_rad)*sin(d_rad)*cos(radians(brng)))
    lon = lon1_rad + atan2(sin(radians(brng))*sin(d_rad)*cos(lat1_rad), 
                          cos(d_rad)-sin(lat1_rad)*sin(lat))
    
    return degrees(lat), degrees(lon)

def distance_along_line(lat1, lon1, lat2, lon2, lat3, lon3, lat4, lon4):
    """
    Calculate distance between two points (lat3,lon3 and lat4,lon4)
    along the direction of a line defined by (lat1,lon1 and lat2,lon2).
    Returns the distance between the points' projections onto the line.
    """
    # Project both points onto the line
    proj3 = project_point_to_line(lat1, lon1, lat2, lon2, lat3, lon3)
    proj4 = project_point_to_line(lat1, lon1, lat2, lon2, lat4, lon4)
    
    # Calculate distance between the projected points
    return haversine_distance(proj3[0], proj3[1], proj4[0], proj4[1])

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
