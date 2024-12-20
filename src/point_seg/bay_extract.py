import rclpy
from rclpy.node import Node
import numpy as np
import ros2_numpy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import matplotlib.pyplot as plt
from sensor_msgs_py.point_cloud2 import read_points, create_cloud
from utils import *


class PointCloudSubscriber(Node):
    def __init__(self):
        super().__init__('pointcloud_subscriber')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/lidar_points_1',  # Replace with your point cloud topic
            self.listener_callback,
            10
        )
        self.subscription
        self.point_cloud_data = None

    def listener_callback(self, msg):
        # Read the point cloud data
        # self.point_cloud_data = pc2.read_points(msg, field_names=['x', 'y', 'z', "intensity"], skip_nans=True)
        pc = ros2_numpy.numpify(msg)
        self.points=pc['xyz']
        self.process_point_cloud()

    def process_point_cloud(self):
        if self.points is None:
            return
        fov =[-60,60] #degrees
        resolution = 0.5 #degree

        # Convert to numpy array
        point_cloud_np = np.array(self.points, dtype=np.float64)
        points_2d = point_cloud_np[:, :2]

        # Precompute polar coordinates
        polar_coords = np.column_stack([
            np.arctan2(points_2d[:, 1], points_2d[:, 0]) * 180 / np.pi,
            np.linalg.norm(points_2d, axis=1)
        ])

        # Filter points within FoV
        fov_mask = (polar_coords[:, 0] >= fov[0]) & (polar_coords[:, 0] <= fov[1])
        valid_points = points_2d[fov_mask]
        valid_polar = polar_coords[fov_mask]

        # Bin angles
        rounded_angles = (np.round(valid_polar[:, 0] / resolution) * resolution).astype(int)

        # Find the closest point for each bin
        bins = {}
        for angle, dist, point in zip(rounded_angles, valid_polar[:, 1], valid_points):
            if angle not in bins or bins[angle][1] > dist:
                bins[angle] = (point, dist)

        # Extract the closest points
        closest_points = np.array([point for point, _ in bins.values()])

        # Filter and publish
        closest_points = closest_points[closest_points[:, 0] > 0.2]
        closest_points_3d = np.hstack((closest_points, np.zeros((closest_points.shape[0], 1))))
        assign_z_to_2d_points(points_2d, points_3d)
        shoreline_publisher.pointcloud_publish(closest_points_3d)



class ShorelinePublisher(Node):
    def __init__(self):
        super().__init__('mesh_to_pointcloud_publisher')
        self.publisher = self.create_publisher(PointCloud2, '/shoreline_pointcloud', 1)
        self.timer = self.create_timer(0.1, self.pointcloud_publish)  # Publish every 0.1 second


    def pointcloud_publish(self, points):
        
        # Convert combined points to PointCloud2 message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'hesai_at128'  # Update to the appropriate frame_id

        pc_data = pc2.create_cloud_xyz32(header, points)
        
        # Publish the PointCloud2 message
        self.publisher.publish(pc_data)
        self.get_logger().info('Published Combined PointCloud2 Message')




def main(args=None):
    rclpy.init(args=args)
    pointcloud_subscriber = PointCloudSubscriber()
    global shoreline_publisher
    shoreline_publisher = ShorelinePublisher()
    try:
        rclpy.spin(pointcloud_subscriber)
        rclpy.spin(shoreline_publisher)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    finally:
        pointcloud_subscriber.destroy_node()
        shoreline_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
