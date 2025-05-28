import rclpy
from rclpy.node import Node
import numpy as np
import ros2_numpy
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, Imu
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs_py.point_cloud2 import create_cloud
import yaml
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from bay_extract.local2gps import ship_local_to_gps
from pynput import keyboard


class ConfigHandler(FileSystemEventHandler):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def on_modified(self, event):
        if event.src_path.endswith('cluster_config.yaml'):
            self.callback()

class PointCloudSubscriber(Node):
    def __init__(self):
        super().__init__('pointcloud_subscriber')
        self.config_path = os.path.join(os.path.dirname(__file__), 'cluster_config.yaml')
        self.cluster_number = 1
        self.filter_window = 5  # Default filter window size
        self.line_history = {}  # Stores history of fitted lines per cluster
        self.load_config()
        
        # Setup config file watcher
        self.event_handler = ConfigHandler(self.load_config)
        self.observer = Observer()
        self.observer.schedule(self.event_handler, os.path.dirname(self.config_path))
        self.observer.start()
        
        # Subscribe to both lidar topics with separate callbacks
        self.subscription1 = self.create_subscription(
            PointCloud2,
            '/lidar_points_1',
            self.lidar1_callback,
            10
        )
        self.subscription2 = self.create_subscription(
            PointCloud2,
            '/lidar_points_4',
            self.lidar4_callback,
            10
        )

        self.subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10)
        self.points1 = None
        self.points2 = None
        self.imu_rot_yaw = None

    def lidar1_callback(self, msg):
        pc = ros2_numpy.numpify(msg)
        # Downsample to 1/10 by taking every 10th point
        self.points1 = pc['xyz'][::10]
        self.check_and_process(msg.header.stamp)

    def lidar4_callback(self, msg):
        pc = ros2_numpy.numpify(msg)
        # Downsample to 1/10 by taking every 10th point
        self.points2 = pc['xyz'][::10]
        self.check_and_process(msg.header.stamp)
    def imu_callback(self, msg):
        # Get z-axis angular velocity directly
        self.imu_rot_yaw = msg.angular_velocity.z
        # print(f'Received angular velocity z: {self.imu_rot_yaw:.2f} rad/s')
        
    def check_and_process(self, stamp):
        # Only process when we have data from both lidars
        if self.points1 is not None and self.points2 is not None:
            # Merge the point clouds and filter by z >= -0.5
            self.points = np.concatenate((self.points1, self.points2))
            self.points = self.points[self.points[:, 2] >= -0.5]  # Keep only points with z >= -0.5
            self.process_point_cloud(stamp)

    def load_config(self):
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
                self.cluster_number = config.get('cluster_number', 1)
                self.filter_window = config.get('filter_window', 5)
                self.get_logger().info(f"Loaded config - cluster_number: {self.cluster_number}, filter_window: {self.filter_window}")
        except Exception as e:
            self.get_logger().error(f"Error loading config: {str(e)}")
            self.cluster_number = 1

    def process_point_cloud(self, timestamp):
        global long_ship, lat_ship, heading_anlge, init_shore_flag  # Declare global scope first
        if self.points is None:
            return
            
        fov = [-90,90]  # degrees
        resolution = 0.5  # degree

        # Convert to numpy array and get 2D points
        point_cloud_np = np.array(self.points, dtype=np.float64)
        points_2d = point_cloud_np[:, :2]

        # Get polar coordinates and filter by FOV
        polar_coords = np.column_stack([
            np.arctan2(points_2d[:, 1], points_2d[:, 0]) * 180 / np.pi,
            np.linalg.norm(points_2d, axis=1)
        ])
        fov_mask = (polar_coords[:, 0] >= fov[0]) & (polar_coords[:, 0] <= fov[1])
        valid_points = points_2d[fov_mask]
        valid_polar = polar_coords[fov_mask]

        # Bin angles and find closest point in each bin
        rounded_angles = (np.round(valid_polar[:, 0] / resolution) * resolution).astype(int)
        bins = {}
        for angle, dist, point in zip(rounded_angles, valid_polar[:, 1], valid_points):
            if angle not in bins or bins[angle][1] > dist:
                bins[angle] = (point, dist)

        # Get and sort closest points
        closest_points = np.array([point for point, _ in bins.values()])
        angles = np.arctan2(closest_points[:,1], closest_points[:,0])
        closest_points = closest_points[angles.argsort()]

        # Filter points
        closest_points = closest_points[closest_points[:, 0] > 0.2]
        
        # Publish closest points
        closest_points_3d = np.hstack((closest_points, np.zeros((closest_points.shape[0], 1))))
        shoreline_publisher.publish_closest_points(closest_points_3d)
        
        # Cluster points using KMeans
        if len(closest_points) > 1:
            kmeans = KMeans(n_clusters=min(self.cluster_number, len(closest_points)), random_state=196)
            clusters = kmeans.fit_predict(closest_points)
            
            # Fit RANSAC line to each cluster and calculate perpendicular lines
            cluster_lines = []
            perp_lines = []
            for cluster_id in np.unique(clusters):
                cluster_points = closest_points[clusters == cluster_id]
                if len(cluster_points) > 1:
                    X = cluster_points[:, 0].reshape(-1, 1)
                    y = cluster_points[:, 1]
                    ransac = RANSACRegressor(random_state=42)
                    ransac.fit(X, y)
                    
                    # Get RANSAC line parameters and apply smoothing
                    new_m = ransac.estimator_.coef_[0]
                    new_c = ransac.estimator_.intercept_
                    
                    # Update line history for this cluster
                    if cluster_id not in self.line_history:
                        self.line_history[cluster_id] = []
                    self.line_history[cluster_id].append((new_m, new_c))
                    
                    # Keep only last N entries
                    if len(self.line_history[cluster_id]) > self.filter_window:
                        self.line_history[cluster_id].pop(0)
                    
                    # Calculate smoothed line parameters (weighted average favoring recent values)
                    weights = np.linspace(0.1, 1.0, len(self.line_history[cluster_id]))
                    weights /= weights.sum()
                    
                    m_history = np.array([x[0] for x in self.line_history[cluster_id]])
                    c_history = np.array([x[1] for x in self.line_history[cluster_id]])
                    
                    m = np.dot(m_history, weights)
                    c = np.dot(c_history, weights)
                    
                    # Generate points along fitted line
                    x_vals = np.linspace(X.min(), X.max(), 10000)
                    y_vals = ransac.predict(x_vals.reshape(-1, 1))
                    line_points = np.column_stack([x_vals, y_vals])
                    cluster_lines.append(line_points)
                    
                    # Calculate perpendicular lines from both (4.7, 30) and (3.7, -9.4)
                    if m != 0:  # Avoid division by zero
                        
                        # For lidar_1
                        x0_origin, y0_origin = 4.7, 30
                        x0_inter_origin = (m*(y0_origin - c) + x0_origin)/(m**2 + 1)
                        y0_inter_origin = m * x0_inter_origin + c
                        fore_gps = ship_local_to_gps(y0_origin, x0_origin, long_ship, lat_ship, heading_anlge)
                        fore_gps_inter = ship_local_to_gps(y0_inter_origin, x0_inter_origin, long_ship, lat_ship, heading_anlge)
                        distance_origin_0 = abs(m*x0_origin - y0_origin + c) / np.sqrt(m**2 + 1)
                        self.get_logger().info(f"for_to_shore: {distance_origin_0:.3f} meters")
                        
                        perp_x_origin = np.linspace(x0_origin, x0_inter_origin, 100)
                        perp_y_origin = np.linspace(y0_origin, y0_inter_origin, 100)
                        perp_points_origin = np.column_stack([perp_x_origin, perp_y_origin])
                        perp_lines.append(perp_points_origin)
                        
                        # For lidar_4
                        x1_origin, y1_origin = 3.7, -9.4
                        x1_inter_origin = (m*(y1_origin - c) + x1_origin)/(m**2 + 1)
                        y1_inter_origin = m * x1_inter_origin + c
                        aft_gps = ship_local_to_gps(y1_origin, x1_origin, long_ship, lat_ship, heading_anlge)
                        aft_gps_inter = ship_local_to_gps(y1_inter_origin, x1_inter_origin, long_ship, lat_ship, heading_anlge)
                        # print(f"Perpendicular point for (3.7, -9.4): {perpen_point_1}")
                        
                        distance_origin_1 = abs(m*x1_origin - y1_origin + c) / np.sqrt(m**2 + 1)
                        self.get_logger().info(f"aft_to_shore: {distance_origin_1:.3f} meters")
                        
                        perp_x_origin = np.linspace(x1_origin, x1_inter_origin, 100)
                        perp_y_origin = np.linspace(y1_origin, y1_inter_origin, 100)
                        perp_points_origin = np.column_stack([perp_x_origin, perp_y_origin])
                        perp_lines.append(perp_points_origin)

                        # Initialize the shoreline points once when the code runs
                        if init_shore_flag:
                            init_shore_flag = False
                            # Convert local coordinates to GPS coordinates
                            shoreline_long0, shoreline_lat0 = ship_local_to_gps(y0_inter_origin, x0_inter_origin, long_ship, lat_ship, heading_anlge)
                            shoreline_long1, shoreline_lat1 = ship_local_to_gps(y1_inter_origin, x1_inter_origin, long_ship, lat_ship, heading_anlge)
                            self.get_logger().info(f"Initialized shoreline points: {shoreline_long0}, {shoreline_lat0} and {shoreline_long1}, {shoreline_lat1}")

                        shoreline_long0, shoreline_lat0 = ship_local_to_gps(y0_inter_origin, x0_inter_origin, long_ship, lat_ship, heading_anlge)
                        # Check for enter key press using pynput
                        def on_press(key):
                            if key == keyboard.Key.enter:
                                shoreline_long0, shoreline_lat0 = ship_local_to_gps(y0_inter_origin, x0_inter_origin, long_ship, lat_ship, heading_anlge)
                                shoreline_long1, shoreline_lat1 = ship_local_to_gps(y1_inter_origin, x1_inter_origin, long_ship, lat_ship, heading_anlge)
                                self.get_logger().info(f"Shoreline points Updated: {shoreline_long0}, {shoreline_lat0} and {shoreline_long1}, {shoreline_lat1}")
                                return False  # Stop listener
                        
                        # Start non-blocking listener
                        listener = keyboard.Listener(on_press=on_press)
                        listener.start()


            
            # Combine all line points for visualization
            if cluster_lines:
                line_points_3d = np.vstack(cluster_lines)
                line_points_3d = np.hstack((line_points_3d, np.zeros((line_points_3d.shape[0], 1))))
                shoreline_publisher.pointcloud_publish(line_points_3d)
                
                if perp_lines:
                    perp_points_3d = np.vstack(perp_lines)
                    perp_points_3d = np.hstack((perp_points_3d, np.zeros((perp_points_3d.shape[0], 1))))
                    shoreline_publisher.publish_perpendicular_lines(perp_points_3d)
                return
        
        # Fallback to original behavior if clustering fails
        shoreline_publisher.pointcloud_publish(closest_points_3d)


class ShorelinePublisher(Node):
    def __init__(self):
        super().__init__('shoreline_publisher')
        self.publisher = self.create_publisher(PointCloud2, '/shoreline_pointcloud', 1)
        self.perp_publisher = self.create_publisher(PointCloud2, '/perpen_line', 1)
        self.closest_publisher = self.create_publisher(PointCloud2, '/closest_points', 1)

    def pointcloud_publish(self, points):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'ship_center'
        pc_data = pc2.create_cloud_xyz32(header, points)
        self.publisher.publish(pc_data)

    def publish_perpendicular_lines(self, points):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'ship_center'
        pc_data = pc2.create_cloud_xyz32(header, points)
        self.perp_publisher.publish(pc_data)

    def publish_closest_points(self, points):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'ship_center'
        pc_data = pc2.create_cloud_xyz32(header, points)
        self.closest_publisher.publish(pc_data)


def main(args=None):
    rclpy.init(args=args)
    pointcloud_subscriber = PointCloudSubscriber()
    global shoreline_publisher
    shoreline_publisher = ShorelinePublisher()
    try:
        rclpy.spin(pointcloud_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        pointcloud_subscriber.destroy_node()
        shoreline_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    global long_ship, lat_ship, heading_anlge, init_shore_flag
    long_ship = 120.2  # Example longitude of ship center
    lat_ship = 35.5    # Example latitude of ship center
    heading_anlge = 45  # Example heading angle in degrees
    init_shore_flag = True  # Flag to indicate if shoreline points are initialized
    main()
