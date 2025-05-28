import rclpy
from rclpy.node import Node
import numpy as np
import ros2_numpy
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs_py.point_cloud2 import create_cloud
import yaml
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


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
        
        self.subscription = self.create_subscription(
            PointCloud2,
            '/lidar_points_1',
            self.listener_callback,
            10
        )
        self.subscription
        self.point_cloud_data = None

    def listener_callback(self, msg):
        pc = ros2_numpy.numpify(msg)
        self.points = pc['xyz']
        self.process_point_cloud(msg.header.stamp)

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
        if self.points is None:
            return
            
        fov = [-45, 45]  # degrees
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
                    # self.get_logger().info(f"RANSAC line params - slope: {m:.3f}, intercept: {c:.3f}")
                    # Verify line equation by checking first and last points
                    first_point = cluster_points[0]
                    last_point = cluster_points[-1]
                    first_y = m * first_point[0] + c
                    last_y = m * last_point[0] + c
                    # self.get_logger().info(f"Line verification - First point: {first_point[1]:.3f}, Predicted: {first_y:.3f}")
                    # self.get_logger().info(f"Line verification - Last point: {last_point[1]:.3f}, Predicted: {last_y:.3f}")
                    
                    # Generate points along fitted line
                    x_vals = np.linspace(X.min(), X.max(), 100)
                    y_vals = ransac.predict(x_vals.reshape(-1, 1))
                    line_points = np.column_stack([x_vals, y_vals])
                    cluster_lines.append(line_points)
                    
                    # Calculate perpendicular lines from both (0,0) and (0,-20)
                    if m != 0:  # Avoid division by zero
                        perp_m = -1/m
                        
                        # For (0,0)
                        x0_origin, y0_origin = 0, 0
                        x_inter_origin = (m*(y0_origin - c) + x0_origin)/(m**2 + 1)
                        y_inter_origin = m * x_inter_origin + c
                        
                        distance_origin = abs(m*x0_origin - y0_origin + c) / np.sqrt(m**2 + 1)
                        self.get_logger().info(f"origin_to_{cluster_id}: {distance_origin:.3f} meters")
                        
                        perp_x_origin = np.linspace(x0_origin, x_inter_origin, 100)
                        perp_y_origin = np.linspace(y0_origin, y_inter_origin, 100)
                        perp_points_origin = np.column_stack([perp_x_origin, perp_y_origin])
                        perp_lines.append(perp_points_origin)
                        
                        # For (0,-20)
                        x0_offset, y0_offset = 0, -39.4
                        x_inter_offset = (m*(y0_offset - c) + x0_offset)/(m**2 + 1)
                        y_inter_offset = m * x_inter_offset + c
                        
                        distance_offset = abs(m*x0_offset - y0_offset + c) / np.sqrt(m**2 + 1)
                        self.get_logger().info(f"offset_to_{cluster_id}: {distance_offset:.3f} meters")
                        
                        perp_x_offset = np.linspace(x0_offset, x_inter_offset, 100)
                        perp_y_offset = np.linspace(y0_offset, y_inter_offset, 100)
                        perp_points_offset = np.column_stack([perp_x_offset, perp_y_offset])
                        perp_lines.append(perp_points_offset)
            
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
        closest_points_3d = np.hstack((closest_points, np.zeros((closest_points.shape[0], 1))))
        shoreline_publisher.pointcloud_publish(closest_points_3d)


class ShorelinePublisher(Node):
    def __init__(self):
        super().__init__('shoreline_publisher')
        self.publisher = self.create_publisher(PointCloud2, '/shoreline_pointcloud', 1)
        self.perp_publisher = self.create_publisher(PointCloud2, '/perpen_line', 1)

    def pointcloud_publish(self, points):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'hesai_at128'
        pc_data = pc2.create_cloud_xyz32(header, points)
        self.publisher.publish(pc_data)

    def publish_perpendicular_lines(self, points):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'hesai_at128'
        pc_data = pc2.create_cloud_xyz32(header, points)
        self.perp_publisher.publish(pc_data)


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
    main()
