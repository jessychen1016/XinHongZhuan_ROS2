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
import socket
import asyncio
import struct
import numpy as np
from utils.utils import distance_along_line

def gps_deg_to_degmin(degree):
    """
    Convert a GPS coordinate in decimal degrees to (degrees, minutes).

    Args:
        degree (float): The coordinate in decimal degrees.

    Returns:
        tuple: (degrees, minutes), where degrees is int, minutes is float.
    """
    degrees = int(degree)
    minutes = abs((degree - degrees) * 60)
    return degrees, minutes

def build_packet(data_points, device_id=111, cmd_type=1, start_addr=0, packet_num=1):
    m_send_dataNum = len(data_points)
    total_bytes = 7 + 4 * m_send_dataNum  # header + data as per protocol

    # Build header (13 bytes)
    Head = bytearray(13)
    Head[0] = 1
    Head[1] = packet_num  # Communication number
    Head[2] = 0
    Head[3] = 0  # TCP protocol type
    Head[4] = total_bytes & 0xFF
    Head[5] = (total_bytes >> 8) & 0xFF
    Head[6] = device_id  # Device address
    Head[7] = cmd_type   # Command type
    Head[8] = 0
    Head[9] = (start_addr) & 0xFF
    Head[10] = m_send_dataNum & 0xFF
    Head[11] = (m_send_dataNum >> 8) & 0xFF
    Head[12] = 0  # Reserved

    # Pack all data points as float32 (little-endian)
    data_section = b''.join(struct.pack('<f', float(val)) for val in data_points)
    return Head + data_section

class AsyncTCPServer:
    def __init__(self, host='10.1.5.16', port=12345):
        self.host = host
        self.port = port
        self.clients = set()
        self.server = None
        self.writer = None  # Store active writer for sending

    async def send_data(self, data_points):
        if not self.writer or self.writer.is_closing():
            return False
            
        packet = build_packet(data_points)
        self.writer.write(packet)
        await self.writer.drain()
        return True


    async def handle_client(self, reader, writer):
        addr = writer.get_extra_info('peername')
        self.clients.add(writer)
        self.writer = writer  # Store active writer for sending
        try:
            while True:
                data = await reader.read(1024)
                if not data:
                    break
                # print(f"Received data from {addr}: {data}")
                try:
                    # Check for complete packet (header + at least 3 float32 values)
                    if len(data) >= 25:  # 13 byte header + 3*4 byte floats
                        # Extract header (13 bytes)
                        header = data[:13]
                        # Extract data points (remaining bytes must be multiple of 4)
                        data_points = data[13:]
                        if len(data_points) % 4 == 0:
                            # Parse all float32 values
                            num_floats = len(data_points) // 4
                            values = struct.unpack(f'<{num_floats}f', data_points)
                            if num_floats >= 3:
                                
                                global long_ship, lat_ship, heading_anlge, long_flag, lat_flag
                                long_ship = values[1]+values[2]/60.0
                                lat_ship = values[3]+values[4]/60.0
                                long_flag = values[5]+values[6]/60.0
                                lat_flag = values[7]+values[8]/60.0
                                heading_anlge = values[0]

                                print(f"Updated ship position: {long_ship}, {lat_ship}, {heading_anlge}")
                                print(f"Updated flag position: {long_flag}, {lat_flag}, {heading_anlge}")
                                response = build_packet([1])  # Ack
                            else:
                                response = build_packet([0])  # Error - not enough data points
                        else:
                            response = build_packet([0])  # Error - malformed data
                    else:
                        # Default response with sample data
                        response = build_packet([1.0, 2.0, 3.0])
                except Exception as e:
                    print(f"Error processing data from {addr}: {e}")
                    response = build_packet([0])  # Error
                    
                writer.write(response)
                await writer.drain()
        except Exception as e:
            print(f"Client {addr} error: {e}")
        finally:
            self.clients.remove(writer)
            writer.close()
            await writer.wait_closed()

    async def start_server(self):
        self.server = await asyncio.start_server(
            self.handle_client, self.host, self.port)
        addr = self.server.sockets[0].getsockname()
        print(f'Serving on {addr}')

    async def stop_server(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()

class ConfigHandler(FileSystemEventHandler):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def on_modified(self, event):
        if event.src_path.endswith('cluster_config.yaml'):
            self.callback()

class PointCloudSubscriber(Node):
    def __init__(self, tcp_server=None):
        super().__init__('pointcloud_subscriber')
        self.tcp_server = tcp_server
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


        self.shoreline_long0 = None
        self.shoreline_lat0 = None
        self.shoreline_long1 = None
        self.shoreline_lat1 = None

        self.rand_add = 0.001

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

                        # For center
                        center_origin, center_origin = 0.0, 0.0
                        center_inter_origin = (m*(center_origin - c) + center_origin)/(m**2 + 1)
                        center_inter_origin = m * center_inter_origin + c
                        # print("hhhhhhhh", long_ship, lat_ship, heading_anlge)
                        fore_gps_long, fore_gps_lat = ship_local_to_gps(center_origin, center_origin, long_ship, lat_ship, heading_anlge)
                        fore_gps_inter_long, fore_gps_inter_lat = ship_local_to_gps(center_inter_origin, center_inter_origin, long_ship, lat_ship, heading_anlge)
                        distance_origin_0 = abs(m*center_origin - center_origin + c) / np.sqrt(m**2 + 1)
                        # self.get_logger().info(f"for_to_shore: {distance_origin_0:.3f} meters")
                        
                        perp_x_origin = np.linspace(center_origin, center_inter_origin, 100)
                        perp_y_origin = np.linspace(center_origin, center_inter_origin, 100)
                        perp_points_origin = np.column_stack([perp_x_origin, perp_y_origin])
                        perp_lines.append(perp_points_origin)
                        
                        # For lidar_1
                        x0_origin, y0_origin = 4.7, 30
                        x0_inter_origin = (m*(y0_origin - c) + x0_origin)/(m**2 + 1)
                        y0_inter_origin = m * x0_inter_origin + c
                        # print("hhhhhhhh", long_ship, lat_ship, heading_anlge)
                        fore_gps_long, fore_gps_lat = ship_local_to_gps(y0_origin, x0_origin, long_ship, lat_ship, heading_anlge)
                        fore_gps_inter_long, fore_gps_inter_lat = ship_local_to_gps(y0_inter_origin, x0_inter_origin, long_ship, lat_ship, heading_anlge)
                        distance_origin_0 = abs(m*x0_origin - y0_origin + c) / np.sqrt(m**2 + 1)
                        # self.get_logger().info(f"for_to_shore: {distance_origin_0:.3f} meters")
                        
                        perp_x_origin = np.linspace(x0_origin, x0_inter_origin, 100)
                        perp_y_origin = np.linspace(y0_origin, y0_inter_origin, 100)
                        perp_points_origin = np.column_stack([perp_x_origin, perp_y_origin])
                        perp_lines.append(perp_points_origin)
                        
                        # For lidar_4
                        x1_origin, y1_origin = 3.7, -9.4
                        x1_inter_origin = (m*(y1_origin - c) + x1_origin)/(m**2 + 1)
                        y1_inter_origin = m * x1_inter_origin + c
                        aft_gps_long, aft_gps_lat = ship_local_to_gps(y1_origin, x1_origin, long_ship, lat_ship, heading_anlge)
                        aft_gps_inter_long, aft_gps_inter_lat = ship_local_to_gps(y1_inter_origin, x1_inter_origin, long_ship, lat_ship, heading_anlge)
                        
                        distance_origin_1 = abs(m*x1_origin - y1_origin + c) / np.sqrt(m**2 + 1)
                        # self.get_logger().info(f"aft_to_shore: {distance_origin_1:.3f} meters")
                        perp_x_origin = np.linspace(x1_origin, x1_inter_origin, 100)
                        perp_y_origin = np.linspace(y1_origin, y1_inter_origin, 100)
                        perp_points_origin = np.column_stack([perp_x_origin, perp_y_origin])
                        perp_lines.append(perp_points_origin)

                        # Initialize the shoreline points once when the code runs
                        if init_shore_flag:
                            init_shore_flag = False
                            # Convert local coordinates to GPS coordinates
                            self.shoreline_long0, self.shoreline_lat0 = ship_local_to_gps(y0_inter_origin, x0_inter_origin, long_ship, lat_ship, heading_anlge)
                            self.shoreline_long1, self.shoreline_lat1 = ship_local_to_gps(y1_inter_origin, x1_inter_origin, long_ship, lat_ship, heading_anlge)
                            self.get_logger().info(f"Initialized shoreline points: {self.shoreline_long0}, {self.shoreline_lat0} and {self.shoreline_long1}, {self.shoreline_lat1}")
                        # Check for enter key press using pynput
                        def on_press(key):
                            if key == keyboard.Key.enter:
                                self.shoreline_long0, self.shoreline_lat0 = ship_local_to_gps(y0_inter_origin, x0_inter_origin, long_ship, lat_ship, heading_anlge)
                                self.shoreline_long1, self.shoreline_lat1 = ship_local_to_gps(y1_inter_origin, x1_inter_origin, long_ship, lat_ship, heading_anlge)
                                self.get_logger().info(f"Shoreline points Updated: {self.shoreline_long0}, {self.shoreline_lat0} and {self.shoreline_long1}, {self.shoreline_lat1}")
                                return False  # Stop listener
                            
                        # print("XXXXXXXX", self.shoreline_long0, self.shoreline_lat0) 
                        # Create data_points array with distances and other relevant data
                        # self.rand_add = self.rand_add + 0.001
                        center_distance_to_flag = distance_along_line(self.shoreline_lat0, self.shoreline_long0, self.shoreline_lat1, self.shoreline_long1,lat_ship,long_ship,lat_flag,long_flag)
                        print(center_distance_to_flag)
                        data_points = [
                            distance_origin_0,
                            center_distance_to_flag, # center to flag
                            distance_origin_1,
                            0, # deviation
                            0, # fore speed
                            0, # center speed
                            0, # aft speed
                            0, # fore heading angle
                            self.imu_rot_yaw, # v_rot_yaw
                            222, # drift
                            2, # shore point count
                            gps_deg_to_degmin(self.shoreline_lat0)[0], #shoreline_lat0 degree
                            gps_deg_to_degmin(self.shoreline_lat0)[1], #shoreline_lat0 minute
                            1, #shoreline_lat0 hemisphere
                            gps_deg_to_degmin(self.shoreline_long0)[0], #shoreline_long0 degree
                            gps_deg_to_degmin(self.shoreline_long0)[1], #shoreline_long0 minute
                            1, #shoreline_long0 hemisphere
                            0,0,
                            gps_deg_to_degmin(self.shoreline_lat1)[0], #shoreline_lat1 degree
                            gps_deg_to_degmin(self.shoreline_lat1)[1], #shoreline_lat1 minute
                            1, #shoreline_lat1 hemisphere
                            gps_deg_to_degmin(self.shoreline_long1)[0], #shoreline_long1 degree
                            gps_deg_to_degmin(self.shoreline_long1)[1], #shoreline_long1 minute
                            1, #shoreline_long1 hemisphere
                            0,0, #uesless but needed
                            0,0,0,0,0,0,0,0, 
                            0,0,0,0,0,0,0,0,
                            0,0,0,0,0,0,0,0, 
                            0,0,0,0,0,0,0,0,
                            0,0,0,0,0,0,0,0,
                            0,0,0,0,0,0,0,0,
                            gps_deg_to_degmin(fore_gps_lat)[0], #shoreline_lat0 degree
                            gps_deg_to_degmin(fore_gps_lat)[1], #shoreline_lat0 minute
                            gps_deg_to_degmin(fore_gps_long)[0], #shoreline_long0 degree
                            gps_deg_to_degmin(fore_gps_long)[1], #shoreline_long0 minute
                            gps_deg_to_degmin(fore_gps_inter_lat)[0], #shoreline_lat0 degree
                            gps_deg_to_degmin(fore_gps_inter_lat)[1], #shoreline_lat0 minute
                            gps_deg_to_degmin(fore_gps_inter_long)[0], #shoreline_long0 degree
                            gps_deg_to_degmin(fore_gps_inter_long)[1], #shoreline_long0 minute
                            gps_deg_to_degmin(aft_gps_lat)[0], #shoreline_lat0 degree
                            gps_deg_to_degmin(aft_gps_lat)[1], #shoreline_lat0 minute
                            gps_deg_to_degmin(aft_gps_long)[0], #shoreline_long0 degree
                            gps_deg_to_degmin(aft_gps_long)[1], #shoreline_long0 minute
                            gps_deg_to_degmin(aft_gps_inter_lat)[0], #shoreline_lat0 degree
                            gps_deg_to_degmin(aft_gps_inter_lat)[1], #shoreline_lat0 minute
                            gps_deg_to_degmin(aft_gps_inter_long)[0], #shoreline_long0 degree
                            gps_deg_to_degmin(aft_gps_inter_long)[1], #shoreline_long0 minute
                        ]
                        # Send data via TCP if server is available
                        if self.tcp_server:
                            asyncio.create_task(self.tcp_server.send_data(data_points))
                        
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


async def ros_main(args=None):
    rclpy.init(args=args)
    # Start TCP server first
    tcp_server = AsyncTCPServer()
    server_task = asyncio.create_task(tcp_server.start_server())
    # Create subscriber with TCP server reference
    pointcloud_subscriber = PointCloudSubscriber(tcp_server)
    global shoreline_publisher
    shoreline_publisher = ShorelinePublisher()
    
    try:
        while rclpy.ok():
            rclpy.spin_once(pointcloud_subscriber, timeout_sec=0.1)
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        await tcp_server.stop_server()
        pointcloud_subscriber.destroy_node()
        shoreline_publisher.destroy_node()
        rclpy.shutdown()

def main(args=None):
    asyncio.run(ros_main(args))


if __name__ == '__main__':
    global long_ship, lat_ship, heading_anlge, init_shore_flag, long_flag, lat_flag
    long_ship = 120.2  # Example longitude of ship center
    lat_ship = 35.5    # Example latitude of ship center
    long_flag = 120.2  # Example longitude of ship center
    lat_flag = 35.5    # Example latitude of ship center
    heading_anlge = 45  # Example heading angle in degrees
    init_shore_flag = True  # Flag to indicate if shoreline points are initialized
    main()
