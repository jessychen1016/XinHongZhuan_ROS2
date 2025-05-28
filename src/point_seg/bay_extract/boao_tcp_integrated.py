import rclpy
from rclpy.node import Node
import numpy as np
import ros2_numpy
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import yaml
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from bay_extract.local2gps import ship_local_to_gps
from pynput import keyboard
import socket
import struct
import time
import threading

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
        self.filter_window = 5
        self.line_history = {}
        self.load_config()
        
        # TCP communication variables
        self.long_ship = 120.2  # Will be updated from TCP
        self.lat_ship = 35.5    # Will be updated from TCP
        self.heading_angle = 45  # Will be updated from TCP
        self.init_shore_flag = True
        
        # Setup config file watcher
        self.event_handler = ConfigHandler(self.load_config)
        self.observer = Observer()
        self.observer.schedule(self.event_handler, os.path.dirname(self.config_path))
        self.observer.start()
        
        # Subscribe to lidar topics
        self.subscription1 = self.create_subscription(
            PointCloud2, '/lidar_points_1', self.lidar1_callback, 10)
        self.subscription2 = self.create_subscription(
            PointCloud2, '/lidar_points_4', self.lidar4_callback, 10)
        self.points1 = None
        self.points2 = None

        # Start TCP server in a separate thread
        self.tcp_thread = threading.Thread(target=self.start_tcp_server)
        self.tcp_thread.daemon = True
        self.tcp_thread.start()

    def start_tcp_server(self):
        HOST = '127.0.0.1'
        PORT = 12345
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((HOST, PORT))
            except OSError as e:
                if e.errno == 98:  # Address already in use
                    time.sleep(5)
                    s.bind((HOST, PORT))
                else:
                    raise
            s.listen(1)
            self.get_logger().info(f"TCP Server listening on {HOST}:{PORT}")
            conn, addr = s.accept()
            with conn:
                self.get_logger().info(f"TCP Connected by {addr}")
                while True:
                    header = self.recv_all(conn, 13)
                    iLength = header[4] | (header[5] << 8)
                    total_len = iLength + 6
                    if total_len > 13:
                        rest = self.recv_all(conn, total_len - 13)
                        packet = header + rest
                    else:
                        packet = header
                    self.parse_tcp_packet(packet)

    def recv_all(self, sock, length):
        data = b''
        while len(data) < length:
            more = sock.recv(length - len(data))
            if not more:
                raise EOFError('Socket closed before all data received')
            data += more
        return data

    def parse_tcp_packet(self, packet):
        if len(packet) < 13:
            return

        buf = packet
        device_id = buf[6]
        if device_id not in (0x6f, 0xC8):
            return

        iLength = buf[4] | (buf[5] << 8)
        if iLength + 6 > len(packet):
            return

        iDataNum = buf[10] | (buf[11] << 8)
        if (iDataNum * 4) + 7 != iLength:
            return

        data_section = packet[13:13 + iDataNum * 4]
        data_points = [
            struct.unpack('<f', data_section[i*4:(i+1)*4])[0]
            for i in range(iDataNum)
        ]
        
        # Update ship position from TCP data (assuming positions are in data_points)
        if len(data_points) >= 8:
            self.heading_angle = data_points[7]  # 8th element is heading angle
            # Update other position data as needed

    def build_tcp_packet(self, data_points, device_id=111, cmd_type=1, start_addr=0, packet_num=1):
        m_send_dataNum = len(data_points)
        total_bytes = 7 + 4 * m_send_dataNum

        Head = bytearray(13)
        Head[0] = 1
        Head[1] = packet_num
        Head[2] = 0
        Head[3] = 0
        Head[4] = total_bytes & 0xFF
        Head[5] = (total_bytes >> 8) & 0xFF
        Head[6] = device_id
        Head[7] = cmd_type
        Head[8] = 0
        Head[9] = (start_addr) & 0xFF
        Head[10] = m_send_dataNum & 0xFF
        Head[11] = (m_send_dataNum >> 8) & 0xFF
        Head[12] = 0

        data_section = b''.join(struct.pack('<f', float(val)) for val in data_points)
        return Head + data_section

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

    def lidar1_callback(self, msg):
        pc = ros2_numpy.numpify(msg)
        self.points1 = pc['xyz'][::10]  # Downsample
        self.check_and_process(msg.header.stamp)

    def lidar4_callback(self, msg):
        pc = ros2_numpy.numpify(msg)
        self.points2 = pc['xyz'][::10]  # Downsample
        self.check_and_process(msg.header.stamp)

    def check_and_process(self, stamp):
        if self.points1 is not None and self.points2 is not None:
            self.points = np.concatenate((self.points1, self.points2))
            self.points = self.points[self.points[:, 2] >= -0.5]
            self.process_point_cloud(stamp)

    def process_point_cloud(self, timestamp):
        if self.points is None:
            return
            
        fov = [-90, 90]
        resolution = 0.5
        point_cloud_np = np.array(self.points, dtype=np.float64)
        points_2d = point_cloud_np[:, :2]

        # Process points and calculate shoreline
        polar_coords = np.column_stack([
            np.arctan2(points_2d[:, 1], points_2d[:, 0]) * 180 / np.pi,
            np.linalg.norm(points_2d, axis=1)
        ])
        fov_mask = (polar_coords[:, 0] >= fov[0]) & (polar_coords[:, 0] <= fov[1])
        valid_points = points_2d[fov_mask]
        valid_polar = polar_coords[fov_mask]

        # Find closest points
        rounded_angles = (np.round(valid_polar[:, 0] / resolution) * resolution).astype(int)
        bins = {}
        for angle, dist, point in zip(rounded_angles, valid_polar[:, 1], valid_points):
            if angle not in bins or bins[angle][1] > dist:
                bins[angle] = (point, dist)

        closest_points = np.array([point for point, _ in bins.values()])
        angles = np.arctan2(closest_points[:,1], closest_points[:,0])
        closest_points = closest_points[angles.argsort()]
        closest_points = closest_points[closest_points[:, 0] > 0.2]

        # Cluster points and calculate shoreline
        if len(closest_points) > 1:
            kmeans = KMeans(n_clusters=min(self.cluster_number, len(closest_points)), random_state=196)
            clusters = kmeans.fit_predict(closest_points)
            
            for cluster_id in np.unique(clusters):
                cluster_points = closest_points[clusters == cluster_id]
                if len(cluster_points) > 1:
                    X = cluster_points[:, 0].reshape(-1, 1)
                    y = cluster_points[:, 1]
                    ransac = RANSACRegressor(random_state=42)
                    ransac.fit(X, y)
                    
                    # Calculate shoreline data
                    m = ransac.estimator_.coef_[0]
                    c = ransac.estimator_.intercept_
                    
                    # Prepare data to send via TCP
                    shoreline_data = [
                        self.long_ship, self.lat_ship, self.heading_angle,
                        m, c  # Add other calculated parameters as needed
                    ]
                    try:
                        # Send data back via TCP
                        packet = self.build_tcp_packet(shoreline_data)
                        # In a real implementation, you would send this to the connected client
                    except Exception as e:
                        self.get_logger().error(f"Error sending TCP data: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    pointcloud_subscriber = PointCloudSubscriber()
    try:
        rclpy.spin(pointcloud_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        pointcloud_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
