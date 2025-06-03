import socket
import struct
import rclpy
import time
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped
from imu_tools.imu_filter import Simple1DKalman, quaternion_to_rotation_matrix

def build_packet(data_points, device_id=123, cmd_type=1, start_addr=0, packet_num=1):
    m_send_dataNum = len(data_points)
    total_bytes = 7 + 4 * m_send_dataNum  # header + data as per protocol

    # Build header (13 bytes)
    Head = bytearray(13)
    Head[0] = 1
    Head[1] = packet_num  # communication number
    Head[2] = 0
    Head[3] = 0  # TCP protocol type
    Head[4] = total_bytes & 0xFF
    Head[5] = (total_bytes >> 8) & 0xFF
    Head[6] = device_id  # device address
    Head[7] = cmd_type   # command type
    Head[8] = 0
    Head[9] = (start_addr) & 0xFF
    Head[10] = m_send_dataNum & 0xFF
    Head[11] = (m_send_dataNum >> 8) & 0xFF
    Head[12] = 0  # reserved

    # Pack all data points as float32 (little-endian)
    data_section = b''.join(struct.pack('<f', float(val)) for val in data_points)
    return Head + data_section

class IMUTCPClient(Node):
    def __init__(self):
        super().__init__('imu_tcp_client')
        self.subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10)
        
        # IMU processing setup
        self.rate = 25.0
        self.dt = 1.0 / self.rate
        self.gyro_noise_density = 9.383292965348195e-05
        self.gyro_random_walk = 1.0782014013440132e-05
        self.accel_noise_density = 6.317082483789875e-04
        self.accel_random_walk = 7.641488363632991e-05
        
        # Initialize Kalman filters
        self.kalman_gyro = [Simple1DKalman(
            process_var=self.gyro_noise_density ** 2,
            bias_var=self.gyro_random_walk ** 2,
            meas_var=self.gyro_noise_density ** 2,
            dt=self.dt
        ) for _ in range(3)]
        self.kalman_accel = [Simple1DKalman(
            process_var=self.accel_noise_density ** 2,
            bias_var=self.accel_random_walk ** 2,
            meas_var=self.accel_noise_density ** 2,
            dt=self.dt
        ) for _ in range(3)]
        
        self.linear_velocity = np.zeros(3)
        self.prev_stamp = None
        
        # TCP client setup
        self.host = '10.1.5.7'  # Server IP
        self.port = 7546        # Port 9 as requested
        self.socket = None
        self.packet_num = 1
        self.connect_to_server()

    def connect_to_server(self):
        while rclpy.ok():
            try:
                self.get_logger().info('Attempting to connect to TCP server...')
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
                self.get_logger().info('Successfully connected to TCP server')
                return
            except (socket.error, ConnectionRefusedError) as e:
                self.get_logger().error(f'Connection failed: {str(e)}. Retrying in 5 seconds...')
                time.sleep(5)

    def imu_callback(self, msg):
        # Time handling
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.prev_stamp is not None:
            dt = stamp - self.prev_stamp
            if dt < 1e-4 or dt > 1.0:
                dt = self.dt
        else:
            dt = self.dt
        self.prev_stamp = stamp
        
        # Update Kalman filter time steps
        for kf in self.kalman_gyro:
            kf.dt = dt
        for kf in self.kalman_accel:
            kf.dt = dt

        # Get raw measurements
        gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

        # Apply Kalman filtering
        filtered_gyro = np.zeros(3)
        for i in range(3):
            filtered_gyro[i], _ = self.kalman_gyro[i].filter(gyro[i])
        
        filtered_accel = np.zeros(3)
        for i in range(3):
            filtered_accel[i], _ = self.kalman_accel[i].filter(accel[i])

        # Remove gravity and integrate velocity
        q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        R = quaternion_to_rotation_matrix(q)
        g_world = np.array([0, 0, 9.80665])
        g_body = R.T @ g_world
        accel_nogravity = filtered_accel - g_body
        accel_world = R @ accel_nogravity
        self.linear_velocity += accel_world * dt
        
        # Create data array with filtered gyro and linear velocity
        data_points = [0.0] * 6  # 3 for gyro, 3 for velocity  rot_x, rot_y, rot_z, vel_x, vel_y, vel_z
        data_points[0:3] = filtered_gyro.tolist()
        data_points[3:6] = self.linear_velocity.tolist() 
        
        # Build and send packet
        packet = build_packet(data_points, device_id=123, packet_num=self.packet_num)
        try:
            if self.socket is None:
                self.connect_to_server()
            self.socket.sendall(packet)
            time.sleep(1)
            self.get_logger().info(f'Sent filtered gyro: {filtered_gyro.tolist()}, velocity: {self.linear_velocity.tolist()}')
        except (socket.error, ConnectionResetError) as e:
            self.get_logger().error(f'Send failed: {str(e)}. Reconnecting...')
            self.socket = None
            self.connect_to_server()
            return
        
        self.packet_num = (self.packet_num + 1) % 256  # Wrap at 255

def main(args=None):
    rclpy.init(args=args)
    client = IMUTCPClient()
    try:
        rclpy.spin(client)
    except KeyboardInterrupt:
        client.get_logger().info('Shutting down IMU TCP client')
    finally:
        client.socket.close()
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
