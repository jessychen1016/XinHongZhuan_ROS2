import socket
import struct
import rclpy
import time
from rclpy.node import Node
from sensor_msgs.msg import Imu
import math

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
        # Get z-axis angular velocity directly
        z_velocity = msg.angular_velocity.z
        x_acc = msg.linear_acceleration.x
        y_acc = msg.linear_acceleration.y
        # print(f'Received angular velocity z: {z_velocity:.2f} rad/s')
        
        # Create data array with z velocity in position #9 (as per test_tcp_send.py example)
        data_points = [0.0] * 3  # Initialize with zeros
        data_points[0] = z_velocity
        data_points[1] = x_acc
        data_points[2] = y_acc
        
        # Build and send packet
        packet = build_packet(data_points, device_id=123, packet_num=self.packet_num)
        try:
            if self.socket is None:
                self.connect_to_server()
            self.socket.sendall(packet)
            time.sleep(1)
            self.get_logger().info(f'Sent angular velocity z: {z_velocity:.2f} rad/s')
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
