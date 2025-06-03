import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped
import numpy as np

def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    R = np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
        [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y]
    ])
    return R

class Simple1DKalman:
    """
    Minimal 1D Kalman Filter for each axis.
    Models bias as part of the state (x, b): state = [measurement, bias]
    """
    def __init__(self, process_var, bias_var, meas_var, dt):
        # State: [value, bias]
        self.x = np.zeros(2)
        self.P = np.eye(2)
        self.dt = dt
        # Process noise covariance
        self.Q = np.array([[process_var, 0], [0, bias_var]])
        # Measurement noise covariance
        self.R = meas_var

    def predict(self):
        # x_k = F * x_{k-1}, F is identity except for bias
        F = np.array([[1, -self.dt], [0, 1]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        # z: measurement
        H = np.array([1, 0]).reshape((1,2))
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).flatten()
        self.P = (np.eye(2) - K @ H) @ self.P

    def filter(self, z):
        self.predict()
        self.update(z)
        return self.x[0], self.x[1]  # value, bias

class IMUKalmanNode(Node):
    def __init__(self):
        super().__init__('imu_kalman_filter_node')
        # Noise parameters (from your intrinsic)
        self.accel_noise_density = 6.317082483789875e-04
        self.accel_random_walk = 7.641488363632991e-05
        self.gyro_noise_density = 9.383292965348195e-05
        self.gyro_random_walk = 1.0782014013440132e-05

        self.rate = 25.0
        self.dt = 1.0 / self.rate

        # Kalman for each axis (x, y, z): [value, bias]
        # For simplicity, model process_var as noise_density^2, bias_var as random_walk^2, meas_var as noise_density^2
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

        self.subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10)
        self.pub_angular = self.create_publisher(Vector3Stamped, '/imu/angular_velocity_filtered', 10)
        self.pub_velocity = self.create_publisher(Vector3Stamped, '/imu/linear_velocity_integrated', 10)
        self.get_logger().info('IMU Kalman Filter Node started.')

    def imu_callback(self, msg: Imu):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.prev_stamp is not None:
            dt = stamp - self.prev_stamp
            if dt < 1e-4 or dt > 1.0:
                dt = self.dt
        else:
            dt = self.dt
        self.prev_stamp = stamp
        for kf in self.kalman_gyro:
            kf.dt = dt
        for kf in self.kalman_accel:
            kf.dt = dt

        # Get measurements
        gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

        # Kalman filtering for gyroscope (angular velocity)
        filtered_gyro = np.zeros(3)
        gyro_bias = np.zeros(3)
        for i in range(3):
            filtered_gyro[i], gyro_bias[i] = self.kalman_gyro[i].filter(gyro[i])

        # Kalman filtering for accelerometer (linear acceleration)
        filtered_accel = np.zeros(3)
        accel_bias = np.zeros(3)
        for i in range(3):
            filtered_accel[i], accel_bias[i] = self.kalman_accel[i].filter(accel[i])

        # Remove gravity (in body frame) using quaternion orientation
        q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        R = quaternion_to_rotation_matrix(q)
        g_world = np.array([0, 0, 9.80665])
        g_body = R.T @ g_world
        accel_nogravity = filtered_accel - g_body

        # Transform to world frame
        accel_world = R @ accel_nogravity

        # Integrate acceleration to velocity
        self.linear_velocity += accel_world * dt

        # Publish filtered angular velocity
        ang_msg = Vector3Stamped()
        ang_msg.header = msg.header
        ang_msg.vector.x, ang_msg.vector.y, ang_msg.vector.z = filtered_gyro.tolist()
        self.pub_angular.publish(ang_msg)

        # Publish velocity
        vel_msg = Vector3Stamped()
        vel_msg.header = msg.header
        vel_msg.vector.x, vel_msg.vector.y, vel_msg.vector.z = self.linear_velocity.tolist()
        self.pub_velocity.publish(vel_msg)

def main(args=None):
    rclpy.init(args=args)
    node = IMUKalmanNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()