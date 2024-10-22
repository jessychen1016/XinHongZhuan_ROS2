import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, PointField
from sensor_msgs_py.point_cloud2 import read_points, create_cloud
import numpy as np
import ros2_numpy
import cv2
from cv_bridge import CvBridge

class LidarColorizerNode(Node):
    def __init__(self):
        super().__init__('lidar_colorizer_node')
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/lidar_points', self.pointcloud_callback, 10)
        self.image_sub = self.create_subscription(
            Image, '/camera/image', self.image_callback, 10)
        self.semantic_image_sub = self.create_subscription(
            Image, '/camera/semantic_image', self.semantic_image_callback, 10)
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, '/lidar/colorized_points', 10)
        self.semantic_pointcloud_pub = self.create_publisher(
            PointCloud2, '/lidar/semantic_points', 10)
        self.rgbd_pub = self.create_publisher(
            Image, '/camera/rgbd_image', 30)
        self.depth_pub = self.create_publisher(
            Image, '/camera/depth_image', 30)
        
        self.bridge = CvBridge()
        self.camera_image = None
        self.camera_image_rgb = None

        # # Camera intrinsics (provided intrinsics) 480p
        # self.K = np.array([[461.93834,   0.     , 318.05872], 
        #                    [0.     , 464.33664, 231.33221], 
        #                    [0.     ,   0.     ,   1.     ]])

        # # Distortion coefficients (provided distortion coefficients) 480p
        # self.dist_coeffs = np.array([0.183058, -0.240628, -0.001085, -0.002892, 0.000000])

        # Camera intrinsics (provided intrinsics) 1080p
        self.K = np.array([[1043.02215,    0.     ,  963.4692 ], 
                           [0.     , 1043.30157,  528.77189], 
                           [0.     ,   0.     ,   1.     ]])

        # Distortion coefficients (provided distortion coefficients) 1080p
        self.dist_coeffs = np.array([0.153638, -0.143077, 0.003250, -0.001801, 0.000000])



        # Transformation from LiDAR to Camera (provided transformation)
        self.T_camera_lidar = self.create_transformation_matrix(
            translation=[-0.30219897627830505, 0.06125678867101669, 0.04833770543336868],
            quaternion=[-0.5033807312037768, 0.47727347938784354, -0.5103808193751581, 0.5082610304402359]
        )
        self.T_lidar_camera = np.linalg.inv(self.T_camera_lidar)

    def image_callback(self, msg):
        
        # self.camera_image = cv2.bitwise_not(self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8'))
        self.camera_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # convert to rgb8
        self.camera_image_rgb = cv2.cvtColor(self.camera_image, cv2.COLOR_BGR2RGB)

    def semantic_image_callback(self, msg):
        
        # self.camera_image = cv2.bitwise_not(self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8'))
        self.semantic_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # convert to rgb8
        self.semantic_image_rgb = cv2.cvtColor(self.semantic_image, cv2.COLOR_BGR2RGB)


    def pointcloud_callback(self, msg):
        if self.camera_image_rgb is None:
            return
        
        # Extract the point cloud data
        pc = ros2_numpy.numpify(msg)
        points = np.array(pc['xyz'], dtype=np.float64)
        # np.array(self.points, dtype=np.float64)
        # points = np.array([p[:3] for p in read_points(msg, field_names=("x", "y", "z"), skip_nans=True)])

        # Convert to homogeneous coordinates
        num_points = points.shape[0]
        homogeneous_points = np.hstack((points, np.ones((num_points, 1))))

        # Apply the transformation (from LiDAR to camera)
        points_camera = self.T_lidar_camera @ homogeneous_points.T

        # Project points onto the image plane
        uvs = self.K @ points_camera[:3, :]
        uvs /= points_camera[2, :]  # Normalize by Z (depth)

        # Distort the image points using the distortion coefficients
        uvs_undistorted = self.undistort_points(uvs[:2, :])

        # Filter points within camera bounds and in front of the camera
        valid_mask = (uvs_undistorted[0, :] >= 0) & (uvs_undistorted[0, :] < self.camera_image.shape[1]) & \
                     (uvs_undistorted[1, :] >= 0) & (uvs_undistorted[1, :] < self.camera_image.shape[0]) & \
                     (points_camera[2, :] > 0)
        valid_uvs = uvs_undistorted[:, valid_mask]
        valid_points = points_camera[:, valid_mask]

        # Generate the depth map from the Z coordinates
        depth_map = np.zeros((self.camera_image_rgb.shape[0], self.camera_image_rgb.shape[1]), dtype=np.float32)
        depth_map[valid_uvs[1, :].astype(int), valid_uvs[0, :].astype(int)] = valid_points[2, :]
        print(depth_map.max())
        print(depth_map.min())
        print(depth_map.shape)
        # Normalize depth map for visualization (optional)
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Create a 4-channel image (RGB + depth)
        rgbd_image = np.dstack((self.camera_image_rgb, depth_map_normalized))
        # Convert to ROS Image message and publish
        rgbd_msg = self.bridge.cv2_to_imgmsg(rgbd_image, encoding='rgba8')
        # Publish depth map as a separate grayscale image
        depth_msg = self.bridge.cv2_to_imgmsg(depth_map, encoding='32FC1')  # Publish as 32-bit float image
        


        # Assign colors to the valid points
        colors = 255 - self.camera_image_rgb[valid_uvs[1, :].astype(int), valid_uvs[0, :].astype(int)]
        semantic_colors = 255 - self.semantic_image_rgb[valid_uvs[1, :].astype(int), valid_uvs[0, :].astype(int)]
        # print(colors.dtype)
        # for i in range(colors.shape[0]):
        #     if colors[i,0] == 0 and colors[i,1] == 0 and colors[i,2] == 0:
        #         colors[i,:] = [254,254,254]
        
        
        colorized_points = np.hstack((valid_points[:3, :].T, colors))
        semantic_points = np.hstack((valid_points[:3, :].T, semantic_colors))
        points_ready = self.transform_back(colorized_points)
        semantic_points_ready = self.transform_back(semantic_points)
        # Publish the colorized point cloud
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1),
        ]
        header = msg.header
        pc_data = create_cloud(header, fields, points_ready)
        semantic_pc_data = create_cloud(header, fields, semantic_points_ready)
        depth_msg.header.stamp = self.get_clock().now().to_msg()
        depth_msg.header.frame_id = "camera_frame"
        
        self.pointcloud_pub.publish(pc_data)
        self.semantic_pointcloud_pub.publish(semantic_pc_data)
        self.rgbd_pub.publish(rgbd_msg)
        self.depth_pub.publish(depth_msg)

    def create_transformation_matrix(self, translation, quaternion):
        """ Create a 4x4 transformation matrix from translation and quaternion """
        # Create rotation matrix from quaternion
        rotation = self.quaternion_to_rotation_matrix(quaternion)

        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = translation
        return T

    def quaternion_to_rotation_matrix(self, q):
        """ Convert quaternion [qx, qy, qz, qw] to a 3x3 rotation matrix """
        qx, qy, qz, qw = q
        return np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ])

    def undistort_points(self, uvs):
        """ Undistort 2D points using the distortion coefficients and camera intrinsics """
        uvs = uvs.T.reshape(-1, 1, 2)  # Reshape to (N, 1, 2) format required by OpenCV
        uvs_undistorted = cv2.undistortPoints(uvs, self.K, self.dist_coeffs, None, self.K)
        return uvs_undistorted.squeeze().T  # Return in shape (2, N)
    
    def transform_back(self, point_cloud):
        positions = point_cloud[:, :3]  # Nx3 for (x, y, z)
        colors = point_cloud[:, 3:]      # Nx3 for (r, g, b)

        # Homogenize the positions (add a column of ones)
        ones = np.ones((positions.shape[0], 1))
        homogeneous_positions = np.hstack((positions, ones))  # Nx4

        # Apply the transformation
        transformed_positions = homogeneous_positions @ self.T_camera_lidar.T  # Nx4

        # # Extract the transformed x, y, z
        # transformed_x = transformed_positions[:, 0]
        # transformed_y = transformed_positions[:, 1]
        # transformed_z = transformed_positions[:, 2]

        # Combine transformed positions with the original colors
        transformed_point_cloud = np.hstack((transformed_positions[:, :3], colors))
        return transformed_point_cloud

def main(args=None):
    rclpy.init(args=args)
    node = LidarColorizerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
