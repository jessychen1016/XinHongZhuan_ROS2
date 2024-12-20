import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class FisheyeRectification(Node):
    def __init__(self):
        super().__init__('fisheye_rectification')
        self.subscription = self.create_subscription( Image, '/camera/image1', self.image_callback, 10)
        self.camera_info_sub = self.create_subscription( CameraInfo, '/camera1/camera_info', self.camera_info_callback, 10)
        self.publisher = self.create_publisher( Image, '/camera1/image_rect', 10)
        self.camera_info_publisher = self.create_publisher(CameraInfo, 'camera1/camera_rect_info', 10)
        self.bridge = CvBridge()
        self.intrinsics = None
        self.dist_coeffs = None
        self.new_intrinsics = None
        self.roi = None
        self.h = None
        self.w = None


        self.camera_info_timer = self.create_timer(1.0, self.publish_camera_info)  # Publish camera info at 1 Hz

    def camera_info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            # print(self.dist_coeffs)
            # print(self.intrinsics)
            # Create a new intrinsic matrix for the rectified image
            self.h, self.w = msg.height, msg.width
            self.new_intrinsics, self.roi = cv2.getOptimalNewCameraMatrix(self.intrinsics, self.dist_coeffs, (self.w, self.h), alpha=1.5)  # Alpha=0 for no black borders
            

    def publish_camera_info(self):
        # Create a fake CameraInfo message
        camera_info_msg = CameraInfo()
        camera_info_msg.header.stamp = self.get_clock().now().to_msg()
        camera_info_msg.header.frame_id = "camera_frame"
        camera_info_msg.width = self.w
        camera_info_msg.height = self.h
        # print(self.new_intrinsics[0,0])
        # Set intrinsic parameters for building19
        camera_info_msg.k = [ self.new_intrinsics[0,0], 0., self.new_intrinsics[0,2],
            0.     ,  self.new_intrinsics[1,1], self.new_intrinsics[1,2],
            0.     ,    0.     ,    1.      ]
        camera_info_msg.distortion_model = "plumb_bob"
        camera_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        camera_info_msg.d = [0.,0.,0.,0.,0.]

        # Publish the CameraInfo message
        self.camera_info_publisher.publish(camera_info_msg)

    def image_callback(self, msg):
        if self.intrinsics is None or self.new_intrinsics is None:
            self.get_logger().warn("Camera intrinsics not available yet.")
            return

        # Convert ROS Image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = cv_image.shape[:2]

        # Rectify the image
        map1, map2 = cv2.initUndistortRectifyMap(
            self.intrinsics, self.dist_coeffs, np.eye(3),
            self.new_intrinsics, (w, h), cv2.CV_16SC2
        )
        rectified_image = cv2.remap(cv_image, map1, map2, interpolation=cv2.INTER_LINEAR)
        x, y, w, h = self.roi
        rectified_image = rectified_image[y:y+h, x:x+w]
        # Publish the rectified image
        rectified_msg = self.bridge.cv2_to_imgmsg(rectified_image, encoding='bgr8')
        self.publisher.publish(rectified_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FisheyeRectification()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
