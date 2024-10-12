import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import cv2
from cv_bridge import CvBridge
from threading import Thread, Lock
import zmq
import base64
import numpy as np

class MultiTopicPublisher(Node):
    def __init__(self):
        super().__init__('multi_topic_publisher')

        self.image_height = 480
        self.image_width = 640
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.PUB)
        self.socket.bind("tcp://*:5555")  # Binding to port 5555 for sending images

        self.receive_context = zmq.Context()
        self.receive_socket = self.receive_context.socket(zmq.SUB)
        self.receive_socket.connect("tcp://localhost:5556")  # Binding to port 5555 for sending images
        self.receive_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages\

        # Create publishers for multiple topics
        self.image_publisher = self.create_publisher(Image, 'camera/image', 10)
        self.semantic_publisher = self.create_publisher(Image, 'camera/semantic_image', 10)
        self.camera_info_publisher = self.create_publisher(CameraInfo, 'camera/camera_info', 10)

        # Initialize OpenCV VideoCapture for the USB camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 1)

    
        self.lock = Lock()

        # Create a separate thread for capturing frames
        self.frame = None
        self.capture_thread = Thread(target=self.capture_frames)
        self.capture_thread.start()

        # Create timers for publishing messages at different frequencies
        self.image_timer = self.create_timer(0.03, self.publish_image)        # Publish images at 10 Hz
        self.camera_info_timer = self.create_timer(0.1, self.publish_camera_info)  # Publish camera info at 1 Hz

        # Create a CvBridge to convert between OpenCV images and ROS Image messages
        self.br = CvBridge()

    def capture_frames(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            # equilize the frame, it is tooo bright
            if ret:
                with self.lock:
                    self.frame = frame
            # cv2.imshow('frame', frame)
            # cv2.waitKey(1)

    def publish_image(self):
        with self.lock:
            if self.frame is not None:
                

                '''send the image through zmq to other scritps that does not support ROS2'''
                _, buffer = cv2.imencode('.jpg', self.frame)
                img_as_text = base64.b64encode(buffer).decode('utf-8')
                self.socket.send_string(f"{img_as_text}")
                
                '''recieve segmentated images'''
                self.receive_socket.setsockopt(zmq.RCVTIMEO, 1000)
                try:
                    img_as_text_receive = self.receive_socket.recv_string()
                    # Decode the base64 string back to binary
                    img_data_receive = base64.b64decode(img_as_text_receive)
                    # Convert the binary data to a NumPy array and decode the JPEG
                    np_img_receive = np.frombuffer(img_data_receive, dtype=np.uint8)
                    img_segment = cv2.imdecode(np_img_receive, cv2.IMREAD_COLOR)
                except zmq.Again:
                    print("No image received, skipping...")
                    img_segment = np.random.randint(0, 255, (self.image_height, self.image_width, 3), dtype=np.uint8)

                '''send original image'''
                # Convert the OpenCV image (BGR) to a ROS Image message
                image_message = self.br.cv2_to_imgmsg(self.frame, encoding="bgr8")
                image_message.header.stamp = self.get_clock().now().to_msg()
                image_message.header.frame_id = "camera_frame"
                # Publish the image
                self.image_publisher.publish(image_message)

                '''send segmented image'''
                image_message_segment = self.br.cv2_to_imgmsg(img_segment, encoding="bgr8")
                image_message_segment.header.stamp = self.get_clock().now().to_msg()
                image_message_segment.header.frame_id = "camera_frame"
                # Publish the image
                self.semantic_publisher.publish(image_message_segment)


    def publish_camera_info(self):
        # Create a fake CameraInfo message
        camera_info_msg = CameraInfo()
        camera_info_msg.header.stamp = self.get_clock().now().to_msg()
        camera_info_msg.header.frame_id = "camera_frame"
        camera_info_msg.width = 640
        camera_info_msg.height = 480
        camera_info_msg.k = [461.93834,   0.     , 318.05872,
           0.     , 464.33664, 231.33221,
           0.     ,   0.     ,   1.     ]
        camera_info_msg.d = [0.183058, -0.240628, -0.001085, -0.002892, 0.000000]
        camera_info_msg.distortion_model = "plumb_bob"
        camera_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        camera_info_msg.p = [476.64185,   0.     , 315.65162,   0.     ,
           0.     , 480.2319 , 230.46985,   0.     ,
           0.     ,   0.     ,   1.     ,   0.     ]

        # Publish the CameraInfo message
        self.camera_info_publisher.publish(camera_info_msg)

def main(args=None):
    rclpy.init(args=args)

    # Create the multi-topic publisher node
    multi_topic_publisher = MultiTopicPublisher()

    # Spin the node so it keeps running
    rclpy.spin(multi_topic_publisher)

    # Clean up on exit
    multi_topic_publisher.cap.release()
    multi_topic_publisher.capture_thread.join()
    multi_topic_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
