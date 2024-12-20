import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import cv2
from cv_bridge import CvBridge
from threading import Thread, Event
import zmq
import base64
import numpy as np
from utils.gstreamer_camera import GStreamerCamera  # Import your GStreamerCamera class

class MultiTopicPublisher(Node):
    def __init__(self):
        super().__init__('multi_topic_publisher')

        self.image_height = 1080
        self.image_width = 1920
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.PUB)
        self.socket.bind("tcp://*:5555")  # Binding to port 5555 for sending images

        self.receive_context = zmq.Context()
        self.receive_socket = self.receive_context.socket(zmq.SUB)
        self.receive_socket.connect("tcp://localhost:5556")  # Binding to port 5556 for receiving images
        self.receive_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages

        # Create publishers for multiple topics
        self.image_publisher = self.create_publisher(Image, 'camera/image', 1)
        self.semantic_publisher = self.create_publisher(Image, 'camera/semantic_image', 1)
        self.camera_info_publisher = self.create_publisher(CameraInfo, 'camera/camera_info', 10)

        # Initialize the GStreamer camera
        self.gstreamer_camera = GStreamerCamera(device='/dev/video2', width=self.image_width, height=self.image_height, show_img=False)  # Adjust the device as needed
        self.gstreamer_camera.start()  # Start the GStreamer pipeline

        # Create a CvBridge to convert between OpenCV images and ROS Image messages
        self.br = CvBridge()

        # Create a shared variable to hold the latest frame
        self.latest_frame = None
        self.frame_event = Event()  # Event to signal when a new frame is available

        # Start a separate thread for retrieving frames from the GStreamer camera
        self.frame_thread = Thread(target=self.retrieve_frames)
        self.frame_thread.start()

        # Create timers for publishing messages at different frequencies
        self.image_timer = self.create_timer(0.05, self.publish_image)  # Publish images at 10 Hz
        self.camera_info_timer = self.create_timer(1.0, self.publish_camera_info)  # Publish camera info at 1 Hz

    def retrieve_frames(self):
        """Continuously retrieve frames from the GStreamer camera."""
        while rclpy.ok():  # Continue until ROS is shutdown
            frame = self.gstreamer_camera.get_frame()
            if frame is not None:
                self.latest_frame = frame
                self.frame_event.set()  # Signal that a new frame is available
            else:
                self.frame_event.clear()  # Clear the event if no frame is available

    def publish_image(self):
        # Check if a new frame is available
        if self.frame_event.is_set() and self.latest_frame is not None:
            # Send the image through ZMQ to other scripts that do not support ROS2
            _, buffer = cv2.imencode('.jpg', self.latest_frame)
            img_as_text = base64.b64encode(buffer).decode('utf-8')
            self.socket.send_string(f"{img_as_text}")

            # Receive segmented images
            self.receive_socket.setsockopt(zmq.RCVTIMEO, 1000)
            try:
                img_as_text_receive = self.receive_socket.recv_string()
                # Decode the base64 string back to binary
                img_data_receive = base64.b64decode(img_as_text_receive)
                # Convert the binary data to a NumPy array and decode the JPEG
                np_img_receive = np.frombuffer(img_data_receive, dtype=np.uint8)
                img_segment = cv2.imdecode(np_img_receive, cv2.IMREAD_COLOR)
            except zmq.Again:
                print("No semantic image received, skipping...")
                img_segment = np.random.randint(0, 255, (self.image_height, self.image_width, 3), dtype=np.uint8)

            # Send original image
            image_message = self.br.cv2_to_imgmsg(self.latest_frame, encoding="bgr8")
            image_message.header.stamp = self.get_clock().now().to_msg()
            image_message.header.frame_id = "camera_frame"
            self.image_publisher.publish(image_message)

            # Send segmented image
            image_message_segment = self.br.cv2_to_imgmsg(img_segment, encoding="bgr8")
            image_message_segment.header.stamp = self.get_clock().now().to_msg()
            image_message_segment.header.frame_id = "camera_frame"
            self.semantic_publisher.publish(image_message_segment)

    def publish_camera_info(self):
        # Create a fake CameraInfo message
        camera_info_msg = CameraInfo()
        camera_info_msg.header.stamp = self.get_clock().now().to_msg()
        camera_info_msg.header.frame_id = "camera_frame"
        camera_info_msg.width = self.image_width
        camera_info_msg.height = self.image_height
        
        # Set intrinsic parameters for 1080p
        camera_info_msg.k = [1043.02215,    0.     ,  963.4692 ,
                              0.     , 1043.30157,  528.77189,
                              0.     ,    0.     ,    1.     ]
        camera_info_msg.d = [0.153638, -0.143077, 0.003250, -0.001801, 0.000000]
        camera_info_msg.distortion_model = "plumb_bob"
        camera_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        camera_info_msg.p = [1075.50488,    0.     ,  958.35386,    0.     ,
                              0.     , 1085.85059,  531.53889,    0.     ,
                              0.     ,    0.     ,    1.     ,    0.     ]

        # Publish the CameraInfo message
        self.camera_info_publisher.publish(camera_info_msg)

    def destroy_node(self):
        # Stop the frame retrieval thread
        self.gstreamer_camera.stop()  # Stop the GStreamer camera
        self.frame_thread.join()  # Wait for the thread to finish
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    # Create the multi-topic publisher node
    multi_topic_publisher = MultiTopicPublisher()

    # Spin the node so it keeps running
    rclpy.spin(multi_topic_publisher)

    # Clean up on exit
    multi_topic_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
