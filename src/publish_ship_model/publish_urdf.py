import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import trimesh
import numpy as np
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R




class ShipVisualizer(Node):
    def __init__(self):
        super().__init__('ship_visualizer')

        # Publisher for PointCloud2 to visualize the OBJ in RViz
        self.publisher = self.create_publisher(PointCloud2, '/ship_pointcloud', 1)

        # TF broadcaster for the transform
        self.tf_broadcaster = TransformBroadcaster(self)

        # Set timer for periodic publishing
        self.timer = self.create_timer(1.0, self.publish_transform_and_pointcloud)

        # Ship Model Position (center of the 'hesai_128' frame)
        self.model_position = [-4.0, -5.0, -6.0]  

        # Load the OBJ file using trimesh
        self.obj_filename = './ship_model/boat_v2_L2.123cb2c2d48f-7519-476c-8274-b5bcd578c751/12219_boat_v2_L2.obj'
        self.scene = trimesh.load_mesh(self.obj_filename)

        if isinstance(self.scene, trimesh.Scene):
            self.mesh = self.scene.dump()[0]  # Extract the first mesh from the scene
        else:
            self.mesh = self.scene 

        # Sample points from the OBJ model
        self.points = self.mesh.sample(20000)  # Sampling 1000 points from the model surface
        self.scale_factor = 0.02  # Scale factor to make the model smaller (change as needed)
        self.points *= self.scale_factor  # Apply scaling to the point clo1

    def publish_transform_and_pointcloud(self):
        # Publish the transform from hesai_128 to the ship model (e.g., center of frame)
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "hesai_at128"  # Reference frame
        transform.child_frame_id = "ship_link"  # The child frame ID for the ship model
        # XYZ Euler angles in radians rpy
        euler_angles = [0, 0, -np.pi/2]  # Replace with your values

        # Create a Rotation object from Euler angles
        rotation = R.from_euler('xyz', euler_angles)

        # Convert to quaternion
        quaternion = rotation.as_quat()


        # Set the transform (e.g., place ship at the origin of the hesai_128 frame)
        transform.transform.translation.x = self.model_position[0]
        transform.transform.translation.y = self.model_position[1]
        transform.transform.translation.z = self.model_position[2]
        transform.transform.rotation.x = quaternion[0]
        transform.transform.rotation.y = quaternion[1]
        transform.transform.rotation.z = quaternion[2]
        transform.transform.rotation.w = quaternion[3]

        # Publish the transform
        self.tf_broadcaster.sendTransform(transform)

        # Visualize the ship model as a PointCloud
        self.publish_ship_pointcloud()

    def publish_ship_pointcloud(self):
        # Create PointCloud2 message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'ship_link'  # Frame ID set to the ship's link

        # Convert the points to the PointCloud2 format
        cloud_msg = pc2.create_cloud_xyz32(header, self.points.astype(np.float32))

        # Publish the PointCloud2 message
        self.publisher.publish(cloud_msg)


def main(args=None):
    rclpy.init(args=args)
    ship_visualizer = ShipVisualizer()

    try:
        rclpy.spin(ship_visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        ship_visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
