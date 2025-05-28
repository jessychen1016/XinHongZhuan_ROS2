import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import pclpy
from pclpy import pcl


class LidarPointCloudMerger(Node):
    def __init__(self):
        super().__init__('lidar_pointcloud_merger')

        # Subscribers for the two LiDAR point clouds
        self.subscription_1 = self.create_subscription(
            PointCloud2,
            '/lidar_points_1',
            self.callback_lidar_1,
            10
        )
        self.subscription_2 = self.create_subscription(
            PointCloud2,
            '/lidar_points_4',
            self.callback_lidar_4,
            10
        )

        # Publisher for the merged point cloud
        self.publisher = self.create_publisher(PointCloud2, '/lidar_points', 10)

        # Placeholders for the point clouds
        self.pc1 = None
        self.pc4 = None

    def callback_lidar_1(self, msg):
        """
        Callback function for the first LiDAR point cloud.
        Convert the message to PCL format.
        """
        self.pc1 = self.convert_to_pcl(msg)
        self.publish_merged_pointcloud(msg.header)

    def callback_lidar_4(self, msg):
        """
        Callback function for the second LiDAR point cloud.
        Convert the message to PCL format.
        """
        self.pc4 = self.convert_to_pcl(msg)
        self.publish_merged_pointcloud(msg.header)

    def convert_to_pcl(self, msg):
        """
        Convert a ROS PointCloud2 message to a PCL PointCloud object.
        """
        points = list(pc2.read_points(msg, skip_nans=True, field_names=None))
        pcl_cloud = pcl.PointCloud.PointXYZ()
        pcl_cloud.from_array(points)
        return pcl_cloud

    def publish_merged_pointcloud(self, header):
        """
        Merge the two point clouds using PCL and publish the result.
        """
        if self.pc1 is None or self.pc4 is None:
            return  # Wait until both point clouds are available

        # Merge the point clouds
        merged_cloud = pcl.PointCloud.PointXYZ()
        merged_cloud += self.pc1
        merged_cloud += self.pc4

        # Convert the merged PCL point cloud back to ROS PointCloud2
        merged_msg = self.convert_to_ros(merged_cloud, header)

        # Publish the merged point cloud
        self.publisher.publish(merged_msg)

    def convert_to_ros(self, pcl_cloud, header):
        """
        Convert a PCL PointCloud object to a ROS PointCloud2 message.
        """
        points = pcl_cloud.to_array()
        fields = [
            pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
        ]
        return pc2.create_cloud(header, fields, points)


def main(args=None):
    rclpy.init(args=args)

    lidar_pointcloud_merger = LidarPointCloudMerger()

    try:
        rclpy.spin(lidar_pointcloud_merger)
    except KeyboardInterrupt:
        pass
    finally:
        lidar_pointcloud_merger.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()