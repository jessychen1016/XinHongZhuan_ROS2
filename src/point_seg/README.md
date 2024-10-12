# for plane seg
cd point_seg/
python3 -m point_seg.seg

# for start usb cam 
cd publish_image/
python3 retrieve_image_usb.py

# for colorization:
cd colorization/
python3 colorize_pointcloud.py 

# for camera claibration
ros2 run camera_calibration cameracalibrator --size 10x7 --square 0.015 --approximate 0.1 image:=/camera/image camera:=/camera_info

