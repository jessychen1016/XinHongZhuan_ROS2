import cv2
import numpy as np

# Load the distorted image
img = cv2.imread('/home/jessy/Calibration/xinhongzhuan/camera1_fisheye_20241201_filtered/left-0012.png')  # Replace with your image path
h, w = img.shape[:2]

# Camera parameters (obtained from calibration)
K = np.array([[553.3816, 0, 1287.174209],  # Example intrinsic matrix
              [0, 552.1519005, 975.1275],
              [0, 0, 1]])
D = np.array([-0.27474227, 0.068477488, -0.68332e-04, -0.0014, -0.00715405])  # Example distortion coefficients

# Compute the optimal new camera matrix
new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)  # Alpha=0 for no black borders

# Compute rectification maps
map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_16SC2)

# Rectify the image
rectified_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

# Crop the image (optional, if alpha=0)
x, y, w, h = roi
rectified_img = rectified_img[y:y+h, x:x+w]
# crop the image from the center with 50% of the original size
# rectified_img = rectified_img[int(h/4):int(3*h/4), int(w/4):int(3*w/4)]
# resize rectified_img to 50%
rectified_img = cv2.resize(rectified_img, (0, 0), fx=0.5, fy=0.5)
print(new_K)

# Display the original and rectified images
cv2.imshow('Original Image', img)
cv2.imshow('Rectified Image', rectified_img)

# Save the rectified image (optional)
cv2.imwrite('rectified_image.jpg', rectified_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
