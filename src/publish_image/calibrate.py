import cv2
import numpy as np
import glob
import os

# Define checkerboard dimensions (inner corners)
CHECKERBOARD = (7,10)  # Replace with your checkerboard dimensions
square_size = 20  # Replace with your checkerboard square size in mm

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

# Prepare object points (3D points in real-world coordinates)
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1, 2)
objp *= square_size  # Scale by square size

# Arrays to store 3D object points and 2D image points
objpoints = []  # 3D points
imgpoints = []  # 2D points

# Path to folder containing images
image_folder = '/home/jessy/Calibration/xinhongzhuan/camera1_fisheye_20241201_filtered/'  # Replace with your folder path

# Load all images in the folder
images = glob.glob(os.path.join(image_folder, '*.png'))  # Adjust file extension if needed

if len(images) == 0:
    print("No images found in the folder!")
    exit()

print(f"Found {len(images)} images for calibration.")

# Process each image
for img_file in images:
    img = cv2.imread(img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        corners_refined = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
        imgpoints.append(corners_refined)

        # Visualize corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners_refined, ret)
        cv2.imshow('Checkerboard Detection', img)
        cv2.waitKey(100)  # Display for 100 ms
    else:
        print(f"Checkerboard not detected in image: {img_file}")

cv2.destroyAllWindows()

# Perform fisheye calibration
K = np.zeros((3, 3))  # Intrinsic matrix
D = np.zeros((4, 1))  # Distortion coefficients
rvecs, tvecs = [], []  # Rotation and translation vectors

print("Starting fisheye calibration...")
ret, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    K,
    D,
    rvecs,
    tvecs,
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
)

if ret:
    print("Calibration successful!")
    print("Intrinsic Matrix (K):\n", K)
    print("Distortion Coefficients (D):\n", D)
    
    # Save calibration results
    np.savez('fisheye_calib_data.npz', K=K, D=D, rvecs=rvecs, tvecs=tvecs)
    print("Calibration data saved to 'fisheye_calib_data.npz'.")
else:
    print("Calibration failed.")
