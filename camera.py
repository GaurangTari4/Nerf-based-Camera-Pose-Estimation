import numpy as np
import cv2
import os

pattern_size = (7, 7)  
square_size = 0.025  

objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

obj_points = []  # 3D points in real-world coordinates
img_points = []  # 2D points in image plane coordinates

# Specify the directory containing your images
img_dir = 'captured_images'


# Get a list of all image files in the directory
img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]

# Loop over all image files
for img_file in img_files:
    # Load the image
    img_path = os.path.join(img_dir, img_file)
    frame = cv2.imread(img_path)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        obj_points.append(objp)
        img_points.append(corners)
        cv2.drawChessboardCorners(frame, pattern_size, corners, ret)
        cv2.imshow('Calibration', frame)
        cv2.waitKey(500)  # Display each image for 500ms

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Save all the parameters
np.savez('calibration_results.npz', ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
