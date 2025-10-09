import numpy as np
import cv2
import glob

# Define the dimensions of the checkerboard
CHECKERBOARD = (9, 6)   # (number of inner corners per row and column)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Create object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

# Load images (support png/jpg/jpeg)
images = sorted(glob.glob('images/*.png') + glob.glob('images/*.jpg') + glob.glob('images/*.jpeg'))

if len(images) == 0:
    raise SystemExit("No images found in the 'images' directory. Put calibration images there (png/jpg/jpeg).")

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Warning: could not read image '{fname}', skipping.")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp.copy())
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and (optionally) display the corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    else:
        print(f"Checkerboard not detected in '{fname}'")

cv2.destroyAllWindows()

if len(objpoints) == 0 or len(imgpoints) == 0:
    raise SystemExit("No checkerboard corners were detected in any image. Check your images and CHECKERBOARD size.")

# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(f"Calibration RMS error: {ret}")
print("\nIntrinsic Matrix:")
print(mtx)
print("\nDistortion Coefficients:")
print(dist.ravel())

# Save parameters for later use
npz_name = "camera_parameters.npz"
np.savez(npz_name, intrinsic_matrix=mtx, distortion=dist, rvecs=rvecs, tvecs=tvecs)
print(f"Saved camera parameters to {npz_name}")
