import cv2
import numpy as np

# Define the dimensions of checkerboard
CHECKERBOARD = (9, 6)  # inner corners (columns, rows)
square_size = 25  # millimeters (adjust if needed)

# Prepare 3D object points (0,0,0), (1,0,0), ... multiplied by square size
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# Create 3D axis for visualization (length = 100mm)
axis = np.float32([[100,0,0], [0,100,0], [0,0,-100]]).reshape(-1,3)

# Open camera
cap = cv2.VideoCapture(0)

# You can use pre-calculated calibration values, or estimate fresh
# For simplicity, let's use a dummy intrinsic matrix (approx for a 640x480 webcam)
# Ideally, replace this with real calibration values
focal_length = 800
center = (320, 240)
camera_matrix = np.array([[focal_length, 0, center[0]],
                          [0, focal_length, center[1]],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))  # assume no lens distortion

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        # Refine corner positions
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        # Solve for pose (rotation and translation vectors)
        success, rvecs, tvecs = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)

        # Project 3D axis points onto the image
        imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, dist_coeffs)

        # Draw corners
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)

        # Draw 3D axes
        corner = tuple(corners2[0].ravel().astype(int))
        frame = cv2.line(frame, corner, tuple(imgpts[0].ravel().astype(int)), (0,0,255), 5)  # X-axis (red)
        frame = cv2.line(frame, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 5)  # Y-axis (green)
        frame = cv2.line(frame, corner, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 5)  # Z-axis (blue)

    cv2.imshow('Pose Estimation - Checkerboard', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
