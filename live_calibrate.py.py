#!/usr/bin/env python3
import cv2
import numpy as np
from datetime import datetime
import os

# Default checkerboard and camera settings
PATTERN_COLS = 9     # inner corners along width (columns)
PATTERN_ROWS = 6     # inner corners along height (rows)
SQUARE_SIZE_M = 0.024  # meters per square (e.g., 24mm = 0.024)
CAMERA_ID = 0

WINDOW_NAME = "Camera Calibration"
MAX_EXTRINSICS_TO_PRINT = 10  # how many views' R,t to print (avoid spam)

def create_object_points(cols, rows, square_size):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)
    return objp

def find_corners(gray, pattern_size):
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH |
        cv2.CALIB_CB_NORMALIZE_IMAGE |
        cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return False, None
    # Refine to sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners

def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist):
    total_error = 0.0
    total_points = 0
    per_view_errors = []
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
        per_view_errors.append(float(err))
        total_error += err * len(projected)
        total_points += len(projected)
    mean_error = total_error / total_points if total_points > 0 else float("nan")
    return float(mean_error), per_view_errors

def rvec_tvec_to_R_t(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    return R, t

def print_intrinsics(K, dist, img_size, rms, mean_err):
    w, h = img_size
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    fovx = np.degrees(2 * np.arctan(w / (2 * fx)))
    fovy = np.degrees(2 * np.arctan(h / (2 * fy)))

    print("\n== Intrinsic parameters ==")
    print(f"Image size:            {w} x {h}")
    print("Camera matrix K:")
    print(K)
    print("Distortion coefficients (k1,k2,p1,p2,k3[,k4,k5,k6]):")
    print(dist.ravel())
    print(f"Focal lengths (fx, fy): {fx:.4f}, {fy:.4f}")
    print(f"Principal point (cx, cy): {cx:.4f}, {cy:.4f}")
    print(f"Approx. FOV (deg):     {fovx:.2f} x {fovy:.2f}")
    print(f"RMS reprojection error: {rms:.6f}")
    print(f"Mean per-point error:   {mean_err:.6f}")

def print_extrinsics(rvecs, tvecs, unit_label="m", max_views=10):
    n = len(rvecs)
    print(f"\n== Extrinsics per captured view ({n} views) ==")
    for i in range(min(n, max_views)):
        R, t = rvec_tvec_to_R_t(rvecs[i], tvecs[i])
        angle = np.linalg.norm(rvecs[i])
        axis = (rvecs[i].flatten() / angle) if angle > 1e-12 else np.array([0, 0, 1])
        print(f"\nView {i}:")
        print("R (world->camera):")
        print(R)
        print(f"t (world->camera) [{unit_label}]: {t.ravel()}")
        print(f"Axis-angle: angle={np.degrees(angle):.2f} deg, axis={axis}")
    if n > max_views:
        print(f"... {n - max_views} more views omitted (increase MAX_EXTRINSICS_TO_PRINT to show more)")

def save_calibration(stem, K, dist, rvecs, tvecs, rms, img_size, pattern_size, square_size):
    # Create results directory if it doesn't exist
    results_dir = "live_calibration_results"
    os.makedirs(results_dir, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    npz_path = os.path.join(results_dir, f"{stem}_{ts}.npz")
    yml_path = os.path.join(results_dir, f"{stem}_{ts}.yml")

    rvecs_arr = np.stack([rv.flatten() for rv in rvecs]) if len(rvecs) else np.empty((0, 3))
    tvecs_arr = np.stack([tv.flatten() for tv in tvecs]) if len(tvecs) else np.empty((0, 3))

    np.savez(
        npz_path,
        camera_matrix=K,
        dist_coeffs=dist,
        rvecs=rvecs_arr,
        tvecs=tvecs_arr,
        rms=rms,
        image_width=img_size[0],
        image_height=img_size[1],
        pattern_cols=pattern_size[0],
        pattern_rows=pattern_size[1],
        square_size=square_size,
    )

    fs = cv2.FileStorage(yml_path, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", K)
    fs.write("dist_coeffs", dist)
    fs.write("image_width", int(img_size[0]))
    fs.write("image_height", int(img_size[1]))
    fs.write("pattern_cols", int(pattern_size[0]))
    fs.write("pattern_rows", int(pattern_size[1]))
    fs.write("square_size_m", float(square_size))
    fs.write("rms_error", float(rms))
    fs.release()

    print(f"\nSaved calibration:")
    print(f"- {npz_path}")
    print(f"- {yml_path}")

def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {CAMERA_ID}")

    cols, rows = PATTERN_COLS, PATTERN_ROWS
    pattern_size = (cols, rows)
    objp = create_object_points(cols, rows, SQUARE_SIZE_M)

    objpoints = []
    imgpoints = []
    img_size = None

    cv2.namedWindow(WINDOW_NAME)
    click_state = {"capture": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param["capture"] = True

    cv2.setMouseCallback(WINDOW_NAME, on_mouse, click_state)

    print("Live calibration:")
    print("- Show the checkerboard at different angles, positions, and distances.")
    print("- Left-click on the window or press 'c' to capture a sample when corners are visible.")
    print("- Press 'q' to finish and calibrate.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        found, corners = find_corners(gray, pattern_size)

        vis = frame.copy()
        if found:
            cv2.drawChessboardCorners(vis, pattern_size, corners, True)

        msg = f"Samples: {len(objpoints)}  [Left-click/C=capture, Q=finish]"
        cv2.putText(vis, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if found else (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, vis)

        key = cv2.waitKey(1) & 0xFF
        request_capture = (key == ord('c')) or click_state["capture"]

        if request_capture:
            click_state["capture"] = False
            if found:
                objpoints.append(objp.copy())
                imgpoints.append(corners)
                print(f"Captured sample {len(objpoints)}")
            else:
                print("No checkerboard detected in this frame. Try again.")

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 5:
        print(f"\nNot enough valid captures ({len(objpoints)}). Take at least 10–20 from varied angles.")
        return

    # Calibrate
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    mean_err, per_view = compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist)

    print_intrinsics(K, dist, img_size, rms, mean_err)
    print_extrinsics(rvecs, tvecs, unit_label="m", max_views=MAX_EXTRINSICS_TO_PRINT)

    save_calibration("calibration", K, dist, rvecs, tvecs, rms, img_size, pattern_size, SQUARE_SIZE_M)

if __name__ == "__main__":
    main()