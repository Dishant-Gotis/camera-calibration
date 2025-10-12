#!/usr/bin/env python3
import cv2
import numpy as np
from datetime import datetime
import os
import glob

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

def create_image_with_data_overlay(image, sample_num, total_samples, corners=None, pattern_size=None):
    """Create an image with calibration data overlay on the left side"""
    h, w = image.shape[:2]
    
    # Create overlay area on the left (200px wide)
    overlay_w = 200
    overlay = np.zeros((h, overlay_w, 3), dtype=np.uint8)
    overlay.fill(30)  # Dark background
    
    # Combine overlay with original image
    result = np.hstack([overlay, image])
    
    # Text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_color = (255, 255, 255)  # White
    line_thickness = 1
    line_spacing = 20
    y_start = 20
    
    # Sample info
    cv2.putText(result, f"Sample: {sample_num}", (5, y_start), font, font_scale, font_color, line_thickness)
    cv2.putText(result, f"Total: {total_samples}", (5, y_start + line_spacing), font, font_scale, font_color, line_thickness)
    
    # Timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")
    cv2.putText(result, f"Time: {timestamp}", (5, y_start + 2*line_spacing), font, font_scale, font_color, line_thickness)
    
    # Pattern detection status
    status = "DETECTED" if corners is not None else "NOT DETECTED"
    status_color = (0, 255, 0) if corners is not None else (0, 0, 255)
    cv2.putText(result, f"Pattern: {status}", (5, y_start + 3*line_spacing), font, font_scale, status_color, line_thickness)
    
    # Pattern size info
    if pattern_size:
        cv2.putText(result, f"Size: {pattern_size[0]}x{pattern_size[1]}", (5, y_start + 4*line_spacing), font, font_scale, font_color, line_thickness)
    
    return result

def save_calibration(stem, K, dist, rvecs, tvecs, rms, img_size, pattern_size, square_size, captured_images=None):
    # Create results directory if it doesn't exist
    results_dir = "live_calibration_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create images directory if it doesn't exist
    images_dir = "live_calibrate_p_img"
    os.makedirs(images_dir, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create timestamped subfolder for images
    session_images_dir = os.path.join(images_dir, f"session_{ts}")
    os.makedirs(session_images_dir, exist_ok=True)
    
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

    # Save final calibration data overlay image
    if captured_images and len(captured_images) > 0:
        # Create final summary image with complete calibration data
        final_overlay = create_final_calibration_overlay(captured_images[0], K, dist, rms, img_size, pattern_size, square_size)
        final_image_path = os.path.join(session_images_dir, f"final_calibration_{ts}.jpg")
        cv2.imwrite(final_image_path, final_overlay)
        print(f"- Final calibration image: {final_image_path}")
    
    print(f"\nSaved calibration:")
    print(f"- {npz_path}")
    print(f"- {yml_path}")
    print(f"- Images folder: {session_images_dir}")

def create_final_calibration_overlay(image, K, dist, rms, img_size, pattern_size, square_size):
    """Create final image with complete calibration data overlay"""
    h, w = image.shape[:2]
    
    # Create overlay area on the left (300px wide for more data)
    overlay_w = 300
    overlay = np.zeros((h, overlay_w, 3), dtype=np.uint8)
    overlay.fill(30)  # Dark background
    
    # Combine overlay with original image
    result = np.hstack([overlay, image])
    
    # Text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    font_color = (255, 255, 255)  # White
    line_thickness = 1
    line_spacing = 18
    y_start = 20
    
    # Title
    cv2.putText(result, "FINAL CALIBRATION", (5, y_start), font, 0.5, (0, 255, 255), 1)
    y_start += 30
    
    # Image info
    cv2.putText(result, f"Image: {img_size[0]}x{img_size[1]}", (5, y_start), font, font_scale, font_color, line_thickness)
    cv2.putText(result, f"Pattern: {pattern_size[0]}x{pattern_size[1]}", (5, y_start + line_spacing), font, font_scale, font_color, line_thickness)
    cv2.putText(result, f"Square: {square_size*1000:.1f}mm", (5, y_start + 2*line_spacing), font, font_scale, font_color, line_thickness)
    
    y_start += 4*line_spacing
    
    # Camera matrix
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    cv2.putText(result, f"fx: {fx:.2f}", (5, y_start), font, font_scale, font_color, line_thickness)
    cv2.putText(result, f"fy: {fy:.2f}", (5, y_start + line_spacing), font, font_scale, font_color, line_thickness)
    cv2.putText(result, f"cx: {cx:.2f}", (5, y_start + 2*line_spacing), font, font_scale, font_color, line_thickness)
    cv2.putText(result, f"cy: {cy:.2f}", (5, y_start + 3*line_spacing), font, font_scale, font_color, line_thickness)
    
    y_start += 5*line_spacing
    
    # Distortion coefficients
    dist_coeffs = dist.ravel()
    cv2.putText(result, "Distortion:", (5, y_start), font, font_scale, font_color, line_thickness)
    for i, coeff in enumerate(dist_coeffs[:4]):  # Show first 4 coefficients
        cv2.putText(result, f"k{i+1}: {coeff:.4f}", (5, y_start + (i+1)*line_spacing), font, font_scale, font_color, line_thickness)
    
    y_start += 6*line_spacing
    
    # Error metrics
    cv2.putText(result, f"RMS Error: {rms:.4f}", (5, y_start), font, font_scale, (0, 255, 0) if rms < 1.0 else (0, 165, 255), line_thickness)
    
    # FOV calculation
    fovx = np.degrees(2 * np.arctan(w / (2 * fx)))
    fovy = np.degrees(2 * np.arctan(h / (2 * fy)))
    cv2.putText(result, f"FOV: {fovx:.1f}x{fovy:.1f}", (5, y_start + line_spacing), font, font_scale, font_color, line_thickness)
    
    return result

def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {CAMERA_ID}")

    cols, rows = PATTERN_COLS, PATTERN_ROWS
    pattern_size = (cols, rows)
    objp = create_object_points(cols, rows, SQUARE_SIZE_M)

    objpoints = []
    imgpoints = []
    captured_images = []  # Store captured images for saving
    img_size = None
    
    # Create session timestamp for this calibration run
    session_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

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
                
                # Create image with data overlay and save it
                sample_num = len(objpoints)
                overlay_image = create_image_with_data_overlay(vis, sample_num, len(objpoints) + 1, corners, pattern_size)
                captured_images.append(overlay_image)
                
                # Save individual captured image
                images_dir = "live_calibrate_p_img"
                os.makedirs(images_dir, exist_ok=True)
                session_images_dir = os.path.join(images_dir, f"session_{session_start_time}")
                os.makedirs(session_images_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%H%M%S")
                image_filename = f"sample_{sample_num:03d}_{timestamp}.jpg"
                image_path = os.path.join(session_images_dir, image_filename)
                cv2.imwrite(image_path, overlay_image)
                
                print(f"Captured sample {sample_num} - saved to {image_path}")
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

    save_calibration("calibration", K, dist, rvecs, tvecs, rms, img_size, pattern_size, SQUARE_SIZE_M, captured_images)

if __name__ == "__main__":
    main()