import argparse
import os
import time
import numpy as np
import cv2
import glob

# Define the dimensions of the checkerboard (inner corners)
CHECKERBOARD = (9, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare a single object points array for the checkerboard (z=0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)


def calibrate_and_save(objpoints, imgpoints, image_size, out_path=None):
    if len(objpoints) < 3 or len(imgpoints) < 3:
        print("Need at least 3 valid views of the checkerboard to calibrate.\n"
              f"Got {len(objpoints)}")
        return False

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    print(f"Calibration RMS error: {ret}")
    print("\nIntrinsic Matrix:")
    print(mtx)
    print("\nDistortion Coefficients:")
    print(dist.ravel())

    if out_path is None:
        ts = time.strftime('%Y%m%d-%H%M%S')
        out_path = f"camera_parameters_{ts}.npz"

    np.savez(out_path, intrinsic_matrix=mtx, distortion=dist, rvecs=rvecs, tvecs=tvecs)
    print(f"Saved camera parameters to {out_path}")
    return True


def calibrate_from_images(images_dir):
    images = sorted(glob.glob(os.path.join(images_dir, '*.png')) +
                    glob.glob(os.path.join(images_dir, '*.jpg')) +
                    glob.glob(os.path.join(images_dir, '*.jpeg')))

    if len(images) == 0:
        print("No images found in the images directory. Skipping image-based calibration.")
        return None

    objpoints = []
    imgpoints = []
    image_size = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: could not read image '{fname}', skipping.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
        if ret:
            objpoints.append(objp.copy())
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
        else:
            print(f"Checkerboard not detected in '{fname}'")

    cv2.destroyAllWindows()

    if image_size is None:
        print("No valid images were processed.")
        return None

    success = calibrate_and_save(objpoints, imgpoints, image_size)
    return success


def calibrate_from_camera(cam_id=0, required_samples=15, show_window=True):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"Unable to open camera id {cam_id}")
        return False

    objpoints = []
    imgpoints = []
    image_size = None

    instructions = "Press 'c' to capture when checkerboard visible, 'q' to finish."
    print(instructions)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

        if found:
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners, found)
            cv2.putText(display, "Checkerboard detected - press 'c' to capture", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(display, f"Captured: {len(objpoints)}/{required_samples}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if show_window:
            cv2.imshow('calibration', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # ESC or q to quit
            break
        if key == ord('c') and found:
            # refine and store
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp.copy())
            imgpoints.append(corners2)
            print(f"Captured view {len(objpoints)}")
            # small visual feedback
            cv2.circle(display, tuple(corners2[0].ravel().astype(int)), 5, (255, 0, 0), -1)
            if show_window:
                cv2.imshow('calibration', display)
                cv2.waitKey(200)

            if len(objpoints) >= required_samples:
                print("Required number of samples captured.")
                break

    cap.release()
    cv2.destroyAllWindows()

    if image_size is None:
        print("No frames captured from camera.")
        return False

    return calibrate_and_save(objpoints, imgpoints, image_size)


def parse_args():
    parser = argparse.ArgumentParser(description='Camera calibration utility')
    parser.add_argument('--camera', action='store_true', help='Use local webcam for interactive calibration')
    parser.add_argument('--cam-id', type=int, default=0, help='Camera device id (default 0)')
    parser.add_argument('--samples', type=int, default=15, help='Number of samples to capture from camera')
    parser.add_argument('--images-dir', default='images', help='Directory with calibration images')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.camera:
        # interactive webcam calibration
        ok = calibrate_from_camera(cam_id=args.cam_id, required_samples=args.samples)
        if not ok:
            print("Camera calibration failed or was cancelled.")
    else:
        # fallback to images directory calibration
        ok = calibrate_from_images(args.images_dir)
        if not ok:
            print("Image-based calibration failed or was cancelled.")
