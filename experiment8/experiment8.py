"""
============================================================
Experiment 8: Application of Optical Flow for Real-Time
              Object Tracking and Motion Analysis
============================================================
Objective:
    Apply optical flow techniques for object tracking and
    motion analysis using Python, OpenCV, NumPy, and Matplotlib.

Ten steps covered:
    1.  Load video / webcam
    2.  Shi–Tomasi corner detection
    3.  Lucas–Kanade sparse optical flow
    4.  Motion trajectories and displacement vectors
    5.  Direction and magnitude calculation
    6.  Farnebäck dense optical flow
    7.  Factors affecting tracking accuracy
    8.  Motion analysis (displacement, speed)
    9.  Performance comparison (optical flow vs. frame-by-frame)
   10.  Observations and practical applications
   +   Save output images and comparison figure

Libraries used:
    - opencv-python  (cv2)  : video I/O, optical flow
    - numpy          (np)   : array operations
    - matplotlib     (plt)  : display and figure saving
    - math                  : atan2, sqrt, degrees
    - time                  : performance timing
    - os                    : directory / path utilities
    - tkinter               : file-picker dialog
============================================================
"""

import os
import sys
import math
import time


# ================================================================
# SECTION 0 – Dependency Check
# ================================================================

def check_dependencies():
    """
    Verify that all required third-party libraries are available.
    Prints installation instructions and exits if anything is missing.
    """
    missing = []
    for module, package in [
        ("cv2",        "opencv-python"),
        ("numpy",      "numpy"),
        ("matplotlib", "matplotlib"),
    ]:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print("ERROR: The following required libraries are not installed:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

# --------------- Imports (guaranteed available after check) ---------------
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ================================================================
# SECTION 1 – Video / Webcam Loading
# ================================================================

def pick_video_path():
    """
    Open a tkinter file-picker dialog so the user can browse for a video.
    Returns the selected path, or '' if cancelled / tkinter unavailable.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        path = filedialog.askopenfilename(
            title="Select a Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("MP4",         "*.mp4"),
                ("AVI",         "*.avi"),
                ("MOV",         "*.mov"),
                ("All files",   "*.*"),
            ],
        )
        root.destroy()
        return path
    except Exception:
        return ""


def open_video():
    """
    Prompt the user to select a video file or use the webcam.
    Validates the source and prints metadata.

    Returns:
        tuple[cv2.VideoCapture, str, str]:
            cap        – opened VideoCapture object
            video_path – path string (or "webcam")
            output_dir – output directory next to the video
    """
    print("=" * 60)
    print("  Experiment 8 – Optical Flow: Object Tracking &")
    print("                  Motion Analysis")
    print("=" * 60)
    print("\nStep 1: Select video source")
    print("  [F] Open file picker")
    print("  [P] Enter video path manually")
    print("  [W] Use webcam (camera 0)")
    choice = input("  Your choice (F/P/W): ").strip().upper()

    video_path = ""
    use_webcam = False

    if choice == "W":
        use_webcam = True
        video_path = "webcam"
    elif choice == "F":
        print("  Opening file picker …")
        video_path = pick_video_path()
        if not video_path:
            print("  No file selected.  Switching to manual entry.")
            choice = "P"

    if (choice == "P" or not video_path) and not use_webcam:
        video_path = input("  Enter the full video path: ").strip().strip('"').strip("'")

    if not video_path:
        print("ERROR: No video source provided.  Exiting.")
        sys.exit(1)

    if not use_webcam and not os.path.isfile(video_path):
        print(f"ERROR: File not found → {video_path}")
        sys.exit(1)

    # Open the capture
    cap = cv2.VideoCapture(0 if use_webcam else video_path)

    if not cap.isOpened():
        src = "webcam" if use_webcam else video_path
        print(f"ERROR: Could not open video source → {src}")
        print("       Check that the file exists and the codec is supported,")
        print("       or that a webcam is connected.")
        sys.exit(1)

    # Print metadata
    fps          = cap.get(cv2.CAP_PROP_FPS)
    vid_width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n  Video source opened successfully!")
    print(f"  Source       : {video_path}")
    print(f"  Resolution   : {vid_width} × {vid_height} px")
    print(f"  Frame rate   : {fps:.2f} fps")
    if not use_webcam:
        print(f"  Total frames : {total_frames}")

    # Output directory
    if use_webcam:
        output_dir = os.path.join(os.path.expanduser("~"), "cv_exp8_output")
    else:
        vid_dir    = os.path.dirname(os.path.abspath(video_path))
        output_dir = os.path.join(vid_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output dir   : {output_dir}")

    return cap, video_path, output_dir


# ================================================================
# SECTION 2 – Shi–Tomasi Corner Detection
# ================================================================

def shi_tomasi_detection(cap):
    """
    Read the first frame and detect Shi–Tomasi corner features.

    Shi–Tomasi selects corner points that exhibit high intensity
    variation in multiple directions, making them stable landmarks
    for tracking across frames.

    Parameters (from the PDF):
        maxCorners  = 100
        qualityLevel= 0.3
        minDistance = 7
        blockSize   = 7

    Args:
        cap (cv2.VideoCapture): Opened video capture at position 0.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            frame (BGR), gray, frame_rgb, corners (or None)
    """
    ret, frame = cap.read()
    if not ret or frame is None:
        print("ERROR: Could not read the first frame from the video.")
        cap.release()
        sys.exit(1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7,
    )

    # Annotate frame
    corner_frame = frame.copy()
    if corners is not None:
        for pt in corners:
            x, y = pt.ravel().astype(int)
            cv2.circle(corner_frame, (x, y), 5, (0, 0, 255), -1)
        print(f"\n  Shi-Tomasi Corner Detection completed!")
        print(f"    Feature points detected : {len(corners)}")
    else:
        print("  WARNING: No corners detected. Try a different video frame.")

    # Display (BGR → RGB)
    corner_rgb = cv2.cvtColor(corner_frame, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(corner_rgb)
    ax.set_title("Shi-Tomasi Corner Detection",
                 fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, gray, frame_rgb, corners, corner_frame


# ================================================================
# SECTION 3 – Lucas–Kanade Sparse Optical Flow
# ================================================================

def lucas_kanade_flow(cap, old_frame, old_gray):
    """
    Read the next consecutive frame and compute Lucas–Kanade sparse
    optical flow from the initial feature points.

    Lucas–Kanade assumes pixels in a small window around each feature
    point move rigidly between frames.  It solves for (dx, dy) per point.

    Drawing conventions (from PDF):
        Green line  : old position → new position (thickness 2)
        Red circle  : new position (radius 5, filled)

    Args:
        cap      (cv2.VideoCapture): Capture, positioned at frame 2.
        old_frame(np.ndarray): First BGR frame.
        old_gray (np.ndarray): Grayscale first frame.

    Returns:
        tuple: new_frame, new_gray, good_old, good_new, lk_result_bgr
    """
    ret, new_frame = cap.read()
    if not ret or new_frame is None:
        print("ERROR: Could not read the second frame.")
        return None, None, None, None, None

    new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

    # Detect initial feature points in the first frame
    p0 = cv2.goodFeaturesToTrack(
        old_gray,
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7,
    )

    if p0 is None or len(p0) == 0:
        print("  WARNING: No initial feature points for Lucas–Kanade.")
        return new_frame, new_gray, None, None, new_frame.copy()

    # Calculate optical flow
    p1, status, error = cv2.calcOpticalFlowPyrLK(
        old_gray, new_gray, p0, None
    )

    if p1 is None or status is None:
        print("  WARNING: Lucas–Kanade returned no results.")
        return new_frame, new_gray, None, None, new_frame.copy()

    good_new = p1[status == 1]
    good_old = p0[status == 1]

    print(f"\n  Lucas-Kanade Optical Flow applied successfully!")
    print(f"    Tracked feature points : {len(good_new)}")

    # Draw results on a copy of the second frame
    lk_result = new_frame.copy()
    for new_pt, old_pt in zip(good_new, good_old):
        a, b = new_pt.ravel().astype(int)
        c, d = old_pt.ravel().astype(int)
        cv2.line(lk_result, (c, d), (a, b), (0, 255, 0), 2)
        cv2.circle(lk_result, (a, b), 5, (0, 0, 255), -1)

    # Display
    lk_rgb = cv2.cvtColor(lk_result, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(lk_rgb)
    ax.set_title("Lucas-Kanade Sparse Optical Flow",
                 fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    return new_frame, new_gray, good_old, good_new, lk_result


# ================================================================
# SECTION 4 – Motion Trajectories and Displacement Vectors
# ================================================================

def draw_trajectories(old_frame, good_old, good_new):
    """
    Visualize the tracked motion paths with:
        - Green line      : path from old → new position
        - Blue arrow      : displacement direction
        - Red filled circle: new position

    Drawing conventions (from PDF):
        Green line  : (0, 255, 0), thickness 2
        Blue arrow  : (255, 0, 0) BGR, thickness 2, tipLength=0.3
        Red circle  : (0, 0, 255), radius 5, filled

    Args:
        old_frame (np.ndarray): Second BGR frame (used as canvas).
        good_old  (np.ndarray): Old positions of tracked points.
        good_new  (np.ndarray): New positions of tracked points.

    Returns:
        np.ndarray: Annotated BGR image.
    """
    traj_frame = old_frame.copy()

    if good_old is None or good_new is None or len(good_new) == 0:
        print("  WARNING: No tracked points for trajectory visualization.")
    else:
        for new_pt, old_pt in zip(good_new, good_old):
            a, b = new_pt.ravel().astype(int)
            c, d = old_pt.ravel().astype(int)

            # Green motion trail
            cv2.line(traj_frame, (c, d), (a, b), (0, 255, 0), 2)

            # Blue displacement arrow
            cv2.arrowedLine(traj_frame, (c, d), (a, b),
                            (255, 0, 0), 2, tipLength=0.3)

            # Red new-position circle
            cv2.circle(traj_frame, (a, b), 5, (0, 0, 255), -1)

    # Display
    traj_rgb = cv2.cvtColor(traj_frame, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(traj_rgb)
    ax.set_title("Motion Trajectories and Displacement Vectors",
                 fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    print("  ✔  Motion trajectories and displacement vectors visualized.")
    return traj_frame


# ================================================================
# SECTION 5 – Direction and Magnitude Calculation
# ================================================================

def calculate_motion_metrics(good_old, good_new):
    """
    Calculate displacement, magnitude, and direction for up to the
    first five successfully tracked feature points.

    Formulas:
        dx        = x_new - x_old
        dy        = y_new - y_old
        magnitude = sqrt(dx² + dy²)
        direction = degrees(atan2(dy, dx))

    Args:
        good_old (np.ndarray): Old positions.
        good_new (np.ndarray): New positions.

    Returns:
        list[dict]: Per-point metrics dictionaries.
    """
    print("\n" + "=" * 60)
    print("  Motion Direction and Magnitude (first 5 tracked points)")
    print("=" * 60)

    results = []

    if good_old is None or good_new is None or len(good_new) == 0:
        print("  No tracked points available for metric calculation.")
        return results

    n_points = min(5, len(good_new))

    for i in range(n_points):
        x_new, y_new = good_new[i].ravel()
        x_old, y_old = good_old[i].ravel()

        dx = float(x_new - x_old)
        dy = float(y_new - y_old)
        magnitude = math.sqrt(dx ** 2 + dy ** 2)
        direction = math.degrees(math.atan2(dy, dx))

        print(f"\n  Point {i + 1}:")
        print(f"    Displacement : ({dx:.2f}, {dy:.2f}) px")
        print(f"    Magnitude    : {magnitude:.2f} px")
        print(f"    Direction    : {direction:.2f}°")

        results.append({
            "dx": dx, "dy": dy,
            "magnitude": magnitude,
            "direction": direction,
        })

    print()
    return results


# ================================================================
# SECTION 6 – Farnebäck Dense Optical Flow
# ================================================================

def farneback_flow(old_frame, old_gray, new_gray):
    """
    Compute Farnebäck dense optical flow and visualize it with HSV.

    Unlike sparse optical flow, Farnebäck estimates a (dx, dy) motion
    vector for EVERY pixel in the frame using polynomial expansion.

    HSV visualization encoding:
        Hue        → motion direction (angle)
        Saturation → maximum (255) so colours are vivid
        Value      → normalized motion magnitude

    Parameters (from PDF):
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0

    Args:
        old_frame (np.ndarray): First BGR frame (for hsv shape).
        old_gray  (np.ndarray): Grayscale frame 1.
        new_gray  (np.ndarray): Grayscale frame 2.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            farneback_result (BGR for saving),
            farneback_rgb    (RGB for display / comparison)
    """
    flow = cv2.calcOpticalFlowFarneback(
        old_gray, new_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Build HSV array with same shape as the source frame
    hsv = np.zeros_like(cv2.cvtColor(old_frame, cv2.COLOR_BGR2HSV))
    hsv[..., 1] = 255
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # BGR for saving / OpenCV operations
    farneback_result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    avg_magnitude = float(np.mean(magnitude))

    print(f"\n  Farneback Dense Optical Flow applied successfully!")
    print(f"    Average motion magnitude : {avg_magnitude:.4f} px")

    # Display
    farneback_rgb = cv2.cvtColor(farneback_result, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(farneback_rgb)
    ax.set_title("Farneback Dense Optical Flow",
                 fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    return farneback_result, farneback_rgb, magnitude


# ================================================================
# SECTION 7 – Factors Affecting Tracking Accuracy
# ================================================================

def print_tracking_factors():
    """
    Print general observations about factors that affect optical flow
    tracking accuracy, following the PDF's observations section.
    """
    print("\n" + "=" * 60)
    print("  Impact on Optical Flow Tracking Accuracy")
    print("=" * 60)

    factors = [
        ("1. Object Speed",
         "Fast-moving objects may move further than the Lucas–Kanade\n"
         "     search window can capture between frames, causing tracking\n"
         "     errors or complete loss of feature points."),

        ("2. Illumination Changes",
         "Sudden changes in brightness alter the pixel intensity values\n"
         "     that optical flow depends on, reducing tracking accuracy."),

        ("3. Occlusion",
         "When a tracked object is partially or completely hidden\n"
         "     (e.g., behind another object), the algorithm loses the\n"
         "     feature point and cannot recover it automatically."),

        ("4. Camera Movement",
         "Motion of the camera itself creates apparent motion for all\n"
         "     pixels, adding background optical flow that is difficult\n"
         "     to separate from the tracked object's motion."),
    ]

    for title, explanation in factors:
        print(f"\n  {title}:")
        print(f"     {explanation}")
    print()


# ================================================================
# SECTION 8 – Motion Analysis
# ================================================================

def motion_analysis(good_old, good_new, fps):
    """
    Analyse the tracked motion to compute:
        - Per-point displacements
        - Total displacement (sum of magnitudes)
        - Average displacement
        - Average speed in pixels per second (if FPS is valid)
        - Number of successfully tracked points

    Args:
        good_old (np.ndarray): Old feature positions.
        good_new (np.ndarray): New feature positions.
        fps      (float)     : Frames per second from video metadata.
    """
    print("\n" + "=" * 60)
    print("  Motion Analysis Results")
    print("=" * 60)

    if good_old is None or good_new is None or len(good_new) == 0:
        print("  No tracked points available for motion analysis.")
        return

    n = len(good_new)
    magnitudes = []

    for new_pt, old_pt in zip(good_new, good_old):
        x_new, y_new = new_pt.ravel()
        x_old, y_old = old_pt.ravel()
        dx = float(x_new - x_old)
        dy = float(y_new - y_old)
        magnitudes.append(math.sqrt(dx ** 2 + dy ** 2))

    total_disp = sum(magnitudes)
    avg_disp   = total_disp / n if n > 0 else 0.0

    print(f"\n  Tracked points         : {n}")
    print(f"  Total displacement     : {total_disp:.2f} px")
    print(f"  Average displacement   : {avg_disp:.2f} px")

    if fps and fps > 0:
        # Two consecutive frames → one inter-frame interval = 1/fps seconds
        avg_speed = avg_disp * fps
        print(f"  Average speed          : {avg_speed:.2f} px/s  "
              f"(at {fps:.2f} fps, one inter-frame interval = {1/fps:.4f} s)")
    else:
        print("  Average speed          : Cannot be estimated "
              "(FPS unavailable or invalid)")

    print(f"\n  Note: Analysis covers one consecutive-frame pair.")
    print(f"        For complete trajectory analysis, process all frames.")
    print()


# ================================================================
# SECTION 9 – Performance Comparison
# ================================================================

def performance_comparison(video_path, max_frames=30):
    """
    Compare the processing time of optical flow tracking versus
    frame-by-frame Shi–Tomasi corner detection as a baseline.

    Methodology:
        Optical flow  : Read consecutive frame pairs; run LK flow.
        Baseline      : Run Shi–Tomasi detection independently on
                        each frame (no inter-frame state).

    Both methods process the same number of frames from the video.
    Python's time.perf_counter() provides sub-millisecond accuracy.

    Args:
        video_path (str): Path to the video file.
        max_frames (int): Maximum frames to process for timing.
    """
    print("\n" + "=" * 60)
    print("  Comparison Results")
    print("=" * 60)
    print(f"  (Processing up to {max_frames} consecutive frames)")

    if video_path == "webcam":
        print("  Timing comparison skipped for webcam input.")
        return

    if not os.path.isfile(video_path):
        print("  Video file not available for timing comparison.")
        return

    # ---- Baseline: Shi–Tomasi on every frame independently ----
    cap_b  = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, f = cap_b.read()
        if not ret or f is None:
            break
        frames.append(f)
    cap_b.release()

    n_frames = len(frames)
    if n_frames < 2:
        print("  Not enough frames for comparison.")
        return

    t0 = time.perf_counter()
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        cv2.goodFeaturesToTrack(g, maxCorners=100,
                                qualityLevel=0.3,
                                minDistance=7, blockSize=7)
    baseline_time = time.perf_counter() - t0

    # ---- Optical flow: LK between consecutive pairs ----
    t0 = time.perf_counter()
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100,
                                 qualityLevel=0.3,
                                 minDistance=7, blockSize=7)
    for f in frames[1:]:
        curr_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if p0 is not None and len(p0) > 0:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
            if p1 is not None and st is not None:
                p0 = p1[st == 1].reshape(-1, 1, 2)
        prev_gray = curr_gray
    of_time = time.perf_counter() - t0

    print(f"\n  Frames processed          : {n_frames}")
    print(f"  Baseline  (corner/frame)  : {baseline_time * 1000:.2f} ms  "
          f"({baseline_time / n_frames * 1000:.3f} ms/frame)")
    print(f"  Optical Flow (LK)         : {of_time * 1000:.2f} ms  "
          f"({of_time / n_frames * 1000:.3f} ms/frame)")

    faster = "Optical Flow" if of_time < baseline_time else "Baseline (corner detection)"
    diff   = abs(baseline_time - of_time) * 1000
    print(f"\n  Faster method (this run)  : {faster}  "
          f"(by {diff:.2f} ms total)")
    print(f"\n  Note: Timing comparison is approximate.  Results depend on")
    print(f"        video content, system load, and frame complexity.")
    print()


# ================================================================
# SECTION 10 – Observations and Practical Applications
# ================================================================

def print_observations_and_applications():
    """
    Print experiment observations and real-world application areas,
    following the PDF's observations section.
    """
    print("\n" + "=" * 60)
    print("  Observations")
    print("=" * 60)

    observations = [
        "1. Lucas–Kanade successfully tracked selected feature points "
        "when suitable corner points were available in the first frame.",
        "2. Motion trajectories showed the path followed by tracked "
        "points between consecutive frames.",
        "3. Displacement and direction of motion were calculated for "
        "each tracked point.",
        "4. Farnebäck Dense Optical Flow provided dense motion "
        "information covering every pixel in the frame.",
        "5. Fast motion and occlusion can reduce the number of "
        "successfully tracked points.",
        "6. Illumination changes and camera movement affect the "
        "accuracy of optical flow results.",
    ]

    for obs in observations:
        print(f"\n  • {obs}")

    print("\n\n" + "=" * 60)
    print("  PRACTICAL APPLICATIONS")
    print("=" * 60)

    applications = [
        ("Object Tracking",
         "Track objects (vehicles, sports balls, people) across video frames."),
        ("Traffic Monitoring",
         "Estimate vehicle speed and count traffic flow on roads."),
        ("Video Surveillance",
         "Detect and follow moving subjects in security footage."),
        ("Autonomous Vehicles",
         "Perceive environment motion to assist navigation and obstacle avoidance."),
        ("Human Activity Recognition",
         "Analyse body motion patterns to classify gestures and activities."),
        ("Sports Analytics",
         "Track player and ball movement to extract performance statistics."),
        ("Robotics and Navigation",
         "Estimate ego-motion and detect moving obstacles for robot navigation."),
    ]

    for app, desc in applications:
        print(f"\n  • {app}")
        print(f"      {desc}")

    print()


# ================================================================
# SECTION 11 – Save Output Images
# ================================================================

def save_outputs(corner_frame, lk_result, traj_frame,
                 farneback_result, comparison_fig, output_dir):
    """
    Save all required output images to the output directory.

    Files:
        shi_tomasi_corner_detection.jpg
        lucas_kanade_sparse_optical_flow.jpg
        motion_trajectories.jpg
        farneback_dense_optical_flow.jpg
        optical_flow_comparison.png

    Args:
        corner_frame     (np.ndarray): Shi–Tomasi annotated frame (BGR).
        lk_result        (np.ndarray): Lucas–Kanade result (BGR) or None.
        traj_frame       (np.ndarray): Trajectory visualization (BGR).
        farneback_result (np.ndarray): Farnebäck result (BGR).
        comparison_fig   (Figure)    : 3-panel Matplotlib figure.
        output_dir       (str)       : Output directory path.
    """
    print(f"\n  Saving output files to: {output_dir}")

    all_saved = True

    def write_bgr(filename, image):
        nonlocal all_saved
        if image is None:
            print(f"    ⚠  Skipped (no data): {filename}")
            return
        path = os.path.join(output_dir, filename)
        ok   = cv2.imwrite(path, image)
        if ok:
            print(f"    ✔  Saved: {filename}  →  {path}")
        else:
            print(f"    ✘  FAILED: {filename}")
            all_saved = False

    write_bgr("shi_tomasi_corner_detection.jpg",       corner_frame)
    write_bgr("lucas_kanade_sparse_optical_flow.jpg",  lk_result)
    write_bgr("motion_trajectories.jpg",               traj_frame)
    write_bgr("farneback_dense_optical_flow.jpg",      farneback_result)

    # Comparison figure via Matplotlib
    comp_path = os.path.join(output_dir, "optical_flow_comparison.png")
    try:
        comparison_fig.savefig(comp_path, dpi=150, bbox_inches="tight")
        print(f"    ✔  Saved: optical_flow_comparison.png  →  {comp_path}")
    except Exception as e:
        print(f"    ✘  FAILED comparison figure: {e}")
        all_saved = False

    if all_saved:
        print("\n  All output files saved successfully.")
    else:
        print("\n  WARNING: Some files could not be saved.")


# ================================================================
# SECTION 12 – Final Comparison Figure (3-panel)
# ================================================================

def build_comparison_figure(corner_frame, lk_result,
                             farneback_rgb):
    """
    Create a 3-panel side-by-side comparison figure:
        Panel 1 – Shi–Tomasi Corner Detection
        Panel 2 – Lucas–Kanade Sparse Optical Flow
        Panel 3 – Farnebäck Dense Optical Flow

    Args:
        corner_frame  (np.ndarray): Annotated BGR frame (Shi–Tomasi).
        lk_result     (np.ndarray): LK result BGR, or None.
        farneback_rgb (np.ndarray): Dense flow in RGB.

    Returns:
        matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Optical Flow Comparison",
                 fontsize=15, fontweight="bold")

    # Shi–Tomasi
    axes[0].imshow(cv2.cvtColor(corner_frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Shi-Tomasi Corner Detection",
                      fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Lucas–Kanade
    if lk_result is not None:
        axes[1].imshow(cv2.cvtColor(lk_result, cv2.COLOR_BGR2RGB))
    else:
        axes[1].text(0.5, 0.5, "No LK result",
                     ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_title("Lucas-Kanade Sparse Optical Flow",
                      fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # Farnebäck
    axes[2].imshow(farneback_rgb)
    axes[2].set_title("Farneback Dense Optical Flow",
                      fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    print("  ✔  Final comparison figure displayed.")
    return fig


# ================================================================
# MAIN PROGRAM
# ================================================================

def main():
    """
    Entry point – orchestrates Experiment 8 step by step:

        Step 1 : Open video / webcam
        Step 2 : Shi–Tomasi corner detection
        Step 3 : Lucas–Kanade sparse optical flow
        Step 4 : Motion trajectories and displacement vectors
        Step 5 : Direction and magnitude calculation
        Step 6 : Farnebäck dense optical flow
        Step 7 : Factors affecting tracking accuracy
        Step 8 : Motion analysis
        Step 9 : Performance comparison
        Step 10: Observations and practical applications
        Step 11: Final 3-panel comparison figure
        Step 12: Save all output images
    """

    # ---- Step 1: Open video ----------------------------------------
    cap, video_path, output_dir = open_video()

    # Retrieve FPS before consuming frames
    fps = cap.get(cv2.CAP_PROP_FPS)

    # ---- Step 2: Shi–Tomasi ----------------------------------------
    print("\nStep 2: Shi–Tomasi corner detection …")
    old_frame, old_gray, _, corners, corner_frame = shi_tomasi_detection(cap)

    # ---- Step 3: Lucas–Kanade --------------------------------------
    print("\nStep 3: Lucas–Kanade sparse optical flow …")
    new_frame, new_gray, good_old, good_new, lk_result = lucas_kanade_flow(
        cap, old_frame, old_gray
    )
    cap.release()   # release after reading the two frames we need

    # ---- Step 4: Trajectories & displacement vectors ---------------
    print("\nStep 4: Motion trajectories and displacement vectors …")
    traj_frame = draw_trajectories(
        new_frame if new_frame is not None else old_frame,
        good_old, good_new,
    )

    # ---- Step 5: Direction and magnitude ---------------------------
    print("\nStep 5: Calculating motion direction and magnitude …")
    calculate_motion_metrics(good_old, good_new)

    # ---- Step 6: Farnebäck dense optical flow ----------------------
    print("\nStep 6: Farnebäck dense optical flow …")
    if new_gray is not None:
        farneback_result, farneback_rgb, magnitude = farneback_flow(
            old_frame, old_gray, new_gray
        )
    else:
        print("  WARNING: Frame 2 unavailable; skipping Farnebäck.")
        farneback_result = old_frame.copy()
        farneback_rgb    = cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB)
        magnitude        = np.zeros(old_gray.shape, dtype=np.float32)

    # ---- Step 7: Factors affecting accuracy ------------------------
    print("\nStep 7: Factors affecting tracking accuracy …")
    print_tracking_factors()

    # ---- Step 8: Motion analysis -----------------------------------
    print("\nStep 8: Motion analysis …")
    motion_analysis(good_old, good_new, fps)

    # ---- Step 9: Performance comparison ----------------------------
    print("\nStep 9: Performance comparison …")
    performance_comparison(video_path)

    # ---- Step 10: Observations and applications --------------------
    print("\nStep 10: Observations and practical applications …")
    print_observations_and_applications()

    # ---- Step 11: Final 3-panel comparison figure ------------------
    print("\nStep 11: Building final comparison figure …")
    comp_fig = build_comparison_figure(corner_frame, lk_result, farneback_rgb)

    # ---- Step 12: Save outputs -------------------------------------
    print("\nStep 12: Saving output images …")
    save_outputs(corner_frame, lk_result, traj_frame,
                 farneback_result, comp_fig, output_dir)

    # ---- Done ------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Experiment 8 completed successfully!")
    print(f"  Output images saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
