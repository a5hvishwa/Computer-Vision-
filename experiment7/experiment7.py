"""
============================================================
Experiment 7: Motion Tracking Using Optical Flow
============================================================
Objective:
    Implement and compare two optical flow techniques using
    Python, OpenCV, NumPy, and Matplotlib.

Techniques covered:
    1.  Load and display the first two video frames
    2.  Shi–Tomasi corner detection (feature point selection)
    3.  Lucas–Kanade Sparse Optical Flow (track selected points)
    4.  Farnebäck Dense Optical Flow (motion for every pixel)
    5.  Side-by-side comparison of both methods
    6.  Save output images

Libraries used:
    - opencv-python  (cv2)  : video I/O, optical flow
    - numpy          (np)   : array operations
    - matplotlib     (plt)  : displaying frames and flow
    - tkinter               : file-picker dialog
    - os                    : directory / path utilities
============================================================
"""

import os
import sys


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
# SECTION 1 – Video Loading
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


def load_video_frames():
    """
    Prompt the user to select a video, then read the first two
    consecutive frames.  Prints video metadata.

    Returns:
        tuple: (frame1, frame2, gray1, gray2, frame1_rgb, frame2_rgb,
                video_path, output_dir)
            frame1 / frame2       : BGR frames (unchanged originals)
            gray1  / gray2        : Grayscale versions
            frame1_rgb/frame2_rgb : RGB versions for Matplotlib
            video_path            : Selected video path
            output_dir            : Output directory (next to video)
    """
    print("=" * 60)
    print("  Experiment 7 – Motion Tracking Using Optical Flow")
    print("=" * 60)
    print("\nStep 1: Select your input video")
    print("  [F] Open file picker")
    print("  [P] Enter path manually")
    choice = input("  Your choice (F/P): ").strip().upper()

    video_path = ""

    if choice == "F":
        print("  Opening file picker …")
        video_path = pick_video_path()
        if not video_path:
            print("  No file selected.  Switching to manual entry.")
            choice = "P"

    if choice == "P" or not video_path:
        video_path = input("  Enter the full video path: ").strip().strip('"').strip("'")

    if not video_path:
        print("ERROR: No video path provided.  Exiting.")
        sys.exit(1)

    if not os.path.isfile(video_path):
        print(f"ERROR: File not found → {video_path}")
        sys.exit(1)

    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: OpenCV could not open the video → {video_path}")
        print("       Ensure the file is a valid MP4/AVI/MOV and that")
        print("       your OpenCV build supports the required codec.")
        sys.exit(1)

    # Print video metadata
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    vid_width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n  Video opened successfully!")
    print(f"  Path         : {video_path}")
    print(f"  Resolution   : {vid_width} × {vid_height} px")
    print(f"  Frame rate   : {fps:.2f} fps")
    print(f"  Total frames : {total_frames}")

    # Read the first two frames
    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()

    cap.release()  # we only need two frames; release immediately

    if not ret1 or frame1 is None:
        print("ERROR: Could not read Frame 1 from the video.")
        sys.exit(1)

    if not ret2 or frame2 is None:
        print("ERROR: Could not read Frame 2.  "
              "The video may have fewer than 2 frames.")
        sys.exit(1)

    print(f"\n  Frame 1 shape : {frame1.shape}")
    print(f"  Frame 2 shape : {frame2.shape}")

    # Convert to grayscale (required for optical flow algorithms)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Convert to RGB for Matplotlib display
    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    # Output directory next to the video file
    video_dir  = os.path.dirname(os.path.abspath(video_path))
    output_dir = os.path.join(video_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n  Output directory: {output_dir}")

    return (frame1, frame2, gray1, gray2,
            frame1_rgb, frame2_rgb, video_path, output_dir)


# ================================================================
# SECTION 2 – Display First Two Video Frames
# ================================================================

def display_frames(frame1_rgb, frame2_rgb):
    """
    Display the first and second video frames side by side.

    Args:
        frame1_rgb (np.ndarray): Frame 1 in RGB.
        frame2_rgb (np.ndarray): Frame 2 in RGB.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Input Video Frames", fontsize=14, fontweight="bold")

    axes[0].imshow(frame1_rgb)
    axes[0].set_title("Frame 1", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(frame2_rgb)
    axes[1].set_title("Frame 2", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    print("  ✔  Both frames displayed.")


# ================================================================
# SECTION 3 – Shi–Tomasi Feature Point Detection
# ================================================================

def detect_feature_points(gray1):
    """
    Detect trackable corner features in the first grayscale frame
    using the Shi–Tomasi corner detector.

    Shi–Tomasi selects corners by evaluating the minimum eigenvalue of
    the 2×2 gradient matrix at each pixel.  Corners are regions where
    intensity changes rapidly in multiple directions, making them
    reliable landmarks for tracking across frames.

    Parameters (from the PDF):
        maxCorners  = 100   (at most 100 strongest corners)
        qualityLevel= 0.3   (threshold = 0.3 × max corner quality)
        minDistance = 7     (minimum pixel distance between corners)
        blockSize   = 7     (neighbourhood for corner calculation)

    Args:
        gray1 (np.ndarray): Grayscale first frame.

    Returns:
        np.ndarray | None: Detected feature points, shape (N, 1, 2),
                           or None if no features are found.
    """
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7,
    )

    p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)

    if p0 is None or len(p0) == 0:
        print("  WARNING: No Shi–Tomasi feature points detected.")
        print("           Try a different video with more visual texture.")
        return None

    print(f"\n  Shi–Tomasi Feature Detection:")
    print(f"    Feature points detected : {len(p0)}")
    print("  ✔  Feature point detection done.")
    return p0


# ================================================================
# SECTION 4 – Lucas–Kanade Sparse Optical Flow
# ================================================================

def apply_lucas_kanade(frame2, gray1, gray2, p0):
    """
    Compute and visualize Lucas–Kanade sparse optical flow.

    The Lucas–Kanade method assumes that pixels in a small neighbourhood
    (window) move by the same amount between frames.  It solves a system
    of equations to find the velocity (dx, dy) for each tracked point.

    Only points that the PDF's feature detector selected are tracked —
    hence the term "sparse" (a subset of pixels, not all of them).

    Parameters (from the PDF):
        winSize  = (15, 15)      (search window around each point)
        maxLevel = 2             (image pyramid levels for large motions)
        criteria = EPS|COUNT, 10 iterations, 0.03 tolerance

    Visualization:
        - Green lines  : motion trail from old position to new position
        - Red circles  : new position of each tracked point

    Args:
        frame2 (np.ndarray): Second video frame (BGR).
        gray1  (np.ndarray): Grayscale frame 1.
        gray2  (np.ndarray): Grayscale frame 2.
        p0     (np.ndarray): Feature points detected in frame 1.

    Returns:
        np.ndarray | None: BGR result image, or None if tracking fails.
    """
    if p0 is None:
        print("  WARNING: No input points for Lucas–Kanade.  Skipping.")
        return None

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            10,
            0.03,
        ),
    )

    # Calculate optical flow from frame 1 → frame 2
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, p0, None, **lk_params
    )

    # Guard: failed tracking
    if p1 is None or st is None:
        print("  WARNING: Lucas–Kanade tracking returned no results.")
        return None

    # Keep only points that were tracked successfully (status == 1)
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    print(f"\n  Lucas–Kanade Sparse Optical Flow:")
    print(f"    Input points    : {len(p0)}")
    print(f"    Tracked points  : {len(good_new)}")

    if len(good_new) == 0:
        print("  WARNING: No points were successfully tracked.")
        return None

    # ---- Draw tracking results on a copy of frame 2 ----
    result_lk = frame2.copy()

    for new_pt, old_pt in zip(good_new, good_old):
        a, b = new_pt.ravel().astype(int)   # new position
        c, d = old_pt.ravel().astype(int)   # old position

        # Green line: motion trail
        cv2.line(result_lk, (c, d), (a, b), (0, 255, 0), 2)

        # Red filled circle: new position
        cv2.circle(result_lk, (a, b), 4, (0, 0, 255), -1)

    # Display (BGR → RGB for Matplotlib)
    result_lk_rgb = cv2.cvtColor(result_lk, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(result_lk_rgb)
    ax.set_title("Lucas-Kanade Sparse Optical Flow",
                 fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    print("  ✔  Lucas–Kanade sparse optical flow done.")
    return result_lk   # BGR for saving / comparison


# ================================================================
# SECTION 5 – Farnebäck Dense Optical Flow
# ================================================================

def apply_farneback(frame1, gray1, gray2):
    """
    Compute and visualize Farnebäck dense optical flow.

    Unlike sparse optical flow, the Farnebäck algorithm estimates a
    motion vector (dx, dy) for EVERY pixel in the frame.  It
    approximates the local image intensities with polynomial expansions
    and uses them to compute the flow field.

    Visualization uses an HSV color space:
        Hue        → motion direction (angle of the flow vector)
        Saturation → set to maximum so colours are fully visible
        Value      → motion magnitude (brighter = faster movement)

    The HSV image is converted to RGB for Matplotlib display.

    Parameters (from the PDF):
        pyr_scale = 0.5      (downsample ratio between pyramid levels)
        levels    = 3        (number of pyramid levels)
        winsize   = 15       (averaging window size)
        iterations= 3        (iterations at each pyramid level)
        poly_n    = 5        (pixel neighbourhood for polynomial expansion)
        poly_sigma= 1.2      (Gaussian σ for polynomial expansion)
        flags     = 0

    Args:
        frame1 (np.ndarray): First BGR frame (for HSV array shape).
        gray1  (np.ndarray): Grayscale frame 1.
        gray2  (np.ndarray): Grayscale frame 2.

    Returns:
        np.ndarray: Dense optical flow visualization in RGB.
    """
    flow = cv2.calcOpticalFlowFarneback(
        gray1,
        gray2,
        None,
        0.5,    # pyr_scale
        3,      # levels
        15,     # winsize
        3,      # iterations
        5,      # poly_n
        1.2,    # poly_sigma
        0,      # flags
    )

    print(f"\n  Farnebäck Dense Optical Flow:")
    print(f"    Flow field shape : {flow.shape}  (H × W × 2)")

    # ---- Compute magnitude and angle from the (dx, dy) flow vectors ----
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # ---- Build the HSV visualization ----
    # Use frame1 shape to create a 3-channel array of the correct size
    hsv = np.zeros(
        (gray1.shape[0], gray1.shape[1], 3), dtype=np.uint8
    )

    # Hue encodes direction (0–360° mapped to 0–180 in OpenCV HSV)
    hsv[..., 0] = angle * 180 / np.pi / 2

    # Full saturation so colours are vivid
    hsv[..., 1] = 255

    # Value (brightness) encodes magnitude, normalized to 0–255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV → RGB for Matplotlib display
    dense_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Display
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(dense_flow)
    ax.set_title("Farneback Dense Optical Flow",
                 fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    print("  ✔  Farnebäck dense optical flow done.")
    return dense_flow   # RGB


# ================================================================
# SECTION 6 – Side-by-Side Comparison
# ================================================================

def display_comparison(result_lk, dense_flow):
    """
    Display Lucas–Kanade and Farnebäck results side by side.

    Args:
        result_lk  (np.ndarray): LK result in BGR (or None).
        dense_flow (np.ndarray): Dense flow visualization in RGB.

    Returns:
        matplotlib.figure.Figure: The comparison figure (for saving).
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Optical Flow Comparison",
                 fontsize=15, fontweight="bold")

    # Panel 1 – Lucas–Kanade
    if result_lk is not None:
        lk_rgb = cv2.cvtColor(result_lk, cv2.COLOR_BGR2RGB)
        axes[0].imshow(lk_rgb)
    else:
        axes[0].text(0.5, 0.5, "No LK result available",
                     ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set_title("Lucas-Kanade Sparse Optical Flow",
                      fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2 – Farnebäck
    axes[1].imshow(dense_flow)
    axes[1].set_title("Farneback Dense Optical Flow",
                      fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    print("  ✔  Comparison figure displayed.")
    return fig


# ================================================================
# SECTION 7 – Save Output Images
# ================================================================

def save_outputs(frame1, frame2, result_lk, dense_flow,
                 comparison_fig, output_dir):
    """
    Save all five required output files.

    Files saved:
        frame1.jpg                          – first video frame (BGR)
        frame2.jpg                          – second video frame (BGR)
        lucas_kanade_sparse_optical_flow.jpg – LK result (BGR)
        farneback_dense_optical_flow.jpg    – dense flow (RGB→BGR)
        optical_flow_comparison.png         – side-by-side figure

    Args:
        frame1         (np.ndarray): First BGR frame.
        frame2         (np.ndarray): Second BGR frame.
        result_lk      (np.ndarray): LK result (BGR) or None.
        dense_flow     (np.ndarray): Dense flow RGB image.
        comparison_fig (Figure)    : Matplotlib comparison figure.
        output_dir     (str)       : Directory to write files into.
    """
    print(f"\n  Saving output files to: {output_dir}")

    all_saved = True

    def write(filename, image):
        """Write a BGR image with cv2.imwrite and report success."""
        nonlocal all_saved
        path = os.path.join(output_dir, filename)
        ok = cv2.imwrite(path, image)
        if ok:
            print(f"    ✔  Saved: {filename}")
        else:
            print(f"    ✘  FAILED to save: {filename}")
            all_saved = False

    # frame1 and frame2 – already BGR
    write("frame1.jpg", frame1)
    write("frame2.jpg", frame2)

    # LK result – BGR (or skip if unavailable)
    if result_lk is not None:
        write("lucas_kanade_sparse_optical_flow.jpg", result_lk)
    else:
        print("    ⚠  Skipped: lucas_kanade_sparse_optical_flow.jpg "
              "(no LK result available)")

    # Dense flow – RGB → BGR for cv2.imwrite
    dense_bgr = cv2.cvtColor(dense_flow, cv2.COLOR_RGB2BGR)
    write("farneback_dense_optical_flow.jpg", dense_bgr)

    # Comparison figure – save via Matplotlib
    comp_path = os.path.join(output_dir, "optical_flow_comparison.png")
    try:
        comparison_fig.savefig(comp_path, dpi=150, bbox_inches="tight")
        print(f"    ✔  Saved: optical_flow_comparison.png")
    except Exception as e:
        print(f"    ✘  FAILED to save comparison figure: {e}")
        all_saved = False

    if all_saved:
        print("\n  All output files saved successfully.")
    else:
        print("\n  WARNING: Some files could not be saved.  "
              "Check path and permissions.")


# ================================================================
# MAIN PROGRAM
# ================================================================

def main():
    """
    Entry point – orchestrates Experiment 7 step by step:

        Step 1 : Load video → extract first two frames
        Step 2 : Display Frame 1 and Frame 2 side by side
        Step 3 : Shi–Tomasi feature point detection
        Step 4 : Lucas–Kanade sparse optical flow
        Step 5 : Farnebäck dense optical flow
        Step 6 : Side-by-side comparison
        Step 7 : Save all output images
    """

    # ---- Step 1: Load video and extract frames ----------------------
    (frame1, frame2, gray1, gray2,
     frame1_rgb, frame2_rgb,
     video_path, output_dir) = load_video_frames()

    # ---- Step 2: Display the two frames ----------------------------
    print("\nStep 2: Displaying first two frames …")
    display_frames(frame1_rgb, frame2_rgb)

    # ---- Step 3: Shi–Tomasi feature detection ----------------------
    print("\nStep 3: Detecting Shi–Tomasi feature points …")
    p0 = detect_feature_points(gray1)

    # ---- Step 4: Lucas–Kanade sparse optical flow ------------------
    print("\nStep 4: Computing Lucas–Kanade sparse optical flow …")
    result_lk = apply_lucas_kanade(frame2, gray1, gray2, p0)

    # ---- Step 5: Farnebäck dense optical flow ----------------------
    print("\nStep 5: Computing Farnebäck dense optical flow …")
    dense_flow = apply_farneback(frame1, gray1, gray2)

    # ---- Step 6: Side-by-side comparison ---------------------------
    print("\nStep 6: Displaying comparison figure …")
    comparison_fig = display_comparison(result_lk, dense_flow)

    # ---- Step 7: Save outputs --------------------------------------
    print("\nStep 7: Saving output images …")
    save_outputs(frame1, frame2, result_lk, dense_flow,
                 comparison_fig, output_dir)

    # ---- Done ------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Experiment 7 completed successfully!")
    print(f"  Output images saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
