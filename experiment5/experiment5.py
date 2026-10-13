"""
============================================================
Experiment 5: Feature Extraction and Feature Matching
              Using SIFT and HOG
============================================================
Objective:
    Implement feature extraction (SIFT, HOG) and feature
    matching using Python, OpenCV, scikit-image, NumPy,
    and Matplotlib.

Operations covered:
    1.  Load image, convert to RGB and grayscale
    2.  SIFT keypoint detection and descriptor extraction
    3.  HOG feature extraction and visualization
    4.  Create a 1.2× resized second image
    5.  SIFT on both images
    6.  Brute-Force k-NN matching + Lowe's ratio test
    7.  SIFT feature-matching visualization
    8.  Display original colour image
    9.  Save output images

Libraries used:
    - opencv-python  (cv2)             : image I/O, SIFT, BFMatcher
    - scikit-image   (skimage)         : HOG features and exposure
    - numpy          (np)              : array operations
    - matplotlib     (plt)             : displaying images / plots
    - tkinter                          : file-picker dialog
    - os                               : directory / path utilities

Install any missing libraries with:
    pip install opencv-python scikit-image numpy matplotlib
    (Note: SIFT is available in standard opencv-python >= 4.5)
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
    checks = [
        ("cv2",             "opencv-python"),
        ("numpy",           "numpy"),
        ("matplotlib",      "matplotlib"),
        ("skimage",         "scikit-image"),
    ]
    for module, package in checks:
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

    # Extra check: SIFT requires opencv with contrib OR opencv >= 4.5
    import cv2
    if not hasattr(cv2, "SIFT_create"):
        print("ERROR: cv2.SIFT_create() is not available.")
        print("       Install the contrib build:")
        print("         pip install opencv-contrib-python")
        sys.exit(1)

check_dependencies()

# --------------- Imports (guaranteed available after check) ---------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure


# ================================================================
# SECTION 1 – Image Loading
# ================================================================

def pick_image_path():
    """
    Open a tkinter file-picker dialog so the user can browse for an
    image.  Returns the selected path, or an empty string if cancelled
    or if tkinter is unavailable on this system.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()                    # hide the empty root window
        root.attributes("-topmost", True)  # bring dialog to the front

        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("JPEG",        "*.jpg *.jpeg"),
                ("PNG",         "*.png"),
                ("BMP",         "*.bmp"),
                ("All files",   "*.*"),
            ],
        )
        root.destroy()
        return file_path

    except Exception:
        return ""  # tkinter unavailable — caller falls back to manual entry


def load_image():
    """
    Prompt the user to select or type an image path, then load it with
    OpenCV.  Returns both BGR and RGB versions, plus the grayscale image.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, str]:
            img (BGR), img_rgb (RGB), gray, image_path
    """
    print("=" * 60)
    print("  Experiment 5 – Feature Extraction & Feature Matching")
    print("                 (SIFT and HOG)")
    print("=" * 60)
    print("\nStep 1: Select your input image")
    print("  [F] Open file picker")
    print("  [P] Enter path manually")
    choice = input("  Your choice (F/P): ").strip().upper()

    image_path = ""

    if choice == "F":
        print("  Opening file picker …")
        image_path = pick_image_path()
        if not image_path:
            print("  No file selected.  Switching to manual entry.")
            choice = "P"

    if choice == "P" or not image_path:
        image_path = input("  Enter the full image path: ").strip().strip('"').strip("'")

    # ---- Validate ----
    if not image_path:
        print("ERROR: No image path provided.  Exiting.")
        sys.exit(1)

    if not os.path.isfile(image_path):
        print(f"ERROR: File not found → {image_path}")
        sys.exit(1)

    supported_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in supported_ext:
        print(f"WARNING: Extension '{ext}' may not be supported.")
        print(f"         Supported formats: {', '.join(supported_ext)}")

    # ---- Load with OpenCV (BGR) ----
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: OpenCV could not read the image → {image_path}")
        print("       The file may be corrupt or in an unsupported format.")
        sys.exit(1)

    # ---- Prepare colour formats ----
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # for Matplotlib display
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # for feature extraction

    # ---- Print metadata ----
    height, width = img.shape[:2]
    print(f"\n  Image loaded successfully!")
    print(f"  Path       : {image_path}")
    print(f"  Width      : {width} px")
    print(f"  Height     : {height} px")
    print(f"  Dimensions : {img.shape}")
    print(f"  Data type  : {img.dtype}")

    # ---- Display original colour image ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(img_rgb)
    ax.set_title("Original Butterfly Image", fontsize=14,
                 fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    return img, img_rgb, gray, image_path


# ================================================================
# SECTION 2 – SIFT Keypoint Detection and Feature Extraction
# ================================================================

def apply_sift(img, img_rgb, gray):
    """
    Detect SIFT keypoints and compute 128-dimensional descriptors.

    SIFT (Scale-Invariant Feature Transform) identifies distinctive
    local image patches that remain stable under changes in scale,
    rotation, and illumination.  Each keypoint is described by a
    128-element gradient histogram vector.

    Args:
        img     (np.ndarray): Original BGR image (for drawing).
        img_rgb (np.ndarray): RGB version for Matplotlib display.
        gray    (np.ndarray): Grayscale image for SIFT detection.

    Returns:
        tuple[list, np.ndarray, cv2.SIFT]: keypoints, descriptors, sift object
    """
    # Create the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute 128-D descriptors simultaneously
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # ---- Guard: no keypoints found ----
    if keypoints is None or len(keypoints) == 0:
        print("  WARNING: No SIFT keypoints found in the image.")
        print("           Try a different image with more visual texture.")
        return [], None, sift

    print(f"\n  SIFT Results:")
    print(f"    Number of keypoints : {len(keypoints)}")
    if descriptors is not None:
        print(f"    Descriptor shape    : {descriptors.shape}")

    # ---- Draw keypoints with scale and orientation rings ----
    sift_image = cv2.drawKeypoints(
        img_rgb,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    # ---- Display ----
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(sift_image)
    ax.set_title("SIFT Keypoints", fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    print("  ✔  SIFT keypoint detection and descriptor extraction done.")
    return keypoints, descriptors, sift


# ================================================================
# SECTION 3 – HOG Feature Extraction
# ================================================================

def apply_hog(img_rgb, gray):
    """
    Extract Histogram of Oriented Gradients (HOG) features and
    display the visualization alongside the original image.

    HOG divides the image into small cells (8×8 pixels) and computes
    a histogram of edge/gradient orientations in each cell.  These
    histograms are normalised across 2×2 blocks and concatenated into
    a feature vector.  HOG is widely used for pedestrian detection
    and object shape representation.

    Parameters (from the PDF):
        orientations    = 9        (number of gradient direction bins)
        pixels_per_cell = (8, 8)   (size of each cell)
        cells_per_block = (2, 2)   (number of cells per block)
        block_norm      = "L2-Hys" (block normalization method)
        visualize       = True     (return a visual HOG image)

    Args:
        img_rgb (np.ndarray): Original RGB image for display.
        gray    (np.ndarray): Grayscale image for HOG computation.

    Returns:
        tuple[np.ndarray, np.ndarray]: hog_features, hog_image_rescaled
    """
    hog_features, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True,
    )

    # Rescale the HOG visualization for clearer display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    print(f"\n  HOG Results:")
    print(f"    HOG feature vector length : {len(hog_features)}")

    # ---- Side-by-side display ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("HOG Feature Extraction", fontsize=14, fontweight="bold")

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(hog_image_rescaled, cmap="gray")
    axes[1].set_title("HOG Visualization", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    print("  ✔  HOG feature extraction and visualization done.")
    return hog_features, hog_image_rescaled


# ================================================================
# SECTION 4 – Create Resized Second Image (1.2×)
# ================================================================

def create_resized_image(img, img_rgb):
    """
    Create a 1.2× scaled version of the original image for feature
    matching.  This simulates capturing the same scene at a slightly
    different zoom level, letting us test whether SIFT descriptors
    remain consistent across scale changes.

    Scale factor: fx=1.2, fy=1.2  (from the PDF)

    Args:
        img     (np.ndarray): Original BGR image.
        img_rgb (np.ndarray): Original RGB image (for display).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            img2 (BGR), img2_rgb (RGB), gray2 (grayscale)
    """
    # Resize with scale factors (None keeps the output size auto-computed)
    img2     = cv2.resize(img, None, fx=1.2, fy=1.2)
    gray2    = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    print(f"\n  Resized image shape : {img2.shape}  (1.2× scale)")

    # ---- Side-by-side display ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Original vs. Resized Image", fontsize=14, fontweight="bold")

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(img2_rgb)
    axes[1].set_title("Resized Image (1.2×)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    print("  ✔  Resized image (1.2×) created and displayed.")
    return img2, img2_rgb, gray2


# ================================================================
# SECTION 5 – SIFT on Both Images
# ================================================================

def detect_sift_both(sift, gray, gray2):
    """
    Run SIFT on the original and resized grayscale images.

    Args:
        sift  (cv2.SIFT): SIFT detector (created in Section 2).
        gray  (np.ndarray): Original grayscale image.
        gray2 (np.ndarray): Resized grayscale image.

    Returns:
        tuple: kp1, des1, kp2, des2
    """
    kp1, des1 = sift.detectAndCompute(gray,  None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    print(f"\n  SIFT on Both Images:")
    print(f"    Original Image Keypoints : {len(kp1) if kp1 else 0}")
    print(f"    Second Image Keypoints   : {len(kp2) if kp2 else 0}")

    return kp1, des1, kp2, des2


# ================================================================
# SECTION 6 – Brute-Force Matching + Lowe's Ratio Test
# ================================================================

def match_features(des1, des2):
    """
    Match SIFT descriptors between two images using Brute-Force k-NN
    matching and filter with Lowe's ratio test.

    How it works:
        1. For each descriptor in image 1, find the 2 closest descriptors
           in image 2 (k=2).
        2. Lowe's ratio test: keep a match only if the closest distance
           is significantly smaller than the second-closest distance
           (threshold = 0.75).  This rejects ambiguous matches.

    Args:
        des1 (np.ndarray): Descriptors from image 1  (shape: N×128).
        des2 (np.ndarray): Descriptors from image 2  (shape: M×128).

    Returns:
        list[cv2.DMatch]: Filtered good matches after ratio test.
    """
    # ---- Guard: missing descriptors ----
    if des1 is None or des2 is None:
        print("  WARNING: One or both descriptor sets are empty. "
              "Skipping matching.")
        return []

    if len(des1) < 2 or len(des2) < 2:
        print("  WARNING: Not enough descriptors for k-NN matching.")
        return []

    # Create a Brute-Force matcher
    bf = cv2.BFMatcher()

    # k=2: for each query descriptor, find the 2 nearest neighbours
    matches = bf.knnMatch(des1, des2, k=2)

    # ---- Lowe's ratio test ----
    good_matches = []
    for match_pair in matches:
        # Guard against pairs that don't have 2 neighbours
        if len(match_pair) < 2:
            continue
        m, n = match_pair
        # Keep only unambiguous matches
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"\n  Feature Matching Results:")
    print(f"    Total matches (before ratio test) : {len(matches)}")
    print(f"    Good matches  (after ratio test)  : {len(good_matches)}")

    return good_matches


# ================================================================
# SECTION 7 – Visualize SIFT Feature Matching
# ================================================================

def visualize_matches(img, kp1, img2, kp2, good_matches):
    """
    Draw lines between matching keypoints in the original and resized
    images, then display the result.

    Shows up to 50 matches (or fewer if not enough are available).

    Args:
        img         (np.ndarray): Original BGR image.
        kp1         (list)      : Keypoints from image 1.
        img2        (np.ndarray): Resized BGR image.
        kp2         (list)      : Keypoints from image 2.
        good_matches(list)      : Filtered good matches.

    Returns:
        np.ndarray: Matched image in BGR format (for saving).
    """
    if not good_matches:
        print("  WARNING: No good matches to visualize.")
        return None

    # Use at most 50 matches for a clean visualization
    draw_matches = good_matches[:50]

    matched_image = cv2.drawMatches(
        img,
        kp1,
        img2,
        kp2,
        draw_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Convert BGR → RGB for Matplotlib
    matched_image_rgb = cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.imshow(matched_image_rgb)
    ax.set_title(
        f"SIFT Feature Matching  (showing {len(draw_matches)} of "
        f"{len(good_matches)} good matches)",
        fontsize=14, fontweight="bold", pad=10,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"  ✔  Feature matching visualization done "
          f"({len(draw_matches)} matches drawn).")
    return matched_image   # BGR for cv2.imwrite


# ================================================================
# SECTION 8 – Display Original Colour Image (final view)
# ================================================================

def display_original_color(img_rgb):
    """
    Display the original colour image with the title
    'Original Colour Image' as the final standalone figure.

    Args:
        img_rgb (np.ndarray): RGB image.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(img_rgb)
    ax.set_title("Original Colour Image", fontsize=14,
                 fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
    print("  ✔  Original colour image displayed.")


# ================================================================
# SECTION 9 – Save Output Images
# ================================================================

def save_images(img, img_rgb, sift_image_bgr, hog_image_rescaled,
                img2, matched_image, input_path):
    """
    Save all required output images to an 'output/' directory located
    next to the input image.

    Files saved:
        original_image.jpg
        sift_keypoints.jpg
        hog_visualization.jpg
        resized_image.jpg
        sift_feature_matching.jpg

    Args:
        img              (np.ndarray): Original BGR image.
        img_rgb          (np.ndarray): Original RGB image (not saved directly;
                                       kept for reference).
        sift_image_bgr   (np.ndarray): SIFT keypoints drawn on BGR image.
        hog_image_rescaled(np.ndarray): HOG visualization (float, rescaled).
        img2             (np.ndarray): Resized BGR image.
        matched_image    (np.ndarray): Feature-matching result (BGR).
        input_path       (str)       : Path of the original input image.

    Returns:
        str: Absolute path to the output directory.
    """
    input_dir  = os.path.dirname(os.path.abspath(input_path))
    output_dir = os.path.join(input_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  Saving output images to: {output_dir}")

    # ---- Convert HOG float image → uint8 for imwrite ----
    hog_uint8 = np.uint8(
        cv2.normalize(hog_image_rescaled, None, 0, 255, cv2.NORM_MINMAX)
    )

    files_to_save = [
        ("original_image.jpg",         img),
        ("sift_keypoints.jpg",         sift_image_bgr),
        ("hog_visualization.jpg",      hog_uint8),
        ("resized_image.jpg",          img2),
        ("sift_feature_matching.jpg",  matched_image),
    ]

    all_saved = True
    for filename, image in files_to_save:
        if image is None:
            print(f"    ⚠  Skipped (no data): {filename}")
            continue
        save_path = os.path.join(output_dir, filename)
        success   = cv2.imwrite(save_path, image)
        if success:
            print(f"    ✔  Saved: {filename}")
        else:
            print(f"    ✘  FAILED to save: {filename}")
            all_saved = False

    if all_saved:
        print("\n  All output files saved successfully.")
    else:
        print("\n  WARNING: Some files could not be saved.  "
              "Check the path and permissions.")

    return output_dir


# ================================================================
# MAIN PROGRAM
# ================================================================

def main():
    """
    Entry point – orchestrates Experiment 5 step by step:

        Step 1 : Load image → BGR, RGB, grayscale
        Step 2 : SIFT keypoint detection and descriptor extraction
        Step 3 : HOG feature extraction and visualization
        Step 4 : Create 1.2× resized image
        Step 5 : SIFT on both images
        Step 6 : Brute-Force k-NN matching + Lowe's ratio test
        Step 7 : Visualize SIFT feature matches
        Step 8 : Display original colour image
        Step 9 : Save output images
    """

    # ---- Step 1: Load image ----------------------------------------
    img, img_rgb, gray, image_path = load_image()

    # ---- Step 2: SIFT keypoint detection ---------------------------
    print("\nStep 2: SIFT keypoint detection and descriptor extraction …")
    keypoints, descriptors, sift = apply_sift(img, img_rgb, gray)

    # Convert SIFT keypoint image back to BGR for saving
    sift_image_bgr = cv2.drawKeypoints(
        img, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    ) if keypoints else img.copy()

    # ---- Step 3: HOG feature extraction ----------------------------
    print("\nStep 3: HOG feature extraction and visualization …")
    hog_features, hog_image_rescaled = apply_hog(img_rgb, gray)

    # ---- Step 4: Create 1.2× resized image -------------------------
    print("\nStep 4: Creating 1.2× resized second image …")
    img2, img2_rgb, gray2 = create_resized_image(img, img_rgb)

    # ---- Step 5: SIFT on both images --------------------------------
    print("\nStep 5: Detecting SIFT features in both images …")
    kp1, des1, kp2, des2 = detect_sift_both(sift, gray, gray2)

    # ---- Step 6: Feature matching -----------------------------------
    print("\nStep 6: Brute-Force k-NN matching with Lowe's ratio test …")
    good_matches = match_features(des1, des2)

    # ---- Step 7: Visualize matches ----------------------------------
    print("\nStep 7: Visualizing SIFT feature matches …")
    matched_image = visualize_matches(img, kp1, img2, kp2, good_matches)

    # ---- Step 8: Display original colour image ----------------------
    print("\nStep 8: Displaying original colour image …")
    display_original_color(img_rgb)

    # ---- Step 9: Save output images ---------------------------------
    print("\nStep 9: Saving output images …")
    output_dir = save_images(
        img, img_rgb, sift_image_bgr,
        hog_image_rescaled, img2, matched_image, image_path,
    )

    # ---- Done -------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Experiment 5 completed successfully!")
    print(f"  Output images saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
