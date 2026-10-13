"""
============================================================
Experiment 6: Image Segmentation Techniques Using
              OpenCV and K-Means Clustering
============================================================
Objective:
    Implement multiple image segmentation techniques using
    Python, OpenCV, NumPy, scikit-learn, and Matplotlib.

Techniques covered:
    1.  Load image (colour, grayscale)
    2.  Grayscale conversion + Gaussian blur
    3.  Global Thresholding      (fixed threshold = 127)
    4.  Otsu's Thresholding      (automatic threshold)
    5.  Adaptive Thresholding    (local Gaussian threshold)
    6.  Watershed Segmentation   (marker-based)
    7.  K-Means Color Segmentation (K = 3, scikit-learn)
    8.  Combined 4×2 comparison grid
    9.  Save individual results (8 PNG files)
   10.  Save combined figure (Experiment_6_All_Results.png)
   11.  Create ZIP archive      (Experiment_6_Results.zip)
   12.  Print experiment summary

Libraries used:
    - opencv-python  (cv2)             : image I/O, thresholding, watershed
    - numpy          (np)              : array operations
    - matplotlib     (plt)             : display and figure saving
    - scikit-learn   (sklearn)         : K-Means clustering
    - zipfile                          : ZIP archive creation
    - tkinter                          : file-picker dialog
    - os                               : directory / path utilities
============================================================
"""

import os
import sys
import zipfile


# ================================================================
# SECTION 0 – Dependency Check
# ================================================================

def check_dependencies():
    """
    Verify that all required third-party libraries are installed.
    Prints installation instructions and exits if anything is missing.
    """
    missing = []
    checks = [
        ("cv2",        "opencv-python"),
        ("numpy",      "numpy"),
        ("matplotlib", "matplotlib"),
        ("sklearn",    "scikit-learn"),
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

check_dependencies()

# --------------- Imports (guaranteed available after check) ---------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# ================================================================
# SECTION 1 – Image Loading
# ================================================================

def pick_image_path():
    """
    Open a tkinter file-picker dialog so the user can browse for an image.
    Returns the selected path, or '' if cancelled / tkinter is unavailable.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        path = filedialog.askopenfilename(
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
        return path

    except Exception:
        return ""


def load_image():
    """
    Prompt the user to select or type an image path, then load it.
    Returns BGR image, RGB image, grayscale image, and the file path.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, str]:
            img (BGR), img_rgb, gray, image_path
    """
    print("=" * 60)
    print("  Experiment 6 – Image Segmentation Techniques")
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

    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: OpenCV could not read the image → {image_path}")
        sys.exit(1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = img.shape[:2]
    print(f"\n  Image loaded successfully!")
    print(f"  Path       : {image_path}")
    print(f"  Dimensions : {img.shape}  (H × W × C)")
    print(f"  Width      : {width} px  |  Height : {height} px")

    # Display the original colour image
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(img_rgb)
    ax.set_title("Original Image", fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    return img, img_rgb, gray, image_path


# ================================================================
# SECTION 2 – Grayscale and Gaussian Blur
# ================================================================

def grayscale_and_blur(gray):
    """
    Apply Gaussian blur to the grayscale image.

    Gaussian blur reduces noise before thresholding, which avoids
    creating many tiny spurious segments from isolated noisy pixels.
    Kernel size 15×15 with sigma=0 (auto-computed).

    Args:
        gray (np.ndarray): Grayscale uint8 image.

    Returns:
        np.ndarray: Gaussian-blurred grayscale image.
    """
    blur = cv2.GaussianBlur(gray, (15, 15), 0)

    # Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Grayscale and Gaussian Blur", fontsize=14, fontweight="bold")

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Original Grayscale Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(blur, cmap="gray")
    axes[1].set_title("Gaussian Blurred Image", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    print("  ✔  Gaussian blur applied (kernel 15×15).")
    return blur


# ================================================================
# SECTION 3 – Global Thresholding
# ================================================================

def apply_global_threshold(blur):
    """
    Apply binary thresholding with a fixed threshold value of 127.

    Pixels above 127 become white (255); pixels at or below 127
    become black (0).  This is a simple segmentation method that
    works well when the image has a clear bimodal intensity distribution.

    Args:
        blur (np.ndarray): Gaussian-blurred grayscale image.

    Returns:
        np.ndarray: Binary threshold image.
    """
    ret, global_thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

    print(f"\n  Global Thresholding:")
    print(f"    Fixed threshold value used : {int(ret)}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(global_thresh, cmap="gray")
    ax.set_title("Global Thresholding", fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    print("  ✔  Global thresholding done.")
    return global_thresh, int(ret)


# ================================================================
# SECTION 4 – Otsu's Thresholding
# ================================================================

def apply_otsu_threshold(blur):
    """
    Apply Otsu's automatic thresholding.

    Otsu's method analyses the image histogram and selects the
    threshold that minimises within-class variance (or maximises
    between-class variance).  It is ideal for images with a clear
    bimodal histogram (two intensity peaks).

    Args:
        blur (np.ndarray): Gaussian-blurred grayscale image.

    Returns:
        tuple[np.ndarray, float]: Binary image and the Otsu threshold value.
    """
    otsu_value, otsu_thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    print(f"\n  Otsu's Thresholding:")
    print(f"    Automatically selected threshold : {otsu_value:.1f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(otsu_thresh, cmap="gray")
    ax.set_title("Otsu's Thresholding", fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    print("  ✔  Otsu's thresholding done.")
    return otsu_thresh, otsu_value


# ================================================================
# SECTION 5 – Adaptive Thresholding
# ================================================================

def apply_adaptive_threshold(blur):
    """
    Apply Gaussian adaptive thresholding.

    Instead of using a single global threshold, adaptive thresholding
    calculates a different threshold for each small rectangular region
    (block) of the image.  This handles uneven illumination well.

    Parameters (from PDF):
        maxValue    = 255                       (output intensity for foreground)
        adaptiveMethod = ADAPTIVE_THRESH_GAUSSIAN_C  (weighted mean of neighbourhood)
        thresholdType  = THRESH_BINARY
        blockSize   = 11                        (neighbourhood size, must be odd)
        C           = 2                         (constant subtracted from the mean)

    Args:
        blur (np.ndarray): Gaussian-blurred grayscale image.

    Returns:
        np.ndarray: Adaptively thresholded binary image.
    """
    adaptive_thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(adaptive_thresh, cmap="gray")
    ax.set_title("Adaptive Thresholding", fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    print("  ✔  Adaptive thresholding done.")
    return adaptive_thresh


# ================================================================
# SECTION 6 – Watershed Segmentation
# ================================================================

def apply_watershed(img, gray):
    """
    Segment the image using the marker-based Watershed algorithm.

    Concept:
        The image is treated like a topographic map.  Flooding starts
        from labelled "marker" regions.  Where flood regions from
        different markers would meet, watershed "boundaries" are drawn.

    Steps:
        A. Otsu threshold (inverted) → binary foreground mask
        B. Morphological opening → remove noise
        C. Dilation → sure background
        D. Distance transform → distance of each pixel from background
        E. Threshold distance map → sure foreground (definite objects)
        F. Subtract → unknown region (boundary candidates)
        G. Connected components → labelled markers
        H. cv2.watershed() draws boundaries (marked as -1)

    Args:
        img  (np.ndarray): Original BGR image.
        gray (np.ndarray): Grayscale image.

    Returns:
        np.ndarray: Watershed result as an RGB image with red boundaries.
    """
    # A. Binary mask (inverted Otsu so objects are white)
    _, thresh_ws = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # B. Morphological opening to remove small noise (2 iterations)
    kernel  = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh_ws, cv2.MORPH_OPEN, kernel, iterations=2)

    # C. Sure background — dilated region contains all objects
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # D. Distance transform — value = distance of each pixel from background
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # E. Sure foreground — pixels far enough from any background region
    max_dist = dist_transform.max()
    if max_dist > 0:
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * max_dist, 255, 0)
    else:
        # Flat or uniform image — no meaningful foreground
        print("  WARNING: Distance transform is all-zero. "
              "Watershed may not segment this image well.")
        sure_fg = np.zeros_like(dist_transform)

    sure_fg = np.uint8(sure_fg)

    # F. Unknown region (between sure foreground and sure background)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # G. Connected components → labelled markers
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1           # shift so background = 1 (not 0)
    markers[unknown == 255] = 0     # unknown region = 0 (to be decided by watershed)

    # H. Apply watershed on a copy of the original BGR image
    watershed_bgr = img.copy()
    markers = cv2.watershed(watershed_bgr, markers)

    # Convert to RGB and paint watershed boundaries red
    watershed_rgb = cv2.cvtColor(watershed_bgr, cv2.COLOR_BGR2RGB)
    watershed_rgb[markers == -1] = [255, 0, 0]   # boundary pixels → red

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(watershed_rgb)
    ax.set_title("Watershed Segmentation", fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"  ✔  Watershed segmentation done  "
          f"(found {num_labels - 1} component(s) before boundaries).")
    return watershed_rgb


# ================================================================
# SECTION 7 – K-Means Color Segmentation
# ================================================================

def apply_kmeans(img_rgb, K=3):
    """
    Segment the image into K color regions using K-Means clustering.

    K-Means groups all pixels into K clusters based on their RGB values.
    Each pixel is then replaced by the mean color of its cluster,
    producing a simplified colour-segmented image.

    Processing at reduced resolution (500 px wide) keeps computation
    fast without changing the segmentation quality significantly.

    Parameters (from PDF):
        K            = 3        (number of color clusters)
        random_state = 42       (for reproducibility)
        n_init       = 10       (run K-Means 10 times, keep best)
        scale_width  = 500 px

    Args:
        img_rgb (np.ndarray): Original RGB image.
        K       (int)       : Number of clusters.

    Returns:
        np.ndarray: K-Means segmented RGB image (resized dimensions).
    """
    # A. Resize to 500 px wide, preserve aspect ratio
    scale_width  = 500
    scale_height = int(img_rgb.shape[0] * scale_width / img_rgb.shape[1])
    small_img    = cv2.resize(img_rgb, (scale_width, scale_height))

    # B. Flatten pixels into rows of [R, G, B]
    pixel_data = small_img.reshape((-1, 3))
    pixel_data = np.float32(pixel_data)

    # C. Fit K-Means
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixel_data)

    # D. Replace each pixel with its cluster centre colour
    centers       = np.uint8(kmeans.cluster_centers_)
    segmented_data = centers[labels]
    kmeans_image  = segmented_data.reshape(small_img.shape)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(kmeans_image)
    ax.set_title(f"K-Means Segmentation (K={K})",
                 fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"  ✔  K-Means segmentation done  (K={K}, image resized to "
          f"{scale_width}×{scale_height}).")
    return kmeans_image


# ================================================================
# SECTION 8 – Combined 4×2 Comparison Grid
# ================================================================

def plot_combined_grid(img_rgb, gray, blur, global_thresh,
                       otsu_thresh, adaptive_thresh,
                       watershed_rgb, kmeans_image):
    """
    Display all eight results in a 4-row × 2-column grid.

    Layout:
        Row 1 : Original Image          |  Grayscale Image
        Row 2 : Gaussian Blurred Image  |  Global Thresholding
        Row 3 : Otsu's Thresholding     |  Adaptive Thresholding
        Row 4 : Watershed Segmentation  |  K-Means Segmentation (K=3)

    Args:
        img_rgb       : Original RGB image
        gray          : Grayscale image
        blur          : Gaussian-blurred grayscale image
        global_thresh : Global binary threshold image
        otsu_thresh   : Otsu's binary threshold image
        adaptive_thresh: Adaptive threshold image
        watershed_rgb : Watershed result (RGB with red boundaries)
        kmeans_image  : K-Means segmented image (RGB)

    Returns:
        matplotlib.figure.Figure: The combined figure (for saving).
    """
    grid_items = [
        (img_rgb,        "Original Image",               None),
        (gray,           "Grayscale Image",              "gray"),
        (blur,           "Gaussian Blurred Image",       "gray"),
        (global_thresh,  "Global Thresholding",          "gray"),
        (otsu_thresh,    "Otsu's Thresholding",          "gray"),
        (adaptive_thresh,"Adaptive Thresholding",        "gray"),
        (watershed_rgb,  "Watershed Segmentation",       None),
        (kmeans_image,   "K-Means Segmentation (K=3)",  None),
    ]

    fig = plt.figure(figsize=(18, 20))
    fig.suptitle(
        "Experiment 6 – Image Segmentation Techniques",
        fontsize=16, fontweight="bold", y=1.01,
    )

    for idx, (image, title, cmap) in enumerate(grid_items, start=1):
        ax = plt.subplot(4, 2, idx)
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    print("  ✔  Combined 4×2 comparison grid displayed.")
    return fig


# ================================================================
# SECTION 9 – Save Individual Images
# ================================================================

def save_individual_images(img_rgb, gray, blur, global_thresh,
                           otsu_thresh, adaptive_thresh,
                           watershed_rgb, kmeans_image,
                           output_dir):
    """
    Save all eight segmentation results as individual 300-DPI PNG files.

    Filenames match the PDF's output section exactly:
        01_Original.png
        02_Grayscale.png
        03_Gaussian_Blur.png
        04_Global_Threshold.png
        05_Otsu_Threshold.png
        06_Adaptive_Threshold.png
        07_Watershed.png
        08_KMeans.png

    Args:
        (all segmentation arrays)
        output_dir (str): Directory where files are saved.

    Returns:
        list[str]: Absolute paths of all saved files.
    """
    def save_rgb(image, filename, title=""):
        """Save an RGB image to a PNG file at 300 DPI."""
        path = os.path.join(output_dir, filename)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(image)
        if title:
            ax.set_title(title, fontsize=12)
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    def save_gray(image, filename, title=""):
        """Save a grayscale image to a PNG file at 300 DPI."""
        path = os.path.join(output_dir, filename)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(image, cmap="gray")
        if title:
            ax.set_title(title, fontsize=12)
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    files_spec = [
        (img_rgb,         "01_Original.png",         "Original Image",              "rgb"),
        (gray,            "02_Grayscale.png",         "Grayscale Image",             "gray"),
        (blur,            "03_Gaussian_Blur.png",     "Gaussian Blurred Image",      "gray"),
        (global_thresh,   "04_Global_Threshold.png",  "Global Thresholding",         "gray"),
        (otsu_thresh,     "05_Otsu_Threshold.png",    "Otsu's Thresholding",         "gray"),
        (adaptive_thresh, "06_Adaptive_Threshold.png","Adaptive Thresholding",       "gray"),
        (watershed_rgb,   "07_Watershed.png",         "Watershed Segmentation",      "rgb"),
        (kmeans_image,    "08_KMeans.png",            "K-Means Segmentation (K=3)",  "rgb"),
    ]

    saved_paths = []
    all_saved   = True

    for image, filename, title, mode in files_spec:
        try:
            if mode == "rgb":
                path = save_rgb(image, filename, title)
            else:
                path = save_gray(image, filename, title)
            print(f"    ✔  Saved: {filename}")
            saved_paths.append(path)
        except Exception as e:
            print(f"    ✘  FAILED to save {filename}: {e}")
            all_saved = False

    if not all_saved:
        print("  WARNING: Some individual images could not be saved.")

    return saved_paths


# ================================================================
# SECTION 10 – Save Combined Figure
# ================================================================

def save_combined_figure(img_rgb, gray, blur, global_thresh,
                         otsu_thresh, adaptive_thresh,
                         watershed_rgb, kmeans_image,
                         output_dir):
    """
    Render and save the 4×2 combined comparison figure at 300 DPI.

    Output filename: Experiment_6_All_Results.png

    Args:
        (all segmentation arrays)
        output_dir (str): Directory where the file is saved.

    Returns:
        str: Absolute path of the saved combined figure.
    """
    grid_items = [
        (img_rgb,         "Original Image",               None),
        (gray,            "Grayscale Image",              "gray"),
        (blur,            "Gaussian Blurred Image",       "gray"),
        (global_thresh,   "Global Thresholding",          "gray"),
        (otsu_thresh,     "Otsu's Thresholding",          "gray"),
        (adaptive_thresh, "Adaptive Thresholding",        "gray"),
        (watershed_rgb,   "Watershed Segmentation",       None),
        (kmeans_image,    "K-Means Segmentation (K=3)",  None),
    ]

    fig = plt.figure(figsize=(18, 20))
    fig.suptitle(
        "Experiment 6 – Image Segmentation Techniques",
        fontsize=16, fontweight="bold", y=1.01,
    )

    for idx, (image, title, cmap) in enumerate(grid_items, start=1):
        ax = plt.subplot(4, 2, idx)
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()

    combined_path = os.path.join(output_dir, "Experiment_6_All_Results.png")
    fig.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"    ✔  Saved: Experiment_6_All_Results.png")
    return combined_path


# ================================================================
# SECTION 11 – Create ZIP Archive
# ================================================================

def create_zip_archive(combined_path, individual_paths, output_dir):
    """
    Bundle the combined figure and all eight individual images into a
    single ZIP file named Experiment_6_Results.zip.

    Because this is a local program (not Google Colab), the ZIP is
    saved to disk and its absolute path is printed so you can open it
    directly from your file manager.

    Args:
        combined_path    (str)      : Path to Experiment_6_All_Results.png.
        individual_paths (list[str]): Paths to the eight individual PNGs.
        output_dir       (str)      : Directory where ZIP is written.

    Returns:
        str: Absolute path of the created ZIP file.
    """
    zip_path   = os.path.join(output_dir, "Experiment_6_Results.zip")
    all_files  = [combined_path] + individual_paths

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in all_files:
            if os.path.isfile(file_path):
                zf.write(file_path, os.path.basename(file_path))
            else:
                print(f"  WARNING: File not found for ZIP → {file_path}")

    # Verify archive contents
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

    expected = 9  # 1 combined + 8 individual
    print(f"\n  ZIP archive created: {zip_path}")
    print(f"  Files in archive   : {len(names)} / {expected} expected")
    for name in names:
        print(f"    • {name}")

    if len(names) < expected:
        print("  WARNING: Archive is missing some files.")

    return zip_path


# ================================================================
# SECTION 12 – Print Experiment Summary
# ================================================================

def print_summary(img, global_thresh_value, otsu_value,
                  output_dir, individual_paths, zip_path, K=3):
    """
    Print a clear experiment summary at the end of the program.

    Args:
        img                 (np.ndarray): Original BGR image.
        global_thresh_value (int)       : Fixed threshold value used.
        otsu_value          (float)     : Otsu's automatically selected value.
        output_dir          (str)       : Output directory path.
        individual_paths    (list[str]) : Saved individual image paths.
        zip_path            (str)       : Path of the ZIP archive.
        K                   (int)       : Number of K-Means clusters.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 6 – SUMMARY")
    print("=" * 60)
    print(f"  Status                   : Completed successfully")
    print(f"  Original image dimensions: {img.shape}")
    print(f"  Fixed global threshold   : {global_thresh_value}")
    print(f"  Otsu's threshold         : {otsu_value:.1f} (automatic)")
    print(f"  K-Means clusters         : K = {K}")
    print(f"\n  Output directory         : {output_dir}")
    print(f"\n  Generated image files:")
    for path in individual_paths:
        print(f"    • {os.path.basename(path)}")
    print(f"    • Experiment_6_All_Results.png")
    print(f"\n  ZIP archive              : {zip_path}")
    print("=" * 60)


# ================================================================
# MAIN PROGRAM
# ================================================================

def main():
    """
    Entry point – orchestrates Experiment 6 step by step:

        Step 1 : Load image
        Step 2 : Grayscale + Gaussian blur
        Step 3 : Global thresholding
        Step 4 : Otsu's thresholding
        Step 5 : Adaptive thresholding
        Step 6 : Watershed segmentation
        Step 7 : K-Means color segmentation
        Step 8 : Combined 4×2 comparison grid
        Step 9 : Save individual images
        Step 10: Save combined figure
        Step 11: Create ZIP archive
        Step 12: Print experiment summary
    """

    # ---- Step 1: Load image ----------------------------------------
    img, img_rgb, gray, image_path = load_image()

    # Set up output directory next to the input image
    input_dir  = os.path.dirname(os.path.abspath(image_path))
    output_dir = os.path.join(input_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n  Output directory: {output_dir}")

    # ---- Step 2: Grayscale + Gaussian blur -------------------------
    print("\nStep 2: Grayscale conversion and Gaussian blur …")
    blur = grayscale_and_blur(gray)

    # ---- Step 3: Global thresholding --------------------------------
    print("\nStep 3: Applying global thresholding …")
    global_thresh, global_thresh_value = apply_global_threshold(blur)

    # ---- Step 4: Otsu's thresholding --------------------------------
    print("\nStep 4: Applying Otsu's thresholding …")
    otsu_thresh, otsu_value = apply_otsu_threshold(blur)

    # ---- Step 5: Adaptive thresholding ------------------------------
    print("\nStep 5: Applying adaptive thresholding …")
    adaptive_thresh = apply_adaptive_threshold(blur)

    # ---- Step 6: Watershed segmentation -----------------------------
    print("\nStep 6: Applying Watershed segmentation …")
    watershed_rgb = apply_watershed(img, gray)

    # ---- Step 7: K-Means segmentation -------------------------------
    print("\nStep 7: Applying K-Means color segmentation (K=3) …")
    K = 3
    kmeans_image = apply_kmeans(img_rgb, K=K)

    # ---- Step 8: Combined comparison grid ---------------------------
    print("\nStep 8: Displaying combined 4×2 comparison grid …")
    plot_combined_grid(img_rgb, gray, blur, global_thresh,
                       otsu_thresh, adaptive_thresh,
                       watershed_rgb, kmeans_image)

    # ---- Step 9: Save individual images -----------------------------
    print("\nStep 9: Saving individual images …")
    print(f"  Saving to: {output_dir}")
    individual_paths = save_individual_images(
        img_rgb, gray, blur, global_thresh,
        otsu_thresh, adaptive_thresh,
        watershed_rgb, kmeans_image,
        output_dir,
    )

    # ---- Step 10: Save combined figure ------------------------------
    print("\nStep 10: Saving combined comparison figure …")
    combined_path = save_combined_figure(
        img_rgb, gray, blur, global_thresh,
        otsu_thresh, adaptive_thresh,
        watershed_rgb, kmeans_image,
        output_dir,
    )

    # ---- Step 11: Create ZIP archive --------------------------------
    print("\nStep 11: Creating ZIP archive …")
    zip_path = create_zip_archive(combined_path, individual_paths, output_dir)

    # ---- Step 12: Print summary -------------------------------------
    print_summary(img, global_thresh_value, otsu_value,
                  output_dir, individual_paths, zip_path, K=K)


if __name__ == "__main__":
    main()
