"""
============================================================
Experiment 3: Image Smoothing and Edge Detection Using OpenCV
============================================================
Objective:
    Implement image smoothing (blurring) and edge detection
    techniques using Python, OpenCV, NumPy, and Matplotlib.

Operations covered:
    1. Image Loading and Grayscale Conversion
    2. Gaussian Blur
    3. Median Filter
    4. Average Filter
    5. Laplacian Edge Detection
    6. Sobel X and Sobel Y Edge Detection
    7. Sobel Combined Edge Map
    8. Final 2×4 Comparison Grid
    9. Observations
   10. Saving Output Images

Libraries used:
    - opencv-python  (cv2)  : image I/O and processing
    - numpy          (np)   : numerical / array operations
    - matplotlib     (plt)  : displaying images
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
# SECTION 1 – Image Loading
# ================================================================

def pick_image_path():
    """
    Open a tkinter file-picker dialog so the user can browse for an
    image.  Returns the selected path, or an empty string if cancelled
    or if tkinter is not available on this system.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()                    # hide the empty root window
        root.attributes("-topmost", True)  # bring the dialog to the front

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
        return ""  # tkinter unavailable — caller will fall back to manual input


def load_image():
    """
    Prompt the user to select or type an image path, then load it with
    OpenCV.  Prints metadata on success; exits with a clear message on
    failure.

    Returns:
        tuple[np.ndarray, str]: BGR image array and its absolute path.
    """
    print("=" * 60)
    print("  Experiment 3 – Image Smoothing and Edge Detection")
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

    # ---- Validate the path ----
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

    # ---- Load with OpenCV ----
    img = cv2.imread(image_path)

    if img is None:
        print(f"ERROR: OpenCV could not read the image → {image_path}")
        print("       The file may be corrupt or in an unsupported format.")
        sys.exit(1)

    # ---- Print image metadata ----
    height, width = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    print(f"\n  Image loaded successfully!")
    print(f"  Path      : {image_path}")
    print(f"  Width     : {width} px")
    print(f"  Height    : {height} px")
    print(f"  Dimensions: {img.shape}")
    print(f"  Channels  : {channels}")
    print(f"  Data type : {img.dtype}")

    return img, image_path


# ================================================================
# SECTION 2 – Helper: BGR → RGB for Matplotlib
# ================================================================

def bgr_to_rgb(img):
    """
    Convert a BGR image (OpenCV default) to RGB so Matplotlib
    renders colours correctly.

    Args:
        img (np.ndarray): BGR image.

    Returns:
        np.ndarray: RGB image.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ================================================================
# SECTION 3 – Grayscale Conversion
# ================================================================

def convert_to_grayscale(img):
    """
    Convert the BGR image to a single-channel grayscale image.
    Also displays the original colour image on its own figure.

    Args:
        img (np.ndarray): Original BGR image.

    Returns:
        np.ndarray: Grayscale image (uint8).
    """
    # Display the original image (BGR → RGB for correct Matplotlib colours)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(bgr_to_rgb(img))
    ax.set_title("Original Image", fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("  ✔  Grayscale conversion done.")
    return gray


# ================================================================
# SECTION 4 – Gaussian Blur
# ================================================================

def apply_gaussian_blur(img):
    """
    Smooth the image using a Gaussian kernel.

    A Gaussian filter gives more weight to pixels near the centre of
    the kernel, producing a natural-looking blur that reduces noise.

    Kernel size: 51 × 51  (must be odd, as required by OpenCV)
    Sigma      : 0         (OpenCV calculates it automatically)

    Args:
        img (np.ndarray): Original BGR image.

    Returns:
        np.ndarray: Gaussian-blurred BGR image.
    """
    gaussian = cv2.GaussianBlur(img, (51, 51), 0)

    # Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Gaussian Blur", fontsize=14, fontweight="bold")

    axes[0].imshow(bgr_to_rgb(img))
    axes[0].set_title("Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(bgr_to_rgb(gaussian))
    axes[1].set_title("Gaussian Blur", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    print("  ✔  Gaussian blur applied.")
    return gaussian


# ================================================================
# SECTION 5 – Median Filter
# ================================================================

def apply_median_filter(img):
    """
    Smooth the image using a median filter.

    For each pixel, the median filter replaces its value with the
    median of the values in the surrounding 31×31 neighbourhood.
    This is particularly effective against salt-and-pepper noise
    because the median is resistant to outlier values.

    Kernel size: 31  (must be a positive odd integer)

    Args:
        img (np.ndarray): Original BGR image.

    Returns:
        np.ndarray: Median-filtered BGR image.
    """
    median = cv2.medianBlur(img, 31)

    # Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Median Filter", fontsize=14, fontweight="bold")

    axes[0].imshow(bgr_to_rgb(img))
    axes[0].set_title("Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(bgr_to_rgb(median))
    axes[1].set_title("Median Filter", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    print("  ✔  Median filter applied.")
    return median


# ================================================================
# SECTION 6 – Average Filter
# ================================================================

def apply_average_filter(img):
    """
    Smooth the image using an averaging (box) filter.

    Each output pixel is the simple average of all pixels in a
    25×25 neighbourhood.  This reduces noise but can blur edges.

    Kernel size: 25 × 25

    Args:
        img (np.ndarray): Original BGR image.

    Returns:
        np.ndarray: Average-filtered BGR image.
    """
    average = cv2.blur(img, (25, 25))

    # Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Average Filter", fontsize=14, fontweight="bold")

    axes[0].imshow(bgr_to_rgb(img))
    axes[0].set_title("Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(bgr_to_rgb(average))
    axes[1].set_title("Average Filter", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    print("  ✔  Average filter applied.")
    return average


# ================================================================
# SECTION 7 – Laplacian Edge Detection
# ================================================================

def apply_laplacian(img, gray):
    """
    Detect edges using the Laplacian operator.

    The Laplacian computes the second spatial derivative of the image.
    Regions where intensity changes rapidly (edges) produce large
    Laplacian values.

    Steps:
        1. Compute Laplacian on the grayscale image using 64-bit floats
           (cv2.CV_64F) so negative values are not clipped.
        2. Convert absolute values to uint8 for display.

    Args:
        img  (np.ndarray): Original BGR image (for side-by-side display).
        gray (np.ndarray): Grayscale image.

    Returns:
        np.ndarray: Laplacian edge map (uint8).
    """
    # Compute Laplacian in 64-bit float to preserve negative edge values
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Convert absolute float values to displayable 8-bit image
    laplacian = cv2.convertScaleAbs(laplacian)

    # Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Laplacian Edge Detection", fontsize=14, fontweight="bold")

    axes[0].imshow(bgr_to_rgb(img))
    axes[0].set_title("Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(laplacian, cmap="gray")
    axes[1].set_title("Laplacian", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    print("  ✔  Laplacian edge detection done.")
    return laplacian


# ================================================================
# SECTION 8 – Sobel Edge Detection
# ================================================================

def apply_sobel(gray):
    """
    Detect edges using the Sobel operator in X and Y directions.

    Sobel X (ksize=3) computes the horizontal gradient → emphasises
    vertical edges (intensity changes left-to-right).

    Sobel Y (ksize=3) computes the vertical gradient → emphasises
    horizontal edges (intensity changes top-to-bottom).

    The combined map uses bitwise OR to merge both gradient directions
    into one comprehensive edge map.

    Args:
        gray (np.ndarray): Grayscale image.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            sobel_x, sobel_y, sobel_combined — all uint8.
    """
    # Sobel X – horizontal gradient (detects vertical edges)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x)

    # Sobel Y – vertical gradient (detects horizontal edges)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    # Combined edge map – union of both gradient directions
    sobel_combined = cv2.bitwise_or(sobel_x, sobel_y)

    # Display all three in a single row
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Sobel Edge Detection", fontsize=14, fontweight="bold")

    axes[0].imshow(sobel_x, cmap="gray")
    axes[0].set_title("Sobel X", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(sobel_y, cmap="gray")
    axes[1].set_title("Sobel Y", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(sobel_combined, cmap="gray")
    axes[2].set_title("Sobel Combined", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    print("  ✔  Sobel edge detection (X, Y, Combined) done.")
    return sobel_x, sobel_y, sobel_combined


# ================================================================
# SECTION 9 – Final Comparison Grid (2 × 4)
# ================================================================

def plot_final_comparison(img, gaussian, median, average,
                          laplacian, sobel_x, sobel_y, sobel_combined):
    """
    Display a 2-row × 4-column grid comparing all processed images.

    Row 1: Original | Gaussian Blur | Median Filter | Average Filter
    Row 2: Laplacian | Sobel X      | Sobel Y       | Sobel Combined

    Color images are converted BGR → RGB.
    Edge maps are displayed in grayscale.

    Args:
        img           (np.ndarray): Original BGR image.
        gaussian      (np.ndarray): Gaussian-blurred image (BGR).
        median        (np.ndarray): Median-filtered image (BGR).
        average       (np.ndarray): Average-filtered image (BGR).
        laplacian     (np.ndarray): Laplacian edge map (grayscale uint8).
        sobel_x       (np.ndarray): Sobel X edge map (grayscale uint8).
        sobel_y       (np.ndarray): Sobel Y edge map (grayscale uint8).
        sobel_combined(np.ndarray): Combined Sobel edge map (grayscale uint8).
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    fig.suptitle(
        "Experiment 3 – Image Smoothing and Edge Detection Comparison",
        fontsize=15, fontweight="bold", y=1.01,
    )

    # ---- Row 1: Smoothing / Blurring ----
    row1 = [
        (bgr_to_rgb(img),      "Original",      None),
        (bgr_to_rgb(gaussian), "Gaussian Blur",  None),
        (bgr_to_rgb(median),   "Median Filter",  None),
        (bgr_to_rgb(average),  "Average Filter", None),
    ]
    for ax, (image, title, cmap) in zip(axes[0], row1):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    # ---- Row 2: Edge Detection ----
    row2 = [
        (laplacian,      "Laplacian",      "gray"),
        (sobel_x,        "Sobel X",        "gray"),
        (sobel_y,        "Sobel Y",        "gray"),
        (sobel_combined, "Sobel Combined", "gray"),
    ]
    for ax, (image, title, cmap) in zip(axes[1], row2):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    print("  ✔  Final 2×4 comparison grid displayed.")


# ================================================================
# SECTION 10 – Observations
# ================================================================

def print_observations():
    """
    Print clear, beginner-friendly observations about each technique,
    matching the observations section of the lab PDF.
    """
    print("\n" + "=" * 60)
    print("  Observations")
    print("=" * 60)

    observations = [
        ("Gaussian Blur",
         "Smooths the image using a weighted average that gives more\n"
         "     importance to pixels near the centre of the kernel.\n"
         "     It reduces noise and slightly softens edges."),

        ("Median Filter",
         "Replaces each pixel with the median value of its neighbourhood.\n"
         "     Very effective against salt-and-pepper noise because the\n"
         "     median is not affected by extreme outlier values.\n"
         "     It preserves edges better than simple averaging."),

        ("Average Filter",
         "Replaces each pixel with the simple arithmetic mean of its\n"
         "     neighbourhood.  Smooths the image but can blur edges."),

        ("Laplacian",
         "Computes the second-order derivative of pixel intensities.\n"
         "     Highlights regions where intensity changes rapidly —\n"
         "     i.e., it detects edges and fine details."),

        ("Sobel X",
         "Computes the horizontal (left–right) intensity gradient.\n"
         "     It emphasises vertical edges in the image."),

        ("Sobel Y",
         "Computes the vertical (top–bottom) intensity gradient.\n"
         "     It emphasises horizontal edges in the image."),

        ("Sobel Combined",
         "Merges the Sobel X and Sobel Y results using bitwise OR.\n"
         "     Produces a more complete edge map that captures both\n"
         "     horizontal and vertical edges simultaneously."),
    ]

    for technique, explanation in observations:
        print(f"\n  {technique}:")
        print(f"     {explanation}")

    print()


# ================================================================
# SECTION 11 – Save Output Images
# ================================================================

def save_images(gaussian, median, average,
                laplacian, sobel_x, sobel_y, sobel_combined,
                input_path):
    """
    Save all seven processed images to an 'output/' directory located
    next to the input image file.

    Files saved:
        gaussian_blur.jpg
        median_filter.jpg
        average_filter.jpg
        laplacian.jpg
        sobel_x.jpg
        sobel_y.jpg
        sobel_combined.jpg

    Checks whether each save succeeds and reports accordingly.
    Does NOT overwrite the original input image.

    Args:
        gaussian      (np.ndarray): Gaussian-blurred image (BGR).
        median        (np.ndarray): Median-filtered image (BGR).
        average       (np.ndarray): Average-filtered image (BGR).
        laplacian     (np.ndarray): Laplacian edge map (uint8).
        sobel_x       (np.ndarray): Sobel X edge map (uint8).
        sobel_y       (np.ndarray): Sobel Y edge map (uint8).
        sobel_combined(np.ndarray): Combined Sobel edge map (uint8).
        input_path    (str)       : Path of the original input image.

    Returns:
        str: Path to the output directory.
    """
    # Create output directory next to the input file
    input_dir  = os.path.dirname(os.path.abspath(input_path))
    output_dir = os.path.join(input_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  Saving output images to: {output_dir}")

    files_to_save = [
        ("gaussian_blur.jpg",   gaussian),
        ("median_filter.jpg",   median),
        ("average_filter.jpg",  average),
        ("laplacian.jpg",       laplacian),
        ("sobel_x.jpg",         sobel_x),
        ("sobel_y.jpg",         sobel_y),
        ("sobel_combined.jpg",  sobel_combined),
    ]

    all_saved = True
    for filename, image in files_to_save:
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
    Entry point – orchestrates Experiment 3 step by step:
        1.  Load image
        2.  Grayscale conversion + display original
        3.  Gaussian blur
        4.  Median filter
        5.  Average filter
        6.  Laplacian edge detection
        7.  Sobel X, Y, and combined edge detection
        8.  Final 2×4 comparison grid
        9.  Print observations
       10.  Save output images
    """

    # ---- Step 1: Load image ----------------------------------------
    img, image_path = load_image()

    # Keep an untouched copy of the original for all operations
    original = img.copy()

    # ---- Step 2: Grayscale conversion and original display ----------
    print("\nStep 2: Displaying original image and converting to grayscale …")
    gray = convert_to_grayscale(original)

    # ---- Step 3: Gaussian blur --------------------------------------
    print("\nStep 3: Applying Gaussian blur (kernel 51×51) …")
    gaussian = apply_gaussian_blur(original)

    # ---- Step 4: Median filter --------------------------------------
    print("\nStep 4: Applying median filter (kernel size 31) …")
    median = apply_median_filter(original)

    # ---- Step 5: Average filter -------------------------------------
    print("\nStep 5: Applying average filter (kernel 25×25) …")
    average = apply_average_filter(original)

    # ---- Step 6: Laplacian edge detection ---------------------------
    print("\nStep 6: Applying Laplacian edge detection …")
    laplacian = apply_laplacian(original, gray)

    # ---- Step 7: Sobel edge detection -------------------------------
    print("\nStep 7: Applying Sobel edge detection (X, Y, Combined) …")
    sobel_x, sobel_y, sobel_combined = apply_sobel(gray)

    # ---- Step 8: Final comparison grid ------------------------------
    print("\nStep 8: Displaying final 2×4 comparison grid …")
    plot_final_comparison(
        original, gaussian, median, average,
        laplacian, sobel_x, sobel_y, sobel_combined,
    )

    # ---- Step 9: Print observations ---------------------------------
    print("\nStep 9: Observations")
    print_observations()

    # ---- Step 10: Save output images --------------------------------
    print("Step 10: Saving output images …")
    output_dir = save_images(
        gaussian, median, average,
        laplacian, sobel_x, sobel_y, sobel_combined,
        image_path,
    )

    # ---- Done -------------------------------------------------------
    print("=" * 60)
    print("  Experiment 3 completed successfully!")
    print(f"  Output images saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
