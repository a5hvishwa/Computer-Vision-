"""
============================================================
Experiment 2: Image Enhancement Using Histogram Techniques
============================================================
Objective:
    Implement image enhancement techniques using Python,
    OpenCV, NumPy, and Matplotlib.

Operations covered:
    1. Grayscale Conversion
    2. Histogram Plotting
    3. Contrast Stretching
    4. Histogram Equalization
    5. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    6. Histogram Comparison
    7. Final Image Comparison
    8. Saving Output Images

Libraries used:
    - opencv-python  (cv2)  : image I/O and processing
    - numpy          (np)   : array / numerical operations
    - matplotlib     (plt)  : plotting images and histograms
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
        ("cv2",         "opencv-python"),
        ("numpy",       "numpy"),
        ("matplotlib",  "matplotlib"),
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
    Open a tkinter file-picker dialog so the user can browse for an image.
    Returns the selected path, or an empty string if cancelled / unavailable.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()                       # hide empty root window
        root.attributes("-topmost", True)     # bring dialog to the front

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
        # tkinter not available on this system
        return ""


def load_image():
    """
    Prompt the user to select or enter an image path, then load it
    with OpenCV.  Prints metadata on success and exits on failure.

    Returns:
        tuple[np.ndarray, str]: BGR image array and its file path.
    """
    print("=" * 60)
    print("  Experiment 2 – Image Enhancement Using Histogram Techniques")
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

    # --- Validate the path ---
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

    # --- Load with OpenCV ---
    img = cv2.imread(image_path)

    if img is None:
        print(f"ERROR: OpenCV could not read the image → {image_path}")
        print("       The file may be corrupt or in an unsupported format.")
        sys.exit(1)

    # --- Print image metadata ---
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
# SECTION 2 – Grayscale Conversion
# ================================================================

def convert_to_grayscale(img):
    """
    Convert the BGR image to grayscale using cv2.cvtColor().

    Grayscale reduces a 3-channel colour image to a single channel
    where each pixel value (0–255) represents luminance/brightness.

    Args:
        img (np.ndarray): Original BGR image.

    Returns:
        np.ndarray: Single-channel grayscale image.
    """
    # Convert BGR → Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Display the grayscale image
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(gray, cmap="gray")
    ax.set_title("Grayscale Image", fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    print("  ✔  Grayscale conversion done.")
    return gray


# ================================================================
# SECTION 3 – Histogram Plotting
# ================================================================

def plot_original_histogram(gray):
    """
    Calculate and display the histogram of the grayscale image.

    A histogram shows how many pixels exist at each intensity level
    (0 = black … 255 = white).  A narrow histogram means low contrast;
    a wide, spread-out histogram means good contrast.

    Args:
        gray (np.ndarray): Grayscale image.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # ravel() flattens the 2-D array into 1-D so plt.hist can process it
    ax.hist(gray.ravel(), bins=256, range=[0, 256], color="gray", edgecolor="none")
    ax.set_title("Histogram – Original Image", fontsize=14, fontweight="bold")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    print("  ✔  Original histogram plotted.")


# ================================================================
# SECTION 4 – Contrast Stretching
# ================================================================

def contrast_stretch(img):
    """
    Stretch pixel intensities linearly to the full 0–255 range.

    Formula (from the lab PDF):
        stretched = (img - min_val) * (255.0 / (max_val - min_val))

    This maps the darkest pixel in the image to 0 and the brightest
    to 255, improving overall contrast.

    Special case: if min == max (uniform/flat image), return as-is
    to avoid division by zero.

    Args:
        img (np.ndarray): Grayscale image (uint8).

    Returns:
        np.ndarray: Contrast-stretched image (uint8).
    """
    min_val = np.min(img)   # darkest pixel value
    max_val = np.max(img)   # brightest pixel value

    print(f"  Contrast Stretch → min pixel: {min_val}, max pixel: {max_val}")

    if max_val == min_val:
        # All pixels are the same intensity — no stretching possible
        print("  WARNING: Image has uniform intensity; contrast stretching skipped.")
        return img.copy()

    # Apply the stretching formula
    stretched = (img - min_val) * (255.0 / (max_val - min_val))

    # Convert back to 8-bit unsigned integer
    stretched_img = np.uint8(stretched)

    # Display the result
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(stretched_img, cmap="gray")
    ax.set_title("Contrast Stretched Image", fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    print("  ✔  Contrast stretching done.")
    return stretched_img


# ================================================================
# SECTION 5 – Histogram Equalization
# ================================================================

def apply_histogram_equalization(gray):
    """
    Apply OpenCV's histogram equalization to redistribute pixel
    intensities evenly across 0–255, enhancing contrast globally.

    Also shows a side-by-side comparison of the original vs. equalized
    image, and plots both histograms for comparison.

    Args:
        gray (np.ndarray): Grayscale image.

    Returns:
        np.ndarray: Histogram-equalized image (uint8).
    """
    # Apply histogram equalization
    equalized_img = cv2.equalizeHist(gray)

    # --- Side-by-side image comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Histogram Equalization Comparison",
                 fontsize=14, fontweight="bold")

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(equalized_img, cmap="gray")
    axes[1].set_title("Histogram Equalized", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    # --- Side-by-side histogram comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Histograms: Original vs. Equalized",
                 fontsize=14, fontweight="bold")

    axes[0].hist(gray.ravel(), bins=256, range=[0, 256],
                 color="gray", edgecolor="none")
    axes[0].set_title("Original Histogram")
    axes[0].set_xlabel("Pixel Intensity")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(equalized_img.ravel(), bins=256, range=[0, 256],
                 color="steelblue", edgecolor="none")
    axes[1].set_title("Equalized Histogram")
    axes[1].set_xlabel("Pixel Intensity")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    print("  ✔  Histogram equalization done.")
    return equalized_img


# ================================================================
# SECTION 6 – CLAHE Enhancement
# ================================================================

def apply_clahe(gray):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Unlike global histogram equalization, CLAHE divides the image into
    small tiles (8×8 grid) and equalises each tile independently.
    The clipLimit prevents over-amplification of noise.

    Settings (from the lab PDF):
        clipLimit    = 2.0
        tileGridSize = (8, 8)

    Args:
        gray (np.ndarray): Grayscale image.

    Returns:
        np.ndarray: CLAHE-enhanced image (uint8).
    """
    # Create the CLAHE object with PDF-specified parameters
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply CLAHE to the grayscale image
    clahe_img = clahe.apply(gray)

    # Display the CLAHE-enhanced image
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(clahe_img, cmap="gray")
    ax.set_title("CLAHE Enhanced Image", fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    # Plot the CLAHE histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(clahe_img.ravel(), bins=256, range=[0, 256],
            color="darkorange", edgecolor="none")
    ax.set_title("Histogram – CLAHE Enhanced Image",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    print("  ✔  CLAHE enhancement done.")
    return clahe_img


# ================================================================
# SECTION 7 – Histogram Comparison (All Three)
# ================================================================

def plot_histogram_comparison(gray, equalized_img, clahe_img):
    """
    Display three histograms side-by-side for easy comparison:
        1. Original
        2. Histogram Equalized
        3. CLAHE

    Different colours make it easy to distinguish the distributions.

    Args:
        gray         (np.ndarray): Original grayscale image.
        equalized_img(np.ndarray): Histogram-equalized image.
        clahe_img    (np.ndarray): CLAHE-enhanced image.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle("Histogram Comparison: Original | Equalized | CLAHE",
                 fontsize=14, fontweight="bold")

    # 1 – Original histogram (gray)
    axes[0].hist(gray.ravel(), bins=256, range=[0, 256],
                 color="gray", edgecolor="none")
    axes[0].set_title("Original Histogram")
    axes[0].set_xlabel("Pixel Intensity")
    axes[0].set_ylabel("Frequency")

    # 2 – Equalized histogram (blue)
    axes[1].hist(equalized_img.ravel(), bins=256, range=[0, 256],
                 color="steelblue", edgecolor="none")
    axes[1].set_title("Equalized Histogram")
    axes[1].set_xlabel("Pixel Intensity")
    axes[1].set_ylabel("Frequency")

    # 3 – CLAHE histogram (orange)
    axes[2].hist(clahe_img.ravel(), bins=256, range=[0, 256],
                 color="darkorange", edgecolor="none")
    axes[2].set_title("CLAHE Histogram")
    axes[2].set_xlabel("Pixel Intensity")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    print("  ✔  Histogram comparison plotted.")


# ================================================================
# SECTION 8 – Final Image Comparison Grid
# ================================================================

def plot_final_comparison(gray, stretched_img, equalized_img, clahe_img):
    """
    Display a single-row comparison of all four processed images:
        Original | Contrast Stretched | Histogram Equalized | CLAHE

    Args:
        gray          (np.ndarray): Original grayscale image.
        stretched_img (np.ndarray): Contrast-stretched image.
        equalized_img (np.ndarray): Histogram-equalized image.
        clahe_img     (np.ndarray): CLAHE-enhanced image.
    """
    images = [
        (gray,          "Original"),
        (stretched_img, "Contrast Stretched"),
        (equalized_img, "Histogram Equalized"),
        (clahe_img,     "CLAHE"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Final Image Comparison",
                 fontsize=16, fontweight="bold", y=1.02)

    for ax, (image, title) in zip(axes, images):
        ax.imshow(image, cmap="gray")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    print("  ✔  Final comparison grid displayed.")


# ================================================================
# SECTION 9 – Save Output Images
# ================================================================

def save_images(gray, stretched_img, equalized_img, clahe_img, input_path):
    """
    Save the four processed images to an 'output/' directory located
    next to the input image.

    Files saved:
        original_grayscale.jpg
        contrast_stretched.jpg
        histogram_equalized.jpg
        clahe_enhanced.jpg

    Args:
        gray          (np.ndarray): Original grayscale image.
        stretched_img (np.ndarray): Contrast-stretched image.
        equalized_img (np.ndarray): Histogram-equalized image.
        clahe_img     (np.ndarray): CLAHE-enhanced image.
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
        ("original_grayscale.jpg",  gray),
        ("contrast_stretched.jpg",  stretched_img),
        ("histogram_equalized.jpg", equalized_img),
        ("clahe_enhanced.jpg",      clahe_img),
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
        print("\n  WARNING: Some files could not be saved. "
              "Check the path and permissions.")

    return output_dir


# ================================================================
# MAIN PROGRAM
# ================================================================

def main():
    """
    Entry point – runs Experiment 2 step by step:
        1. Load image
        2. Grayscale conversion
        3. Original histogram
        4. Contrast stretching
        5. Histogram equalization
        6. CLAHE enhancement
        7. Three-histogram comparison
        8. Final four-image comparison
        9. Save output images
    """

    # --- Step 1: Load image -----------------------------------------
    img, image_path = load_image()

    # --- Step 2: Grayscale conversion --------------------------------
    print("\nStep 2: Converting to grayscale …")
    gray = convert_to_grayscale(img)

    # --- Step 3: Original histogram ----------------------------------
    print("\nStep 3: Plotting original histogram …")
    plot_original_histogram(gray)

    # --- Step 4: Contrast stretching ---------------------------------
    print("\nStep 4: Applying contrast stretching …")
    stretched_img = contrast_stretch(gray)

    # --- Step 5: Histogram equalization ------------------------------
    print("\nStep 5: Applying histogram equalization …")
    equalized_img = apply_histogram_equalization(gray)

    # --- Step 6: CLAHE -----------------------------------------------
    print("\nStep 6: Applying CLAHE …")
    clahe_img = apply_clahe(gray)

    # --- Step 7: Histogram comparison --------------------------------
    print("\nStep 7: Plotting three-histogram comparison …")
    plot_histogram_comparison(gray, equalized_img, clahe_img)

    # --- Step 8: Final image comparison ------------------------------
    print("\nStep 8: Displaying final image comparison grid …")
    plot_final_comparison(gray, stretched_img, equalized_img, clahe_img)

    # --- Step 9: Save output images ----------------------------------
    print("\nStep 9: Saving output images …")
    output_dir = save_images(gray, stretched_img, equalized_img,
                             clahe_img, image_path)

    # --- Done --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Experiment 2 completed successfully!")
    print(f"  Output images saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
