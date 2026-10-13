"""
============================================================
Experiment 1: Basic Image Processing using OpenCV in Python
============================================================
Objective:
    Implement basic image-processing operations using
    Python, OpenCV, NumPy, and Matplotlib.

Libraries used:
    - opencv-python  (cv2)      : image I/O and processing
    - numpy          (np)       : array operations
    - matplotlib     (plt)      : displaying images
    - tkinter                   : file-picker dialog
    - os                        : directory / path utilities
============================================================
"""

import os
import sys

# --------------- Dependency check ---------------
def check_dependencies():
    """Verify that all required third-party libraries are available."""
    missing = []
    for module, package in [("cv2", "opencv-python"),
                             ("numpy", "numpy"),
                             ("matplotlib", "matplotlib")]:
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

# --------------- Imports (after dependency check) ---------------
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ================================================================
# SECTION 1 – Image Loading
# ================================================================

def pick_image_path():
    """
    Open a file-picker dialog so the user can browse for an image.
    Falls back to a manual path prompt if tkinter is unavailable.

    Returns:
        str: The selected file path, or an empty string if cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()                       # hide the empty root window
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
        # tkinter not available — fall back to manual input
        return ""


def load_image():
    """
    Prompt the user to select or enter an image path, then load it with
    OpenCV.  Prints image metadata on success.

    Returns:
        tuple[np.ndarray, str]: The loaded BGR image and its file path.
                                Exits the program on failure.
    """
    print("=" * 60)
    print("  Experiment 1 – Basic Image Processing with OpenCV")
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

    # Validate the path
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
        print(f"         Supported: {', '.join(supported_ext)}")

    # Load the image
    img = cv2.imread(image_path)

    if img is None:
        print(f"ERROR: OpenCV could not read the image → {image_path}")
        print("       The file may be corrupt or in an unsupported format.")
        sys.exit(1)

    # Print image metadata
    height, width = img.shape[:2]
    channels      = img.shape[2] if img.ndim == 3 else 1
    print(f"\n  Image loaded successfully!")
    print(f"  Path      : {image_path}")
    print(f"  Width     : {width} px")
    print(f"  Height    : {height} px")
    print(f"  Dimensions: {img.shape}")
    print(f"  Channels  : {channels}")
    print(f"  Data type : {img.dtype}")

    return img, image_path


# ================================================================
# SECTION 2 – Image Processing Operations
# ================================================================

def process_images(img):
    """
    Apply all required image-processing operations on the original
    image and return a dictionary of results.

    Args:
        img (np.ndarray): Original BGR image loaded by OpenCV.

    Returns:
        dict[str, np.ndarray]: Named processed images.
    """
    height, width = img.shape[:2]

    results = {}

    # 1. Original image (keep a clean copy – never modify this)
    results["original"] = img.copy()

    # 2. Grayscale conversion
    results["grayscale"] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. HSV conversion (Hue, Saturation, Value)
    results["hsv"] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 4. LAB conversion (Lightness, a*, b*)
    results["lab"] = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # 5. Resize to 300 × 300 pixels
    results["resized"] = cv2.resize(img, (300, 300))

    # 6. Rotate 90° clockwise
    results["rotated"] = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # 7. Horizontal flip (flip code 1 → around the y-axis)
    results["horizontal_flip"] = cv2.flip(img, 1)

    # 8. Vertical flip (flip code 0 → around the x-axis)
    results["vertical_flip"] = cv2.flip(img, 0)

    # 9. Negative / complement image
    results["negative"] = 255 - img

    # 10. Region of Interest – central half of the image
    #     Start at 1/4 of (height, width), size is 1/2 of (height, width)
    roi_y_start = height // 4
    roi_y_end   = roi_y_start + height // 2
    roi_x_start = width  // 4
    roi_x_end   = roi_x_start + width  // 2
    results["roi"] = img[roi_y_start:roi_y_end, roi_x_start:roi_x_end].copy()

    print("\n  All 10 image-processing operations completed.")
    return results


# ================================================================
# SECTION 3 – Display Helpers
# ================================================================

def bgr_to_rgb(img):
    """Convert a BGR image (OpenCV default) to RGB for Matplotlib."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def display_images(results):
    """
    Display each processed image in its own Matplotlib figure with a
    descriptive title.  Color-space conversions are applied so that
    every plot shows visually meaningful colours.

    Args:
        results (dict[str, np.ndarray]): Output of process_images().
    """
    print("\n  Displaying individual processed images …")

    # Map result keys → (display title, pre-display conversion function)
    display_config = [
        ("original",        "Original Image",           bgr_to_rgb,                  None),
        ("grayscale",       "Grayscale Image",           lambda x: x,                "gray"),
        ("hsv",             "HSV Image",                 lambda x: bgr_to_rgb(cv2.cvtColor(x, cv2.COLOR_HSV2BGR)), None),
        ("lab",             "LAB Image",                 lambda x: bgr_to_rgb(cv2.cvtColor(x, cv2.COLOR_LAB2BGR)), None),
        ("resized",         "Resized Image (300×300)",   bgr_to_rgb,                  None),
        ("rotated",         "Rotated Image (90° CW)",    bgr_to_rgb,                  None),
        ("horizontal_flip", "Horizontal Flip",           bgr_to_rgb,                  None),
        ("vertical_flip",   "Vertical Flip",             bgr_to_rgb,                  None),
        ("negative",        "Negative / Complement",     bgr_to_rgb,                  None),
        ("roi",             "Region of Interest (ROI)",  bgr_to_rgb,                  None),
    ]

    for key, title, convert_fn, cmap in display_config:
        img_to_show = convert_fn(results[key])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(img_to_show, cmap=cmap)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.axis("off")
        plt.tight_layout()
        plt.show()


def display_comparison_grid(results):
    """
    Show a 2 × 3 comparison grid with six key images side-by-side.

    Grid layout:
        Original  |  Grayscale  |  HSV
        LAB       |  Negative   |  ROI

    Args:
        results (dict[str, np.ndarray]): Output of process_images().
    """
    print("\n  Displaying comparison grid …")

    grid_items = [
        ("original",  "Original",  bgr_to_rgb(results["original"]),  None),
        ("grayscale", "Grayscale", results["grayscale"],              "gray"),
        ("hsv",       "HSV",       bgr_to_rgb(cv2.cvtColor(results["hsv"], cv2.COLOR_HSV2BGR)), None),
        ("lab",       "LAB",       bgr_to_rgb(cv2.cvtColor(results["lab"], cv2.COLOR_LAB2BGR)), None),
        ("negative",  "Negative",  bgr_to_rgb(results["negative"]),   None),
        ("roi",       "ROI",       bgr_to_rgb(results["roi"]),        None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        "Experiment 1 – Image Processing Comparison",
        fontsize=16, fontweight="bold", y=1.02,
    )

    for ax, (_, title, img_to_show, cmap) in zip(axes.flat, grid_items):
        ax.imshow(img_to_show, cmap=cmap)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ================================================================
# SECTION 4 – Save Processed Images
# ================================================================

def save_images(results, input_path):
    """
    Save all required output images to an 'output/' directory located
    in the same folder as the input image.

    Files saved:
        output.jpg          – original image
        output.png          – original image
        grayscale.jpg
        resized.jpg
        rotated.jpg
        horizontal_flip.jpg
        vertical_flip.jpg
        negative.jpg
        roi.jpg

    Args:
        results (dict[str, np.ndarray]): Output of process_images().
        input_path (str): Path of the original input image.
    """
    # Create the output directory next to the input file
    input_dir  = os.path.dirname(os.path.abspath(input_path))
    output_dir = os.path.join(input_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  Saving processed images to: {output_dir}")

    # (filename, image array) pairs to save
    files_to_save = [
        ("output.jpg",          results["original"]),
        ("output.png",          results["original"]),
        ("grayscale.jpg",       results["grayscale"]),
        ("resized.jpg",         results["resized"]),
        ("rotated.jpg",         results["rotated"]),
        ("horizontal_flip.jpg", results["horizontal_flip"]),
        ("vertical_flip.jpg",   results["vertical_flip"]),
        ("negative.jpg",        results["negative"]),
        ("roi.jpg",             results["roi"]),
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
        print("\n  All files saved successfully.")
    else:
        print("\n  WARNING: Some files could not be saved.  Check the path and permissions.")

    return output_dir


# ================================================================
# MAIN PROGRAM
# ================================================================

def main():
    """
    Entry point – orchestrates the full experiment workflow:
        1. Load image
        2. Process image (10 operations)
        3. Display individual results
        4. Display comparison grid
        5. Save output files
    """

    # -- Step 1: Load image ------------------------------------------
    img, image_path = load_image()

    # -- Step 2: Process image ---------------------------------------
    print("\nStep 2: Applying image-processing operations …")
    results = process_images(img)

    # -- Step 3: Display individual images ---------------------------
    print("\nStep 3: Displaying results")
    print("  (Close each Matplotlib window to advance to the next image)")
    display_images(results)

    # -- Step 4: Display comparison grid ----------------------------
    print("\nStep 4: Displaying comparison grid")
    display_comparison_grid(results)

    # -- Step 5: Save output files -----------------------------------
    print("\nStep 5: Saving processed images")
    output_dir = save_images(results, image_path)

    # -- Done --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Experiment 1 completed successfully!")
    print(f"  Output images saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
