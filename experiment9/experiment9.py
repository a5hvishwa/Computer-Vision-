"""
============================================================
Experiment 9: Object Detection Using YOLOv8
============================================================
Objective:
    Perform object detection on an image using the pretrained
    YOLOv8 Nano model, display annotated results, and save
    all outputs.

Steps covered:
    1.  Check dependencies and install guidance
    2.  Load the pretrained YOLOv8 Nano model (yolov8n.pt)
    3.  Load and display the input image
    4.  Run object detection
    5.  Display annotated detection results
    6.  Print class names and confidence scores
    7.  Count total detected objects
    8.  Display list of detected object names
    9.  Save output files (original, annotated, text report)

Libraries used:
    - ultralytics  : YOLOv8 model, inference, and annotation
    - opencv-python: image I/O and color conversion
    - matplotlib   : displaying images
    - tkinter      : file-picker dialog
    - os           : directory / path utilities

Install missing libraries with:
    pip install ultralytics opencv-python matplotlib
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
    Attempts to auto-install any missing packages using pip.
    On Kali/Debian externally-managed environments uses --break-system-packages.
    Note: 'ultralytics' pulls PyTorch (~700 MB) on first install – be patient.
    """
    import subprocess

    checks = [
        ("cv2",          "opencv-python"),
        ("matplotlib",   "matplotlib"),
        ("ultralytics",  "ultralytics"),
    ]

    for module, package in checks:
        try:
            __import__(module)
        except ImportError:
            print(f"  '{package}' not found.  Installing … (may take several minutes)")
            installed = False
            # Try normal pip first, then --break-system-packages (Kali/Debian)
            for extra in [[], ["--break-system-packages"]]:
                cmd = [sys.executable, "-m", "pip", "install", package] + extra
                # Stream output live so large downloads don't appear frozen
                result = subprocess.run(cmd)
                if result.returncode == 0:
                    installed = True
                    print(f"  ✔  '{package}' installed successfully.")
                    break

            if not installed:
                print(f"\nERROR: Could not install '{package}' automatically.")
                print(f"       Please run manually in your terminal:")
                print(f"         pip install {package} --break-system-packages")
                sys.exit(1)

            # Re-import after installation
            try:
                __import__(module)
            except ImportError:
                print(f"ERROR: '{package}' installed but cannot be imported.")
                print(f"       Restart the terminal and try again.")
                sys.exit(1)

check_dependencies()

# --------------- Imports (guaranteed available after check) ---------------
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ================================================================
# SECTION 1 – Load YOLOv8 Model
# ================================================================

def load_model(model_name="yolov8n.pt"):
    """
    Load the pretrained YOLOv8 Nano model.

    YOLOv8n ('nano') is the smallest and fastest variant of YOLOv8.
    It is pretrained on the COCO dataset, which contains 80 object
    categories (person, car, bus, bottle, etc.).

    Ultralytics automatically downloads the weights (~6 MB) from its
    servers on the first run if they are not already cached locally.
    On subsequent runs the cached weights are used directly.

    Args:
        model_name (str): Model filename (default: "yolov8n.pt").

    Returns:
        YOLO: Loaded model object, or exits on failure.
    """
    print("\nStep 2: Loading YOLOv8 model …")
    try:
        model = YOLO(model_name)
        print(f"  YOLOv8 model loaded successfully!")
        print(f"  Model    : {model_name}")
        print(f"  Classes  : {len(model.names)}  "
              f"(COCO 80-class pretrained)")
        return model
    except Exception as e:
        print(f"ERROR: Failed to load YOLOv8 model → {e}")
        print("       Ensure you have internet access for the first download,")
        print("       or place 'yolov8n.pt' in the working directory.")
        sys.exit(1)


# ================================================================
# SECTION 2 – Image Loading
# ================================================================

def pick_image_path():
    """
    Open a tkinter file-picker dialog.
    Returns the selected path, or '' if cancelled / tkinter unavailable.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        path = filedialog.askopenfilename(
            title="Select an Image for Object Detection",
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
    Prompt the user to select an image.  Falls back to manual entry
    if the file picker is unavailable or no file is selected.

    Returns:
        tuple[str, numpy.ndarray, numpy.ndarray]:
            image_path, img_bgr (OpenCV), img_rgb (for Matplotlib)
    """
    print("\nStep 3: Load input image")
    print("  [F] Open file picker")
    print("  [P] Enter image path manually")
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

    # ---- Load ----
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"ERROR: OpenCV could not read the image → {image_path}")
        sys.exit(1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    height, width = img_bgr.shape[:2]
    print(f"\n  Image loaded successfully!")
    print(f"  Path       : {image_path}")
    print(f"  Dimensions : {width} × {height} px")

    # ---- Display original image ----
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img_rgb)
    ax.set_title("Original Image", fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    return image_path, img_bgr, img_rgb


# ================================================================
# SECTION 3 – Run Object Detection
# ================================================================

def run_detection(model, image_path):
    """
    Run YOLOv8 inference on the selected image.

    YOLO processes the image, applies a neural network, and returns
    bounding boxes, class IDs, and confidence scores for every
    detected object in the scene.

    Args:
        model      (YOLO): Loaded YOLOv8 model.
        image_path (str) : Absolute path to the input image.

    Returns:
        list: Ultralytics Results list, or exits on failure.
    """
    print("\nStep 4: Running YOLOv8 object detection …")
    try:
        results = model(image_path)
        print(f"\n  Object detection completed!")

        # Print inference summary provided by Ultralytics
        if results:
            r = results[0]
            print(f"  Input shape  : {r.orig_shape}  (H × W)")
            n_boxes = len(r.boxes) if r.boxes is not None else 0
            print(f"  Objects found: {n_boxes}")

            # Speed information (ms) when available
            if hasattr(r, "speed") and r.speed:
                spd = r.speed
                print(f"  Preprocess   : {spd.get('preprocess', 0):.1f} ms")
                print(f"  Inference    : {spd.get('inference', 0):.1f} ms")
                print(f"  Postprocess  : {spd.get('postprocess', 0):.1f} ms")

        return results

    except Exception as e:
        print(f"ERROR: Detection failed → {e}")
        sys.exit(1)


# ================================================================
# SECTION 4 – Display Annotated Results
# ================================================================

def display_results(results):
    """
    Generate and display the annotated image with bounding boxes,
    class labels, and confidence scores.

    results[0].plot() renders all detections directly onto the image
    using the Ultralytics built-in visualizer.

    Args:
        results (list): Ultralytics Results list.

    Returns:
        numpy.ndarray: Annotated image in BGR format (for saving).
    """
    print("\nStep 5: Displaying detection results …")

    # Ultralytics returns an annotated BGR image
    result_image_bgr = results[0].plot()

    # Convert to RGB for Matplotlib
    result_image_rgb = cv2.cvtColor(result_image_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(result_image_rgb)
    ax.set_title("YOLOv8 Object Detection Results",
                 fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    print("  ✔  Annotated detection image displayed.")
    return result_image_bgr   # BGR — for cv2.imwrite


# ================================================================
# SECTION 5 – Print Class Names and Confidence Scores
# ================================================================

def print_detections(results, model):
    """
    Iterate over detected bounding boxes and print the class name and
    confidence score for each one.

    Each box object contains:
        box.cls   – tensor with the class ID (integer index)
        box.conf  – tensor with the confidence score (0.0 to 1.0)

    model.names is a dictionary mapping class IDs to class names,
    e.g. {0: 'person', 5: 'bus', ...}.

    Args:
        results (list): Ultralytics Results list.
        model   (YOLO): Loaded YOLOv8 model.

    Returns:
        list[str]: List of detected class names (may contain duplicates).
    """
    print("\nStep 6: Detected class names and confidence scores:")

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print("  (No objects detected in this image)")
        return []

    detected_names = []

    for box in boxes:
        class_id   = int(box.cls)
        confidence = float(box.conf)
        class_name = model.names[class_id]

        print(f"  {class_name}: {confidence:.2f}")
        detected_names.append(class_name)

    return detected_names


# ================================================================
# SECTION 6 – Count Detected Objects
# ================================================================

def count_objects(results):
    """
    Count and print the total number of detected bounding boxes.

    Args:
        results (list): Ultralytics Results list.

    Returns:
        int: Total number of detections.
    """
    total_objects = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"\nStep 7: Total objects detected: {total_objects}")

    if total_objects == 0:
        print("  (No objects were detected.  Try a different image or "
              "a model with a lower confidence threshold.)")

    return total_objects


# ================================================================
# SECTION 7 – Print Detected Object List
# ================================================================

def print_object_list(detected_names):
    """
    Print the list of detected object class names.
    Duplicate names appear for each detected instance of that class,
    following the PDF's output format.

    Args:
        detected_names (list[str]): Class names from print_detections().
    """
    print("\nStep 8: Detected objects:")
    print(f"  {detected_names}")


# ================================================================
# SECTION 8 – Save Outputs
# ================================================================

def save_outputs(image_path, img_bgr, result_image_bgr,
                 detected_names, total_objects, results, model):
    """
    Save all required output files to an 'output/' directory located
    next to the input image.

    Files saved:
        original_image.jpg       – original input image (unchanged)
        yolov8_detection.jpg     – annotated detection image
        detection_results.txt    – text report with detections

    Args:
        image_path       (str)          : Input image path.
        img_bgr          (numpy.ndarray): Original BGR image.
        result_image_bgr (numpy.ndarray): Annotated BGR image.
        detected_names   (list[str])    : Detected class names.
        total_objects    (int)          : Total detection count.
        results          (list)         : Raw Ultralytics results.
        model            (YOLO)         : YOLOv8 model (for names dict).

    Returns:
        str: Absolute path to the output directory.
    """
    input_dir  = os.path.dirname(os.path.abspath(image_path))
    output_dir = os.path.join(input_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nStep 9: Saving output files to: {output_dir}")

    all_saved = True

    def write_bgr(filename, image):
        nonlocal all_saved
        path = os.path.join(output_dir, filename)
        ok   = cv2.imwrite(path, image)
        if ok:
            print(f"  ✔  Saved: {filename}  →  {path}")
        else:
            print(f"  ✘  FAILED to save: {filename}")
            all_saved = False
        return path

    # Save original (do NOT overwrite the source file)
    write_bgr("original_image.jpg", img_bgr)

    # Save annotated detection image
    write_bgr("yolov8_detection.jpg", result_image_bgr)

    # Save text report
    txt_path = os.path.join(output_dir, "detection_results.txt")
    try:
        with open(txt_path, "w") as f:
            f.write("=" * 50 + "\n")
            f.write("  Experiment 9 – YOLOv8 Detection Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input image : {image_path}\n")
            f.write(f"Total objects detected: {total_objects}\n\n")
            f.write("Detected objects (class : confidence):\n")

            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    cid  = int(box.cls)
                    conf = float(box.conf)
                    name = model.names[cid]
                    f.write(f"  {name}: {conf:.2f}\n")

            f.write("\nDetected object names list:\n")
            f.write(f"  {detected_names}\n")

        print(f"  ✔  Saved: detection_results.txt  →  {txt_path}")
    except Exception as e:
        print(f"  ✘  FAILED to save detection_results.txt: {e}")
        all_saved = False

    if all_saved:
        print("\n  All output files saved successfully.")
    else:
        print("\n  WARNING: Some files could not be saved.")

    return output_dir


# ================================================================
# MAIN PROGRAM
# ================================================================

def main():
    """
    Entry point – orchestrates Experiment 9 step by step:

        Step 1 : Dependency check (done at import time)
        Step 2 : Load YOLOv8 Nano model
        Step 3 : Load input image
        Step 4 : Run YOLOv8 object detection
        Step 5 : Display annotated results
        Step 6 : Print class names and confidence scores
        Step 7 : Count total detected objects
        Step 8 : Print detected object names list
        Step 9 : Save original image, annotated image, and text report
    """
    print("=" * 60)
    print("  Experiment 9 – Object Detection Using YOLOv8")
    print("=" * 60)

    # ---- Step 2: Load model ----------------------------------------
    model = load_model("yolov8n.pt")

    # ---- Step 3: Load image ----------------------------------------
    image_path, img_bgr, img_rgb = load_image()

    # ---- Step 4: Run detection -------------------------------------
    results = run_detection(model, image_path)

    # ---- Step 5: Display annotated results -------------------------
    result_image_bgr = display_results(results)

    # ---- Step 6: Print class names and confidence scores -----------
    detected_names = print_detections(results, model)

    # ---- Step 7: Count objects -------------------------------------
    total_objects = count_objects(results)

    # ---- Step 8: Print object list ---------------------------------
    print_object_list(detected_names)

    # ---- Step 9: Save outputs --------------------------------------
    output_dir = save_outputs(
        image_path, img_bgr, result_image_bgr,
        detected_names, total_objects, results, model,
    )

    # ---- Done ------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Experiment 9 completed successfully!")
    print(f"  Output files saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
