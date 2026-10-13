"""
============================================================
Experiment 4: Discrete Fourier Transform and
              Frequency-Domain Image Filtering
============================================================
Objective:
    Implement DFT-based frequency-domain filtering using
    Python, OpenCV, NumPy, and Matplotlib.

Operations covered:
    1. Load image in grayscale
    2. Convert to float32
    3. Compute DFT using OpenCV (cv2.dft)
    4. Shift zero-frequency component to centre (np.fft.fftshift)
    5. Calculate and display the Magnitude Spectrum
    6. Low-Pass Filter  (keep low frequencies → smooth image)
    7. High-Pass Filter (keep high frequencies → edge image)
    8. Inverse DFT for both filtered spectra
    9. Final 2×2 comparison grid
   10. Save output images

Libraries used:
    - opencv-python  (cv2)  : DFT, image I/O, drawing
    - numpy          (np)   : array operations, fftshift
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
# SECTION 1 – Image Loading (Grayscale)
# ================================================================

def pick_image_path():
    """
    Open a tkinter file-picker dialog so the user can browse for an image.
    Returns the selected path, or an empty string if cancelled or
    if tkinter is unavailable on this system.
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
    Prompt the user to select or type an image path, then load it in
    GRAYSCALE mode using OpenCV.  Prints image metadata on success;
    exits with a clear message on failure.

    The DFT experiment uses a single-channel (grayscale) image
    because frequency-domain filtering operates on scalar pixel values.

    Returns:
        tuple[np.ndarray, str]: Grayscale image (uint8) and its path.
    """
    print("=" * 60)
    print("  Experiment 4 – Discrete Fourier Transform &")
    print("                  Frequency-Domain Filtering")
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

    # ---- Load in grayscale (single channel required for DFT) ----
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"ERROR: OpenCV could not read the image → {image_path}")
        print("       The file may be corrupt or in an unsupported format.")
        sys.exit(1)

    # ---- Print image metadata ----
    print(f"\n  Image loaded successfully! (grayscale)")
    print(f"  Path      : {image_path}")
    print(f"  Shape     : {img.shape}  (height × width)")
    print(f"  Data type : {img.dtype}")

    # ---- Display the original grayscale image ----
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(img, cmap="gray")
    ax.set_title("Original Grayscale Image", fontsize=14,
                 fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    return img, image_path


# ================================================================
# SECTION 2 – Convert to float32
# ================================================================

def convert_to_float32(img):
    """
    Convert the uint8 grayscale image to 32-bit floating-point format.

    Why float32?
        OpenCV's DFT (cv2.dft) requires floating-point input.
        Float32 supports the large intermediate values that arise during
        frequency-domain calculations, whereas uint8 (0–255) would
        cause overflow or loss of precision.

    Args:
        img (np.ndarray): Grayscale uint8 image.

    Returns:
        np.ndarray: float32 version of the image.
    """
    img_float = np.float32(img)
    print("  ✔  Converted image to float32 for DFT computation.")
    return img_float


# ================================================================
# SECTION 3 – Compute the DFT
# ================================================================

def compute_dft(img_float):
    """
    Compute the 2-D Discrete Fourier Transform (DFT) of the image.

    cv2.dft() decomposes the spatial-domain image into its frequency
    components.  The flag DFT_COMPLEX_OUTPUT returns both the real and
    imaginary parts as a two-channel array:
        dft[:, :, 0]  →  Real component
        dft[:, :, 1]  →  Imaginary component

    Args:
        img_float (np.ndarray): float32 grayscale image.

    Returns:
        np.ndarray: Complex DFT output of shape (H, W, 2).
    """
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    print(f"  ✔  DFT computed successfully.  Output shape: {dft.shape}")
    return dft


# ================================================================
# SECTION 4 – Shift Zero-Frequency Component to Centre
# ================================================================

def shift_dft(dft):
    """
    Shift the zero-frequency (DC) component from the top-left corner
    to the centre of the spectrum using np.fft.fftshift().

    Why shift?
        By default, the DFT places the lowest frequencies (DC and near-DC)
        at the corners of the output array.  Shifting them to the centre
        makes it intuitive to design and apply circular frequency masks,
        and makes the magnitude spectrum much easier to visualise.

    Args:
        dft (np.ndarray): Raw DFT output (H, W, 2).

    Returns:
        np.ndarray: Frequency-shifted DFT (H, W, 2).
    """
    dft_shift = np.fft.fftshift(dft)
    print("  ✔  Zero-frequency component shifted to the centre.")
    return dft_shift


# ================================================================
# SECTION 5 – Magnitude Spectrum
# ================================================================

def compute_magnitude_spectrum(dft_shift):
    """
    Calculate the magnitude spectrum and apply logarithmic scaling.

    Steps:
        1. Compute magnitude from the real and imaginary channels.
        2. Apply   20 * log(magnitude + 1)   to compress the wide
           dynamic range so that both small and large components are
           visible in the plot.  (+1 avoids log(0).)

    The magnitude spectrum shows the strength (energy) of each
    frequency component.  The bright centre represents strong
    low-frequency content; the edges represent high frequencies.

    Args:
        dft_shift (np.ndarray): Centred DFT (H, W, 2).

    Returns:
        np.ndarray: Log-scaled magnitude spectrum (float64).
    """
    # cv2.magnitude combines real and imaginary channels: sqrt(r²+i²)
    magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0],
                                       dft_shift[:, :, 1])

    # Logarithmic scaling to enhance visibility of low-energy components
    magnitude_spectrum = 20 * np.log(magnitude_spectrum + 1)

    # Display the magnitude spectrum
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(magnitude_spectrum, cmap="gray")
    ax.set_title("Magnitude Spectrum", fontsize=14,
                 fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    print("  ✔  Magnitude spectrum calculated and displayed.")
    return magnitude_spectrum


# ================================================================
# SECTION 6 – Low-Pass Filter
# ================================================================

def apply_low_pass_filter(dft_shift, img_shape, radius=50):
    """
    Apply a circular Low-Pass Filter (LPF) in the frequency domain.

    How it works:
        A circular mask is created with 1s inside a circle of the
        given radius (centred on the image) and 0s outside.
        Multiplying the DFT by this mask retains only the low-frequency
        components (near the centre) and zeroes out the high-frequency
        components (near the edges).

    Effect on the image:
        Low frequencies encode the overall brightness and slowly
        varying regions.  Retaining them (and removing high frequencies)
        produces a smoothed, blurred version of the original image —
        similar to a Gaussian blur in the spatial domain.

    Args:
        dft_shift  (np.ndarray): Centred DFT (H, W, 2).
        img_shape  (tuple)     : (rows, cols) of the original image.
        radius     (int)       : Filter radius in pixels (default 50,
                                 as specified in the lab PDF).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            filtered DFT, inverse-shifted DFT, reconstructed uint8 image.
    """
    rows, cols = img_shape
    crow, ccol = rows // 2, cols // 2   # centre of the frequency array

    # ---- Create a two-channel mask (matches DFT's (H, W, 2) shape) ----
    mask_low = np.zeros((rows, cols, 2), np.float32)

    # Draw a filled white circle at the centre — this is the pass region
    # (1, 1) sets both channels of the mask to 1 inside the circle
    cv2.circle(mask_low, (ccol, crow), radius, (1, 1), -1)

    # ---- Apply the mask in the frequency domain ----
    filtered_dft_low = dft_shift * mask_low

    # ---- Inverse shift: move DC component back to the corners ----
    f_ishift_low = np.fft.ifftshift(filtered_dft_low)

    # ---- Inverse DFT: reconstruct the spatial-domain image ----
    img_low_complex = cv2.idft(f_ishift_low)

    # ---- Compute magnitude of the complex output ----
    img_low = cv2.magnitude(img_low_complex[:, :, 0],
                            img_low_complex[:, :, 1])

    # ---- Normalize to 0–255 and convert to uint8 ----
    img_low = cv2.normalize(img_low, None, 0, 255, cv2.NORM_MINMAX)
    img_low = np.uint8(img_low)

    # ---- Display the low-pass filtered image ----
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(img_low, cmap="gray")
    ax.set_title("Low-Pass Filtered Image", fontsize=14,
                 fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"  ✔  Low-pass filter applied (radius = {radius}).")
    return filtered_dft_low, f_ishift_low, img_low


# ================================================================
# SECTION 7 – High-Pass Filter
# ================================================================

def apply_high_pass_filter(dft_shift, img_shape, radius=50):
    """
    Apply a circular High-Pass Filter (HPF) in the frequency domain.

    How it works:
        A mask is created with 1s everywhere EXCEPT inside a filled
        circle at the centre.  This zeroes out the low-frequency
        components and retains the high-frequency components.

    Effect on the image:
        High frequencies encode fine detail, sharp edges, and rapid
        intensity changes.  Retaining them (and removing low frequencies)
        produces an image that highlights edges and textures while
        removing the background (DC/slow-varying) content.

    Args:
        dft_shift  (np.ndarray): Centred DFT (H, W, 2).
        img_shape  (tuple)     : (rows, cols) of the original image.
        radius     (int)       : Filter radius in pixels (default 50,
                                 as specified in the lab PDF).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            filtered DFT, inverse-shifted DFT, reconstructed uint8 image.
    """
    rows, cols = img_shape
    crow, ccol = rows // 2, cols // 2

    # ---- Create a two-channel mask filled with ones ----
    mask_high = np.ones((rows, cols, 2), np.float32)

    # Draw a filled black circle at the centre — this is the STOP region
    # (0, 0) sets both channels of the mask to 0 inside the circle
    cv2.circle(mask_high, (ccol, crow), radius, (0, 0), -1)

    # ---- Apply the mask ----
    filtered_dft_high = dft_shift * mask_high

    # ---- Inverse shift ----
    f_ishift_high = np.fft.ifftshift(filtered_dft_high)

    # ---- Inverse DFT ----
    img_high_complex = cv2.idft(f_ishift_high)

    # ---- Magnitude of the complex output ----
    img_high = cv2.magnitude(img_high_complex[:, :, 0],
                             img_high_complex[:, :, 1])

    # ---- Normalize and convert to uint8 ----
    img_high = cv2.normalize(img_high, None, 0, 255, cv2.NORM_MINMAX)
    img_high = np.uint8(img_high)

    # ---- Display the high-pass filtered image ----
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(img_high, cmap="gray")
    ax.set_title("High-Pass Filtered Image", fontsize=14,
                 fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"  ✔  High-pass filter applied (radius = {radius}).")
    return filtered_dft_high, f_ishift_high, img_high


# ================================================================
# SECTION 8 – Inverse DFT Confirmation
# ================================================================

def confirm_inverse_dft(f_ishift_low, f_ishift_high):
    """
    Confirm that the Inverse Discrete Fourier Transform (IDFT) was
    applied to both filtered spectra and print a summary message.

    The actual IDFT computation is done inside apply_low_pass_filter()
    and apply_high_pass_filter() to avoid redundant calculations.
    This function documents and confirms that step.

    Args:
        f_ishift_low  (np.ndarray): Inverse-shifted low-pass DFT.
        f_ishift_high (np.ndarray): Inverse-shifted high-pass DFT.
    """
    # Verify both IDFT inputs are non-empty numpy arrays
    assert f_ishift_low.shape  == f_ishift_high.shape, \
        "Shape mismatch between low-pass and high-pass inverse-shift arrays."

    print("\n  Inverse Fourier Transform Summary:")
    print(f"    Low-pass  IDFT input shape : {f_ishift_low.shape}")
    print(f"    High-pass IDFT input shape : {f_ishift_high.shape}")
    print("  ✔  Inverse DFT completed successfully for both filters.")


# ================================================================
# SECTION 9 – Final 2×2 Comparison Grid
# ================================================================

def plot_final_comparison(img, magnitude_spectrum, img_low, img_high):
    """
    Display a 2×2 comparison grid following the PDF layout (page 7).

    Layout:
        [1] Original Image        [2] Magnitude Spectrum
        [3] Low-Pass Filtered     [4] High-Pass Filtered

    Args:
        img               (np.ndarray): Original grayscale image (uint8).
        magnitude_spectrum(np.ndarray): Log-scaled magnitude spectrum.
        img_low           (np.ndarray): Low-pass filtered image (uint8).
        img_high          (np.ndarray): High-pass filtered image (uint8).
    """
    plt.figure(figsize=(14, 10))

    # Subplot 1 – Original grayscale image
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Image", fontsize=13, fontweight="bold")
    plt.axis("off")

    # Subplot 2 – Magnitude spectrum
    plt.subplot(2, 2, 2)
    plt.imshow(magnitude_spectrum, cmap="gray")
    plt.title("Magnitude Spectrum", fontsize=13, fontweight="bold")
    plt.axis("off")

    # Subplot 3 – Low-pass filtered image
    plt.subplot(2, 2, 3)
    plt.imshow(img_low, cmap="gray")
    plt.title("Low-Pass Filtered Image", fontsize=13, fontweight="bold")
    plt.axis("off")

    # Subplot 4 – High-pass filtered image
    plt.subplot(2, 2, 4)
    plt.imshow(img_high, cmap="gray")
    plt.title("High-Pass Filtered Image", fontsize=13, fontweight="bold")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("  ✔  Final 2×2 comparison grid displayed.")


# ================================================================
# SECTION 10 – Save Output Images
# ================================================================

def save_images(img, magnitude_spectrum, img_low, img_high, input_path):
    """
    Save the four result images to an 'output/' directory located
    next to the input image.

    Files saved:
        original_grayscale.jpg
        magnitude_spectrum.jpg
        low_pass_filtered.jpg
        high_pass_filtered.jpg

    The magnitude spectrum (float64) is normalised to 0–255 before
    saving so that cv2.imwrite() can store it as a standard JPEG.

    Args:
        img               (np.ndarray): Original grayscale image.
        magnitude_spectrum(np.ndarray): Log-scaled magnitude spectrum.
        img_low           (np.ndarray): Low-pass filtered image (uint8).
        img_high          (np.ndarray): High-pass filtered image (uint8).
        input_path        (str)       : Full path of the input image.

    Returns:
        str: Absolute path to the output directory.
    """
    input_dir  = os.path.dirname(os.path.abspath(input_path))
    output_dir = os.path.join(input_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  Saving output images to: {output_dir}")

    # Normalise float magnitude spectrum → uint8 for imwrite
    mag_save = cv2.normalize(magnitude_spectrum, None, 0, 255,
                             cv2.NORM_MINMAX)
    mag_save = np.uint8(mag_save)

    files_to_save = [
        ("original_grayscale.jpg", img),
        ("magnitude_spectrum.jpg", mag_save),
        ("low_pass_filtered.jpg",  img_low),
        ("high_pass_filtered.jpg", img_high),
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
    Entry point – orchestrates Experiment 4 step by step:

        Step 1 : Load image in grayscale
        Step 2 : Convert to float32
        Step 3 : Compute DFT
        Step 4 : Shift zero-frequency component to centre
        Step 5 : Calculate and display magnitude spectrum
        Step 6 : Apply Low-Pass Filter
        Step 7 : Apply High-Pass Filter
        Step 8 : Confirm Inverse DFT
        Step 9 : Final 2×2 comparison grid
        Step 10: Save output images
    """

    # ---- Step 1: Load image ----------------------------------------
    img, image_path = load_image()

    # Keep the original unchanged; use rows/cols throughout
    rows, cols = img.shape

    # ---- Step 2: Convert to float32 --------------------------------
    print("\nStep 2: Converting image to float32 …")
    img_float = convert_to_float32(img)

    # ---- Step 3: Compute DFT ----------------------------------------
    print("\nStep 3: Computing DFT …")
    dft = compute_dft(img_float)

    # ---- Step 4: Shift zero-frequency to centre --------------------
    print("\nStep 4: Shifting zero-frequency component to centre …")
    dft_shift = shift_dft(dft)

    # ---- Step 5: Magnitude spectrum --------------------------------
    print("\nStep 5: Computing magnitude spectrum …")
    magnitude_spectrum = compute_magnitude_spectrum(dft_shift)

    # ---- Step 6: Low-Pass Filter -----------------------------------
    print("\nStep 6: Applying Low-Pass Filter (radius = 50) …")
    _, f_ishift_low, img_low = apply_low_pass_filter(
        dft_shift, img.shape, radius=50
    )

    # ---- Step 7: High-Pass Filter ----------------------------------
    print("\nStep 7: Applying High-Pass Filter (radius = 50) …")
    _, f_ishift_high, img_high = apply_high_pass_filter(
        dft_shift, img.shape, radius=50
    )

    # ---- Step 8: Inverse DFT confirmation --------------------------
    print("\nStep 8: Confirming Inverse DFT …")
    confirm_inverse_dft(f_ishift_low, f_ishift_high)

    # ---- Step 9: Final comparison grid -----------------------------
    print("\nStep 9: Displaying final 2×2 comparison grid …")
    plot_final_comparison(img, magnitude_spectrum, img_low, img_high)

    # ---- Step 10: Save output images --------------------------------
    print("\nStep 10: Saving output images …")
    output_dir = save_images(img, magnitude_spectrum,
                             img_low, img_high, image_path)

    # ---- Done -------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Experiment 4 completed successfully!")
    print(f"  Output images saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
