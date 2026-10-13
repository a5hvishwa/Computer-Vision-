"""
============================================================
Experiment 10: Handwritten Digit Classification Using CNN
============================================================
Objective:
    Build, train, and evaluate a Convolutional Neural Network
    (CNN) on the MNIST handwritten digit dataset using
    TensorFlow/Keras, and analyse its performance.

Steps covered:
    1.  Check / auto-install dependencies
    2.  Load the MNIST dataset
    3.  Visualize sample training images
    4.  Preprocess (normalize + reshape)
    5.  Build the CNN architecture
    6.  Compile the model
    7.  Train for 5 epochs
    8.  Evaluate on test data + classification report
    9.  Predict on 10 unseen test images
   10.  Identify and display misclassified images
   11.  Plot training/validation accuracy curve
   12.  Plot training/validation loss curve
   13.  Generate confusion matrix heatmap
   14.  Model performance analysis
   15.  Practical applications
   16.  Save model and all output files

Libraries used:
    tensorflow / keras  : CNN model building and training
    numpy               : numerical operations
    matplotlib          : image and plot visualizations
    scikit-learn        : classification report + confusion matrix
    seaborn             : confusion matrix heatmap
    os, time            : directory management and timing
============================================================
"""

import os
import sys
import time
import subprocess


# ================================================================
# SECTION 0 – Dependency Check / Auto-Install
# ================================================================

def _install(package):
    """
    Attempt to install a pip package, trying --break-system-packages
    as a fallback for externally-managed environments (Kali/Debian).
    """
    for extra in [[], ["--break-system-packages"]]:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package, "--quiet"] + extra,
            capture_output=True,
        )
        if result.returncode == 0:
            return True
    return False


def check_dependencies():
    """
    Verify required libraries are available; auto-install if missing.
    """
    checks = [
        ("tensorflow", "tensorflow"),
        ("numpy",      "numpy"),
        ("matplotlib", "matplotlib"),
        ("sklearn",    "scikit-learn"),
        ("seaborn",    "seaborn"),
    ]
    for module, package in checks:
        try:
            __import__(module)
        except ImportError:
            print(f"  '{package}' not found.  Installing …")
            if _install(package):
                print(f"  ✔  '{package}' installed.")
                try:
                    __import__(module)
                except ImportError:
                    print(f"ERROR: '{package}' installed but cannot be imported.")
                    print("       Restart the terminal and try again.")
                    sys.exit(1)
            else:
                print(f"ERROR: Could not install '{package}'.")
                print(f"       Run manually:  pip install {package} --break-system-packages")
                sys.exit(1)

check_dependencies()

# --------------- Imports (guaranteed available after check) ---------------
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe for saving figs
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import classification_report, confusion_matrix

# Suppress verbose TF logging (keep warnings, suppress info/debug)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ================================================================
# SECTION 1 – Output Directory Setup
# ================================================================

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "experiment_10_results",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(fig, filename):
    """
    Save a Matplotlib figure to the output directory and close it.

    Args:
        fig      (Figure) : Matplotlib figure object.
        filename (str)    : Output filename (without directory).

    Returns:
        str: Absolute path where the file was saved.
    """
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Saved: {filename}  →  {path}")
    return path


# ================================================================
# SECTION 2 – Load the MNIST Dataset
# ================================================================

def load_mnist():
    """
    Download (if needed) and load the MNIST handwritten digit dataset
    from Keras.

    MNIST contains:
        60,000 training images (28×28 px, grayscale, 0–255)
        10,000 test images

    Returns:
        tuple: (x_train, y_train, x_test, y_test)
    """
    print("\nStep 2: Loading MNIST dataset …")
    try:
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    except Exception as e:
        print(f"ERROR: Could not load MNIST dataset → {e}")
        print("       Ensure you have internet access for the first download.")
        sys.exit(1)

    print(f"  Training images shape  : {x_train.shape}")
    print(f"  Training labels shape  : {y_train.shape}")
    print(f"  Testing  images shape  : {x_test.shape}")
    print(f"  Testing  labels shape  : {y_test.shape}")
    print("  ✔  MNIST dataset loaded.")

    return x_train, y_train, x_test, y_test


# ================================================================
# SECTION 3 – Visualize Sample Training Images
# ================================================================

def visualize_samples(x_train, y_train):
    """
    Display the first 10 MNIST training images in a 2×5 grid.

    MNIST digits are 28×28 px grayscale images representing handwritten
    digits 0–9.  Each pixel value is an integer in [0, 255].

    Args:
        x_train (np.ndarray): Training images.
        y_train (np.ndarray): Training labels.
    """
    print("\nStep 3: Visualizing sample training images …")

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle("Sample MNIST Training Digits", fontsize=14, fontweight="bold")

    for i in range(10):
        ax = fig.add_subplot(2, 5, i + 1)
        ax.imshow(x_train[i], cmap="gray")
        ax.set_title(f"Digit: {y_train[i]}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    save_fig(fig, "sample_mnist_digits.png")
    print("  ✔  Sample digits figure saved.")


# ================================================================
# SECTION 4 – Preprocess the Dataset
# ================================================================

def preprocess(x_train, y_train, x_test, y_test):
    """
    Normalize pixel values to [0, 1] and reshape images to add the
    grayscale channel dimension required by the CNN.

    Normalization:
        Dividing by 255 maps each pixel from [0, 255] → [0.0, 1.0].
        This stabilizes gradient descent and speeds up convergence.

    Reshaping:
        The CNN expects input of shape (samples, height, width, channels).
        MNIST is grayscale so channels = 1.

    Args:
        x_train, y_train, x_test, y_test: Raw MNIST arrays.

    Returns:
        tuple: Preprocessed (x_train, y_train, x_test, y_test).
    """
    print("\nStep 4: Preprocessing dataset …")

    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test  = x_test.reshape(-1,  28, 28, 1)

    print(f"  Preprocessed training shape : {x_train.shape}")
    print(f"  Preprocessed testing shape  : {x_test.shape}")
    print("  Labels kept as integers (sparse categorical cross-entropy).")
    print("  ✔  Preprocessing complete.")

    return x_train, y_train, x_test, y_test


# ================================================================
# SECTION 5 – Build the CNN Model
# ================================================================

def build_model():
    """
    Build the CNN model following the architecture in the PDF.

    Architecture explanation:
        Conv2D(32, 3×3, ReLU)   : Learns 32 local feature maps.
        MaxPooling2D(2×2)        : Halves spatial dimensions; reduces params.
        Conv2D(64, 3×3, ReLU)   : Learns 64 higher-level features.
        MaxPooling2D(2×2)        : Further spatial reduction.
        Flatten                  : Converts 2-D feature maps to 1-D vector.
        Dense(64, ReLU)          : Fully connected layer for classification.
        Dense(10, Softmax)       : Output probabilities for 10 digit classes.

    ReLU (Rectified Linear Unit): Introduces non-linearity; ignores negatives.
    Softmax                     : Converts raw scores to class probabilities
                                  that sum to 1.

    Returns:
        tf.keras.Model: Compiled-ready Sequential CNN.
    """
    print("\nStep 5: Building CNN model …")

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu",
                      input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])

    print("\n  CNN Architecture:")
    model.summary()
    print("\n  ✔  CNN model built.")
    return model


# ================================================================
# SECTION 6 – Compile the Model
# ================================================================

def compile_model(model):
    """
    Compile the CNN with the settings from the PDF.

    Optimizer – Adam:
        Adaptive learning rate optimizer; generally the best default
        choice for most deep-learning tasks.

    Loss – Sparse Categorical Cross-Entropy:
        Suitable when labels are integers (not one-hot vectors).
        Measures how far predicted probabilities are from the true class.

    Metric – Accuracy:
        Fraction of correctly classified examples in each batch.

    Args:
        model (tf.keras.Model): Uncompiled CNN.
    """
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("\n  CNN model compiled successfully!")


# ================================================================
# SECTION 7 – Train the Model
# ================================================================

def train_model(model, x_train, y_train):
    """
    Train the CNN for 5 epochs with a 10% validation split.

    Epochs       : One complete pass through all 60,000 training images.
    Batch size   : Number of images processed before updating weights.
    Validation   : 10% of training data (6,000 images) reserved to
                   monitor generalisation during training.

    Args:
        model   (tf.keras.Model): Compiled CNN.
        x_train (np.ndarray)    : Preprocessed training images.
        y_train (np.ndarray)    : Training labels (integers).

    Returns:
        tf.keras.callbacks.History: Training history object.
    """
    print("\nStep 7: Training CNN (5 epochs, batch_size=64) …")
    t_start = time.perf_counter()

    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=1,
    )

    t_elapsed = time.perf_counter() - t_start
    print(f"\n  ✔  Training complete in {t_elapsed:.1f} s.")

    # Print final training metrics
    final_train_acc  = history.history["accuracy"][-1]
    final_val_acc    = history.history["val_accuracy"][-1]
    final_train_loss = history.history["loss"][-1]
    final_val_loss   = history.history["val_loss"][-1]

    print(f"  Final train accuracy      : {final_train_acc:.4f}")
    print(f"  Final validation accuracy : {final_val_acc:.4f}")
    print(f"  Final train loss          : {final_train_loss:.4f}")
    print(f"  Final validation loss     : {final_val_loss:.4f}")

    return history


# ================================================================
# SECTION 8 – Evaluate on Test Data
# ================================================================

def evaluate_model(model, x_test, y_test):
    """
    Evaluate the trained model on the 10,000 test images and generate
    per-class classification metrics.

    Metrics:
        Precision  : Of all predicted positives, how many were correct?
        Recall     : Of all actual positives, how many were found?
        F1-score   : Harmonic mean of precision and recall.
        Support    : Number of actual samples in each class.

    Args:
        model  (tf.keras.Model): Trained CNN.
        x_test (np.ndarray)    : Preprocessed test images.
        y_test (np.ndarray)    : True test labels.

    Returns:
        tuple[float, float, np.ndarray]:
            test_accuracy, test_loss, y_pred (predicted class indices)
    """
    print("\nStep 8: Evaluating on test data …")

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_accuracy * 100:.2f} %")

    # Generate class predictions
    print("\n  Generating predictions …")
    y_probability = model.predict(x_test, verbose=0)
    y_pred        = np.argmax(y_probability, axis=1)

    # Classification report
    print("\n  Classification Report:")
    print("  " + "-" * 56)
    report_str = classification_report(
        y_test, y_pred,
        target_names=[str(d) for d in range(10)],
        digits=4,
        zero_division=0,
    )
    print(report_str)

    return test_accuracy, test_loss, y_pred


# ================================================================
# SECTION 9 – Predict Unseen Test Images
# ================================================================

def visualize_predictions(x_test, y_test, y_pred):
    """
    Display 10 test images with actual and predicted labels in a 2×5 grid.

    Args:
        x_test (np.ndarray): Preprocessed test images (N,28,28,1).
        y_test (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    """
    print("\nStep 9: Displaying predictions on 10 test images …")

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle("CNN Predictions on Test Images",
                 fontsize=14, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        ax.imshow(x_test[i].reshape(28, 28), cmap="gray")
        color = "green" if y_pred[i] == y_test[i] else "red"
        ax.set_title(
            f"Actual: {y_test[i]}\nPredicted: {y_pred[i]}",
            fontsize=9, color=color,
        )
        ax.axis("off")

    plt.tight_layout()
    save_fig(fig, "correct_predictions.png")
    print("  ✔  Prediction figure saved.")


# ================================================================
# SECTION 10 – Misclassified Images
# ================================================================

def visualize_misclassified(x_test, y_test, y_pred):
    """
    Find all misclassified examples and display up to 10 in a 2×5 grid.

    Misclassified examples reveal which digit pairs the model confuses most
    often (e.g., 4 vs. 9, 3 vs. 5) and guide future improvements.

    Args:
        x_test (np.ndarray): Preprocessed test images.
        y_test (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        np.ndarray: Indices of misclassified images.
    """
    print("\nStep 10: Identifying misclassified images …")

    wrong = np.where(y_pred != y_test)[0]
    print(f"  Total Misclassified Images: {len(wrong)}")

    if len(wrong) == 0:
        print("  (Perfect classification — no misclassified images!)")
        return wrong

    n_show = min(10, len(wrong))
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle(f"Misclassified Images (showing {n_show} of {len(wrong)})",
                 fontsize=14, fontweight="bold")

    for i, index in enumerate(wrong[:10]):
        ax = axes.flat[i]
        ax.imshow(x_test[index].reshape(28, 28), cmap="gray")
        ax.set_title(
            f"Actual: {y_test[index]}\nPred: {y_pred[index]}",
            fontsize=9, color="red",
        )
        ax.axis("off")

    # Hide any unused subplots
    for j in range(n_show, 10):
        axes.flat[j].axis("off")

    plt.tight_layout()
    save_fig(fig, "misclassified_images.png")
    print("  ✔  Misclassified images figure saved.")

    return wrong


# ================================================================
# SECTION 11 – Training / Validation Accuracy Plot
# ================================================================

def plot_accuracy(history):
    """
    Plot training and validation accuracy curves over all epochs.

    Rising training accuracy and stable/rising validation accuracy
    indicates healthy learning.  A large gap between the two may
    indicate overfitting.

    Args:
        history (History): Keras training history object.
    """
    print("\nStep 11: Plotting accuracy curves …")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["accuracy"],     label="Training Accuracy",   marker="o")
    ax.plot(history.history["val_accuracy"], label="Validation Accuracy", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training and Validation Accuracy",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "training_validation_accuracy.png")
    print("  ✔  Accuracy plot saved.")


# ================================================================
# SECTION 12 – Training / Validation Loss Plot
# ================================================================

def plot_loss(history):
    """
    Plot training and validation loss curves over all epochs.

    Decreasing loss on both sets means the model is learning.
    If training loss falls but validation loss rises, the model is
    overfitting (memorising training data rather than generalising).

    Args:
        history (History): Keras training history object.
    """
    print("\nStep 12: Plotting loss curves …")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["loss"],     label="Training Loss",   marker="o")
    ax.plot(history.history["val_loss"], label="Validation Loss", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "training_validation_loss.png")
    print("  ✔  Loss plot saved.")


# ================================================================
# SECTION 13 – Confusion Matrix
# ================================================================

def plot_confusion_matrix(y_test, y_pred):
    """
    Compute and display the 10×10 confusion matrix as a heatmap.

    Diagonal cells (row == column) represent correct predictions.
    Off-diagonal cells show which digits are confused with each other.
    Bright off-diagonal entries point to the most common errors.

    Args:
        y_test (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    """
    print("\nStep 13: Generating confusion matrix …")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10),
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("Actual Label",    fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "confusion_matrix.png")
    print("  ✔  Confusion matrix saved.")

    return cm


# ================================================================
# SECTION 14 – Model Performance Analysis
# ================================================================

def performance_analysis(test_accuracy, test_loss, y_test, y_pred,
                         wrong, cm, history):
    """
    Print a structured performance analysis of the trained CNN.

    Args:
        test_accuracy (float)      : Final test accuracy (0–1).
        test_loss     (float)      : Final test loss.
        y_test        (np.ndarray) : True labels.
        y_pred        (np.ndarray) : Predicted labels.
        wrong         (np.ndarray) : Indices of misclassified samples.
        cm            (np.ndarray) : 10×10 confusion matrix.
        history       (History)    : Training history.
    """
    correct = len(y_test) - len(wrong)

    print("\n" + "=" * 60)
    print("  MODEL PERFORMANCE ANALYSIS")
    print("=" * 60)
    print(f"\n  Final test accuracy   : {test_accuracy * 100:.2f} %")
    print(f"  Final test loss       : {test_loss:.4f}")
    print(f"  Correctly classified  : {correct} / {len(y_test)}")
    print(f"  Misclassified         : {len(wrong)} / {len(y_test)}")

    # Most confused digit pair (off-diagonal maximum)
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)  # ignore diagonal (correct predictions)
    max_idx = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
    print(f"\n  Most confused pair    : Actual {max_idx[0]} → Predicted {max_idx[1]}"
          f"  ({cm_copy[max_idx]} times)")

    print("\n  Possible factors affecting accuracy:")
    factors = [
        "Differences in individual handwriting styles.",
        "Visually similar digits (e.g., 4 vs. 9, 3 vs. 5, 7 vs. 1).",
        "Variations in stroke thickness and image quality.",
        "Limited model capacity (small CNN trained for only 5 epochs).",
        "Potential overfitting or underfitting depending on data variety.",
    ]
    for f in factors:
        print(f"    • {f}")
    print()


# ================================================================
# SECTION 15 – Practical Applications
# ================================================================

def print_applications():
    """
    Print real-world applications of CNN-based digit/image classification.
    """
    print("\n" + "=" * 60)
    print("  PRACTICAL APPLICATIONS OF CNN-BASED IMAGE CLASSIFICATION")
    print("=" * 60)

    apps = [
        ("Handwritten Digit Recognition",
         "Digitise handwritten numbers in forms, notebooks, and surveys."),
        ("Postal Code Recognition",
         "Automatically read ZIP codes on envelopes for mail sorting."),
        ("Bank Cheque Digit Recognition",
         "Extract and verify numerical amounts written on cheques."),
        ("Automatic Form Processing",
         "Convert filled paper forms into searchable digital data."),
        ("Document Digitization",
         "Scan historical documents and recognise handwritten content."),
        ("Number Recognition in Scanned Docs",
         "Extract serial numbers, dates, and codes from scanned pages."),
        ("General Image Classification",
         "Classify objects, medical images, satellite imagery, and more."),
    ]

    for app, desc in apps:
        print(f"\n  • {app}")
        print(f"      {desc}")
    print()


# ================================================================
# SECTION 16 – Save Model and Classification Report
# ================================================================

def save_model_and_report(model, test_accuracy, test_loss,
                          y_test, y_pred, wrong):
    """
    Save the trained Keras model and a text classification report.

    Files saved:
        mnist_cnn_model.keras       – trained CNN weights
        classification_report.txt   – metrics text summary

    Args:
        model         (tf.keras.Model): Trained CNN.
        test_accuracy (float)         : Test accuracy (0–1).
        test_loss     (float)         : Test loss.
        y_test        (np.ndarray)    : True labels.
        y_pred        (np.ndarray)    : Predicted labels.
        wrong         (np.ndarray)    : Misclassified indices.
    """
    print("\nStep 16: Saving model and classification report …")

    # ---- Save Keras model ----
    model_path = os.path.join(OUTPUT_DIR, "mnist_cnn_model.keras")
    try:
        model.save(model_path)
        print(f"  ✔  Model saved  →  {model_path}")
    except Exception as e:
        print(f"  ✘  Model save failed: {e}")

    # ---- Save text report ----
    txt_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    try:
        with open(txt_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("  Experiment 10 – CNN Digit Classification Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"  Test Accuracy       : {test_accuracy * 100:.2f} %\n")
            f.write(f"  Test Loss           : {test_loss:.4f}\n")
            f.write(f"  Correctly Classified: {len(y_test) - len(wrong)}\n")
            f.write(f"  Misclassified       : {len(wrong)}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(
                y_test, y_pred,
                target_names=[str(d) for d in range(10)],
                digits=4,
                zero_division=0,
            ))
        print(f"  ✔  Report saved →  {txt_path}")
    except Exception as e:
        print(f"  ✘  Report save failed: {e}")


# ================================================================
# MAIN PROGRAM
# ================================================================

def main():
    """
    Orchestrates Experiment 10 step by step.
    """
    print("=" * 60)
    print("  Experiment 10 – Handwritten Digit Classification")
    print("                  Using CNN (MNIST)")
    print("=" * 60)
    print(f"\n  TensorFlow version : {tf.__version__}")
    print(f"  Output directory   : {OUTPUT_DIR}")

    # Detect GPU availability
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"  GPU available      : {len(gpus)} device(s) detected")
    else:
        print("  GPU                : Not detected — running on CPU")

    # ---- Step 2: Load dataset ----------------------------------------
    x_train_raw, y_train, x_test_raw, y_test = load_mnist()

    # ---- Step 3: Visualize samples -----------------------------------
    visualize_samples(x_train_raw, y_train)

    # ---- Step 4: Preprocess ------------------------------------------
    x_train, y_train, x_test, y_test = preprocess(
        x_train_raw, y_train, x_test_raw, y_test
    )

    # ---- Step 5: Build model -----------------------------------------
    model = build_model()

    # ---- Step 6: Compile model ---------------------------------------
    print("\nStep 6: Compiling CNN …")
    compile_model(model)

    # ---- Step 7: Train model -----------------------------------------
    history = train_model(model, x_train, y_train)

    # ---- Step 8: Evaluate --------------------------------------------
    test_accuracy, test_loss, y_pred = evaluate_model(model, x_test, y_test)

    # ---- Step 9: Prediction visualization ---------------------------
    visualize_predictions(x_test, y_test, y_pred)

    # ---- Step 10: Misclassified images -------------------------------
    wrong = visualize_misclassified(x_test, y_test, y_pred)

    # ---- Step 11: Accuracy plot --------------------------------------
    plot_accuracy(history)

    # ---- Step 12: Loss plot ------------------------------------------
    plot_loss(history)

    # ---- Step 13: Confusion matrix -----------------------------------
    cm = plot_confusion_matrix(y_test, y_pred)

    # ---- Step 14: Performance analysis --------------------------------
    performance_analysis(test_accuracy, test_loss,
                         y_test, y_pred, wrong, cm, history)

    # ---- Step 15: Practical applications -----------------------------
    print_applications()

    # ---- Step 16: Save model and report ------------------------------
    save_model_and_report(model, test_accuracy, test_loss,
                          y_test, y_pred, wrong)

    # ---- Done ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Experiment 10 completed successfully!")
    print(f"  All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
