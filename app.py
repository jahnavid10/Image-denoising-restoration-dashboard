import os
import time
from pathlib import Path
from typing import Dict, List

# Set Keras backend before importing keras
os.environ["KERAS_BACKEND"] = "tensorflow"

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import (
    Layer,
    Dense,
    GlobalAveragePooling2D,
    Reshape,
    Add,
    Activation,
    Lambda,
    Multiply,
)
from keras.models import load_model
from keras.saving import register_keras_serializable


# Custom RCAN Layer
@register_keras_serializable()
class ChannelAttention(Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = int(input_shape[-1])
        reduced_channels = max(1, channel // self.ratio)

        self.shared_dense_1 = Dense(reduced_channels, activation="relu")
        self.shared_dense_2 = Dense(channel)

        super().build(input_shape)

    def call(self, x):
        avg_pool = GlobalAveragePooling2D()(x)
        avg_pool = Reshape((1, 1, -1))(avg_pool)
        avg_pool = self.shared_dense_1(avg_pool)
        avg_pool = self.shared_dense_2(avg_pool)

        max_pool = Lambda(
            lambda tensor: tf.reduce_max(tensor, axis=[1, 2], keepdims=True)
        )(x)
        max_pool = self.shared_dense_1(max_pool)
        max_pool = self.shared_dense_2(max_pool)

        attention = Add()([avg_pool, max_pool])
        attention = Activation("sigmoid")(attention)

        return Multiply()([x, attention])

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config


# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_files"

MODEL_ORDER = ["autoencoder", "cbdnet", "ircnn", "unet", "rcan"]

MODEL_PATHS: Dict[str, Path] = {
    "autoencoder": MODEL_DIR / "autoencoder.h5",
    "cbdnet": MODEL_DIR / "cbdnet.h5",
    "ircnn": MODEL_DIR / "ircnn.h5",
    "unet": MODEL_DIR / "unet.h5",
    "rcan": MODEL_DIR / "rcan.h5",
}



# Model Loading
def load_all_models() -> Dict[str, object]:
    models = {}

    for model_name in MODEL_ORDER:
        model_path = MODEL_PATHS[model_name]

        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue

        try:
            print(f"Loading model: {model_name}")

            models[model_name] = load_model(
                str(model_path),
                compile=False,
                custom_objects={"ChannelAttention": ChannelAttention},
                safe_mode=False,
            )

            print(f"Loaded successfully: {model_name}")

        except Exception as e:
            print(f"Failed to load {model_name}: {e}")

    print(f"Final loaded models: {list(models.keys())}")
    return models


MODELS = load_all_models()


# Image Helpers
def resize_normalize(img_rgb: np.ndarray, size=(256, 256)) -> np.ndarray:
    """
    Resize uploaded image to the model input size and normalize it.

    All models were trained on 256 x 256 images.
    """
    resized = cv2.resize(img_rgb, size)
    return resized.astype("float32") / 255.0


def restore_image(model, img_norm: np.ndarray) -> np.ndarray:
    """
    Run model prediction and convert output back to uint8 RGB image.
    """
    pred = model.predict(np.expand_dims(img_norm, axis=0), verbose=0)
    restored = np.squeeze(pred, axis=0)

    restored = np.clip(restored * 255.0, 0, 255).astype("uint8")

    # If model outputs grayscale image: H x W
    if restored.ndim == 2:
        restored = cv2.cvtColor(restored, cv2.COLOR_GRAY2RGB)

    # If model outputs single-channel image: H x W x 1
    if restored.ndim == 3 and restored.shape[-1] == 1:
        restored = cv2.cvtColor(restored, cv2.COLOR_GRAY2RGB)

    return restored


def resize_panel_image(img: np.ndarray, max_single_width: int = 430) -> np.ndarray:
    """
    Resize image for cleaner dashboard display.
    """
    h, w = img.shape[:2]

    if w <= max_single_width:
        return img

    scale = max_single_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img, (new_w, new_h))


def make_side_by_side(
    noisy: np.ndarray,
    restored: np.ndarray,
    model_name: str = "",
    max_single_width: int = 430,
) -> np.ndarray:
    """
    Create a clear side-by-side visual:
    Noisy Input | Restored Output
    """
    noisy_display = resize_panel_image(
        noisy,
        max_single_width=max_single_width,
    )

    h, w = noisy_display.shape[:2]
    restored_display = cv2.resize(restored, (w, h))

    combined = np.concatenate([noisy_display, restored_display], axis=1)

    label_height = 48
    canvas = np.ones(
        (combined.shape[0] + label_height, combined.shape[1], 3),
        dtype=np.uint8,
    ) * 255

    canvas[label_height:, :, :] = combined

    title = model_name.upper() if model_name else "MODEL OUTPUT"

    cv2.putText(
        canvas,
        title,
        (20, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "Noisy Input",
        (20, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "Restored Output",
        (w + 20, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    return canvas


def get_image_size_message(image: np.ndarray) -> str:
    """
    Create a message explaining whether uploaded image matches 256 x 256.
    """
    h, w = image.shape[:2]

    if h == 256 and w == 256:
        return (
            "Uploaded image size: 256 × 256 pixels. "
            "This matches the model training size."
        )

    return (
        f"Uploaded image size: {w} × {h} pixels. "
        "The image was resized to 256 × 256 before prediction. "
        "For best results, upload a noisy image that is already 256 × 256 pixels."
    )


# Metrics
def calculate_no_reference_metrics(img_rgb: np.ndarray) -> dict:
    """
    Calculate no-reference image property metrics.

    Since no clean ground-truth image is uploaded, true PSNR, SSIM,
    MSE, or MAE cannot be calculated.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    brightness_mean = float(np.mean(gray))
    contrast_std = float(np.std(gray))

    sharpness_laplacian_var = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
    histogram = histogram / np.sum(histogram)
    histogram = histogram[histogram > 0]

    entropy = float(-np.sum(histogram * np.log2(histogram)))

    total_pixels = gray.size
    underexposed_pixels = np.sum(gray < 30)
    overexposed_pixels = np.sum(gray > 225)

    underexposed_percent = float((underexposed_pixels / total_pixels) * 100)
    overexposed_percent = float((overexposed_pixels / total_pixels) * 100)

    return {
        "brightness_mean": round(brightness_mean, 4),
        "contrast_std": round(contrast_std, 4),
        "sharpness_laplacian_variance": round(sharpness_laplacian_var, 4),
        "entropy": round(entropy, 4),
        "underexposed_percent": round(underexposed_percent, 4),
        "overexposed_percent": round(overexposed_percent, 4),
    }


def calculate_metric_changes(input_metrics: dict, restored_metrics: dict) -> dict:
    """
    Calculate restored metric minus uploaded noisy image metric.
    """
    changes = {}

    for key in input_metrics:
        changes[f"{key}_change"] = round(
            restored_metrics[key] - input_metrics[key],
            4,
        )

    return changes


def build_metrics_dataframe(
    model_name: str,
    input_metrics: dict,
    restored_metrics: dict,
    changes: dict,
    inference_time_ms: float,
) -> pd.DataFrame:
    """
    Build detailed metrics table for one model.
    """
    rows = []

    for metric_name in input_metrics:
        rows.append(
            {
                "model": model_name,
                "metric": metric_name,
                "uploaded_noisy_image": input_metrics[metric_name],
                "restored_image": restored_metrics[metric_name],
                "change_after_restoration": changes[f"{metric_name}_change"],
            }
        )

    rows.append(
        {
            "model": model_name,
            "metric": "inference_time_ms",
            "uploaded_noisy_image": "-",
            "restored_image": inference_time_ms,
            "change_after_restoration": "-",
        }
    )

    return pd.DataFrame(rows)


def build_summary_row(
    model_name: str,
    changes: dict,
    inference_time_ms: float,
) -> dict:
    """
    Build compact summary row for compare-all-models tab.
    """
    return {
        "model": model_name,
        "brightness_change": changes["brightness_mean_change"],
        "contrast_change": changes["contrast_std_change"],
        "sharpness_change": changes["sharpness_laplacian_variance_change"],
        "entropy_change": changes["entropy_change"],
        "underexposed_change": changes["underexposed_percent_change"],
        "overexposed_change": changes["overexposed_percent_change"],
        "inference_time_ms": inference_time_ms,
    }


# Gradio Functions
def run_single_model(image: np.ndarray, model_name: str):
    """
    Run one selected model on uploaded noisy image.
    """
    if image is None:
        raise gr.Error("Please upload an image.")

    if not MODELS:
        raise gr.Error("No models were loaded. Check your model_files folder.")

    if model_name not in MODELS:
        raise gr.Error(f"Model '{model_name}' is not available.")

    noisy_original = image.astype("uint8")

    size_message = get_image_size_message(noisy_original)

    noisy_norm = resize_normalize(noisy_original)

    start_time = time.time()
    restored_256 = restore_image(MODELS[model_name], noisy_norm)
    inference_time_ms = round((time.time() - start_time) * 1000, 4)

    h, w = noisy_original.shape[:2]
    restored_fullsize = cv2.resize(restored_256, (w, h))

    comparison_image = make_side_by_side(
        noisy_original,
        restored_fullsize,
        model_name=model_name,
        max_single_width=520,
    )

    input_metrics = calculate_no_reference_metrics(noisy_original)
    restored_metrics = calculate_no_reference_metrics(restored_fullsize)
    changes = calculate_metric_changes(input_metrics, restored_metrics)

    metrics_df = build_metrics_dataframe(
        model_name,
        input_metrics,
        restored_metrics,
        changes,
        inference_time_ms,
    )

    return comparison_image, metrics_df, size_message


def compare_all_models_clean(image: np.ndarray):
    """
    Run uploaded noisy image through all loaded models.
    """
    if image is None:
        raise gr.Error("Please upload an image.")

    if not MODELS:
        raise gr.Error("No models were loaded. Check your model_files folder.")

    noisy_original = image.astype("uint8")

    size_message = get_image_size_message(noisy_original)

    noisy_norm = resize_normalize(noisy_original)

    h, w = noisy_original.shape[:2]

    output_images = {model_name: None for model_name in MODEL_ORDER}
    detailed_metric_tables: List[pd.DataFrame] = []
    summary_rows = []

    for model_name in MODEL_ORDER:
        if model_name not in MODELS:
            continue

        model = MODELS[model_name]

        start_time = time.time()
        restored_256 = restore_image(model, noisy_norm)
        inference_time_ms = round((time.time() - start_time) * 1000, 4)

        restored_fullsize = cv2.resize(restored_256, (w, h))

        comparison_image = make_side_by_side(
            noisy_original,
            restored_fullsize,
            model_name=model_name,
            max_single_width=430,
        )

        output_images[model_name] = comparison_image

        input_metrics = calculate_no_reference_metrics(noisy_original)
        restored_metrics = calculate_no_reference_metrics(restored_fullsize)
        changes = calculate_metric_changes(input_metrics, restored_metrics)

        metrics_df = build_metrics_dataframe(
            model_name,
            input_metrics,
            restored_metrics,
            changes,
            inference_time_ms,
        )

        detailed_metric_tables.append(metrics_df)

        summary_rows.append(
            build_summary_row(
                model_name,
                changes,
                inference_time_ms,
            )
        )

    summary_df = pd.DataFrame(summary_rows)

    if detailed_metric_tables:
        detailed_df = pd.concat(detailed_metric_tables, ignore_index=True)
    else:
        detailed_df = pd.DataFrame()

    return (
        output_images["autoencoder"],
        output_images["cbdnet"],
        output_images["ircnn"],
        output_images["unet"],
        output_images["rcan"],
        summary_df,
        detailed_df,
        size_message,
    )


# Gradio Dashboard
available_models = list(MODELS.keys())

custom_css = """
footer {
    visibility: hidden;
}

.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
}

#main-title {
    text-align: center;
    padding: 18px;
    border-radius: 14px;
    background: linear-gradient(90deg, #1f2937, #374151);
    color: white;
    margin-bottom: 18px;
}

#main-title h1 {
    margin-bottom: 6px;
}

.small-note {
    font-size: 14px;
}
"""

with gr.Blocks(title="Image Denoising Dashboard", css=custom_css) as demo:
    gr.Markdown(
        """
        <div id="main-title">
            <h1>Image Denoising and Restoration Dashboard</h1>
            <p>
                Upload a noisy image, run trained restoration models,
                and compare reconstructed outputs with image quality metrics.
            </p>
        </div>
        """
    )

    with gr.Accordion("About this Dashboard", open=True):
        gr.Markdown(
            """
            This dashboard is designed for **image denoising and restoration** using deep learning models.

            The uploaded noisy image is passed through trained restoration models such as
            **Autoencoder, CBDNet, IRCNN, U-Net, and RCAN**. Each model attempts to reconstruct
            a cleaner version of the input image.

            The models were trained using images of size **256 × 256 pixels**.
            Therefore, the best and most reliable results are expected when the uploaded noisy image
            is also **256 × 256 pixels**.

            Images with other dimensions can still be uploaded. However, they are resized to
            **256 × 256** before model prediction because the models expect this fixed input size.
            After reconstruction, the output is resized back to the uploaded image size for display.

            Since only a noisy image is uploaded and no clean ground-truth image is provided,
            the dashboard reports **no-reference image quality metrics**, including brightness,
            contrast, sharpness, entropy, exposure percentage, and inference time.
            """
        )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                ### Supported Models

                - Autoencoder  
                - CBDNet  
                - IRCNN  
                - U-Net  
                - RCAN  
                """
            )

        with gr.Column(scale=1):
            gr.Markdown(
                """
                ### Recommended Input

                For best restoration quality, upload a noisy image with:

                - **Width:** 256 pixels  
                - **Height:** 256 pixels  
                - **Channels:** RGB image  
                """
            )

    with gr.Tab("Run One Model"):
        gr.Markdown(
            """
            ### Upload one image and test one selected denoising model.

            **Recommended image size:** 256 × 256 pixels.  
            The models were trained on 256 × 256 images, so images with the same size usually give the most optimal results.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Noisy Image",
                    type="numpy",
                    height=350,
                )

                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0] if available_models else None,
                    label="Choose Model",
                )

                run_button = gr.Button("Restore Image", variant="primary")

            with gr.Column(scale=2):
                output_image = gr.Image(
                    label="Noisy Input | Restored Output",
                    type="numpy",
                    height=520,
                )

        image_size_info = gr.Textbox(
            label="Image Size Information",
            interactive=False,
            lines=2,
        )

        output_metrics = gr.Dataframe(
            label="Detailed Metrics",
            interactive=False,
        )

        run_button.click(
            fn=run_single_model,
            inputs=[input_image, model_dropdown],
            outputs=[output_image, output_metrics, image_size_info],
        )

    with gr.Tab("Compare All Models"):
        gr.Markdown(
            """
            ### Compare all loaded models

            Upload one noisy image and run it through all available models.
            The outputs are separated into tabs so the interface stays clean.

            **Recommended image size:** 256 × 256 pixels for best restoration quality.
            """
        )

        compare_input = gr.Image(
            label="Upload Noisy Image",
            type="numpy",
            height=350,
        )

        compare_button = gr.Button("Compare All Models", variant="primary")

        compare_size_info = gr.Textbox(
            label="Image Size Information",
            interactive=False,
            lines=2,
        )

        compare_summary = gr.Dataframe(
            label="Summary Comparison",
            interactive=False,
        )

        gr.Markdown("### Model Outputs")

        with gr.Tabs():
            with gr.Tab("Autoencoder"):
                autoencoder_output = gr.Image(
                    label="Autoencoder Output",
                    type="numpy",
                    height=520,
                )

            with gr.Tab("CBDNet"):
                cbdnet_output = gr.Image(
                    label="CBDNet Output",
                    type="numpy",
                    height=520,
                )

            with gr.Tab("IRCNN"):
                ircnn_output = gr.Image(
                    label="IRCNN Output",
                    type="numpy",
                    height=520,
                )

            with gr.Tab("U-Net"):
                unet_output = gr.Image(
                    label="U-Net Output",
                    type="numpy",
                    height=520,
                )

            with gr.Tab("RCAN"):
                rcan_output = gr.Image(
                    label="RCAN Output",
                    type="numpy",
                    height=520,
                )

        with gr.Accordion("Detailed Metrics for All Models", open=False):
            compare_detailed = gr.Dataframe(
                label="Detailed Metrics",
                interactive=False,
            )

        compare_button.click(
            fn=compare_all_models_clean,
            inputs=[compare_input],
            outputs=[
                autoencoder_output,
                cbdnet_output,
                ircnn_output,
                unet_output,
                rcan_output,
                compare_summary,
                compare_detailed,
                compare_size_info,
            ],
        )

    with gr.Accordion("Metric Meaning", open=False):
        gr.Markdown(
            """
            - **brightness_mean**: Average grayscale brightness.
            - **contrast_std**: Standard deviation of grayscale intensity.
            - **sharpness_laplacian_variance**: Higher value usually means sharper image.
            - **entropy**: Amount of visual information/detail.
            - **underexposed_percent**: Percentage of very dark pixels.
            - **overexposed_percent**: Percentage of very bright pixels.
            - **change values**: Restored image metric minus noisy image metric.
            - **inference_time_ms**: Time taken by the model to reconstruct the image.
            """
        )


if __name__ == "__main__":
    demo.launch()