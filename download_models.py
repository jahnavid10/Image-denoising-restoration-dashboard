from pathlib import Path
from huggingface_hub import hf_hub_download


MODEL_REPO_ID = "dulalajahnavi10/Image-Denoising-Dashboard"

MODEL_FILES = [
    "autoencoder.h5",
    "cbdnet.h5",
    "ircnn.h5",
    "unet.h5",
    "rcan.h5",
]

MODEL_DIR = Path("model_files")
MODEL_DIR.mkdir(exist_ok=True)


def download_models():
    for model_file in MODEL_FILES:
        print(f"Downloading {model_file}...")

        hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=model_file,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
        )

        print(f"Downloaded {model_file}")

    print("All model files downloaded successfully.")


if __name__ == "__main__":
    download_models()
