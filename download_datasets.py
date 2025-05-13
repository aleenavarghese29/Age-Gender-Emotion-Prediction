import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Helper function to download and extract
def download_and_extract(dataset, output_dir, zip_name):
    if not os.path.exists(output_dir):
        print(f"📥 Downloading {dataset}...")
        api.dataset_download_files(dataset, path=".", quiet=False)
        print(f"📦 Extracting {zip_name}...")
        with zipfile.ZipFile(f"{zip_name}.zip", 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"✅ Extracted to '{output_dir}/'")
    else:
        print(f"✅ {output_dir}/ already exists.")

# UTKFace (Age & Gender)
download_and_extract("jangedoo/utkface-new", "utkface_dataset", "utkface-new")

# FER2013 (Emotion)
download_and_extract("msambare/fer2013", "fer2013_dataset", "fer2013")
