import os
import zipfile
import kaggle

# ✅ Set your Kaggle credentials
os.environ['KAGGLE_USERNAME'] = 'aleenads'
os.environ['KAGGLE_KEY'] = 'ac1a32800478812375078666823cef15'

# ✅ Download UTKFace dataset zip (if not already downloaded)
if not os.path.exists("utkface-new.zip"):
    print("📥 Downloading UTKFace dataset...")
    kaggle.api.dataset_download_files('jangedoo/utkface-new', path='.', unzip=False)
else:
    print("✅ Dataset already downloaded.")

# ✅ Unzip dataset (if not already unzipped)
if not os.path.exists("utkface_dataset"):
    print("📦 Extracting dataset...")
    with zipfile.ZipFile("utkface-new.zip", 'r') as zip_ref:
        zip_ref.extractall("utkface_dataset")
    print("✅ Dataset extracted to 'utkface_dataset/'")
else:
    print("✅ Dataset already extracted.")
