MedMNISTv2 Dataset Download Instructions
The MedMNISTv2 dataset is a lightweight benchmark for 2D and 3D biomedical image classification. In this project, we use the 2D subsets (e.g., BloodMNIST, BreastMNIST, PathMNIST, RetinaMNIST).

**Option 1: Using Hugging Face Datasets**
The easiest way to access the data is via the Hugging Face datasets library. You can load a specific subset (e.g., BloodMNIST) with:

**Python**
from datasets import load_dataset
ds = load_dataset("albertvillanova/medmnist-v2", "bloodmnist")
To load another subset, replace "bloodmnist" with the desired dataset identifier (e.g., "breastmnist", "pathmnist", "retinamnist").

**Option 2: Manual Download**
Visit the official MedMNIST website: https://medmnist.com/
Navigate to the download section and select the desired subsets.
Follow the provided instructions to download the files.
Organize the downloaded data into subfolders under the data/ directory, for example:

data/
├── BloodMNIST/
│   ├── train_images.npy
│   ├── train_labels.npy
│   ├── val_images.npy
│   ├── val_labels.npy
│   └── test_images.npy, test_labels.npy
├── BreastMNIST/
├── PathMNIST/
└── RetinaMNIST/
Ensure that your code (or data loader scripts) point to the correct paths where these datasets are stored.

**Additional Information**
For more details on each dataset (image resolutions, number of classes, etc.), please refer to the MedMNISTv2 documentation on their website.

