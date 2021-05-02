import os
from pathlib import Path

DATASET_FOLDER = Path("dataset")
DATASET_RAW_IMGS_FOLDER = DATASET_FOLDER.joinpath("RAW_IMGS")

# Create utility folders if it doesn't exists
TEMP_FOLDERS = [DATASET_RAW_IMGS_FOLDER]
for temp_folder in TEMP_FOLDERS:
    if not temp_folder.exists():
        os.mkdir(temp_folder)
