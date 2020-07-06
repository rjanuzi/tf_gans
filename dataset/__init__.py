import os
from pathlib import Path

DATASET_RAW_IMGS_FOLDER = Path(r'.\dataset\RAW_IMGS')

# Create utility folders if it doesn't exists
TEMP_FOLDERS = [DATASET_RAW_IMGS_FOLDER]
for temp_folder in TEMP_FOLDERS:
    if not temp_folder.exists():
        os.mkdir(temp_folder)