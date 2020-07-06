import pandas as pd

from dataset import DATASET_RAW_IMGS_FOLDER
from dataset.remote import get_remote_imgs_list

DATASET_INDEX_PATH = r'%s\dataset_index.csv' % DATASET_RAW_IMGS_FOLDER

def generate_dataset_index():
    index_df = pd.DataFrame()
    imgs_info = get_remote_imgs_list(10)

    for info in imgs_info:
        for k, v in info.items():
            if k not in index_df.columns:
                index_df[k] = [v]
            else:
                index_df[k].append(v)
    
    print(info)