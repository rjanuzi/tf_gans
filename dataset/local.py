import numpy as np
import pandas as pd
from PIL import Image

from dataset import DATASET_RAW_IMGS_FOLDER
from dataset.remote import download_img, get_remote_imgs_list

DATASET_INDEX_PATH = r'%s\dataset_index.csv' % DATASET_RAW_IMGS_FOLDER

class NotInitiated(Exception):
    pass

class DownloadError(Exception):
    pass

def __persist_index(index_df):
    index_df.to_csv(DATASET_INDEX_PATH, sep=';', index=False)

def generate_dataset_index():
    temp_data_dict = {
                        'id': [],
                        'name': [],
                        'img_type': [],
                        'pixels_x': [],
                        'pixels_y': [],
                        'age': [],
                        'sex': [],
                        'body_location': [],
                        'benign_malignant': [],
                        'diagnosis': [],
                        'diagnosis_confirm_type': [],
                        'melanocytic': [],
                        'downloaded': []
    }
    imgs_info = get_remote_imgs_list()

    for info in imgs_info:
        temp_data_dict['id'].append(info.get('_id'))
        temp_data_dict['name'].append(info.get('name'))

        if 'meta' in info.keys() and 'acquisition' in info['meta'].keys():
            acquisition = info.get('meta').get('acquisition')
            temp_data_dict['img_type'].append(acquisition.get('image_type'))
            temp_data_dict['pixels_x'].append(acquisition.get('pixelsX'))
            temp_data_dict['pixels_y'].append(acquisition.get('pixelsY'))
        else:
            temp_data_dict['img_type'].append(None)
            temp_data_dict['pixels_x'].append(None)
            temp_data_dict['pixels_y'].append(None)

        if 'meta' in info.keys() and 'clinical' in info['meta'].keys():
            clinical = info.get('meta').get('clinical')
            temp_data_dict['age'].append(clinical.get('age_approx'))
            temp_data_dict['sex'].append(clinical.get('sex'))
            temp_data_dict['body_location'].append(clinical.get('anatom_site_general'))
            temp_data_dict['benign_malignant'].append(clinical.get('benign_malignant'))
            temp_data_dict['diagnosis'].append(clinical.get('diagnosis'))
            temp_data_dict['diagnosis_confirm_type'].append(clinical.get('diagnosis_confirm_type'))
            temp_data_dict['melanocytic'].append(clinical.get('melanocytic'))  
        else:
            temp_data_dict['age'].append(None)
            temp_data_dict['sex'].append(None)
            temp_data_dict['body_location'].append(None)
            temp_data_dict['benign_malignant'].append(None)
            temp_data_dict['diagnosis'].append(None)
            temp_data_dict['diagnosis_confirm_type'].append(None)
            temp_data_dict['melanocytic'].append(None)

        temp_data_dict['downloaded'] = False 
    
    df = pd.DataFrame(temp_data_dict)
    __persist_index(df)

def read_dataset_index():
    return pd.read_csv(DATASET_INDEX_PATH, sep=';')

def look_local_imgs():
    '''
    Using the dataset_index, look into the RAW_IMGS folders for
    the downloaded imgs, updating the 'downloaded' columns of the
    index to True or False.
    '''
    raise NotImplementedError()

def get_imgs_names(init_if_need=False):
    try:
        return read_dataset_index()['name']
    except:
        if init_if_need:
            generate_dataset_index()
            return read_dataset_index()['name']
        else:
            raise NotInitiated()

def __read_img_data(img_path):
    return np.array(Image.open(img_path))

def get_img_data(img_name, download_if_need=True):
    img_path = r'%s\%s.jpg' % (DATASET_RAW_IMGS_FOLDER, img_name)
    dataset_index = read_dataset_index()
    row_index = dataset_index.index[dataset_index['name']==img_name].tolist()

    if not len(row_index):
        return None
    
    row_index = row_index[0]

    if dataset_index.iloc[row_index]['downloaded']:
        return __read_img_data(img_path)
    elif download_if_need:
        if download_img(id=dataset_index.iloc[row_index]['id'], img_name=img_name):
            dataset_index.at[row_index, 'downloaded'] = True
            __persist_index(dataset_index)
            
            return __read_img_data(img_path)
    
    return None

def download_all_imgs():
    imgs_downloaded = 0
    dataset_index = read_dataset_index()

    for index, row in dataset_index.iterrows():
        if row['downloaded']:
            continue

        if not download_img(id=row['id'], img_name=row['name']):
            raise DownloadError()

        dataset_index.at[index, 'downloaded'] = True

        imgs_downloaded += 1
        if imgs_downloaded % 100 == 0:
            print('%d imgs downloaded' % imgs_downloaded)
            __persist_index(dataset_index)
