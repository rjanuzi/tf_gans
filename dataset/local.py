import traceback
from random import shuffle

import numpy as np
import pandas as pd
from PIL import Image

import dataset.image_util as image_util
from dataset import DATASET_FOLDER, DATASET_RAW_IMGS_FOLDER
from dataset.remote import download_img, get_remote_imgs_list

RAW_DATASET_INDEX_PATH = r'%s\dataset_index.csv' % DATASET_FOLDER
TRAINING_DATASET_INDEX_PATH = r'%s\training_dataset_index.csv' % DATASET_FOLDER
VALIDATION_DATASET_INDEX_PATH = r'%s\validation_dataset_index.csv' % DATASET_FOLDER
TEST_DATASET_INDEX_PATH = r'%s\test_dataset_index.csv' % DATASET_FOLDER

NEW_WIDTH = 150
NEW_HEIGHT = 112
MAX_DATASET_LOOPS = 3

class NotInitiated(Exception):
    pass

class DownloadError(Exception):
    pass

def __persist_index(index_df, path):
    index_df.to_csv(path, sep=';', index=False)

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
    __persist_index(df, RAW_DATASET_INDEX_PATH)

def read_dataset_index(path=RAW_DATASET_INDEX_PATH):
    try:
        return pd.read_csv(path, sep=';')
    except FileNotFoundError:
        generate_dataset_index()
        return pd.read_csv(path, sep=';')

def look_local_imgs():
    '''
    Using the dataset_index, look into the RAW_IMGS folders for
    the downloaded imgs, updating the 'downloaded' columns of the
    index to True or False.
    '''
    raise NotImplementedError()

def get_imgs_names(init_if_need=False):
    try:
        return read_dataset_index(RAW_DATASET_INDEX_PATH)['name']
    except:
        if init_if_need:
            generate_dataset_index()
            return read_dataset_index(RAW_DATASET_INDEX_PATH)['name']
        else:
            raise NotInitiated()

def __read_img_data(img_path):
    return np.array(Image.open(img_path))

def get_img_data(img_name, data_set_folder=DATASET_RAW_IMGS_FOLDER, download_if_need=True):
    img_path = r'%s\%s.jpg' % (data_set_folder, img_name)
    dataset_index = read_dataset_index(RAW_DATASET_INDEX_PATH)
    row_index = dataset_index.index[dataset_index['name']==img_name].tolist()

    if not len(row_index):
        return None
    
    row_index = row_index[0]

    if dataset_index.iloc[row_index]['downloaded']:
        try:
            return __read_img_data(img_path)
        except:
            print(traceback.format_exc())
            print('Downloading %s again...' % img_name)
            if download_img(id=dataset_index.iloc[row_index]['id'], img_name=img_name):         
                return __read_img_data(img_path)

    elif download_if_need:
        if download_img(id=dataset_index.iloc[row_index]['id'], img_name=img_name):
            dataset_index.at[row_index, 'downloaded'] = True
            __persist_index(dataset_index, RAW_DATASET_INDEX_PATH)
            
            return __read_img_data(img_path)
    
    return None

def download_all_imgs():
    imgs_downloaded = 0
    dataset_index = read_dataset_index(RAW_DATASET_INDEX_PATH)

    for index, row in dataset_index.iterrows():
        if row['downloaded']:
            continue

        if not download_img(id=row['id'], img_name=row['name']):
            raise DownloadError()

        dataset_index.at[index, 'downloaded'] = True

        imgs_downloaded += 1
        if imgs_downloaded % 100 == 0:
            print('%d imgs downloaded' % imgs_downloaded)
            __persist_index(dataset_index, RAW_DATASET_INDEX_PATH)

def generate_classification_targets():
    ds = read_dataset_index(RAW_DATASET_INDEX_PATH)
    
    # Apply the following filters:
    #   1) Only Demoscopic imgs
    #   2) Only confirmed benign or malignant
    #   3) Removed empty diagnosis
    ds = ds[(ds['img_type'] == 'dermoscopic') & 
            ((ds['benign_malignant'] == 'benign') | (ds['benign_malignant'] == 'malignant')) &
            (ds['diagnosis'] != '')]

    # Generate the malignant classification target column
    ds['is_malignant'] = ds.apply(lambda row: row['benign_malignant'] == 'malignant', axis=1)

    # Generate the melanoma classification target column
    ds['is_melanoma'] = ds.apply(lambda row: row['diagnosis'] == 'melanoma', axis=1)

    # Generate the malignant_melanoma classification target column
    ds['is_malignant_melanoma'] = ds.apply(lambda row: (row['benign_malignant'] == 'malignant') and (row['diagnosis'] == 'melanoma'), axis=1)

    return ds

def generate_training_datasets(training_set_proportion=0.8, validation_set_proportion=0.1, target_col='is_malignant_melanoma'):
    ds = generate_classification_targets()

    ds = ds.reindex(np.random.permutation(ds.index))
    ds_positives = ds[ds[target_col]]
    ds_negatives = ds[ds[target_col] != True]

    training_ds = pd.DataFrame()
    validation_ds = pd.DataFrame()
    test_ds = pd.DataFrame()

    # Add positives to datasets
    training_idx = int(len(ds_positives.index)*training_set_proportion)
    validation_idx = int(len(ds_positives.index)*validation_set_proportion)+training_idx

    training_ds = training_ds.append(other=ds_positives[:training_idx], verify_integrity=True)
    validation_ds = validation_ds.append(other=ds_positives[training_idx:validation_idx], verify_integrity=True)
    test_ds = test_ds.append(other=ds_positives[validation_idx:], verify_integrity=True)

    # Add negatives to datasets
    training_idx = int(len(ds_negatives.index)*training_set_proportion)
    validation_idx = int(len(ds_negatives.index)*validation_set_proportion)+training_idx

    training_ds = training_ds.append(other=ds_negatives[:training_idx], verify_integrity=True)
    validation_ds = validation_ds.append(other=ds_negatives[training_idx:validation_idx], verify_integrity=True)
    test_ds = test_ds.append(other=ds_negatives[validation_idx:], verify_integrity=True)

    __persist_index(index_df=training_ds.reindex(np.random.permutation(training_ds.index)), path=TRAINING_DATASET_INDEX_PATH)
    __persist_index(index_df=validation_ds.reindex(np.random.permutation(validation_ds.index)), path=VALIDATION_DATASET_INDEX_PATH)
    __persist_index(index_df=test_ds.reindex(np.random.permutation(test_ds.index)), path=TEST_DATASET_INDEX_PATH)

def __get_dataset_pairs(index_df, target_col, batch_size, resize=True):

    # Provide (imgs, targets), shuffling after a loop over all the data
    dataset_loops = 0
    while dataset_loops < MAX_DATASET_LOOPS:
        index_df = index_df.reindex(np.random.permutation(index_df.index))

        last_idx = 0
        for batch_idx in range(batch_size, len(index_df.index), batch_size):
            imgs = []
            targets = []
            sliced_df = index_df[last_idx:batch_idx]
            for _, row in sliced_df.iterrows():
                if resize:
                    temp_img = image_util.resize(get_img_data(row['name']), new_width=600, new_height=450)
                else:
                    temp_img = get_img_data(row['name'])
                
                imgs.append(temp_img)
                targets.append(1 if row[target_col] else 0)
            
            # Update the lower bound slice
            last_idx = batch_idx
            
            yield (np.array(imgs, dtype='float32'), np.array(targets, dtype='float32'))
        
        dataset_loops += 1

def make_train_generator(target_col='is_malignant_melanoma', batch_size=50):
    ds = read_dataset_index(TRAINING_DATASET_INDEX_PATH)

    # TODO - Execute a data augmentation in the dataset (adding extra information when we need to execute some operation before provide the img)

    return __get_dataset_pairs(ds, target_col, batch_size)

def make_validation_generator(target_col='is_malignant_melanoma', batch_size=50):
    ds = read_dataset_index(VALIDATION_DATASET_INDEX_PATH)
    return __get_dataset_pairs(ds, target_col, batch_size)

def make_test_generator(target_col='is_malignant_melanoma', batch_size=50):
    ds = read_dataset_index(TEST_DATASET_INDEX_PATH)
    return __get_dataset_pairs(ds, target_col, batch_size)