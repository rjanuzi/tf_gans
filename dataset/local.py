import traceback
from multiprocessing.dummy import Pool
from random import shuffle

import numpy as np
import pandas as pd
from PIL import Image

from dataset import DATASET_FOLDER, DATASET_RAW_IMGS_FOLDER
from dataset.remote import download_img, get_remote_imgs_list

RAW_DATASET_INDEX_PATH = r"%s/dataset_index.csv" % DATASET_FOLDER
TRAINING_DATASET_INDEX_PATH = r"%s/training_dataset_index.csv" % DATASET_FOLDER
VALIDATION_DATASET_INDEX_PATH = r"%s/validation_dataset_index.csv" % DATASET_FOLDER
TEST_DATASET_INDEX_PATH = r"%s/test_dataset_index.csv" % DATASET_FOLDER


class NotInitiated(Exception):
    pass


class DownloadError(Exception):
    pass


def __persist_index(index_df, path):
    index_df.to_csv(path, sep=";", index=False)


def generate_dataset_index():
    temp_data_dict = {
        "id": [],
        "name": [],
        "img_type": [],
        "pixels_x": [],
        "pixels_y": [],
        "age": [],
        "sex": [],
        "body_location": [],
        "benign_malignant": [],
        "diagnosis": [],
        "diagnosis_confirm_type": [],
        "melanocytic": [],
        "downloaded": [],
    }

    print("Querying the imgs list from isic-archive.")

    imgs_info = []
    for offset in range(0, 100000, 1000):
        new_list = get_remote_imgs_list(list_limit=1000, offset=offset)
        if not new_list:
            break
        imgs_info += new_list

    print("Data fetched.")

    for info in imgs_info:
        temp_data_dict["id"].append(info.get("_id"))
        temp_data_dict["name"].append(info.get("name"))

        if "meta" in info.keys() and "acquisition" in info["meta"].keys():
            acquisition = info.get("meta").get("acquisition")
            temp_data_dict["img_type"].append(acquisition.get("image_type"))
            temp_data_dict["pixels_x"].append(acquisition.get("pixelsX"))
            temp_data_dict["pixels_y"].append(acquisition.get("pixelsY"))
        else:
            temp_data_dict["img_type"].append(None)
            temp_data_dict["pixels_x"].append(None)
            temp_data_dict["pixels_y"].append(None)

        if "meta" in info.keys() and "clinical" in info["meta"].keys():
            clinical = info.get("meta").get("clinical")
            temp_data_dict["age"].append(clinical.get("age_approx"))
            temp_data_dict["sex"].append(clinical.get("sex"))
            temp_data_dict["body_location"].append(clinical.get("anatom_site_general"))
            temp_data_dict["benign_malignant"].append(clinical.get("benign_malignant"))
            temp_data_dict["diagnosis"].append(clinical.get("diagnosis"))
            temp_data_dict["diagnosis_confirm_type"].append(
                clinical.get("diagnosis_confirm_type")
            )
            temp_data_dict["melanocytic"].append(clinical.get("melanocytic"))
        else:
            temp_data_dict["age"].append(None)
            temp_data_dict["sex"].append(None)
            temp_data_dict["body_location"].append(None)
            temp_data_dict["benign_malignant"].append(None)
            temp_data_dict["diagnosis"].append(None)
            temp_data_dict["diagnosis_confirm_type"].append(None)
            temp_data_dict["melanocytic"].append(None)

        temp_data_dict["downloaded"] = False

    print("Persisting in dataset_index.")
    df = pd.DataFrame(temp_data_dict)
    __persist_index(df, RAW_DATASET_INDEX_PATH)


def read_dataset_index(path=RAW_DATASET_INDEX_PATH):
    try:
        return pd.read_csv(path, sep=";")
    except FileNotFoundError:
        generate_dataset_index()
        return pd.read_csv(path, sep=";")


def look_local_imgs():
    """
    Using the dataset_index, look into the RAW_IMGS folders for
    the downloaded imgs, updating the 'downloaded' columns of the
    index to True or False.
    """
    raise NotImplementedError()


def get_imgs_names(init_if_need=False):
    try:
        return read_dataset_index(RAW_DATASET_INDEX_PATH)["name"]
    except:
        if init_if_need:
            generate_dataset_index()
            return read_dataset_index(RAW_DATASET_INDEX_PATH)["name"]
        else:
            raise NotInitiated()


def __read_img_data(img_path, resize_params):
    img = Image.open(img_path)
    if resize_params:
        img = img.resize(size=(resize_params["new_width"], resize_params["new_height"]))

    # TODO - Add the other operations here

    return np.array(img, dtype="float32") / 255.0


def get_img_path(img_name, dataset_folder=DATASET_RAW_IMGS_FOLDER):
    return r"{}/{}.jpg".format(dataset_folder, img_name)


def get_img_data(
    img_name,
    data_set_folder=DATASET_RAW_IMGS_FOLDER,
    download_if_need=True,
    resize_params=None,
):
    img_path = get_img_path(img_name, data_set_folder)
    dataset_index = read_dataset_index(RAW_DATASET_INDEX_PATH)
    row_index = dataset_index.index[dataset_index["name"] == img_name].tolist()

    if not len(row_index):
        return None

    row_index = row_index[0]

    if dataset_index.iloc[row_index]["downloaded"]:
        try:
            return __read_img_data(img_path, resize_params)
        except:
            print(traceback.format_exc())
            print("Downloading %s again..." % img_name)
            if download_img(id=dataset_index.iloc[row_index]["id"], img_name=img_name):
                return __read_img_data(img_path, resize_params)

    elif download_if_need:
        if download_img(id=dataset_index.iloc[row_index]["id"], img_name=img_name):
            dataset_index.at[row_index, "downloaded"] = True
            __persist_index(dataset_index, RAW_DATASET_INDEX_PATH)

            return __read_img_data(img_path, resize_params)

    return None


def download_all_imgs():
    print("Downloading all ISIC imgs")
    imgs_downloaded = 0
    dataset_index = read_dataset_index(RAW_DATASET_INDEX_PATH)

    print("Index fetched")

    for index, row in dataset_index.iterrows():
        if row["downloaded"]:
            continue

        if not download_img(id=row["id"], img_name=row["name"]):
            raise DownloadError()

        dataset_index.at[index, "downloaded"] = True

        imgs_downloaded += 1
        if imgs_downloaded % 100 == 0:
            print("%d imgs downloaded" % imgs_downloaded)
            __persist_index(dataset_index, RAW_DATASET_INDEX_PATH)

    print("All images downloaded")


def generate_classification_targets():
    ds = read_dataset_index(RAW_DATASET_INDEX_PATH)

    # Apply the following filters:
    #   1) Only Demoscopic imgs
    #   2) Only confirmed benign or malignant
    #   3) Removed empty diagnosis
    ds = ds[
        (ds["img_type"] == "dermoscopic")
        & (
            (ds["benign_malignant"] == "benign")
            | (ds["benign_malignant"] == "malignant")
        )
        & (ds["diagnosis"] != "")
    ]

    # Generate the malignant classification target column
    ds["is_malignant"] = ds.apply(
        lambda row: row["benign_malignant"] == "malignant", axis=1
    )

    # Generate the melanoma classification target column
    ds["is_melanoma"] = ds.apply(lambda row: row["diagnosis"] == "melanoma", axis=1)

    # Generate the malignant_melanoma classification target column
    ds["is_malignant_melanoma"] = ds.apply(
        lambda row: (row["benign_malignant"] == "malignant")
        and (row["diagnosis"] == "melanoma"),
        axis=1,
    )

    return ds


def generate_training_datasets(
    training_set_proportion=0.8,
    validation_set_proportion=0.1,
    target_col="is_malignant_melanoma",
):
    ds = generate_classification_targets()

    ds = ds.reindex(np.random.permutation(ds.index))
    ds_positives = ds[ds[target_col]]
    ds_negatives = ds[ds[target_col] != True]

    training_ds = pd.DataFrame()
    validation_ds = pd.DataFrame()
    test_ds = pd.DataFrame()

    # Add positives to datasets
    training_idx = int(len(ds_positives.index) * training_set_proportion)
    validation_idx = (
        int(len(ds_positives.index) * validation_set_proportion) + training_idx
    )

    training_ds = training_ds.append(
        other=ds_positives[:training_idx], verify_integrity=True
    )
    validation_ds = validation_ds.append(
        other=ds_positives[training_idx:validation_idx], verify_integrity=True
    )
    test_ds = test_ds.append(other=ds_positives[validation_idx:], verify_integrity=True)

    # Add negatives to datasets
    training_idx = int(len(ds_negatives.index) * training_set_proportion)
    validation_idx = (
        int(len(ds_negatives.index) * validation_set_proportion) + training_idx
    )

    training_ds = training_ds.append(
        other=ds_negatives[:training_idx], verify_integrity=True
    )
    validation_ds = validation_ds.append(
        other=ds_negatives[training_idx:validation_idx], verify_integrity=True
    )
    test_ds = test_ds.append(other=ds_negatives[validation_idx:], verify_integrity=True)

    __persist_index(
        index_df=training_ds.reindex(np.random.permutation(training_ds.index)),
        path=TRAINING_DATASET_INDEX_PATH,
    )
    __persist_index(
        index_df=validation_ds.reindex(np.random.permutation(validation_ds.index)),
        path=VALIDATION_DATASET_INDEX_PATH,
    )
    __persist_index(
        index_df=test_ds.reindex(np.random.permutation(test_ds.index)),
        path=TEST_DATASET_INDEX_PATH,
    )


def __get_dataset_pairs(
    index_df, target_col, batch_size, resize_params, max_dataset_loops
):
    def load_tuples(row):
        row = row[1]  # Ignoring dataframe idx
        temp_img = get_img_data(img_name=row["name"], resize_params=resize_params)

        return temp_img, (1 if row[target_col] else 0)

    # Provide (imgs, targets), shuffling after a loop over all the data
    dataset_loops = 0
    while dataset_loops < max_dataset_loops:
        index_df = index_df.reindex(np.random.permutation(index_df.index))

        last_idx = 0
        for batch_idx in range(batch_size, len(index_df.index), batch_size):
            sliced_df = index_df[last_idx:batch_idx]

            batch_data = None
            with Pool() as pool:
                batch_data = pool.map(load_tuples, sliced_df.iterrows())

            # Update the lower bound slice
            last_idx = batch_idx

            # Generate the X and y lists
            X, y = [], []
            for img, target in batch_data:
                X.append(img)
                y.append(target)

            yield (np.array(X, dtype="float32"), np.array(y, dtype="float32"))

        dataset_loops += 1


def make_train_generator(
    target_col="is_malignant_melanoma",
    batch_size=50,
    resize_params={"new_width": 200, "new_height": 200},
    max_dataset_loops=1,
):
    ds = read_dataset_index(TRAINING_DATASET_INDEX_PATH)

    # TODO - Execute a data augmentation in the dataset (adding extra information when we need to execute some operation before provide the img)

    return __get_dataset_pairs(
        ds, target_col, batch_size, resize_params, max_dataset_loops
    )


def make_validation_generator(target_col="is_malignant_melanoma", batch_size=50):
    ds = read_dataset_index(VALIDATION_DATASET_INDEX_PATH)
    return __get_dataset_pairs(ds, target_col, batch_size)


def make_test_generator(target_col="is_malignant_melanoma", batch_size=50):
    ds = read_dataset_index(TEST_DATASET_INDEX_PATH)
    return __get_dataset_pairs(ds, target_col, batch_size)


def get_dataset_imgs_paths(dataset_index_path=RAW_DATASET_INDEX_PATH):
    ds = read_dataset_index(dataset_index_path)
    return ds["name"].map(lambda img_name: get_img_path(img_name=img_name)).tolist()
