from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import datasets, losses, optimizers

from _telegram import send_img, send_simple_message
from csv_util import read_csv, write_csv
from dataset.local import get_dataset_imgs_paths
from models.gan.proGAN import ProGAN

# Define some constants
SEND_TELEGRAM = True

EXPERIMENTS_INPUT = "progan_experiments_isic_target_128.csv"
# EXPERIMENTS_INPUT = "progan_experiments_mnist_target_32.csv"

DATASET_FORCE_WIDTH = 600
DATASET_FORCE_HEIGHT = 450

DATASET_IMGS_LIMIT = 10000
DATASET_PARAM_CACHE = True
DATASET_PARAM_NORMALIZE = True


class UnknowDataset(Exception):
    pass


def show_img(img_data):
    # When the image is black and white we shall remove the "channel" dimension
    if img_data.shape[-1] == 1 and len(img_data.shape) == 3:
        img_data = img_data[:, :, 0]

    # If the image have some value bigger than 1.0, its not normalized
    img_data = img_data if img_data.max() > 1.0 else (img_data + 1) * 127.5

    # If we have a 3 channel image, we need to use de mode 'RGB'
    im = (
        Image.fromarray(obj=img_data).convert("RGB")
        if len(img_data.shape) == 2
        else Image.fromarray(obj=img_data.astype("int8"), mode="RGB")
    )

    im.show()


def save_img(img_data, path):

    # When the image is black and white we shall remove the "channel" dimension
    if img_data.shape[-1] == 1 and len(img_data.shape) == 3:
        img_data = img_data[:, :, 0]

    # If the image have some value bigger than 1.0, its not normalized
    img_data = img_data if img_data.max() > 1.0 else (img_data + 1) * 127.5

    # If we have a 3 channel image, we need to use de mode 'RGB'
    im = (
        Image.fromarray(obj=img_data).convert("RGB")
        if len(img_data.shape) == 2
        else Image.fromarray(obj=img_data.astype("int8"), mode="RGB")
    )

    im.save(path)


def isic_preprocessing(path, normalize=True):
    image = tf.image.decode_jpeg(tf.io.read_file(path))
    image = tf.image.resize(
        image,
        [DATASET_FORCE_HEIGHT, DATASET_FORCE_WIDTH],
        # method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )  # The most common size in the dataset

    if normalize:
        image = tf.cast(image, tf.float32) * (2.0 / 255) - 1
        # image = tf.image.per_image_standardization(tf.cast(image, tf.float32))

    return image


def prepare_dataset(
    dataset_name, batch_size, cache=True, normalize=True, data_limit=None
):
    if dataset_name.lower() == "mnist":
        (x_train, _), (x_test, _) = datasets.mnist.load_data()
        all_digits = (
            np.concatenate([x_train, x_test])[:data_limit]
            if data_limit
            else np.concatenate([x_train, x_test])
        )

        if normalize:
            all_digits = all_digits.astype("float32") * (2.0 / 255) - 1

        all_digits = np.reshape(all_digits, (-1, 28, 28, 1))

        ds = tf.data.Dataset.from_tensor_slices(all_digits)
        ds = ds.shuffle(buffer_size=1024).batch(batch_size)

        if cache:
            ds = ds.cache()

        # Return th dataset generator and the dataset size
        return ds, all_digits.shape[0]

    elif dataset_name.lower() == "isic":
        files_paths = (
            get_dataset_imgs_paths()[:data_limit]
            if data_limit
            else get_dataset_imgs_paths()
        )
        ds = tf.data.Dataset.list_files(files_paths)
        ds = ds.map(
            lambda file_path: isic_preprocessing(file_path, normalize=normalize)
        )

        ds = ds.shuffle(buffer_size=1024).batch(batch_size)

        if cache:
            ds = ds.cache()

        # Return th dataset generator and the dataset size
        return ds, len(files_paths)

    # elif dataset_name.lower() == '<other_option>':

    else:
        raise UnknowDataset


def prepare_experiment_results_folder(folder_name):
    p = Path(folder_name)
    p = p.joinpath(datetime.now().strftime("%Y%m%d_%H%M"))
    p.mkdir(parents=True, exist_ok=True)

    return p


def run_proGAN_experiment(
    dataset_name,
    channels,
    batch_size,
    latent_dim,
    target_size,
    epochs_to_fade_in,
    epochs_to_stabilize,
    fmap_base,
    fmap_max,
    fmap_decay,
):
    start_time = time()

    # Create the folder to store results
    results_folder = prepare_experiment_results_folder(
        "proGAN_{}_runs".format(dataset_name.lower())
    )

    # Notify the experiment start by telegram
    if SEND_TELEGRAM:
        send_simple_message("Starting experiment in {}".format(results_folder))

    # Gen noise samples
    samples = tf.random.normal(shape=(10, latent_dim), mean=0, stddev=1)

    # Save samples in the results folder
    np.save(file=results_folder.joinpath("samples.npy"), arr=samples.numpy())
    np.savetxt(fname=results_folder.joinpath("samples.txt"), X=samples.numpy())

    # Function to call before each grow - Monitor the images being generated
    def before_grow_callback(gan, dataset_samples):
        """
        Generate images of the current size and send by telegram
        """
        # Gen samples imgs and save it
        sample_imgs = proGAN.generate(samples).numpy()

        # Save samples for the current dataset imgs (check the resizing quality)
        generated_files = []
        # Iterate over 10 imgs from the first batch
        for idx, tmp_img in enumerate(list(dataset_samples)[0][:10]):
            tmp_img_path = results_folder.joinpath(
                "{}x{}_{}_dataset_ref.jpeg".format(
                    gan.current_output_size, gan.current_output_size, idx
                )
            )

            save_img(img_data=tmp_img.numpy(), path=tmp_img_path)
            generated_files.append(tmp_img_path)

        if SEND_TELEGRAM:
            send_simple_message(
                text="{}x{} dataset ref images: ".format(
                    gan.current_output_size, gan.current_output_size
                )
            )
            for tmp_img_path in generated_files[:2]:  # Limit the imgs sent by telegram
                send_img(img_path=tmp_img_path)

        # Save samples for the generated images so far
        generated_files = []
        for idx, tmp_img in enumerate(sample_imgs):
            tmp_img_path = results_folder.joinpath(
                "{}x{}_{}.jpeg".format(
                    gan.current_output_size, gan.current_output_size, idx
                )
            )

            save_img(img_data=tmp_img, path=tmp_img_path)
            generated_files.append(tmp_img_path)

        if SEND_TELEGRAM:
            send_simple_message(
                text="{}x{} images: ".format(
                    gan.current_output_size, gan.current_output_size
                )
            )
            for tmp_img_path in generated_files[:2]:  # Limit the imgs sent by telegram
                send_img(img_path=tmp_img_path)

    # Prepare the dataset
    ds, dataset_size = prepare_dataset(
        dataset_name=dataset_name,
        batch_size=batch_size,
        cache=DATASET_PARAM_CACHE,
        normalize=DATASET_PARAM_NORMALIZE,
        data_limit=DATASET_IMGS_LIMIT,
    )

    # Instantiate a fresh new model
    proGAN = ProGAN(
        latent_dim=latent_dim,
        channels=channels,
        fmap_base=fmap_base,
        fmap_max=fmap_max,
        fmap_decay=fmap_decay,
    )

    # Start the fit & grow process
    proGAN.fit_and_grow(
        dataset=ds,
        dataset_size=dataset_size,
        batch_size=batch_size,
        target_size=target_size,
        disc_optimizer=optimizers.Adam(
            learning_rate=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8
        ),
        gen_optimizer=optimizers.Adam(
            learning_rate=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8
        ),
        loss_fn=losses.BinaryCrossentropy(from_logits=True),
        epochs_to_fade_in=epochs_to_fade_in,
        epochs_to_stabilize=epochs_to_stabilize,
        callback_before_grow=before_grow_callback,
    )

    # Save model
    # TODO

    return {
        "g_loss": "",  # TODO - How to get the last loss values
        "d_loss": "",  # TODO - How to get the last loss values
        "results_folder": str(results_folder),
        "total_time_s": time() - start_time,
    }


# ======================================================================
# Run experiments and store the results
# ======================================================================
def get_fld(fld_name, headers, row):
    return row[headers.index(fld_name)]


def set_fld(fld_name, headers, row, new_val):
    row[headers.index(fld_name)] = new_val


headers, rows = read_csv(EXPERIMENTS_INPUT)
for r in rows:
    # Skip rows where we have "end_time", since the test already had been executed
    if not get_fld("end_time", headers, r):

        set_fld("start_time", headers, r, datetime.now())
        write_csv(EXPERIMENTS_INPUT, headers, rows)

        # Run an experiment
        results = run_proGAN_experiment(
            dataset_name=get_fld("dataset_name", headers, r),
            channels=get_fld("channels", headers, r),
            batch_size=get_fld("batch_size", headers, r),
            latent_dim=get_fld("latent_dim", headers, r),
            target_size=get_fld("target_size", headers, r),
            epochs_to_fade_in=get_fld("epochs_to_fade_in", headers, r),
            epochs_to_stabilize=get_fld("epochs_to_stabilize", headers, r),
            fmap_base=get_fld("fmap_base", headers, r),
            fmap_max=get_fld("fmap_max", headers, r),
            fmap_decay=get_fld("fmap_decay", headers, r),
        )

        set_fld("end_time", headers, r, datetime.now())
        set_fld("total_time_s", headers, r, int(results["total_time_s"]))
        set_fld("g_loss", headers, r, results["g_loss"])
        set_fld("d_loss", headers, r, results["d_loss"])
        set_fld("results_folder", headers, r, results["results_folder"])
        write_csv(EXPERIMENTS_INPUT, headers, rows)

        if SEND_TELEGRAM:
            send_simple_message(
                "Total time {:d} seconds".format(int(results["total_time_s"]))
            )
