from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import datasets, losses, optimizers

from _telegram import send_img, send_simple_message
from csv_util import read_csv, write_csv
from models.gan.proGAN import ProGAN

# Define some constants
EXPERIMENTS_INPUT = "progan_experiments.csv"
SEND_TELEGRAM = True


class UnknowDataset(Exception):
    pass


def prepare_dataset(dataset_name, batch_size, normalize=True):
    if dataset_name.lower() == "mnist":
        (x_train, _), (x_test, _) = datasets.mnist.load_data()
        all_digits = np.concatenate([x_train, x_test])
        # all_digits = np.concatenate([x_train[:1000], x_test[:1000]])

        if normalize:
            all_digits = all_digits.astype("float32") / 255.0

        all_digits = np.reshape(all_digits, (-1, 28, 28, 1))

        ds = tf.data.Dataset.from_tensor_slices(all_digits)
        ds = ds.shuffle(buffer_size=1024).batch(batch_size)

        # Return th dataset generator and the dataset size
        return ds, all_digits.shape[0]

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

    # Function to call before each grow
    def before_grow_callback(gan):
        """
        Generate images of the current size and send by telegram
        """
        # Gen samples imgs and save it
        sample_imgs = proGAN.generate(samples).numpy()

        generated_files = []
        for idx, tmp_img in enumerate(sample_imgs):
            tmp_img_path = results_folder.joinpath(
                "{}x{}_{}.jpeg".format(
                    gan.current_output_size, gan.current_output_size, idx
                )
            )
            tmp_img = tmp_img.reshape(
                tmp_img.shape[0:2]
            )  # Convert to the current output size x size (2D)
            im = Image.fromarray(
                obj=tmp_img, mode="L"
            )  # L: 8-bit pixels, black and white
            im.save(tmp_img_path)
            generated_files.append(tmp_img_path)

        if SEND_TELEGRAM:
            send_simple_message(
                text="{}x{} images: ".format(
                    gan.current_output_size, gan.current_output_size
                )
            )
            for tmp_img_path in generated_files[:1]:  # Limit the imgs sent by telegram
                send_img(img_path=tmp_img_path)

    ds, dataset_size = prepare_dataset(dataset_name=dataset_name, batch_size=batch_size)
    proGAN = ProGAN(
        latent_dim=latent_dim,
        channels=1,
        fmap_base=fmap_base,
        fmap_max=fmap_max,
        fmap_decay=fmap_decay,
    )
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
# RUN EXPERIMENTS
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
