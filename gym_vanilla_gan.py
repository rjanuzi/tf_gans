from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import datasets, optimizers
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.python.keras.backend import sigmoid

from _telegram import send_img, send_simple_message
from models.gan.vanilla import VanillaGAN, d_loss_fn, g_loss_fn

SEND_TELEGRAM = True
EPOCHS = 10
LATENT_DIM = 256
TARGET_IMG_SIZE = 28
MNIST_IMG_SIZE = 28
CHANNELS = 1
K = 1
SAMPLES_COUNT = 10


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


def prepare_dataset(batch_size=32, cache=True, normalize=True, dataset_limit=None):
    (x_train, _), (x_test, _) = datasets.mnist.load_data()
    all_digits = (
        np.concatenate([x_train, x_test])[:dataset_limit]
        if dataset_limit
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


def prepare_experiment_results_folder(folder_name):
    p = Path(folder_name)
    p = p.joinpath(datetime.now().strftime("%Y%m%d_%H%M"))
    p.mkdir(parents=True, exist_ok=True)

    return p


start_time = time()

# Create the folder to store results
results_folder = prepare_experiment_results_folder(
    "vanilla_gan_{}_runs".format("mnist")
)

# Notify the experiment start by telegram
send_simple_message("Starting experiment in {}".format(results_folder))

# Gen noise samples
samples = tf.random.normal(shape=(SAMPLES_COUNT, LATENT_DIM), mean=1, stddev=1)

# Save samples in the results folder
np.save(file=results_folder.joinpath("samples.npy"), arr=samples.numpy())
np.savetxt(fname=results_folder.joinpath("samples.txt"), X=samples.numpy())

# Prepare the dataset
ds, dataset_size = prepare_dataset()
if TARGET_IMG_SIZE != MNIST_IMG_SIZE:
    ds = ds.map(
        lambda img: tf.image.resize(
            img,
            [TARGET_IMG_SIZE, TARGET_IMG_SIZE],
            # method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )
    )

# Take a look into the dataset

generated_files = []
# Iterate over 10 imgs from the first batch
dataset_samples = ds.take(10)
for idx, tmp_img in enumerate(list(dataset_samples)[0][:10]):
    tmp_img_path = results_folder.joinpath(
        "{}x{}_{}_dataset_ref.jpeg".format(TARGET_IMG_SIZE, TARGET_IMG_SIZE, idx)
    )

    save_img(img_data=tmp_img.numpy(), path=tmp_img_path)
    generated_files.append(tmp_img_path)

if SEND_TELEGRAM:
    send_simple_message(
        text="{}x{} dataset ref images: ".format(TARGET_IMG_SIZE, TARGET_IMG_SIZE)
    )
    for tmp_img_path in generated_files[:2]:  # Limit the imgs sent by telegram
        send_img(img_path=tmp_img_path)

# Instantiate a fresh new model and compile
vanilla_gan = VanillaGAN(
    target_img_size=TARGET_IMG_SIZE,
    channels=CHANNELS,
    latent_dim=LATENT_DIM,
    k=K,
    g_layers_units=[128, 256, 256],
    g_layers_activations=[relu, relu, sigmoid],
    d_layers_units=[128, 256, 256],
    d_layers_activations=[relu, relu, relu],
)
vanilla_gan.compile(
    d_optimizer=optimizers.Adam(
        learning_rate=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8
    ),
    g_optimizer=optimizers.Adam(
        learning_rate=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8
    ),
    g_loss_fn=g_loss_fn,
    d_loss_fn=d_loss_fn,
)

# Train
vanilla_gan.fit(ds, epochs=EPOCHS)

# Save samples for the generated images so farg
sample_imgs = vanilla_gan.generate(samples).numpy()
sample_imgs = sample_imgs.reshape(
    (SAMPLES_COUNT, TARGET_IMG_SIZE, TARGET_IMG_SIZE, CHANNELS)
)
generated_files = []
for idx, tmp_img in enumerate(sample_imgs):
    tmp_img_path = results_folder.joinpath(
        "{}x{}_{}.jpeg".format(TARGET_IMG_SIZE, TARGET_IMG_SIZE, idx)
    )

    save_img(img_data=tmp_img, path=tmp_img_path)
    generated_files.append(tmp_img_path)

if SEND_TELEGRAM:
    send_simple_message(text="{}x{} images: ".format(TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    for tmp_img_path in generated_files[:2]:  # Limit the imgs sent by telegram
        send_img(img_path=tmp_img_path)

# Save model
# TODO
