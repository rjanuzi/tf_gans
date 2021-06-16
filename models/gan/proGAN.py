import functools
import math

import tensorflow as tf
from tensorflow.keras import Model, layers


def gradient_penalty(f, real, fake):
    def _gradient_penalty(f, real, fake=None):
        def _interpolate(a, b):
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random.uniform(shape=shape, minval=0.0, maxval=1.0)
            inter = a + alpha * (b - a)
            inter.set_shape(a.shape)
            return inter

        x = _interpolate(real, fake)
        with tf.GradientTape() as t:
            t.watch(x)
            pred = f(x)
        grad = t.gradient(pred, x)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        return gp

    gp = _gradient_penalty(f, real, fake)

    return gp


def number_of_filters(blocks, fmap_base, fmap_max, fmap_decay):
    return min(int(fmap_base / (2.0 ** (blocks * fmap_decay))), fmap_max)


class Generator(Model):
    channels = None
    fade_in_alpha = 0.0
    fade_in_block = None
    layers_sequence = None
    current_upscale_layer = None
    current_to_rgb = None
    new_to_rgb = None
    fmap_base = None
    fmap_max = None
    fmap_decay = None

    def __init__(self, latent_dim, channels, fmap_base, fmap_max, fmap_decay):
        super(Generator, self).__init__()

        self.fmap_base = fmap_base
        self.fmap_max = fmap_max
        self.fmap_decay = fmap_decay

        # Resolution state
        self.channels = channels
        self.resolution = 2
        self.blocks_count = 1

        filters_count = number_of_filters(
            self.blocks_count,
            fmap_base=self.fmap_base,
            fmap_max=self.fmap_max,
            fmap_decay=self.fmap_decay,
        )

        # Config the first block (Resolution 4 = 4x4)
        self.pixel_norm_0 = layers.LayerNormalization(axis=1, epsilon=1e-8)
        self.dense_0 = layers.Dense(units=filters_count * 16, activation="linear")
        self.reshape_1 = layers.Reshape(target_shape=(4, 4, filters_count))
        self.pixel_norm_1 = layers.LayerNormalization(axis=1, epsilon=1e-8)
        self.conv_0 = layers.Conv2D(
            filters=filters_count,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=layers.LeakyReLU(alpha=0.2),
        )
        self.pixel_norm_2 = layers.LayerNormalization(axis=1, epsilon=1e-8)

        # This layer in not added in the layers_sequence
        self.current_to_rgb = layers.Conv2D(
            filters=channels,
            kernel_size=1,
            strides=1,
            padding="same",
            activation="linear",
            trainable=True,
        )

        # Helper list to the caller
        self.layers_sequence = [
            self.pixel_norm_0,
            self.dense_0,
            self.reshape_1,
            self.pixel_norm_1,
            self.conv_0,
            self.pixel_norm_2,
        ]

    def add_block(self):
        self.resolution *= 2
        self.blocks_count += 1

        # Check if we're in middle of a fade-in process
        # If yes, we shall add the "new" block to the layers sequence, since a new
        # one will be generated in this method
        if self.fade_in_block:
            self.layers_sequence += (
                self.fade_in_block
            )  # Concat new block in the default caller's list
            self.current_to_rgb = (
                self.new_to_rgb
            )  # Replace the old to-rgb to with the new one

        # This indicates to the "call" method that we're in a fade-in process
        self.fade_in_block = []  # Helper calling list to fade-in
        self.fade_in_alpha = 0.0

        # Calculate filtes count
        filters_count = number_of_filters(
            self.blocks_count,
            fmap_base=self.fmap_base,
            fmap_max=self.fmap_max,
            fmap_decay=self.fmap_decay,
        )

        # 1) Add upscale layer
        new_layer_name = "upscale_block_%d_1" % self.blocks_count
        new_layer = layers.UpSampling2D(interpolation="nearest", trainable=False)
        self.__dict__[new_layer_name] = new_layer
        self.fade_in_block.append(new_layer)
        self.current_upscale_layer = new_layer  # Let's save the current upscale layer to use in the fade-in process.

        # 2) Add convolutional layer
        new_layer_name = "conv_%d_1" % self.blocks_count
        new_layer = layers.Conv2D(
            filters=filters_count,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=layers.LeakyReLU(alpha=0.2),
        )
        self.__dict__[new_layer_name] = new_layer
        self.fade_in_block.append(new_layer)

        # 3) Add Pixel Normalization
        new_layer_name = "pixel_norm_%d_1" % self.blocks_count
        new_layer = layers.LayerNormalization(axis=1)
        self.__dict__[new_layer_name] = new_layer
        self.fade_in_block.append(new_layer)

        # 4) Add convolutional layer
        new_layer_name = "conv_%d_2" % self.blocks_count
        new_layer = layers.Conv2D(
            filters=filters_count,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=layers.LeakyReLU(alpha=0.2),
        )
        self.__dict__[new_layer_name] = new_layer
        self.fade_in_block.append(new_layer)

        # 5) Add Pixel Normalization
        new_layer_name = "pixel_norm_%d_2" % self.blocks_count
        new_layer = layers.LayerNormalization(axis=1)
        self.__dict__[new_layer_name] = new_layer
        self.fade_in_block.append(new_layer)

        # 6) New To-RGB layer. Not added to the "block sequence"
        self.new_to_rgb = layers.Conv2D(
            filters=self.channels,
            kernel_size=1,
            strides=1,
            padding="same",
            activation="linear",
            trainable=True,
        )

    def call(self, inputs):

        # Check if the fade-in of a new block was completed
        if self.fade_in_alpha >= 1.0 and self.fade_in_block:
            self.layers_sequence += (
                self.fade_in_block
            )  # Concat new block in the default caller's list
            self.current_to_rgb = (
                self.new_to_rgb
            )  # Replace the old to-rgb to with the new one
            self.fade_in_block = []

        # Foward data through all layers
        x = self.layers_sequence[0](inputs)
        for tmp_layer in self.layers_sequence[1:]:
            x = tmp_layer(x)

        # If we don't have a block being faded-in, just call the list
        if not self.fade_in_block:
            return self.current_to_rgb(x)  # Just generate the output image
        else:  # Fade-in ocurring

            # Pass through the new block
            x_new_block = self.fade_in_block[0](x)
            for tmp_layer in self.fade_in_block[1:]:
                x_new_block = tmp_layer(x_new_block)

            # Generate the output to the new block
            x_new_block = self.new_to_rgb(x_new_block)

            # Generate the output to the old block (using the old to_rgb and upscale layers)
            x_last_block = self.current_to_rgb(x)
            x_last_block = self.current_upscale_layer(x_last_block)

            # (new_result * alpha) + (last_result * (1-alpha))
            return (x_new_block * self.fade_in_alpha) + (
                x_last_block * (1.0 - self.fade_in_alpha)
            )


class Discriminator(Model):
    fade_in_alpha = 0.0
    fade_in_block = None
    discriminator_block = None
    layers_sequence = None
    new_downscale_layer = None
    current_from_rgb = None
    new_from_rgb = None
    fmap_base = None
    fmap_max = None
    fmap_decay = None

    def __init__(self, use_minibatch_norm, fmap_base, fmap_max, fmap_decay):
        super(Discriminator, self).__init__()

        self.fmap_base = fmap_base
        self.fmap_max = fmap_max
        self.fmap_decay = fmap_decay

        self.use_minibatch_norm = use_minibatch_norm

        # Resolution state
        self.resolution = 4
        self.blocks_count = 1

        # Calc the number of filters for the layers
        filters_count = number_of_filters(
            self.blocks_count,
            fmap_base=self.fmap_base,
            fmap_max=self.fmap_max,
            fmap_decay=self.fmap_decay,
        )

        # Save the lowest block simplify the fade-in process
        self.discriminator_block = []

        # This layer in not added in the layers_sequence
        self.current_from_rgb = layers.Conv2D(
            filters=filters_count,
            kernel_size=1,
            strides=1,
            padding="same",
            activation=layers.LeakyReLU(alpha=0.2),
            trainable=True,
        )

        if self.use_minibatch_norm:
            # TODO - How to add this?
            self.batch_norm = layers.BatchNormalization(
                trainable=False, virtual_batch_size=4
            )

        filters_count = number_of_filters(
            self.blocks_count,
            fmap_base=self.fmap_base,
            fmap_max=self.fmap_max,
            fmap_decay=self.fmap_decay,
        )
        self.conv_1 = layers.Conv2D(
            filters=filters_count,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=layers.LeakyReLU(alpha=0.2),
        )

        self.conv_2 = layers.Conv2D(
            filters=filters_count,
            kernel_size=4,
            strides=1,
            padding="same",
            activation=layers.LeakyReLU(alpha=0.2),
        )

        # self.dense_1 = layers.Dense(
        #     units=filters_count, activation=layers.LeakyReLU(alpha=0.2)
        # )
        # self.output_faltten = layers.Flatten()

        self.dense_1 = layers.Dense(units=1)  # Output layer

        # This block is static
        self.discriminator_block = [
            self.conv_1,
            self.conv_2,
            self.dense_1,
            # self.output_faltten,
            # self.dense_2,
        ]

        # Helper list to the caller
        self.layers_sequence = []

    def add_block(self):
        self.resolution *= 2  # Doubles resolution
        self.blocks_count += 1

        # Check if we're in middle of a fade-in process
        # If yes, we shall add the "new" block to the layers sequence, since a new
        # one will be generated in this method
        if self.fade_in_block:
            self.layers_sequence = (
                self.fade_in_block + self.layers_sequence
            )  # Concat new block in the default caller's list
            self.current_from_rgb = self.new_from_rgb

        # This indicates to the "call" method that we're in a fade-in process
        self.fade_in_block = []  # Helper calling list to fade-in
        self.fade_in_alpha = 0.0

        # Calc the number of filters for the layers
        filters_count = number_of_filters(
            self.blocks_count,
            fmap_base=self.fmap_base,
            fmap_max=self.fmap_max,
            fmap_decay=self.fmap_decay,
        )

        # This layer in not added in the layers_sequence
        self.new_from_rgb = layers.Conv2D(
            filters=filters_count,
            kernel_size=1,
            strides=1,
            padding="same",
            activation=layers.LeakyReLU(alpha=0.2),
            trainable=True,
        )

        # 1) Conv2D
        new_layer_name = "conv_%d_0" % self.blocks_count
        new_layer = layers.Conv2D(
            filters=filters_count,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=layers.LeakyReLU(alpha=0.2),
        )
        self.__dict__[new_layer_name] = new_layer
        self.fade_in_block = self.fade_in_block + [new_layer]

        # 2) Conv2D
        # We need to match the filters count of this layer (fitlers count of the output of the new block)
        # with the filters count expected for the "from-rgb" layer of the current block, since
        # we need to fade-in this one as input of the current block
        filters_count = number_of_filters(
            self.blocks_count - 1,
            fmap_base=self.fmap_base,
            fmap_max=self.fmap_max,
            fmap_decay=self.fmap_decay,
        )
        new_layer_name = "conv_%d_1" % self.blocks_count
        new_layer = layers.Conv2D(
            filters=filters_count,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=layers.LeakyReLU(alpha=0.2),
        )
        self.__dict__[new_layer_name] = new_layer
        self.fade_in_block = self.fade_in_block + [new_layer]

        # 3) Add downscale layer
        new_layer_name = "downscale_block_%d_0" % self.blocks_count
        new_layer = layers.AveragePooling2D(
            pool_size=(2, 2), strides=(2, 2), padding="same", trainable=False
        )
        self.__dict__[new_layer_name] = new_layer
        self.fade_in_block = self.fade_in_block + [new_layer]
        self.new_downscale_layer = (
            new_layer  # Easy access to this layer in order to do the fade-in
        )

    def call(self, img_input):
        # Check if the fade-in of a new block was completed
        if self.fade_in_alpha >= 1.0 and self.fade_in_block:
            self.layers_sequence = (
                self.fade_in_block + self.layers_sequence
            )  # Concat new block in the default caller's list
            self.current_from_rgb = (
                self.new_from_rgb
            )  # Replace the old from-RGB layer with the new one
            self.fade_in_block = []

        # If we're in a fade-in process
        if self.fade_in_block:
            # Pass through the new block (The last layer is the new_downscale_layer)
            x_new_block = self.new_from_rgb(img_input)
            for tmp_layer in self.fade_in_block:
                x_new_block = tmp_layer(x_new_block)

            # Generate the input to the last added layer. Basicaly we need to downscale (using the
            # new downscale layer) and pass trhough the current from-rgb
            x_last_block = self.new_downscale_layer(
                img_input
            )  # Downscale the input img
            x_last_block = self.current_from_rgb(x_last_block)

            # Generate the effective "x" using the fading-in alpha
            # (new_result * alpha) + (last_result * (1-alpha))
            x = (x_new_block * self.fade_in_alpha) + (
                x_last_block * (1.0 - self.fade_in_alpha)
            )
        else:
            x = self.current_from_rgb(
                img_input
            )  # Pass through the currnto from-RGB before continue

        # Pass through the intermediary layers if exist
        for tmp_layer in self.layers_sequence:
            x = tmp_layer(x)

        # The final block is different from all others, in order to generate the expected classification shape
        for tmp_layer in self.discriminator_block:
            x = tmp_layer(x)

        return x


class ProGAN(Model):
    G = None
    D = None
    latent_dim = None
    d_optimizer = None
    g_optimizer = None
    loss_fn = None
    fade_in_alpha = 1.0
    fading_in_block = False
    fade_in_increment = None
    current_output_size = 4
    steps = 0
    skipping_disc_training = True
    d_updates_per_g_update = 1

    def __init__(
        self,
        latent_dim=48,
        channels=3,
        use_minibatch_norm=False,
        fmap_base=8192,
        fmap_max=512,
        fmap_decay=1.0,
        d_updates_per_g_update=2,
    ):
        # assert 4*4*channels == latent_dim

        super(ProGAN, self).__init__()
        self.G = Generator(
            latent_dim=latent_dim,
            channels=channels,
            fmap_base=fmap_base,
            fmap_max=fmap_max,
            fmap_decay=fmap_decay,
        )
        self.D = Discriminator(
            use_minibatch_norm=use_minibatch_norm,
            fmap_base=fmap_base,
            fmap_max=fmap_max,
            fmap_decay=fmap_decay,
        )
        self.latent_dim = latent_dim
        self.d_updates_per_g_update = d_updates_per_g_update

    def compile(self, d_optimizer, g_optimizer, g_loss_fn, d_loss_fn):
        super(ProGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn

    def train_G(self, batch_size):
        """
        WGAN-GP's train step for Generator
        """
        with tf.GradientTape() as t:
            z = tf.random.normal(shape=(batch_size, 1, 1, self.latent_dim))
            x_fake = self.G(z, training=True)
            x_fake_d_logit = self.D(x_fake, training=True)
            G_loss = self.g_loss_fn(x_fake_d_logit)

        G_grad = t.gradient(G_loss, self.G.trainable_variables)
        self.g_optimizer.apply_gradients(zip(G_grad, self.G.trainable_variables))

        return {"g_loss": G_loss}

    def train_D(self, x_real, batch_size, gp_weight=10.0):
        """
        WGAN-GP's train step for Discriminator
        """
        with tf.GradientTape() as t:
            z = tf.random.normal(shape=(batch_size, 1, 1, self.latent_dim))
            x_fake = self.G(z, training=True)

            x_real_d_logit = self.D(x_real, training=True)
            x_fake_d_logit = self.D(x_fake, training=True)

            x_real_d_loss, x_fake_d_loss = self.d_loss_fn(
                x_real_d_logit, x_fake_d_logit
            )
            gp = gradient_penalty(
                functools.partial(self.D, training=True), x_real, x_fake
            )

            D_loss = (
                x_real_d_loss + x_fake_d_loss
            ) + gp * gp_weight  # Default GP Weight = 10.0

        D_grad = t.gradient(D_loss, self.D.trainable_variables)
        self.d_optimizer.apply_gradients(zip(D_grad, self.D.trainable_variables))

        return {"d_loss": x_real_d_loss + x_fake_d_loss, "gp": gp}

    def train_step(self, real_images):
        """
        The training step is:
            1) Generate images from random noise
            2) Discriminate over the generated images and real images
            3) Calculate Discriminator loss
            4) Update Discriminator weights
            5) Generate new images from random noise
            6) Discriminate over the generated images
            7) Calculae Generator loss
            8) Update Generator weights
        """
        # Update fade-in state
        if self.fading_in_block and self.fade_in_alpha >= 1.0:
            self.fading_in_block = False
            self.fade_in_alpha = 1.0
            self.G.fade_in_alpha = self.fade_in_alpha
            self.D.fade_in_alpha = self.fade_in_alpha

        elif self.fading_in_block:
            self.fade_in_alpha += self.fade_in_increment
            self.G.fade_in_alpha = self.fade_in_alpha
            self.D.fade_in_alpha = self.fade_in_alpha

        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]

        # Train D
        D_loss_dict = self.train_D(real_images, batch_size=batch_size)

        # Train G
        # Skip G trainings
        # if self.d_optimizer.iterations.numpy() % self.d_updates_per_g_update == 0:
        #     G_loss_dict = self.train_G(batch_size=batch_size)
        G_loss_dict = self.train_G(batch_size=batch_size)

        # random_latent_vectors = tf.random.normal(
        #     shape=(batch_size, self.latent_dim), mean=0, stddev=1
        # )

        # # Decode them to fake images
        # generated_images = self.G(random_latent_vectors)

        # # Combine them with real images
        # combined_images = tf.concat([generated_images, real_images], axis=0)

        # # Assemble labels discriminating real from fake images
        # labels = tf.concat(
        #     [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))],
        #     axis=0
        #     # [tf.ones((batch_size, 1)), tf.ones((batch_size, 1))*(-1)], axis=0
        # )

        # # Add random noise to the labels - important trick!
        # # labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # # Train the discriminator
        # # if tf.constant(not self.skipping_disc_training, dtype=tf.bool):
        # if self.skipping_disc_training:
        #     # Train more the generator than the discriminator
        #     with tf.GradientTape() as tape:
        #         predictions = self.D(combined_images)
        #         d_loss = self.loss_fn(labels, predictions)
        #     if self.steps >= 2:
        #         self.steps = 0
        #         grads = tape.gradient(d_loss, self.D.trainable_weights)
        #         self.d_optimizer.apply_gradients(
        #             zip(grads, self.D.trainable_weights)
        #         )
        #     else:
        #         self.steps += 1
        # else:
        #     with tf.GradientTape() as tape:
        #         predictions = self.D(combined_images)
        #         d_loss = self.loss_fn(labels, predictions)
        #     grads = tape.gradient(d_loss, self.D.trainable_weights)
        #     self.d_optimizer.apply_gradients(zip(grads, self.D.trainable_weights))

        # # Sample random points in the latent space
        # # random_latent_vectors = tf.random.normal(
        # #     shape=(batch_size, self.latent_dim), mean=0, stddev=1
        # # )

        # # Assemble labels that say "all real images"
        # is_real_labels = tf.zeros((batch_size, 1))
        # # is_real_labels = tf.ones((batch_size, 1))*(-1)

        # # Train the generator
        # with tf.GradientTape() as tape:
        #     # predictions = self.D(self.G(random_latent_vectors))
        #     predictions = self.D(generated_images)
        #     g_loss = self.loss_fn(is_real_labels, predictions)
        # grads = tape.gradient(g_loss, self.G.trainable_weights)
        # self.g_optimizer.apply_gradients(zip(grads, self.G.trainable_weights))

        # Train more the discriminator than the generator
        # with tf.GradientTape() as tape:
        #     predictions = self.D(self.G(random_latent_vectors))
        #     g_loss = self.loss_fn(is_real_labels, predictions)
        # if self.steps == 10:
        #     self.steps = 0
        #     grads = tape.gradient(g_loss, self.G.trainable_weights)
        #     self.g_optimizer.apply_gradients(zip(grads, self.G.trainable_weights))
        # else:
        #     self.steps += 1

        losses_dict = D_loss_dict
        losses_dict.update(G_loss_dict)

        return losses_dict

    def grow(self):
        """
        Add a new block to generator and discriminator
        """
        self.G.add_block()
        self.D.add_block()
        self.fade_in_alpha = 0.0
        self.fading_in_block = True

    def fit_and_grow(
        self,
        dataset,
        dataset_size,
        batch_size,
        target_size,
        d_optimizer,
        g_optimizer,
        g_loss_fn,
        d_loss_fn,
        epochs_to_fade_in,
        epochs_to_stabilize,
        callback_before_grow=None,
    ):
        assert target_size >= 4
        assert math.log(target_size, 2) == int(math.log(target_size, 2))

        # Calculate the increment to accomplish the number of epochs to use as fade-in of a new block
        self.fade_in_increment = (1 / (dataset_size / batch_size)) * epochs_to_fade_in

        # Determine the total epochs to each training size
        epochs = epochs_to_fade_in + epochs_to_stabilize

        self.compile(
            d_optimizer=d_optimizer,
            g_optimizer=g_optimizer,
            g_loss_fn=g_loss_fn,
            d_loss_fn=d_loss_fn,
        )

        self.current_output_size = 4  # Start size
        while True:
            # Resize dataset the the current network size
            dataset_resized = dataset.map(
                lambda img: tf.image.resize(
                    img,
                    [self.current_output_size, self.current_output_size],
                    # method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                )
            )
            self.fit(dataset_resized, epochs=epochs)

            if callback_before_grow:
                callback_before_grow(self, dataset_resized.take(5))

            # Doubles the output size
            self.current_output_size *= 2  # Doubles the output size
            if self.current_output_size > target_size:
                self.current_output_size = (
                    target_size  # Set the target size as final output size of the model
                )
                break

            # Add layers to Generator and Disciminator
            self.grow()

            # Re-compile the models to recalculate the execution graph (We  add new layers when grow the network)
            self.compile(
                d_optimizer=d_optimizer,
                g_optimizer=g_optimizer,
                g_loss_fn=g_loss_fn,
                d_loss_fn=d_loss_fn,
            )

    def generate(self, noises):
        return self.G(noises)

    def discriminate(self, img_inputs):
        return self.D(img_inputs)

    # def save(self, root_dir):
    #     self.G.save(root_dir.joinpath('generator'))
    #     self.D.save(root_dir.joinpath('discriminator'))
