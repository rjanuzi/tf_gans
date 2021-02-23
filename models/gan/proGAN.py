import tensorflow as tf
from tensorflow.keras import Model, layers, models

# def G_wgan_acgan(G, D, opt, training_set, minibatch_size,
#     cond_weight = 1.0): # Weight of the conditioning term.

#     latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
#     labels = training_set.get_random_labels_tf(minibatch_size)
#     fake_images_out = G.get_output_for(latents, labels, is_training=True)
#     fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
#     loss = -fake_scores_out

#     if D.output_shapes[1][1] > 0:
#         with tf.name_scope('LabelPenalty'):
#             label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
#         loss += label_penalty_fakes * cond_weight
#     return loss

# #----------------------------------------------------------------------------
# # Discriminator loss function used in the paper (WGAN-GP + AC-GAN).

# def D_wgangp_acgan(G, D, opt, training_set, minibatch_size, reals, labels,
#     wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
#     wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
#     wgan_target     = 1.0,      # Target value for gradient magnitudes.
#     cond_weight     = 1.0):     # Weight of the conditioning terms.

#     latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
#     fake_images_out = G.get_output_for(latents, labels, is_training=True)
#     real_scores_out, real_labels_out = fp32(D.get_output_for(reals, is_training=True))
#     fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
#     real_scores_out = tfutil.autosummary('Loss/real_scores', real_scores_out)
#     fake_scores_out = tfutil.autosummary('Loss/fake_scores', fake_scores_out)
#     loss = fake_scores_out - real_scores_out

#     with tf.name_scope('GradientPenalty'):
#         mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
#         mixed_images_out = tfutil.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
#         mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out, is_training=True))
#         mixed_scores_out = tfutil.autosummary('Loss/mixed_scores', mixed_scores_out)
#         mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
#         mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
#         mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
#         mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
#         gradient_penalty = tf.square(mixed_norms - wgan_target)
#     loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

#     with tf.name_scope('EpsilonPenalty'):
#         epsilon_penalty = tfutil.autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
#     loss += epsilon_penalty * wgan_epsilon

#     if D.output_shapes[1][1] > 0:
#         with tf.name_scope('LabelPenalty'):
#             label_penalty_reals = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=real_labels_out)
#             label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
#             label_penalty_reals = tfutil.autosummary('Loss/label_penalty_reals', label_penalty_reals)
#             label_penalty_fakes = tfutil.autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
#         loss += (label_penalty_reals + label_penalty_fakes) * cond_weight
#     return loss

class Generator(Model):
    def __init__(self, latent_dim, channels):
        super(Generator, self).__init__()

        # Resolution state
        self.resolution = 4

        # Config the first block (Resolution 4 = 4x4)
        self.reshape_1 = layers.Reshape(target_shape=(4, 4, channels), input_shape=(latent_dim,))
        self.pixel_norm_1 = layers.LayerNormalization(axis=1)
        self.dense_1 = layers.Dense(units=16, activation='linear')
        self.leaky_relu_1 = layers.LeakyReLU()
        self.pixel_norm_2 = layers.LayerNormalization(axis=1)
        self.conv_1 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')
        self.leaky_relu_2 = layers.LeakyReLU()
        self.pixel_norm_3 = layers.LayerNormalization(axis=1)

        # This layer in not added in the layers_sequence
        self.to_rgb = layers.Conv2D(filters=channels, kernel_size=1, strides=1, padding='same')

        # Helper list to the caller
        self.layers_sequence = [
                                    self.reshape_1,
                                    self.pixel_norm_1,
                                    self.dense_1,
                                    self.leaky_relu_1,
                                    self.pixel_norm_2,
                                    self.conv_1,
                                    self.leaky_relu_2,
                                    self.pixel_norm_3
                                ]

    def add_block(self):
        self.resolution *= 2 # Doubles resolution

        # Simple counter to correctly define layers names
        try:
            self.blocks_count += 1
        except:
            self.blocks_count = 1

        # 1) Add upscale layer
        new_layer_name = 'upscale_block_%d_1' % self.blocks_count
        new_layer = layers.UpSampling2D(interpolation='nearest')
        self.__dict__[new_layer_name] = new_layer
        self.layers_sequence.append(new_layer)

        # 2) Add convolutional layer
        new_layer_name = 'conv_%d_1' % self.blocks_count
        new_layer = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')
        self.__dict__[new_layer_name] = new_layer
        self.layers_sequence.append(new_layer)

        # 3) Add Leaky ReLU layer
        new_layer_name = 'leaky_relu_%d_1' % self.blocks_count
        new_layer = layers.LeakyReLU()
        self.__dict__[new_layer_name] = new_layer
        self.layers_sequence.append(new_layer)

        # 4) Add Pixel Normalization
        new_layer_name = 'pixel_norm_%d_1' % self.blocks_count
        new_layer = layers.LayerNormalization(axis=1)
        self.__dict__[new_layer_name] = new_layer
        self.layers_sequence.append(new_layer)

        # 5) Add convolutional layer
        new_layer_name = 'conv_%d_2' % self.blocks_count
        new_layer = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')
        self.__dict__[new_layer_name] = new_layer
        self.layers_sequence.append(new_layer)

        # 6) Add Leaky ReLU layer
        new_layer_name = 'leaky_relu_%d_2' % self.blocks_count
        new_layer = layers.LeakyReLU()
        self.__dict__[new_layer_name] = new_layer
        self.layers_sequence.append(new_layer)

        # 7) Add Pixel Normalization
        new_layer_name = 'pixel_norm_%d_2' % self.blocks_count
        new_layer = layers.LayerNormalization(axis=1)
        self.__dict__[new_layer_name] = new_layer
        self.layers_sequence.append(new_layer)

        # New to RGB layer
        # self.to_rgb = layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same')

    def call(self, inputs):
        # Foward data through all layers
        x = self.layers_sequence[0](inputs)
        for idx in range(1, len(self.layers_sequence), 1):
            x = self.layers_sequence[idx](x)
        return self.to_rgb(x)

class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Resolution state
        self.resolution = 4

        # This layer in not added in the layers_sequence
        self.from_rgb = layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same')
        self.leaky_relu_0 = layers.LeakyReLU()

        # Config the first block (Resolution 4 = 4x4)
        self.batch_norm = layers.BatchNormalization(trainable=False, virtual_batch_size=4) # Minibatch Standard Deviation Layers (TODO check this layer)
        self.conv_1 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')
        self.leaky_relu_1 = layers.LeakyReLU()
        self.dense_1 = layers.Dense(units=16, activation='linear')
        self.leaky_relu_2 = layers.LeakyReLU()
        self.output_faltten = layers.Flatten()
        self.dense_2 = layers.Dense(units=1, activation='softmax') # Output layer

        # Helper list to the caller
        self.layers_sequence = [
                                    self.conv_1,
                                    self.leaky_relu_1,
                                    self.dense_1,
                                    self.leaky_relu_2,
                                    self.output_faltten,
                                    self.dense_2
                                ]
    
    def add_block(self):
        self.resolution *= 2 # Doubles resolution

        # Simple counter to correctly define layers names
        try:
            self.blocks_count += 1
        except:
            self.blocks_count = 1

        # 1) Add downscale layer
        new_layer_name = 'downscale_block_%d_1' % self.blocks_count
        new_layer = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='same')
        self.__dict__[new_layer_name] = new_layer
        self.layers_sequence = [new_layer] + self.layers_sequence

        # 2) Add 2 pairs of Conv2D -> Leaky Relu
        for _ in range(2):
            # Leaky Relu first
            new_layer_name = 'leaky_relu_%d_1' % self.blocks_count
            new_layer = layers.LeakyReLU()
            self.__dict__[new_layer_name] = new_layer
            self.layers_sequence = [new_layer] + self.layers_sequence

            # Then Conv2D
            new_layer_name = 'conv_%d_1' % self.blocks_count
            new_layer = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')
            self.__dict__[new_layer_name] = new_layer
            self.layers_sequence = [new_layer] + self.layers_sequence

    def call(self, img_input):
        # Foward data through all layers
        x = self.from_rgb(img_input)
        x = self.leaky_relu_0(x)
        for idx in range(0, len(self.layers_sequence)-1, 1):
            x = self.layers_sequence[idx](x)
        return self.layers_sequence[-1](x)

class ProGAN(Model):
    G = None
    D = None
    latent_dim = None
    disc_optimizer = None
    gen_optimizer = None
    loss_fn = None

    def __init__(self, latent_dim=48, channels=3):
        assert 4*4*channels == latent_dim

        super(ProGAN, self).__init__()
        self.G = Generator(latent_dim=latent_dim, channels=channels)
        self.D = Discriminator()
        self.latent_dim = latent_dim
    
    def compile(self, disc_optimizer, gen_optimizer, loss_fn):
        super(ProGAN, self).compile()
        self.disc_optimizer = disc_optimizer
        self.gen_optimizer = gen_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        '''
        The training step is:
            1) Generate images from random noise
            2) Discriminate over the generated images and real images
            3) Calculate Discriminator loss
            4) Update Discriminator weights
            5) Generate new images from random noise
            6) Discriminate over the generated images
            7) Calculae Generator loss
            8) Update Generator weights
        '''

        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.G(random_latent_vectors)
        
        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.D(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.D.trainable_weights)
        self.disc_optimizer.apply_gradients(zip(grads, self.D.trainable_weights))

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.D(self.G(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.G.trainable_weights)
        self.gen_optimizer.apply_gradients(zip(grads, self.G.trainable_weights))
        
        return {"d_loss": d_loss, "g_loss": g_loss}

    def grow(self):
        '''
        Add a new block to generator and discriminator
        '''
        self.G.add_block()
        self.D.add_block()
    
    def generate(self, noises):
        return self.G(noises)

    def discriminate(self, img_inputs):
        return self.D(img_inputs)

# MNIST Example
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from tensorflow.keras import datasets, losses, optimizers
    from datetime import datetime

    tf.config.set_visible_devices([], 'GPU')

    BATCH_SIZE = 32
    # LATENT_DIM = 48
    LATENT_DIM = 16

    # Prepare input data
    (x_train, _), (x_test, _) = datasets.mnist.load_data()
    all_digits = np.concatenate([x_train, x_test])
    all_digits = all_digits.astype('float32') / 255.0
    all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
    dataset = tf.data.Dataset.from_tensor_slices(all_digits)
    dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

    # Instantiate and compile GAN
    proGAN = ProGAN(LATENT_DIM, channels=1) # 4x4   
    proGAN.compile(
                    disc_optimizer=optimizers.Adam(learning_rate=0.01),
                    gen_optimizer=optimizers.Adam(learning_rate=0.01),
                    loss_fn=losses.BinaryCrossentropy(from_logits=True)
    )

    # Pro GAN starts with images 4x4
    # In this step we'll also convert the grayscale (x, y, 1) images to RGB images (x, y, 3) 
    # dataset = dataset.map(lambda img: tf.image.grayscale_to_rgb(tf.image.resize(img, [4, 4])))

    # TESTING
    # ==============================================================================================
    # dataset = dataset.map(lambda img: tf.image.grayscale_to_rgb(tf.image.resize(img, [8, 8])))
    dataset = dataset.map(lambda img: tf.image.resize(img, [8, 8]))
    proGAN.grow() # 8x8
    # proGAN.grow() # 16x16
    # proGAN.grow() # 32x32
    
    # fig, axs = plt.subplots(3, 3)
    # dataset_ite = dataset.as_numpy_iterator()
    # for i in range(3):
    #     for j in range(3):
    #         img = dataset_ite.__next__()[0]
    #         # axs[i][j].imshow(img)
    #         axs[i][j].imshow(img, cmap='gray')

    # fig.show()
    # ==============================================================================================

    # Train with tensorboard
    # log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # proGAN.fit(dataset, epochs=10, callbacks=[tensorboard_callback])

    # Train without tensorboard
    proGAN.fit(dataset, epochs=10)

    # Generate some images with the trained GAN
    samples = tf.random.normal(shape=(9, LATENT_DIM))
    sample_imgs = proGAN.generate(samples).numpy()

    # Plot generated images
    fig, axs = plt.subplots(3, 3)
    tmp_idx = 0
    for i in range(3):
        for j in range(3):
            # axs[i][j].imshow(sample_imgs[tmp_idx])
            axs[i][j].imshow(sample_imgs[tmp_idx], cmap='gray')
            tmp_idx += 1
    fig.show()
