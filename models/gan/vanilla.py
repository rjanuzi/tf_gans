import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import activations
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.layers import Dense, Flatten


def d_loss_fn(real_logit, fake_logit):
    real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=real_logit, labels=tf.ones_like(real_logit)
        )
    )

    fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_logit, labels=tf.zeros_like(fake_logit)
        )
    )

    return real_loss + fake_loss


def g_loss_fn(fake_logit):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_logit, labels=tf.ones_like(fake_logit)
        )
    )


class Generator(Model):
    """
    The "Vanilla GAN"s Generator basically consists in a Multilayer Perceptron
    Net (aka as Dense Layer in Tensorflow/Keras) with common activation functions,
    like Rectifier Linear (ReLU) and Sigmoid.

    In this implementation are provide the following hyperparameters:

    taget_img_size: The desired image size, considering squared images.
    channels: How much layers are expected for the output images
    (Ex: 1 to monocolor or 3 to RGB).
    layers_units: A list with the units count in each Dense layer.
    layers_activations: A list with the activation functions to each Dense layer.
    """

    target_img_size = None
    channels = None
    layers_units = None
    activations = None
    layers_count = None
    layers_sequence = []
    output_layer = None

    def __init__(
        self,
        target_img_size=32,
        channels=3,
        layers_units=[128, 128],
        activations=[relu, sigmoid],
    ):
        super(Generator, self).__init__()

        # Register basic info
        self.target_img_size = target_img_size
        self.channels = channels
        self.layers_units = layers_units
        self.activations = activations
        self.layers_count = len(layers_units)

        # Create the layers sequence
        for idx, units in enumerate(self.layers_units):
            tmp_layer = Dense(units=units, activation=self.activations[idx])
            self.layers_sequence.append(tmp_layer)

        # Create the output layer, its a Dense configuration that generate a image
        self.output_layer = Dense(
            (self.target_img_size ** 2) * self.channels, activation=sigmoid
        )

    def call(self, inputs):
        x = self.layers_sequence[0](inputs)
        for tmp_layer in self.layers_sequence[1:]:
            x = tmp_layer(x)

        return self.output_layer(x)


class Discriminator(Model):
    """
    In original "Vanilla GAN" paper (by Goodfellow) experiments, the Discriminator is built using Maxout
    (Maxout Networks) layers, but for simplicity sake and in order to use only "official" TF lib, this
    implementation uses common Multilayer Perceptron Layers (aka as Dense Layer in Tensorflow/Keras)
    with common activation functions, like Rectifier Linear (ReLU) and Sigmoid.

    In this implementation are provide the following hyperparameters:

    layers_units: A list with the units count in each Dense layer.
    layers_activations: A list with the activation functions to each Dense layer.
    """

    layers_units = None
    activations = None
    flatten_layer = None
    layers_sequence = []
    output_layer = None

    def __init__(self, layers_units=[128, 128], activations=[relu, relu]):
        super(Discriminator, self).__init__()
        self.layers_units = layers_units
        self.activations = activations
        self.layers_count = len(layers_units)

        # Create the flatten layer to guarantee the input shape
        self.flatten_layer = Flatten()

        # Create the layers sequence
        for idx, units in enumerate(self.layers_units):
            tmp_layer = Dense(units=units, activation=self.activations[idx])
            self.layers_sequence.append(tmp_layer)

        # Create the output layer. Basically a simple 1 unit output that will inform if the input image is
        # fake or real
        self.output_layer = Dense(units=1, activation=relu)

    def call(self, inputs):
        x = self.flatten_layer(inputs)
        for tmp_layer in self.layers_sequence:
            x = tmp_layer(x)

        return self.output_layer(x)


class VanillaGAN(Model):
    target_img_size = None
    channels = None
    latent_dim = None
    k = None
    g_layers_units = None
    d_layers_units = None
    g_layers_activations = None
    d_layers_activations = None
    G = None
    D = None
    g_optimizer = None
    d_optimizer = None
    g_loss_fn = None
    d_loss_fn = None

    def __init__(
        self,
        target_img_size=32,
        channels=3,
        latent_dim=128,
        k=1,
        g_layers_units=[128, 128],
        g_layers_activations=[relu, sigmoid],
        d_layers_units=[128, 128],
        d_layers_activations=[relu, relu],
    ):
        super(VanillaGAN, self).__init__()
        self.target_img_size = target_img_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.k = k
        self.g_layers_units = g_layers_units
        self.g_layers_activations = g_layers_activations
        self.d_layers_units = d_layers_units
        self.d_layers_activations = d_layers_activations
        self.G = Generator(
            target_img_size=self.target_img_size,
            channels=self.channels,
            layers_units=self.g_layers_units,
            activations=self.g_layers_activations,
        )
        self.D = Discriminator(
            layers_units=self.d_layers_units, activations=self.d_layers_activations
        )

    def compile(self, g_optimizer, d_optimizer, g_loss_fn, d_loss_fn):
        super(VanillaGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def train_D(self, real_imgs, batch_size):
        with tf.GradientTape() as gt:
            z = tf.random.normal(shape=(batch_size, 1, 1, self.latent_dim))
            fake_imgs = self.G(z, training=True)

            real_imgs_d_logit = self.D(real_imgs, training=True)
            fake_imgs_d_logit = self.D(fake_imgs, training=True)

            d_loss = self.d_loss_fn(real_imgs_d_logit, fake_imgs_d_logit)

        d_grads = gt.gradient(d_loss, self.D.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.D.trainable_variables))

        return d_loss

    def train_G(self, batch_size):
        with tf.GradientTape() as gt:
            z = tf.random.normal(shape=(batch_size, 1, 1, self.latent_dim))
            fake_imgs = self.G(z, training=True)
            fake_imgs_d_logit = self.D(fake_imgs, training=True)
            g_loss = self.g_loss_fn(fake_imgs_d_logit)

        g_grads = gt.gradient(g_loss, self.G.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.G.trainable_variables))

        return g_loss

    def train_step(self, real_imgs):

        # Discover batch size
        batch_size = tf.shape(real_imgs)[0]

        # Train the Discriminator
        d_loss = self.train_D(real_imgs=real_imgs, batch_size=batch_size)

        # Train the Generator at "k" rate in relation to discriminator
        # g_loss = 0
        # if tf.equal(self.d_optimizer.iterations % self.k, 0):
        #     g_loss = self.train_G(batch_size=batch_size)

        g_loss = self.train_G(batch_size=batch_size)

        return {"d_loss": d_loss, "g_loss": g_loss}

    def generate(self, z=None):
        if z == None:
            z = tf.random.normal(shape=(1, 1, 1, self.latent_dim))
        return self.G(z)

    def discriminate(self, imgs):
        return self.D(imgs)
