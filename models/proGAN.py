import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, layers, models

def discriminator_loss(a, b):
    # TODO
    return 0

def generator_loss(a, b):
    # TODO
    return 0

class Generator(models.Sequential):
    def __init__(self):
        super(Generator, self).__init__()

        # Resolution state
        self.resolution = 4

        # Config the first block (Resolution 4 = 4x4)
        self.reshape_1 = layers.Reshape(target_shape=(4, 4, 3), input_shape=(48,))
        self.pixel_norm_1 = layers.LayerNormalization(axis=1)
        self.dense_1 = layers.Dense(units=16, activation='linear')
        self.leaky_relu_1 = layers.LeakyReLU()
        self.pixel_norm_2 = layers.LayerNormalization(axis=1)
        self.conv_1 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')
        self.leaky_relu_2 = layers.LeakyReLU()
        self.pixel_norm_3 = layers.LayerNormalization(axis=1)

        # This layer in not added in the layers_sequence
        self.to_rgb = layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same')

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
        
        # Compile
        self.compile()
    
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

class Discriminator(models.Sequential):
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
        self.dense_2 = layers.Dense(units=2, activation='softmax') # Output layer

        # Helper list to the caller
        self.layers_sequence = [
                                    self.conv_1,
                                    self.leaky_relu_1,
                                    self.dense_1,
                                    self.leaky_relu_2,
                                    self.output_faltten,
                                    self.dense_2
                                ]
        
        # Compile
        self.compile()
    
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

class ProGAN:
    G = Generator()
    D = Discriminator()

    @staticmethod
    def gen_noise():
        return tf.random.normal([1, 48])
    
    @staticmethod
    def gen_random_img(size=(4,4), show=False):
        random_img = np.random.random((1, *size, 3))
        if show:
            plt.imshow(random_img[0, :, :, :])
            plt.title('{}'.format(random_img.shape))
            plt.show()

        return random_img
    
    def grow(self):
        self.G.add_block()
        self.D.add_block()
    
    def generate(self, noise, show=False):
        generated_img = self.G(noise)

        if show:
            plt.imshow(generated_img[0, :, :, :])
            plt.title('{}'.format(generated_img.shape))
            plt.show()

        return generated_img

    def discriminate(self, img_input):
        return self.D(img_input)
    
    def step(self, real_img, Z, show=False):
        '''
        Executes GAN-like step:
            1) Generate a fake img.
            2) Discriminate the fake img.
            3) Discriminate the real img.
            4) calculate the losses.
        '''
        fake_img = self.generate(noise=Z, show=show)
        fake_inference = self.discriminate(fake_img)
        real_img_inference = self.discriminate(real_img)

        return fake_inference, real_img_inference
        # TODO - Return generator and discriminator losses

# Discriminator Run
if __name__ == "__main__":
    proGAN = ProGAN()

    real_imgs = [proGAN.gen_random_img() for _ in range(2)] # TODO Get real imgs
    noises = [proGAN.gen_noise() for _ in range(2)]

    img_sizes = [4, 8, 16, 32, 64, 128, 256, 512]
    for img_size in img_sizes:
        if img_size >= 8:
            proGAN.grow()

            # TODO - With real imgs we will do a resize here
            real_imgs = [proGAN.gen_random_img(size=(img_size, img_size)) for _ in range(2)]

        for real_img, Z in zip(real_imgs, noises):
            print(proGAN.step(real_img, Z, show=True))
