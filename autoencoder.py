from keras import Model
from keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
import numpy as np

class Autoencoder:
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        self.input_shape = input_shape  # [28, 28, 1]
        self.conv_filters = conv_filters  # [32, 64, 64, 64]
        self.conv_kernels = conv_kernels  # [3, 3, 3, 3]
        self.conv_strides = conv_strides  # [1, 2, 2, 1]
        self.latent_space_dim = latent_space_dim  # 2

        self.encoder = None
        self.decoder = None
        self.model = None

        self.num_conv_layers = len(conv_filters)
        self.shape_before_bottleneck = None
        self.model_input = None

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        model_input = self.model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")

    def _build_decoder(self):
        decoder_input = Input(shape=(self.latent_space_dim,), name="decoder_input")
        dense_layer = Dense(np.prod(self.shape_before_bottleneck), name="decoder_dense")(decoder_input)
        reshape_layer = Reshape(self.shape_before_bottleneck, name="decoder_reshape")(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding='same',
            name="decoder_output"
        )(conv_transpose_layers)
        output_layer = Activation("sigmoid", name="sigmoid")(decoder_output)
        self.decoder = Model(decoder_input, output_layer, name="decoder")

    def _add_conv_transpose_layers(self, x):
        for i in reversed(range(1, self.num_conv_layers)):
            x = Conv2DTranspose(
                filters=self.conv_filters[i],
                kernel_size=self.conv_kernels[i],
                strides=self.conv_strides[i],
                padding='same',
                name=f"decoder_conv_transpose_{self.num_conv_layers - i}"
            )(x)
            x = LeakyReLU(name=f"decoder_leakyrelu_{self.num_conv_layers - i}")(x)
            x = BatchNormalization(name=f"decoder_bn_{self.num_conv_layers - i}")(x)
        return x

    def _build_encoder(self):
        encoder_input = Input(shape=self.input_shape, name="encoder_input")
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self.model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_conv_layers(self, encoder_input):
        x = encoder_input
        for i in range(self.num_conv_layers):
            x = Conv2D(
                filters=self.conv_filters[i],
                kernel_size=self.conv_kernels[i],
                strides=self.conv_strides[i],
                padding='same',
                name=f"encoder_conv_layer_{i + 1}"
            )(x)
            x = LeakyReLU(name=f"encoder_leakyrelu_{i + 1}")(x)
            x = BatchNormalization(name=f"encoder_bn_{i + 1}")(x)
        return x

    def _add_bottleneck(self, x):
        self.shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten(name="encoder_flatten")(x)
        x = Dense(self.latent_space_dim, name="encoder_output")(x)
        return x

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss)

    def train(self, x_train, batch_size, epochs):
        self.model.fit(
            x_train,
            x_train,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True
        )

if __name__ == '__main__':
    autoencoder = Autoencoder(
        input_shape=(28, 28, 1),
        conv_filters=[32, 64, 64, 64],
        conv_kernels=[3, 3, 3, 3],
        conv_strides=[1, 2, 2, 1],
        latent_space_dim=2
    )
    autoencoder.summary()
