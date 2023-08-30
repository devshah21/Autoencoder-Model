from keras import Model
from keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose
from keras import backend as K
import numpy as np

class Autoencoder:
    
    
    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        self.input_shape = input_shape # [28, 28, 1]
        self.conv_filters = conv_filters # [2, 4, 8]
        self.conv_kernels = conv_kernels # [3, 5, 3]
        self.conv_strides = conv_strides # [1, 2, 2]
        self.lsd = latent_space_dim # 2
        
        self.encoder = None
        self.decoder = None
        self.model = None
        
        self.num_convlayers = len(conv_filters)
        self._shape_before_bn = None
        
        self._build()
        
    def summary(self):
        self.encoder.summary()
    
    
    def _build(self):
        self._build_encoder()
        self._build_decoder()
        #self._build_autoencoder()
        
    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layer = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layer)
        self.decoder = Model(decoder_input, decoder_input, name = 'decoder')
    
    def _add_decoder_input(self):
        return Input(shape=self.lsd, name='decoder_input')
    
    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bn) # [1, 2, 4] -> 8, the np.prod multiplies all the values
        denselayer = Dense(num_neurons, name = 'decoder_dense')(decoder_input) # this applies the dense layer to the decoder_input
        return denselayer
    
    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bn)(dense_layer)
    
    def _add_conv_transpose_layers(self, x):
        # loop through conv layers in reverse order, then stop at 1st layer
        # we want to mirror the convolutional architecture in encoder
        for i in reversed(range(1, self.num_convlayers)):
            # [0,1,2] -> [2,1,0], but we want to ignore index 0, hence why we start at 
            x = self._add_conv_transpose_layer(i, x) # create 1 layer, not multiple layers
        return x
    
    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self.num_convlayers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding='same',
            name=f'decoder_conv_transpose_{layer_num}'
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f'decoder_relu_{layer_num}')(x)
        x = BatchNormalization(name = f'decoder_bn_{layer_num}')
        
        
        
        
    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers) # bottleneck is the output of the encoder
        self.encoder = Model(encoder_input, bottleneck, name = 'encoder')
        
    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name = 'encoder_input')
    
    def _add_conv_layers(self, encoder_input):
        """This method iterates over the number of convolutional layers specified (self.num_convlayers) and adds each convolutional layer to the encoder architecture."""
        x = encoder_input
        for i in range(self.num_convlayers):
            x = self._add_conv_layer(i, x)
        return x   
           
    def _add_conv_layer(self, layer_index, x):
        """ adds a conv block to a graph of layers consisting of
        conv 2d + ReLU + batch normalization"""
        layernum = layer_index + 1
        
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size = self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding='same',
            name=f'encoder_conv_layer_{layernum}'
        )
        x = conv_layer(x) # applies defined conv layer to input tensor 'x' -> outputs the output of a conv operation
        x = ReLU(name=f'encoder_relu_{layernum}')(x)
        x = BatchNormalization(name=f'encoder_bn_{layernum}')(x)
        return x

    def _add_bottleneck(self, x):
        "flatten data and add bottleneck (Dense Layer)"
        self._shape_before_bn = K.int_shape(x)[1:] # 4 dimensional array [batchsize, width, height, num of channels]
        x = Flatten()(x)
        x = Dense(self.lsd, name = 'encoder_output')(x) # dense layer needs to have number of neurons and that's equal to the latent space dimensions
        return x
        
        
if __name__ == '__main__':
    autoencoder = Autoencoder(
        input_shape=(28,28,1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels = (3,3,3,3),
        conv_strides=(1,2,2,1),
        latent_space_dim=2
    )
    autoencoder.summary()