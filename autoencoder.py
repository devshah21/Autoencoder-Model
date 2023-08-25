from keras import Model
from keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense
from keras import backend as K

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
        
        
    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()
        
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
        
        
    