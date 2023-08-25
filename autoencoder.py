from keras import Model
from keras.layers import Input

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
    
        
        
        
    