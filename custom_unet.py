
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU

kernel_initializer =  'he_uniform' #Try others if you want

################################################################
# encoder block
def conv_block(input, num_filters):
  
    x = Conv3D(num_filters, 3, padding = "same")(input) # use 3x3x3 kernel size
    # print(tf.shape(x))
    x = Activation("relu")(x)
    x = BatchNormalization()(x) # 

    # Feed first conv block into second
    x = Conv3D(num_filters, 3, padding = "same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    return x 

def Encoder_Block(input, num_filters, dropout = 0.3, max_pooling = True):
    x = conv_block(input, num_filters)

    if dropout > 0:     
        x = Dropout(dropout)(x)
    if max_pooling:
        next_layer = MaxPooling3D((2,2,2))(x)
    else:
        next_layer = x

    skip_connection = x
    return next_layer, skip_connection

# decoder block
def Decoder_Block(prev_layer, skip_layer_input, num_filters = 32):
    up = Conv3DTranspose(num_filters, (2,2,2), strides=2,padding = "same")(prev_layer)
    merge = concatenate([up, skip_layer_input])

    x = conv_block(merge, num_filters)

    return x

# Stack the layers and build the u-net model
def unet_model(input_shape, n_classes, dropout = 0.1, max_pooling = True):
    inputs = Input(input_shape)

    s1, p1 = Encoder_Block(inputs, 32)
    s2, p2 = Encoder_Block(s1, 64)
    s3, p3 = Encoder_Block(s2, 128)
    s4, p4 = Encoder_Block(s3, 256)

    b1 = conv_block(s4, 512)

    d1 = Decoder_Block(b1, p4, 256)
    d2 = Decoder_Block(d1, p3, 128)
    d3 = Decoder_Block(d2, p2, 64)
    d4 = Decoder_Block(d3, p1, 32)

    if n_classes ==1: 
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    outputs = Conv3D(n_classes, 1, padding = "same", activation=activation)(d4)
    print(activation)

    model = Model(inputs, outputs, name = "u-net")

    return model

#Test if everything is working ok. 
model = unet_model((128,128,128, 3), 4)
print(model.input_shape)
print(model.output_shape)