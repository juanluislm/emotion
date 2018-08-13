from keras.layers import Input
from keras.regularizers import l2
from keras.layers import Activation, Convolution2D, Dropout, Conv2D, Dense
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from copy import  deepcopy

def ResidualNet(input_shape, num_classes, residual_layers, conv_layers, l2_regularization=0.01, init_strides=(1,1),
                residual_strides=(2,2), conv2d_filters=8, pooling_strides=(2,2) ):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)

    x = Conv2D( conv2d_filters // 2, (3, 3), strides=init_strides, kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = StackConvolutions(x, conv_layers, conv2d_filters, init_strides, regularization)

    x = StackResidualLayers(x, residual_layers, regularization, residual_strides, conv2d_filters, pooling_strides )

    x = Conv2D(num_classes, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)

    model = Model(img_input, output)
    return model

def StackConvolutions(initialized_net, conv_layers, conv2d_filters, init_strides, regularization):

    x = initialized_net

    for i in range(0, conv_layers):

        x = Conv2D( conv2d_filters // 2, (3, 3), strides=init_strides, kernel_regularizer=regularization,
                   use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    return x

def StackResidualLayers(initialized_net, residual_layers, regularization, residual_strides=(2,2), conv2d_filters=8, pooling_strides=(2,2) ):

    x = initialized_net

    for i in range(0, residual_layers):
        cur_filter=conv2d_filters*(i+1)
        residual = Conv2D( cur_filter, (1, 1), strides=residual_strides,
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(cur_filter, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(cur_filter, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((3, 3), strides=pooling_strides, padding='same')(x)
        x = layers.add([x, residual])

    return x

def StackDenseLayers(initialized_net, dense_layers, params, num_classes):

    x = deepcopy(initialized_net)

    if(dense_layers == 0):
        return x

    for i in range(1, dense_layers):

        x = Dense( params // ( 2*(i+1) ) )(x)
        # x = Dropout(drop_rate)(x)
        x = Activation('relu')(x)

    x = Dense(num_classes)(x)

    return x



def DenseResNet(input_shape, num_classes, residual_layers, conv_layers, l2_regularization=0.01, init_strides=(1,1),
                residual_strides=(2,2), conv2d_filters=8, pooling_strides=(2,2), drop_rate=0.2, dense_layers=2 ):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)

    x = Conv2D(conv2d_filters // 2, (3, 3), strides=init_strides, kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = StackConvolutions(x, conv_layers, conv2d_filters, init_strides, regularization)

    x = StackResidualLayers(x, residual_layers, residual_strides, conv2d_filters, pooling_strides)

    x = StackDenseLayers(x, dense_layers, conv2d_filters * conv_layers, num_classes)

    x = Dropout(drop_rate)(x)

    # x = Conv2D(num_classes, (3, 3),
    #            # kernel_regularizer=regularization,
    #            padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)

    model = Model(img_input, output)
    return model

