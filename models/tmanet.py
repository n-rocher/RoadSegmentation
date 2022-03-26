from keras import layers
from keras.models import Model

import tensorflow as tf

tf.keras.backend.set_image_data_format('channels_first')

KEY_CHANNELS = 64  # , 128 # 256
VALUE_CHANNELS = 256  # 512 # 1024

def memory_module(memory_keys, memory_values, query_key, query_value):

    _, sequence_num, key_channels, height, width = memory_keys.shape
    _, _, value_channels, _, _ = memory_values.shape

    # QUERY KEY & MEMORY KEYS
    
    #permutation_memory_keys
    memory_keys = layers.Permute((2, 1, 3, 4), input_shape=(None, sequence_num, key_channels, height, width))(memory_keys)
    
    #view_memory_keys
    memory_keys = layers.Reshape((key_channels, sequence_num * height * width), input_shape=(None, key_channels, sequence_num, height, width))(memory_keys)

    #view_query_key
    query_key = layers.Reshape((key_channels, height * width), input_shape=(None, height, width, key_channels))(query_key)

    #permutation_query_key
    query_key = layers.Permute((2, 1), input_shape=(None, key_channels, height * width))(query_key)


    # dot_query_key_memory_keys
    key_attention = layers.Dot(axes=(2, 1))([query_key, memory_keys]) # BUG DE VARIABLE FICHIER H5
    
    #activation_key_attention
    key_attention = layers.Activation('softmax')(key_attention)



    # QUERY VALUES & MEMORY VALUES

    #permutation_memory_values
    memory_values = layers.Permute((2, 1, 3, 4), input_shape=(None, sequence_num, key_channels, height, width))(memory_values)

    #view_memory_values
    memory_values = layers.Reshape((value_channels, sequence_num * height * width), input_shape=(None, key_channels, sequence_num, height, width))(memory_values)

    #permutation_memory_values_2
    memory_values = layers.Permute((2, 1), input_shape=(None, value_channels, sequence_num * height * width))(memory_values)


    #dot_key_attention_memory_values
    memory = layers.Dot(axes=(2, 1))([key_attention, memory_values]) # BUG DE VARIABLE FICHIER H5


    #permutation_memory
    memory = layers.Permute((2, 1), input_shape=(None, height * width, key_channels))(memory)

    #view_memory
    memory = layers.Reshape((value_channels, height, width), input_shape=(None, key_channels, height * width))(memory)


    #concat_query_value_memory
    query_memory = layers.Concatenate(axis=1)([query_value, memory])

    return query_memory

def TMA_HEAD(image, sequence_imgs, n_output, upsampling=False):

    padding = "same"

    memory_keys_conv_1 = tf.keras.Sequential(
        [
            layers.Conv2D(KEY_CHANNELS, (1, 1), padding=padding),
            layers.BatchNormalization(momentum=0.1),
            layers.Activation("relu"),
            layers.Conv2D(KEY_CHANNELS, (1, 1), padding=padding),
            layers.BatchNormalization(momentum=0.1),
            layers.Activation("relu")
        ]
    )

    memory_keys_conv_3 = tf.keras.Sequential(
        [
            layers.Conv2D(KEY_CHANNELS, (3, 3), padding=padding),
            layers.BatchNormalization(momentum=0.1),
            layers.Activation("relu"),
            layers.Conv2D(KEY_CHANNELS, (3, 3), padding=padding),
            layers.BatchNormalization(momentum=0.1),
            layers.Activation("relu")
        ]
    )

    memory_values_conv_1 = tf.keras.Sequential(
        [
            layers.Conv2D(VALUE_CHANNELS, (1, 1), padding=padding),
            layers.BatchNormalization(momentum=0.1),
            layers.Activation("relu"),
            layers.Conv2D(VALUE_CHANNELS, (1, 1), padding=padding),
            layers.BatchNormalization(momentum=0.1),
            layers.Activation("relu")
        ]
    )

    memory_values_conv_3 = tf.keras.Sequential(
        [
            layers.Conv2D(VALUE_CHANNELS, (3, 3), padding=padding),
            layers.BatchNormalization(momentum=0.1),
            layers.Activation("relu"),
            layers.Conv2D(VALUE_CHANNELS, (3, 3), padding=padding),
            layers.BatchNormalization(momentum=0.1),
            layers.Activation("relu")
        ]
    )

    query_key_conv_1 = tf.keras.Sequential(
        [
            layers.Conv2D(KEY_CHANNELS, (1, 1), padding=padding),
            layers.BatchNormalization(momentum=0.1),
            layers.Activation("relu")
        ],
        name="query_key_conv_1"
    )

    query_key_conv_3 = tf.keras.Sequential(
        [
            layers.Conv2D(KEY_CHANNELS, (3, 3), padding=padding),
            layers.BatchNormalization(momentum=0.1),
            layers.Activation("relu")
        ],
        name="query_key_conv_3"
    )

    query_value_conv_1 = tf.keras.Sequential(
        [
            layers.Conv2D(VALUE_CHANNELS, (1, 1), padding=padding),
            layers.BatchNormalization(momentum=0.1),
            layers.Activation("relu")
        ],
        name="query_value_conv_1"
    )

    query_value_conv_3 = tf.keras.Sequential(
        [
            layers.Conv2D(VALUE_CHANNELS, (3, 3), padding=padding),
            layers.BatchNormalization(momentum=0.1),
            layers.Activation("relu")
        ],
        name="query_value_conv_3"
    )

    bottleneck = tf.keras.Sequential(
        [
            layers.Conv2D(512, (3, 3), padding=padding),
            layers.BatchNormalization(momentum=0.1),
            layers.Activation("relu")
        ],
        name="bottleneck"
    )

    classification_segmentation = tf.keras.Sequential(
        [
            layers.Conv2D(n_output, (1, 1), padding=padding),
            layers.Dropout(0.1)
        ],
        name="classification_segmentation"
    )

    # Assemblage des éléments

    memory_keys = layers.TimeDistributed(memory_keys_conv_1, name="td_memory_keys_1")(sequence_imgs)
    memory_keys = layers.TimeDistributed(memory_keys_conv_3, name="td_memory_keys_3")(memory_keys)

    memory_values = layers.TimeDistributed(memory_values_conv_1, name="td_memory_values_1")(sequence_imgs)
    memory_values = layers.TimeDistributed(memory_values_conv_3, name="td_memory_values_3")(memory_values)

    query_key = query_key_conv_1(image)
    query_key = query_key_conv_3(query_key)

    query_value = query_value_conv_1(image)
    query_value = query_value_conv_3(query_value)

    output = memory_module(memory_keys, memory_values, query_key, query_value)

    output = bottleneck(output)  # bottleneck
    output = classification_segmentation(output)  # cls_seg

    if upsampling is not False:
        output = layers.UpSampling2D(size=(upsampling, upsampling), interpolation="bilinear")(output)

    return output


def TMAnet(image_size, sequence_length, categories):

    from .aunet import Attention_ResUNet

    encoder = Attention_ResUNet(categories, (3,) + image_size, onlyEncoder=True)

    return build_network(image_size, sequence_length, categories, 8, encoder)

def TMAnet_ResNetEncoder(image_size, sequence_length, categories):

    encoder = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(3,) + image_size)

    # for layer in encoder.layers:
    #     layer.trainable = False

    return build_network(image_size, sequence_length, categories, 32, encoder)


def build_network(image_size, sequence_length, categories, upsampling, encoder):

    image_size = (3,) + image_size

    image = layers.Input(image_size)
    sequence_imgs = layers.Input((sequence_length,) + image_size)

    image_encoded = encoder(image)
    sequence_imgs_encoded = layers.TimeDistributed(encoder, name="tb_sequence_resnet")(sequence_imgs)

    tma_head = TMA_HEAD(image_encoded, sequence_imgs_encoded, categories, upsampling=upsampling)

    return Model([image, sequence_imgs], tma_head, name="TMA-" + encoder.name)


def tma_block(image, sequence_imgs, key_channels, value_channels, use_bottleneck=False, bottleneck_size=512):

    padding = "same"

    memory_keys_conv_1 = tf.keras.Sequential([
        layers.Conv2D(key_channels, (1, 1), padding=padding),
        layers.BatchNormalization(momentum=0.1),
        layers.Activation("relu")
    ])

    memory_keys_conv_3 = tf.keras.Sequential([
        layers.Conv2D(key_channels, (3, 3), padding=padding),
        layers.BatchNormalization(momentum=0.1),
        layers.Activation("relu")
    ])

    memory_values_conv_1 = tf.keras.Sequential([
        layers.Conv2D(value_channels, (1, 1), padding=padding),
        layers.BatchNormalization(momentum=0.1),
        layers.Activation("relu")
    ])

    memory_values_conv_3 = tf.keras.Sequential([
        layers.Conv2D(value_channels, (3, 3), padding=padding),
        layers.BatchNormalization(momentum=0.1),
        layers.Activation("relu")
    ])

    query_key_conv_1 = tf.keras.Sequential([
        layers.Conv2D(key_channels, (1, 1), padding=padding),
        layers.BatchNormalization(momentum=0.1),
        layers.Activation("relu")
    ])

    query_key_conv_3 = tf.keras.Sequential([
        layers.Conv2D(key_channels, (3, 3), padding=padding),
        layers.BatchNormalization(momentum=0.1),
        layers.Activation("relu")
    ])

    query_value_conv_1 = tf.keras.Sequential([
        layers.Conv2D(value_channels, (1, 1), padding=padding),
        layers.BatchNormalization(momentum=0.1),
        layers.Activation("relu")
    ])

    query_value_conv_3 = tf.keras.Sequential([
        layers.Conv2D(value_channels, (3, 3), padding=padding),
        layers.BatchNormalization(momentum=0.1),
        layers.Activation("relu")
    ])

    bottleneck = tf.keras.Sequential([
        layers.Conv2D(bottleneck_size, (3, 3), padding=padding),
        layers.BatchNormalization(momentum=0.1),
        layers.Activation("relu")
    ])

    # Assemblage des éléments
    memory_keys = layers.TimeDistributed(memory_keys_conv_1)(sequence_imgs)
    memory_keys = layers.TimeDistributed(memory_keys_conv_3)(memory_keys)

    memory_values = layers.TimeDistributed(memory_values_conv_1)(sequence_imgs)
    memory_values = layers.TimeDistributed(memory_values_conv_3)(memory_values)

    query_key = query_key_conv_1(image)
    query_key = query_key_conv_3(query_key)

    query_value = query_value_conv_1(image)
    query_value = query_value_conv_3(query_value)

    output = memory_module(memory_keys, memory_values, query_key, query_value)

    # bottleneck
    if use_bottleneck is not False:
        output = bottleneck(output)

    return output

def res_conv_model(input_shape, filter_size, size, dropout, batch_norm=False):

    inputs = layers.Input(input_shape, dtype=tf.float32)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(inputs)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation('relu')(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)

    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    shortcut = layers.Conv2D(size, kernel_size=(1, 1), padding='same')(inputs)
    if batch_norm is True:
        shortcut = layers.BatchNormalization(axis=3)(shortcut)

    res_path = layers.add([shortcut, conv])
    res_path = layers.Activation('relu')(res_path)

    return Model(inputs, res_path)

def TMA_ResUnet(image_size, sequence_length, num_classes, dropout_rate=0.0, batch_norm=False):
    '''
    Residual UNet, with Temporal Memory attention 
    '''

    FILTER_NUM = 16  # number of basic filters for the first layer
    FILTER_SIZE = 3  # size of the convolutional filter
    UP_SAMP_SIZE = 2  # size of upsampling filters

    image_size = (3,) + image_size

    in_image = layers.Input(image_size)
    in_sequence_imgs = layers.Input((sequence_length,) + image_size)

    #######################
    # Downsampling layers #
    #######################
    
    # DownRes 1
    block_conv_128 = res_conv_model(image_size, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    block_pool_64 = layers.MaxPooling2D(pool_size=(2, 2))

    img_conv_128 = block_conv_128(in_image)
    img_pool_64 = block_pool_64(img_conv_128)

    seq_conv_128 = layers.TimeDistributed(block_conv_128)(in_sequence_imgs)
    seq_pool_64 = layers.TimeDistributed(block_pool_64)(seq_conv_128)

    # DownRes 2
    block_conv_64 = res_conv_model(img_pool_64.shape[1:], FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    block_pool_32 = layers.MaxPooling2D(pool_size=(2, 2))

    img_conv_64 = block_conv_64(img_pool_64)
    img_pool_32 = block_pool_32(img_conv_64)

    seq_conv_64 = layers.TimeDistributed(block_conv_64)(seq_pool_64)
    seq_pool_32 = layers.TimeDistributed(block_pool_32)(seq_conv_64)

    # DownRes 3
    block_conv_32 = res_conv_model(img_pool_32.shape[1:], FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    block_pool_16 = layers.MaxPooling2D(pool_size=(2, 2))

    img_conv_32 = block_conv_32(img_pool_32)
    img_pool_16 = block_pool_16(img_conv_32)

    seq_conv_32 = layers.TimeDistributed(block_conv_32)(seq_pool_32)
    seq_pool_16 = layers.TimeDistributed(block_pool_16)(seq_conv_32)

    # DownRes 4
    block_conv_16 = res_conv_model(img_pool_16.shape[1:], FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    block_pool_8 = layers.MaxPooling2D(pool_size=(2, 2))

    img_conv_16 = block_conv_16(img_pool_16)
    img_pool_8 = block_pool_8(img_conv_16)

    seq_conv_16 = layers.TimeDistributed(block_conv_16)(seq_pool_16)
    seq_pool_8 = layers.TimeDistributed(block_pool_8)(seq_conv_16)

    # DownRes 5, convolution only
    block_conv_8 = res_conv_model(img_pool_8.shape[1:], FILTER_SIZE, 16 * FILTER_NUM, dropout_rate, batch_norm)

    img_conv_8 = block_conv_8(img_pool_8)
    seq_conv_8 = layers.TimeDistributed(block_conv_8)(seq_pool_8)

    
    #####################
    # Upsampling layers #
    #####################


    # UpRes 6, tma_block + upsampling + double residual convolution
    tma_block_8 = tma_block(img_conv_8, seq_conv_8, FILTER_NUM, 8 * FILTER_NUM)
    tma_block_8 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE))(tma_block_8)
    up_16 = layers.concatenate([tma_block_8, img_conv_16], axis=1)
    up_conv_16 = res_conv_model(up_16.shape[1:], FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)(up_16)
    up_conv_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE))(up_conv_16)


    # UpRes 7
    tma_block_16 = tma_block(img_conv_16, seq_conv_16, FILTER_NUM, 4 * FILTER_NUM)
    tma_block_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE))(tma_block_16)
    up_32 = layers.concatenate([tma_block_16, up_conv_16, img_conv_32], axis=1)
    up_conv_32 = res_conv_model(up_32.shape[1:], FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)(up_32)
    up_conv_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE))(up_conv_32)
   
   
    # UpRes 8
    tma_block_32 = tma_block(img_conv_32, seq_conv_32, FILTER_NUM, 2 * FILTER_NUM)
    tma_block_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE))(tma_block_32)
    up_64 = layers.concatenate([tma_block_32, up_conv_32, img_conv_64], axis=1)
    up_conv_64 = res_conv_model(up_64.shape[1:], FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)(up_64)
    up_conv_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE))(up_conv_64)


    # UpRes 9
    tma_block_64 = tma_block(img_conv_64, seq_conv_64, FILTER_NUM, FILTER_NUM)
    tma_block_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE))(tma_block_64)
    up_128 = layers.concatenate([tma_block_64, up_conv_64, img_conv_128], axis=1)
    up_conv_128 = res_conv_model(up_128.shape[1:], FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)(up_128)

    #########################
    # Classification layers #
    ######################### 

    output = layers.Conv2D(filters=num_classes * 2, kernel_size=(3, 3), padding='same')(up_conv_128)
    output = layers.BatchNormalization()(output)
    output = layers.Activation('relu')(output)

    output = layers.Conv2D(filters=num_classes, kernel_size=(3, 3), padding='same')(output)

    return Model([in_image, in_sequence_imgs], output, name="TMA_ResUnet")


if __name__ == "__main__":

    img_size = (384, 384)

    model = TMA_ResUnet(img_size, 3, 19)

    model.summary(line_length=250)

    # model.save("test.tf")