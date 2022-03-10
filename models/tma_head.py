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
    memory_keys = layers.Permute((2, 1, 3, 4), input_shape=(None, sequence_num, key_channels, height, width), name="permutation_memory_keys")(memory_keys)
    memory_keys = layers.Reshape((key_channels, sequence_num * height * width), input_shape=(None, key_channels, sequence_num, height, width), name="view_memory_keys")(memory_keys)

    query_key = layers.Reshape((key_channels, height * width), input_shape=(None, height, width, key_channels), name="view_query_key")(query_key)
    query_key = layers.Permute((2, 1), input_shape=(None, key_channels, height * width), name="permutation_query_key")(query_key)

    key_attention = layers.Dot(axes=(2, 1), name="dot_query_key_memory_keys")([query_key, memory_keys])
    key_attention = layers.Activation('softmax', name="activation_key_attention")(key_attention)

    # QUERY VALUES & MEMORY VALUES
    memory_values = layers.Permute((2, 1, 3, 4), input_shape=(None, sequence_num, key_channels, height, width), name="permutation_memory_values")(memory_values)
    memory_values = layers.Reshape((value_channels, sequence_num * height * width), input_shape=(None, key_channels, sequence_num, height, width), name="view_memory_values")(memory_values)
    memory_values = layers.Permute((2, 1), input_shape=(None, value_channels, sequence_num * height * width), name="permutation_memory_values_2")(memory_values)

    memory = layers.Dot(axes=(2, 1), name="dot_key_attention_memory_values")([key_attention, memory_values])

    memory = layers.Permute((2, 1), input_shape=(None, height * width, key_channels), name="permutation_memory")(memory)
    memory = layers.Reshape((value_channels, height, width), input_shape=(None, key_channels, height * width), name="view_memory")(memory)

    query_memory = layers.Concatenate(axis=1, name="concat_query_value_memory")([query_value, memory])

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


if __name__ == "__main__":

    img_size = (384, 384)

    model = TMAnet_ResNetEncoder(img_size, 3, 19)

    model.summary(line_length=250)
