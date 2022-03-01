from keras import layers
from keras.models import Model

import tensorflow as tf

tf.keras.backend.set_image_data_format('channels_first')

KEY_CHANNELS = 64 #, 128 # 256
VALUE_CHANNELS = 256 # 512 # 1024

# SI ON VEUT PARTAGER LES WEIGHTS DU KEYS ET MEMORIES
# memory_keys_conv1 = layers.Conv2D(KEY_CHANNELS, (1, 1), padding="same", activation="relu", name="memory_keys_conv1")
# memory_keys_conv2 = layers.Conv2D(KEY_CHANNELS, (3, 3), padding="same", activation="relu", name="memory_keys_conv2")

# memory_keys = layers.TimeDistributed(memory_keys_conv1)(sequence_imgs)
# memory_keys = layers.TimeDistributed(memory_keys_conv2)(memory_keys)

# memory_values_conv1 = layers.Conv2D(VALUE_CHANNELS, (1, 1), padding="same", activation="relu", name="memory_values_conv1")
# memory_values_conv2 = layers.Conv2D(VALUE_CHANNELS, (3, 3), padding="same", activation="relu", name="memory_values_conv2")

# memory_values = layers.TimeDistributed(memory_values_conv1)(sequence_imgs)
# memory_values = layers.TimeDistributed(memory_values_conv2)(memory_values)

# query_key = memory_keys_conv1(x)
# query_key = memory_keys_conv2(query_key)

# query_value = memory_values_conv1(x)
# query_value = memory_values_conv2(query_value)

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

def TMA_HEAD(image, sequence_imgs, n_output):

    # print("image.shape", image.shape)
    # print("sequence_imgs.shape", sequence_imgs.shape)

    padding = "same"

    memory_keys = layers.TimeDistributed(layers.Conv2D(KEY_CHANNELS, (1, 1), padding="same", activation="relu"), name="td_memory_keys_1")(sequence_imgs)
    memory_keys = layers.TimeDistributed(layers.Conv2D(KEY_CHANNELS, (3, 3), padding="same", activation="relu"), name="td_memory_keys_2")(memory_keys)

    memory_values = layers.TimeDistributed(layers.Conv2D(VALUE_CHANNELS, (1, 1), padding="same", activation="relu"), name="td_memory_values_1")(sequence_imgs)
    memory_values = layers.TimeDistributed(layers.Conv2D(VALUE_CHANNELS, (3, 3), padding="same", activation="relu"), name="td_memory_values_2")(memory_values)

    query_key = layers.Conv2D(KEY_CHANNELS, (1, 1), padding="same", activation="relu", name="key_conv_1")(image)
    query_key = layers.Conv2D(KEY_CHANNELS, (3, 3), padding="same", activation="relu", name="key_conv_2")(query_key)

    query_value = layers.Conv2D(VALUE_CHANNELS, (1, 1), padding="same", activation="relu", name="query_value_conv_1")(image)
    query_value = layers.Conv2D(VALUE_CHANNELS, (3, 3), padding="same", activation="relu", name="query_value_conv_2")(query_value)

    # print("memory_keys", memory_keys.shape)
    # print("memory_values", memory_values.shape)
    # print("query_key", query_key.shape)
    # print("query_value", query_value.shape)

    output = memory_module(memory_keys, memory_values, query_key, query_value)

    output = layers.Conv2D(n_output, (3, 3), padding=padding, activation="relu", name="bottleneck")(output)  # bottleneck
    output = layers.Conv2D(n_output, (1, 1), padding=padding, name="cls_seg")(output)  # cls_seg

    return output


def TMAnet(image_size, sequence_length, categories):

    image = layers.Input(image_size)
    sequence_imgs = layers.Input((sequence_length,) + image_size)

    ResNet50 = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=image)

    for layer in ResNet50.layers:
        layer.trainable = False

    image = ResNet50(image)
    sequence_imgs = layers.TimeDistributed(ResNet50, name="tb_sequence_resnet")(sequence_imgs)

    tma_head = TMA_HEAD(image, sequence_imgs, categories)

    return Model([image, sequence_imgs], tma_head, name="TMA-RESNET")

if __name__ == "__main__":

    img_size = (3, 384, 384)

    model = TMAnet(img_size, 3, 19)
   
    model.summary(line_length=250)
