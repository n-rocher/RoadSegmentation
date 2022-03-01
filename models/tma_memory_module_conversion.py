import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from keras import layers

if __name__ == "__main__":

    sequence_num, batch_size, key_channels, height, width = 2, 2, 250, 30, 25

    query_key = np.zeros((batch_size, key_channels, height, width))
    query_key = query_key + 1

    query_value = np.zeros((batch_size, key_channels, height, width))
    query_value = query_value + 10

    memory_keys = np.zeros((batch_size, sequence_num, key_channels, height, width))
    memory_keys[0, 1] = memory_keys[0, 1] + 1

    pt_memory_keys = np.zeros((sequence_num, batch_size, key_channels, height, width))  # TxBxCxHxW
    pt_memory_keys[1, 0] = pt_memory_keys[1, 0] + 1

    pt_memory_keys = torch.from_numpy(pt_memory_keys)




    memory_values = np.zeros((batch_size, sequence_num, key_channels, height, width))
    memory_values[0, 1] = memory_values[0, 1] + 15

    pt_memory_values = np.zeros((sequence_num, batch_size, key_channels, height, width))
    pt_memory_values[1, 0] = pt_memory_values[1, 0] + 15

    pt_memory_values = torch.from_numpy(pt_memory_values)
    pt_query_key = torch.from_numpy(query_key)
    pt_query_value = torch.from_numpy(query_value)

    """
    Memory Module forward.
    Args:
        memory_keys (Tensor): memory keys tensor, shape: TxBxCxHxW => BxTxCxHxW
        memory_values (Tensor): memory values tensor, shape: TxBxCxHxW
        query_key (Tensor): query keys tensor, shape: BxCxHxW
        query_value (Tensor): query values tensor, shape: BxCxHxW

    Returns:
        Concat query and memory tensor.
    """
    sequence_num, batch_size, key_channels, height, width = pt_memory_keys.shape
    _, _, value_channels, _, _ = memory_values.shape

    assert query_key.shape[1] == key_channels and query_value.shape[1] == value_channels

    permutation_memory_keys = layers.Permute((2, 1, 3, 4), input_shape=(None, sequence_num, key_channels, height, width))
    memory_keys = permutation_memory_keys(memory_keys)
    view_memory_keys = layers.Reshape((key_channels, sequence_num * height * width), input_shape=(None, key_channels, sequence_num, height, width))
    memory_keys = view_memory_keys(memory_keys)
    # print(memory_keys, memory_keys.shape)

    pt_memory_keys = pt_memory_keys.permute(1, 2, 0, 3, 4).contiguous()  # BxCxTxHxW
    pt_memory_keys = pt_memory_keys.view(batch_size, key_channels, sequence_num * height * width)  # BxCxT*H*W
    # print(pt_memory_keys, pt_memory_keys.shape)

    # pt_memory_keys == memory_keys

    view_query_key = layers.Reshape((key_channels, height * width), input_shape=(None, height, width, key_channels))
    permutation_query_key = layers.Permute((2, 1), input_shape=(None, key_channels, height * width))
    query_key = permutation_query_key(view_query_key(query_key))
    # print(query_key, query_key.shape)

    pt_query_key = pt_query_key.view(batch_size, key_channels, height * width).permute(0, 2, 1).contiguous()  # BxH*WxCk
    # print(pt_query_key, pt_query_key.shape)

    # pt_query_key == query_key

    dot_query_key_memory_keys = layers.Dot(axes=(2, 1))
    key_attention = dot_query_key_memory_keys([query_key, memory_keys])
    # print(key_attention, key_attention.shape)

    pt_key_attention = torch.bmm(pt_query_key, pt_memory_keys)  # BxH*WxT*H*W 
    # print(pt_key_attention, pt_key_attention.shape)

    # key_attention == pt_key_attention

    key_attention = layers.Activation('softmax')(key_attention)
    pt_key_attention = F.softmax(pt_key_attention, dim=-1)  # BxH*WxT*H*W

    # key_attention == pt_key_attention


    # MEMORY VALUES
    permutation_memory_values = layers.Permute((2, 1, 3, 4), input_shape=(None, sequence_num, key_channels, height, width)) 
    view_memory_values = layers.Reshape((value_channels, sequence_num * height * width), input_shape=(None, key_channels, sequence_num, height, width))
    memory_values = permutation_memory_values(memory_values)
    memory_values = view_memory_values(memory_values)
    permutation_memory_values_2 = layers.Permute((2, 1), input_shape=(None, value_channels, sequence_num * height * width))
    memory_values = permutation_memory_values_2(memory_values)
    # print(memory_values, memory_values.shape)

    pt_memory_values = pt_memory_values.permute(1, 2, 0, 3, 4).contiguous()  # BxCxTxHxW 
    pt_memory_values = pt_memory_values.view(batch_size, value_channels, sequence_num * height * width)
    pt_memory_values = pt_memory_values.permute(0, 2, 1).contiguous()  # BxT*H*WxC
    # print(pt_memory_values, pt_memory_values.shape)

    # pt_memory_values == memory_values


    dot_key_attention_memory_values = layers.Dot(axes=(2, 1))
    memory = dot_key_attention_memory_values([key_attention, memory_values])
    # print(memory, memory.shape)

    pt_memory = torch.bmm(pt_key_attention, pt_memory_values)  # BxH*WxC
    # print(pt_memory, pt_memory.shape)
   
    # memory == pt_memory

    permutation_memory = layers.Permute((2, 1), input_shape=(None, height * width, key_channels), name="permutation_memory")
    view_memory = layers.Reshape((value_channels, height, width), input_shape=(None, key_channels, height * width))
    memory = permutation_memory(memory)
    memory = view_memory(memory)
    # print(memory, memory.shape)

    pt_memory = pt_memory.permute(0, 2, 1).contiguous()  # BxCxH*W
    pt_memory = pt_memory.view(batch_size, value_channels, height, width)  # BxCxHxW
    # print(pt_memory, pt_memory.shape)
 
    # memory == pt_memory

    query_memory = layers.Concatenate(axis=1)([query_value, memory])
    # print(query_memory, query_memory.shape)
    pt_query_memory = torch.cat([pt_query_value, pt_memory], dim=1)
    # print(pt_query_memory, pt_query_memory.shape)

    # query_memory == pt_query_memory

    # return query_memory
