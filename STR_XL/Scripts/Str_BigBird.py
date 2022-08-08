import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.append("../../../deep-learning-dna")
sys.path.append("../")

import wandb

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from IPython.display import display
import math
import string

from Attention import BigBird, Set_Transformer
from common.models import dnabert
from common import dna
from lmdbm import Lmdb
from common.data import DnaSequenceGenerator, DnaLabelType, DnaSampleGenerator, find_dbs
import wandb

import tf_utils as tfu

class Create_Embeddings(keras.layers.Layer):
    def __init__(self, encoder):
        super(Create_Embeddings, self).__init__()
        self.encoder = encoder
        
    
    def subbatch_predict(self, model, batch, subbatch_size, concat=lambda old, new: tf.concat((old, new), axis=0)):
        def predict(i, result=None):
            n = i + subbatch_size
            pred = tf.stop_gradient(model(batch[i:n]))
            if result is None:
                return [n, pred]
            return [n, concat(result, pred)]
        i, result = predict(0)
        batch_size = tf.shape(batch)[0]
        i, result = tf.while_loop(
            cond=lambda i, _: i < batch_size,
            body=predict,
            loop_vars=[i, result],
            parallel_iterations=1)

        return result
    
    def modify_data_for_input(self, data):
        batch_size = tf.shape(data)[0]
        subsample_size = tf.shape(data)[1]
        flat_data = tf.reshape(data, (batch_size*subsample_size, -1))
        encoded = self.subbatch_predict(self.encoder, flat_data, 128)
        return tf.reshape(encoded, (batch_size, subsample_size, -1))
    
    def call(self, data):
        return  self.modify_data_for_input(data)

class Create_BigBird_Masks(keras.layers.Layer):
    def __init__(self, attention_block_size):
        super(Create_BigBird_Masks, self).__init__()
            
        self.mask_layer = BigBird.BigBirdMasks(block_size=attention_block_size)
        
    def call(self, one_batch):

        mask = tf.ones(tf.shape(one_batch)[:-1])
                       
        masks = self.mask_layer(one_batch, mask)      
        
        return masks


class Big_Bird_Attention(keras.layers.Layer):
    def __init__(self, dropout, inner_size, num_heads, key_dim, num_rand_blocks,from_block_size,to_block_size,max_rand_mask_length):
        super(Big_Bird_Attention, self).__init__()
        
        self.dropout = dropout
        self.inner_size = inner_size
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.num_rand_blocks = num_rand_blocks
        self.from_block_size = from_block_size
        self.to_block_size = to_block_size
        self.max_rand_mask_length = max_rand_mask_length
        
        self.attention_layer = BigBird.BigBirdAttention(num_heads=self.num_heads, key_dim=self.key_dim, num_rand_blocks=self.num_rand_blocks,from_block_size=self.from_block_size,to_block_size=self.to_block_size,max_rand_mask_length=self.max_rand_mask_length)
        self.attention_dropout = tf.keras.layers.Dropout(rate=self.dropout)
        self.attention_layer_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12, dtype=tf.float32)
        self.inner_dense = tf.keras.layers.experimental.EinsumDense("abc,cd->abd", output_shape=(None, self.inner_size), bias_axes="d", kernel_initializer=keras.initializers.RandomNormal(stddev=0.1))
        self.inner_activation_layer = tf.keras.layers.Activation("relu")
        self.inner_dropout_layer = tf.keras.layers.Dropout(rate=self.dropout)
        self.output_dense = tf.keras.layers.experimental.EinsumDense("abc,cd->abd", output_shape=(None, inner_size), bias_axes="d", kernel_initializer=keras.initializers.RandomNormal(stddev=0.1))
        self.output_dropout = tf.keras.layers.Dropout(rate=self.dropout)
        self.output_layer_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12)
        
    def call(self, content_stream, mask):

        attention_output = self.attention_layer(content_stream, content_stream, content_stream, mask)

        attention_stream = attention_output
        input_stream = content_stream

        attention_stream = self.attention_dropout(attention_stream)
        attention_stream = self.attention_layer_norm(attention_stream + input_stream)
        inner_output = self.inner_dense(attention_stream)
        inner_output = self.inner_activation_layer(inner_output)
        inner_output = self.inner_dropout_layer(inner_output)
        layer_output = self.output_dense(inner_output)
        layer_output = self.output_dropout(layer_output)
        layer_output = self.output_layer_norm(layer_output + attention_stream)
        attention_output = layer_output
        
        return attention_output

class Set_Big_Bird_Model(keras.Model):
    def __init__(self, embed_dim, stack, encoder, max_files, num_seeds, pooling_num_heads, attention_block_size, dropout, inner_size, attention_num_heads, key_dim, num_rand_blocks, from_block_size, to_block_size, max_rand_mask_length, use_layernorm, pre_layernorm, use_keras_mha):
        super(Set_Big_Bird_Model, self).__init__()
        
        self.embed_dim = embed_dim
        self.stack = stack
        self.encoder = encoder
        self.max_files = max_files
        self.num_seeds = num_seeds 
        self.pooling_num_heads = pooling_num_heads   
        
        self.attention_block_size = attention_block_size
        self.dropout = dropout
        self.inner_size = inner_size
        self.attention_num_heads = attention_num_heads
        self.key_dim = key_dim
        self.num_rand_blocks = num_rand_blocks
        self.from_block_size = from_block_size
        self.to_block_size = to_block_size
        self.max_rand_mask_length = max_rand_mask_length
        
        self.use_layernorm = use_layernorm
        self.pre_layernorm = pre_layernorm
        self.use_keras_mha = use_keras_mha
        
        self.embedding_layer = Create_Embeddings(self.encoder)
        self.linear_layer = keras.layers.Dense(self.embed_dim)
        self.attention_blocks = []
        
        self.mask_layer = Create_BigBird_Masks(attention_block_size)
        
        self.attention_layer = Big_Bird_Attention(self.dropout, self.inner_size, self.attention_num_heads, self.key_dim, self.num_rand_blocks, self.from_block_size, self.to_block_size, self.max_rand_mask_length)
        
        for i in range(self.stack):
            self.attention_blocks.append(self.attention_layer)
                
        self.pooling_layer = Set_Transformer.PoolingByMultiHeadAttention(num_seeds=self.num_seeds,embed_dim=self.embed_dim,num_heads=self.pooling_num_heads,use_layernorm=self.use_layernorm,pre_layernorm=self.pre_layernorm,use_keras_mha=self.use_keras_mha,is_final_block=True)
    
        self.reshape_layer = keras.layers.Reshape((self.embed_dim,))
        
        self.output_layer = keras.layers.Dense(self.max_files)
    
    def call(self, data):
        
            embeddings = self.embedding_layer(data)
            
            linear_transform = self.linear_layer(embeddings)
            
            mask = self.mask_layer(linear_transform)
            
            attention = linear_transform
            
            for attention_block in self.attention_blocks:
                attention = attention_block(attention, mask)
                
            pooling = self.pooling_layer(attention)
        
            reshape = self.reshape_layer(pooling)
            
            output = self.output_layer(reshape)    
            
            return output
