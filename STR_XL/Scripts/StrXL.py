import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.append("../../../deep-learning-dna")
sys.path.append("../")

import wandb

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import math
import string

from Attention import Set_Transformer
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
        result = tf.reshape(encoded, (batch_size, subsample_size, -1))
        return result
    
    def call(self, data):
        return  self.modify_data_for_input(data)

def Cache_Memory(current_state, previous_state, memory_length):
    if memory_length is None or memory_length == 0:
        return None
    else:
        if previous_state is None:
            new_mem = current_state[:, -memory_length:, :]
        else:
            new_mem = tf.concat(
                    [previous_state, current_state], 1)[:, -memory_length:, :]

    return tf.stop_gradient(new_mem)

class Attention(keras.Model):
    def __init__(self, num_induce, embed_dim, num_heads, use_layernorm, pre_layernorm, use_keras_mha):
        super(Attention, self).__init__()
        
        self.num_induce = num_induce
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_layernorm = use_layernorm
        self.pre_layernorm = pre_layernorm
        self.use_keras_mha = use_keras_mha
        
        if self.num_induce == 0:       
            self.attention = (Set_Transformer.SetAttentionBlock(embed_dim=self.embed_dim, num_heads=self.num_heads, use_layernorm=self.use_layernorm,pre_layernorm=self.pre_layernorm,use_keras_mha=self.use_keras_mha))
        else:
            self.attention = Set_Transformer.InducedSetAttentionBlock(embed_dim=self.embed_dim, num_heads=self.num_heads, num_induce=self.num_induce, use_layernorm=self.use_layernorm, pre_layernorm=self.pre_layernorm, use_keras_mha=self.use_keras_mha)
    
    
    def call(self, data, mems):
                
            attention = self.attention([data, mems])
                
            return attention

class TransformerXLBlock(tf.keras.layers.Layer):
    def __init__(self,
                 num_induce, 
                 embed_dim,
                 num_heads,
                 use_layernorm,
                 pre_layernorm,
                 use_keras_mha,):

        super(TransformerXLBlock, self).__init__()
        
        self.num_induce = num_induce
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_layernorm = use_layernorm
        self.pre_layernorm = pre_layernorm
        self.use_keras_mha = use_keras_mha
        
        self.attention = Attention
        
        self.attention_layer = self.attention(self.num_induce, self.embed_dim, self.num_heads, self.use_layernorm, self.pre_layernorm, self.use_keras_mha)

   
    def call(self,
             content_stream,
             state=None):
        
        attention_output = self.attention_layer(content_stream, state)

        return attention_output

class TransformerXL(tf.keras.layers.Layer):
    def __init__(self,
                 mem_switched, 
                 num_layers,
                 num_induce,
                 embed_dim,
                 num_heads,
                 dropout_rate,
                 mem_len=None,
                 use_layernorm=True,
                 pre_layernorm=True, 
                 use_keras_mha=True):
        
        super(TransformerXL, self).__init__()

        self.mem_switched = mem_switched
        self.num_layers = num_layers
        self.num_induce = num_induce
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.mem_len = mem_len
        self.use_layernorm = use_layernorm
        self.pre_layernorm = pre_layernorm
        self.use_keras_mha = use_keras_mha

        self.transformer_xl_layers = []
        
        for i in range(self.num_layers):
            self.transformer_xl_layers.append(
                    TransformerXLBlock(self.num_induce,
                                        self.embed_dim,
                                        self.num_heads,
                                        self.use_layernorm,
                                        self.pre_layernorm, 
                                        self.use_keras_mha))

        self.output_dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)

    def call(self,
             content_stream,
             state=None):
        
        new_mems = []

        if state is None:
            state = [None] * self.num_layers
            
        for i in range(self.num_layers):
            if self.mem_switched == False:
                new_mems.append(Cache_Memory(content_stream, state[i], self.mem_len))
            
            transformer_xl_layer = self.transformer_xl_layers[i]
            
            transformer_xl_output = transformer_xl_layer(content_stream=content_stream,
                                                        state=state[i])
            
            content_stream = self.output_dropout(transformer_xl_output)
            
            if self.mem_switched == True:
                new_mems.append(Cache_Memory(content_stream, state[i], self.mem_len))

        output_stream = content_stream
        return output_stream, new_mems

class XlModel(keras.Model):
    def __init__(self, mem_switched, max_files, encoder, block_size, max_set_len, num_induce, embed_dim, num_layers, num_heads, mem_len, dropout_rate, num_seeds, use_layernorm, pre_layernorm, use_keras_mha):
        super(XlModel, self).__init__()
        
        self.mem_switched = mem_switched
        self.max_files = max_files
        self.encoder = encoder
        self.block_size = block_size
        self.max_set_len = max_set_len
        self.num_induce = num_induce
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mem_len = mem_len
        self.dropout_rate = dropout_rate
        self.num_seeds = num_seeds
        self.use_layernorm = use_layernorm
        self.pre_layernorm = pre_layernorm
        self.use_keras_mha = use_keras_mha
        
        self.embedding_layer = Create_Embeddings(self.encoder)

        self.linear_layer = keras.layers.Dense(self.embed_dim)
        
        self.transformer_xl = TransformerXL(self.mem_switched,
                                            self.num_layers,
                                             self.num_induce,
                                             self.embed_dim,
                                             self.num_heads,
                                             self.dropout_rate,
                                             self.mem_len,
                                             self.use_layernorm,
                                             self.pre_layernorm,
                                             self.use_keras_mha)
        

        self.pooling_layer = Set_Transformer.PoolingByMultiHeadAttention(num_seeds=self.num_seeds,embed_dim=self.embed_dim,num_heads=self.num_heads,use_layernorm=self.use_layernorm,pre_layernorm=self.pre_layernorm, use_keras_mha=self.use_keras_mha, is_final_block=True)
    
        self.reshape_layer = keras.layers.Reshape((self.embed_dim,))
   
        self.output_layer = keras.layers.Dense(self.max_files, activation=keras.activations.softmax)
        
    
    def call(self, x, training=None):        
 
        mems = tf.zeros((self.num_layers, tf.shape(x)[0], self.mem_len, self.embed_dim))
        
        embeddings = self.embedding_layer(x)
            
        linear_transform = self.linear_layer(embeddings)    
            
        for i in range(0, self.max_set_len, self.block_size):
            block = linear_transform[:,i:i+self.block_size]
            
            output, mems = self.transformer_xl(content_stream=block, state=mems)
                
        pooling = self.pooling_layer(output)

        reshape = self.reshape_layer(pooling)

        output = self.output_layer(reshape)          
        
        return output
