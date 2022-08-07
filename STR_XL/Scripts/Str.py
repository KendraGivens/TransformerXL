
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.append("../../deep-learning-dna")

import wandb

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
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
        return tf.reshape(encoded, (batch_size, subsample_size, -1))
    
    def call(self, data):
        return  self.modify_data_for_input(data)

class Set_Transformer_Model(keras.Model):
    def __init__(self, num_induce, embed_dim, attention_num_heads, stack, use_layernorm, pre_layernorm, use_keras_mha, encoder, max_files, num_seeds, pooling_num_heads):
        super(Set_Transformer_Model, self).__init__()
        
        
        self.num_induce = num_induce
        self.embed_dim = embed_dim
        self.attention_num_heads = attention_num_heads
        self.stack = stack
        self.use_layernorm = use_layernorm
        self.pre_layernorm = pre_layernorm
        self.use_keras_mha = use_keras_mha
        self.encoder = encoder
        self.max_files = max_files
        self.num_seeds = num_seeds 
        self.pooling_num_heads = pooling_num_heads
        
        
        self.embedding_layer = Create_Embeddings(self.encoder)
        self.linear_layer = keras.layers.Dense(self.embed_dim)
        self.attention_blocks = []
        
        if self.num_induce == 0:
              for i in range(self.stack):
                self.attention_blocks.append(Set_Transformer.SetAttentionBlock(embed_dim=self.embed_dim,num_heads=self.attention_num_heads,use_layernorm=self.use_layernorm,pre_layernorm=self.pre_layernorm,use_keras_mha=self.use_keras_mha))
        else:
            for i in range(self.stack):
                self.attention_blocks.append(Set_Transformer.InducedSetAttentionBlock(embed_dim=self.embed_dim,num_heads=self.attention_num_heads, num_induce=self.num_induce, use_layernorm=self.use_layernorm,pre_layernorm=self.pre_layernorm,use_keras_mha=self.use_keras_mha))

        self.pooling_layer = Set_Transformer.PoolingByMultiHeadAttention(num_seeds=self.num_seeds,embed_dim=self.embed_dim,num_heads=self.pooling_num_heads,_use_layernorm=self.use_layernorm,pre_layernorm=self.pre_layernorm,use_keras_mha=self.use_keras_mha,is_final_block=True)
    
        self.reshape_layer = keras.layers.Reshape((self.embed_dim,))
        
        self.output_layer = keras.layers.Dense(self.max_files)
    
    def call(self, data):
        
            embeddings = self.embedding_layer(data)
            
            linear_transform = self.linear_layer(embeddings)
            
            attention = linear_transform
            
            for attention_block in self.attention_blocks:
                attention = attention_block([attention, None])
                
            pooling = self.pooling_layer(attention)
        
            reshape = self.reshape_layer(pooling)
            
            output = self.output_layer(reshape)    
            
            return output
