import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.append("../../../deep-learning-dna")
sys.path.append("../")
sys.path.append("../../../deep-learning-dna/common")

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

from core.custom_objects import CustomObject

import tf_utils as tfu

class Create_Embeddings():
    def __init__(self, encoder):
        super(Create_Embeddings, self).__init__()
        self.encoder = encoder
        
    def subbatch_predict(self, model, batch, subbatch_size, concat=lambda old, new: tf.concat((old, new), axis=0)):
        batch_size = tf.shape(batch)[0]
        
        result = tf.zeros((batch_size, 64))
        
        for i in range(0, tf.shape(batch)[0], subbatch_size):
            subbatch = batch[i:i+subbatch_size]
            clamp = tf.minimum(subbatch_size, batch_size-i)
            encoded = self.encoder(subbatch)
            result = tf.tensor_scatter_nd_update(result, tf.expand_dims(tf.range(i, i+clamp), 1), encoded)
        return result
    
    def modify_data_for_input(self, data):
        batch_size = tf.shape(data)[0]
        subsample_size = tf.shape(data)[1]
        flat_data = tf.reshape(data, (batch_size*subsample_size, -1))
        encoded = self.subbatch_predict(self.encoder, flat_data, 128)
        result = tf.reshape(encoded, (batch_size, subsample_size, -1))
        return result
    
    def __call__(self, data):
        embeddings = self.modify_data_for_input(data)
        return embeddings

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

class Attention(keras.layers.Layer):
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

class TransformerXL(keras.layers.Layer):
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
    def __init__(self, mem_switched, max_files, seg_size, max_set_len, num_induce, embed_dim, num_layers, num_heads, mem_len, dropout_rate, num_seeds, use_layernorm, pre_layernorm, use_keras_mha):
        super(XlModel, self).__init__()
        
        self.mem_switched = mem_switched
        self.max_files = max_files
        self.seg_size = seg_size
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
        
    
    def call(self, embeddings, mems, index, training=None):        
        
        linear_transform = self.linear_layer(embeddings)
        
        segment = linear_transform[:, index:index+self.seg_size]
        
        output, mems = self.transformer_xl(content_stream=segment, state=mems)
                
        pooling = self.pooling_layer(output)

        reshape = self.reshape_layer(pooling)

        output = self.output_layer(reshape)          
        
        return output, mems


@tf.function()
def train_step(inputs):
    batch, max_set_len, seg_size = inputs
    
    #Iterate through subbatches
    #Pull out one set at a time
    for i in range (batch_size[0]):
        print("Building:", i)
        n = i + subbatch_size
        one_set = (batch[0][i:n], batch[1][i:n]) 
        x, y = one_set
        i += 1
        
        #Initialize mems
        mems = tf.zeros((num_layers, tf.shape(x)[0], mem_len, embed_dim))
        
        #Initialize embeddings
        embeddings = embedder(x)
        
        #Initialize gradients
        accum_grads = [tf.zeros_like(w) for w in model.trainable_weights]
    
        total_loss = 0.0
        total_accuracy = 0.0
    
        #Split set into segments
        for index in range(0, max_set_len, seg_size):
            
            #Pass entire set (for embeddings) and memories into model
            with tf.GradientTape() as tape:
                
                segment_output, mems = model(embeddings, mems, index, True)

                loss = loss_function(y, segment_output)
                accuracy = accuracy_function(y, segment_output)
                
                total_loss += loss

            #Compute segment level gradients
            grads = tape.gradient(loss, model.trainable_weights)   

            accum_grads = [(gs + ags) for gs, ags in zip(grads, accum_grads)]    

        total_accuracy += accuracy 
            
        #Apply gradients
        model.optimizer.apply_gradients(zip(accum_grads, model.trainable_weights))
        
    return total_loss, total_accuracy/batch_size[0]

@tf.function()
def val_step(inputs):
    batch, max_set_len, seg_size = inputs
    
    #Iterate through subbatches
    #Pull out one set at a time
    for i in range (batch_size[0]):
        n = i + subbatch_size
        one_set = (batch[0][i:n], batch[1][i:n]) 
        x, y = one_set
        i += 1
        
        #Initialize mems
        mems = tf.zeros((num_layers, tf.shape(x)[0], mem_len, embed_dim))
        
        #Initialize embeddings
        embeddings = embedder(x)
    
        total_loss = 0.0
        total_accuracy = 0.0
    
        #Split set into segments
        for index in range(0, max_set_len, seg_size):
            
            #Pass entire set (for embeddings) and memories into model
            segment_output, mems = model(embeddings, mems, index, True)

            loss = loss_function(y, segment_output)
            accuracy = accuracy_function(y, segment_output)

            total_loss += loss
        total_accuracy += accuracy 

    return total_loss, total_accuracy/batch_size[0]


def Training(model, train_dataset, val_dataset, epochs)
    
    loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    accuracy_function = keras.metrics.SparseCategoricalAccuracy()
    
    embedder = Create_Embeddings(pretrained_encoder)    
    
    e = embedder(train_dataset[0][0])
    mems = tf.zeros((num_layers, tf.shape(e)[0], mem_len, embed_dim))
    e = model(e, mems, 0)
    
    for epoch in range(epochs):    
        total_loss = 0.0
        total_accuracy = 0.0

        i = 0

        #Iterate through batches
        for batch in train_dataset:

            i += 1
            #Pass one batch intro train_step
            loss, accuracy = train_step([batch, max_set_len, seg_size])
            
            total_loss += loss
            total_accuracy += accuracy

            print(f"\r{epoch+1}/{epochs} batch: {i}/{len(train_dataset)} Train Loss: {loss} Train Accuracy = {accuracy}", end="")

        total_accuracy = total_accuracy/tf.shape(batch[0])[0]
            
        total_val_loss = 0.0
        total_val_accuracy = 0.0
        i = 0

        #Iterate through batches
        for batch in val_dataset:

            i += 1
            #Pass one batch intro train_step
            val_loss, val_accuracy = val_step([batch, max_set_len, seg_size])
            
            total_val_loss += val_loss
            total_val_accuracy += val_accuracy

            print(f"\r{epoch+1}/{epochs} batch: {i}/{len(train_dataset)} Val Loss: {val_loss} Val Accuracy = {val_accuracy}", end="")
            
        total_val_accuracy = total_val_loss/tf.shape(batch[0])[0]
        
        wandb.run.log({"loss":total_loss, "val_loss":total_val_loss, "accuracy":total_accuracy, "val_accuracy":total_val_accuracy, "epoch":epoch+1}