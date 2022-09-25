import sys
import tf_utils as tfu
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.append("../")
sys.path.append("../../../deep-learning-dna")

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import math
import string

from Attention import Set_Transformer
from common.models import dnabert
from common import dna
from lmdbm import Lmdb
from common.data import DnaSequenceGenerator, DnaLabelType, DnaSampleGenerator, find_dbs
import wandb

from Scripts.Str_BigBird import *

def define_arguments(cli):
    cli.use_strategy()
    
    cli.artifact("--dataset", type=str, required=True)
    cli.artifact("--encoder", type=str, required=True)
    
    cli.argument("--seed", type=int, default = None)
    
    cli.argument("--embed_dim", type=int, default = 48)
    cli.argument("--stack", type=int, default = 4)
    cli.argument("--num_seeds", type=int, default = 1)    
    cli.argument("--pooling_num_heads", type=int, default = 1)
    
    cli.argument("--attention_block_size", type=int, default = 8)
    cli.argument("--dropout", type=int, default = .01)
    cli.argument("--inner_size", type=int, default = 48)
    cli.argument("--attention_num_heads", type=int, default = 6) 
    cli.argument("--key_dim", type=int, default = 48)
    cli.argument("--num_rand_blocks", type=int, default = 4)
    cli.argument("--from_block_size", type=int, default = 8)
    cli.argument("--to_block_size", type=int, default = 8)
    
    cli.argument("--use_layernorm", type=tfu.utils.str_to_bool, default = True)
    cli.argument("--pre_layernorm", type=tfu.utils.str_to_bool, default = True)
    cli.argument("--use_keras_mha", type=tfu.utils.str_to_bool, default = True)  

    cli.argument("--set_len", type=int, default=1000)
    
    cli.argument("--batches_per_epoch", type=int, default=20)
    cli.argument("--validation_batch_size", type=int, default=5)
    
    cli.argument("--save_to", type=str, default=None)
    
    cli.use_training(epochs=10000, batch_size=20)
    
   
def load_dataset(config):
    dataset_path = tfu.scripting.artifact(config, "dataset")
    
    samples = find_dbs(dataset_path + '/train')
    
    split_ratios = [0.8, 0.2]
    set_len = config.set_len
    sequence_len = 150
    kmer = 3
    batch_size = [config.batch_size, config.validation_batch_size]
    batches_per_epoch = config.batches_per_epoch
    augument = True
    labels = DnaLabelType.SampleIds
    rng = tfu.scripting.rng()
    random_samples = samples.copy()

    rng.shuffle(random_samples)

    trimmed_samples, (train_dataset, val_dataset) = DnaSampleGenerator.split(samples=random_samples, split_ratios=split_ratios, 
                                                    subsample_length=set_len, sequence_length=sequence_len, kmer=kmer,
                                                    batch_size=batch_size,batches_per_epoch=batches_per_epoch,augment=augument,labels=labels, rng=rng)


    return trimmed_samples, train_dataset, val_dataset
    
def train(config):
    with tfu.scripting.strategy(config).scope():
        trimmed_samples, train_dataset, val_dataset = load_dataset(config)
        
        
        model_path = tfu.scripting.artifact(config, "encoder")
        pretrained_model = dnabert.DnaBertModel.load(model_path)
        pretrained_model.load_weights(model_path + "/model.h5")
        
        encoder = dnabert.DnaBertEncoderModel(pretrained_model.base)
        encoder.trainable = False
        
        max_files = len(trimmed_samples)
        max_rand_mask_length = config.set_len
        
        model = Set_Big_Bird_Model(config.embed_dim, config.stack, encoder, max_files, config.num_seeds, config.pooling_num_heads, config.attention_block_size, config.dropout, config.inner_size, config.attention_num_heads, config.key_dim, config.num_rand_blocks, config.from_block_size, config.to_block_size, max_rand_mask_length, config.use_layernorm, config.pre_layernorm, config.use_keras_mha)
        model.compile(optimizer=keras.optimizers.Adam(1e-3),loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = [keras.metrics.sparse_categorical_accuracy])

        
        tfu.scripting.run_safely(model.fit, x=train_dataset, validation_data=val_dataset, epochs=config.epochs, initial_epoch=config.initial_epoch, verbose=1, callbacks=[wandb.keras.WandbCallback(save_model=False)])


        if config.save_to != None:
            model.save_weights(tfu.scripting.path_to(config.save_to) + ".h5")
    
def main(argv):
    config = tfu.scripting.init(argv[1:], define_arguments)
    tfu.scripting.random_seed(config.seed)
    
    print(tfu.scripting.initial_epoch(config))
    if tfu.scripting.initial_epoch(config) < config.epochs:
        train(config)
    
    
    
if __name__ == "__main__":
    sys.exit(tfu.scripting.boot(main, sys.argv))

