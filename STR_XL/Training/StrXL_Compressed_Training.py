import sys
import tf_utilities as tfu
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.append("../../../deep-learning-dna")
sys.path.append("../")
sys.path.append("../../../deep-learning-dna/common")

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
import dotenv
import pickle

from Scripts.StrXL_Compressed import *

def define_arguments(cli):
    cli.use_strategy()
    cli.use_wandb()

    cli.artifact("--dataset", type=str, required=True)
    cli.artifact("--encoder", type=str, required=True)
    
    cli.argument("--seed", type=int, default = None)
    
    cli.argument("--mem_switched", type=tfu.utils.str_to_bool, default=False)
    cli.argument("--num_compressed_seeds", type=int, default=50)
    cli.argument("--compressed_len", type=int, default=250)
    cli.argument("--block_size", type=int, default = 250)
    cli.argument("--max_set_len", type=int, default = 1000)
    cli.argument("--num_induce", type=int, default = 0)
    cli.argument("--embed_dim", type=int, default = 64)
    cli.argument("--num_layers", type=int, default = 8)
    cli.argument("--num_heads", type=int, default = 8)
    cli.argument("--mem_len", type=int, default = 250)
    cli.argument("--compressed_mem_len", type=int, default = 250)
    cli.argument("--dropout_rate", type=float, default = 0.01)
    cli.argument("--num_seeds", type=int, default = 1)
    cli.argument("--use_layernorm", type=tfu.utils.str_to_bool, default = True)
    cli.argument("--pre_layernorm", type=tfu.utils.str_to_bool, default = True)
    cli.argument("--use_keras_mha", type=tfu.utils.str_to_bool, default = True)

    cli.argument("--set_len", type=int, default=1000)
    
    cli.argument("--batches_per_epoch", type=int, default=20)
    cli.argument("--validation_batch_size", type=int, default=5)
    
    cli.argument("--save-to", type=str, default=None)
    
    cli.use_training(epochs=700, batch_size=20)

    
def load_dataset(config):
    dataset_path = tfu.scripting.artifact(config, "dataset")
    
    samples = find_dbs(dataset_path)
    
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
                                                    batch_size=batch_size,batches_per_epoch=batches_per_epoch,augment=augument,labels=labels,rng=rng)


    return trimmed_samples, train_dataset, val_dataset 

def train(config, model_path):
    with tfu.scripting.strategy(config).scope():
        trimmed_samples, train_dataset, val_dataset = load_dataset(config)
        
        encoder_path = tfu.scripting.artifact(config, "encoder")
        pretrained_model = dnabert.DnaBertModel.load(encoder_path)
        pretrained_model.load_weights(encoder_path + "/model.h5")
        
        encoder = dnabert.DnaBertEncoderModel(pretrained_model.base)
        encoder.trainable = False
        
        max_files = len(trimmed_samples)
        
        model = XlModel(config.mem_switched, config.num_compressed_seeds, config.compressed_len, max_files, encoder, config.block_size, config.max_set_len, config.num_induce, config.embed_dim, config.num_layers, config.num_heads, config.mem_len, config.dropout_rate, config.num_seeds, config.use_layernorm, config.pre_layernorm, config.use_keras_mha)
      
        model(train_dataset[0][0][:1])
        
        if model_path is not None:
            with open(model_path, "rb") as f:
                model.set_weights(pickle.load(f))
 
        model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer = keras.optimizers.Adam(1e-3),
                        metrics=keras.metrics.SparseCategoricalAccuracy())

        wandb_callback = wandb.keras.WandbCallback(save_model=False)
        wandb_callback.save_model_as_artifact = False
        wandb_callback.save_weights_only: bool = True
        wandb_callback.save_freq: Union[SaveStrategy, int] = "epoch",   

        tfu.scripting.run_safely(model.fit, x=train_dataset, validation_data=val_dataset, epochs=config.epochs + config.initial_epoch, initial_epoch=tfu.scripting.initial_epoch(config), verbose=1, callbacks=[wandb_callback])

    if config.save_to != None:
        filename = tfu.scripting.path_to(config.save_to.format(**config.__dict__)) + ".h5"
        
        with open(filename, "wb") as f:
            pickle.dump(model.get_weights(), f)

def main(argv):
    dotenv.load_dotenv()
    config = tfu.scripting.init(define_arguments)
    tfu.scripting.random_seed(config.seed)
    
    model_path = None
    if tfu.scripting.is_resumed():
        print("Restoring previous model...")
        model_path = tfu.scripting.restore(config.save_to.format(**config.__dict__) + ".h5").name
        print(tfu.scripting.initial_epoch(config))
        
    if tfu.scripting.initial_epoch(config) < config.epochs:
        train(config, model_path)
    
if __name__ == "__main__":
    sys.exit(tfu.scripting.boot(main, sys.argv))

