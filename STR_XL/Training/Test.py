import sys

sys.path.append("../")

import tf_utilities as tfu
import os
import dotenv
import numpy as np
import wandb

from Scripts.Test_Model import *

def define_arguments(cli):
    cli.use_strategy()
    
    cli.use_training(epochs=20, batch_size=10)

    cli.argument("--save_to", type=str, default=None)

def train(config, model_path):

    with tfu.scripting.strategy(config).scope():
    
        train_dataset = np.random.normal(size=[2,20,5])
        val_dataset = np.random.normal(size=[2,10,5])
        
        model = TestModel()
      
        if model_path is not None:
            model = model.load_weights(model_path + ".h5")
 
        model.compile(loss = keras.losses.mse, optimizer = keras.optimizers.Adam(1e-3))

              
        tfu.scripting.run_safely(model.fit, x=train_dataset[0], y=train_dataset[1], validation_data=tuple(val_dataset), epochs=config.epochs + config.initial_epoch, initial_epoch=config.initial_epoch, verbose=1, callbacks=[wandb.keras.WandbCallback(save_model=False)])


        if config.save_to != None:
            model.save_weights(tfu.scripting.path_to(config.save_to) + ".h5")



def main(argv):
    dotenv.load_dotenv()
    config = tfu.scripting.init(define_arguments)

    model_path = None
    if tfu.scripting.is_resumed():
        print("Restoring previous model...")
        model_path = tfu.scripting.restore_dir(config.save_to)

    print(model_path)

    print(tfu.scripting.initial_epoch(config))
    if tfu.scripting.initial_epoch(config) < config.epochs:
        train(config, model_path)    
    
if __name__ == "__main__":
    sys.exit(tfu.scripting.boot(main, sys.argv))































