from autoencoder import set_seed
set_seed()

import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import logging

import numpy as np
from tf.keras.optimizers import Nadam

from dataset.dataset import VexasDataset
from model.autoencoder import Autoencoder, callbacks
from settings import BATCH_SIZE, EPOCHS, SCALING_CONSTANT, CHECKPOINT_PATH, LOGFILE

logging.basicConfig(filename=LOGFILE,
                    filemode='a',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for training magnitude imputer.'
    )
    parser.add_argument(
        '--survey_name', '-sn', dest='survey_name',
        default='DESW', help='Name of the input survey (one of PSW, DESW, SMW)'
    )
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    save_path = f"./{args.survey_name}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    VEXAS = VexasDataset(args.survey_name)
    num_mags = len(VEXAS.magnitude_columns)
    model = Autoencoder(input_dim=num_mags,
                        use_auxiliary_features=True)
    model.compile(Nadam(), 'logcosh')
    model.load_weights(f"{save_path}/{CHECKPOINT_PATH}")
    VEXAS.inference(model, save_path)
