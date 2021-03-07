import warnings
warnings.filterwarnings("ignore")

from model.autoencoder import set_seed
set_seed()

import os
import logging
import argparse

import numpy as np
from tf.keras.optimizers import Nadam

from dataset.dataset import VexasDataset
from model.autoencoder import Autoencoder, callbacks
from utility.plot import plot_scatter
from settings import BATCH_SIZE, EPOCHS, SCALING_CONSTANT, CHECKPOINT_PATH, LOGFILE, EPS


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
    VEXAS = VexasDataset(args.survey_name)
    X, num_mags = VEXAS.select_with_all_magnitudes()
    logging.info(f"[start] {args.survey_name}")
    logging.info(f"[data] Training sample: {X['train_mags'].shape[0]} sources")
    logging.info(f"[data] Validation sample: {X['valid_mags'].shape[0]} sources")

    save_path = f"./{args.survey_name}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model = Autoencoder(input_dim=num_mags,
                        use_auxiliary_features=True)
    model.compile(Nadam, 'logcosh')

    train_data = (X['train_mags'], X['train_aux'])
    valid_data = ((X['valid_mags'], X['valid_aux']), X['valid'])
    history = model.fit(x=train_data,
                        y=X['train'],
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=valid_data,
                        shuffle=True,
                        callbacks=callbacks(save_path))

    model.load_weights(f"{save_path}/{CHECKPOINT_PATH}")
    valid_predicted = model.predict((X['valid_mags'], X['valid_aux']))

    columns = X['valid'].columns
    logging.info("[validation] results:")
    for i in range(valid_predicted.shape[-1]):
        col = columns[i]
        logging.info(f"{col}")
        # plot imputed magnitudes only
        idxs = np.abs(X['valid'][col] - valid_predicted[:, i]) > EPS
        x = X['valid'][idxs][col] * SCALING_CONSTANT
        y = valid_predicted[idxs, i] * SCALING_CONSTANT
        r_2 = plot_scatter(x, y, col, f"{save_path}/{col}.png")
        logging.info(f"[validation] {col}: {r_2}")
