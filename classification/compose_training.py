import os
import argparse

import numpy as np

from tqdm import tqdm

from models.factory import Ensemble
from config.read_config import Config
from dataset.transform import Transform
from dataset.dataset import SpectroscopicSurvey, TrainingSet
from settings import MAIN_SURVEY, AUX_SURVEYS, CONFIG_PATH, TRAIN_PATH



def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for training ensemble to classify VEXAS sources.'
    )
    parser.add_argument(
        '--dataset', '-ds', dest='dataset',
        default='PS', help='VEXAS subsample name (PS, DES, SM)'
    )
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    vexas_dataset = args.dataset
    config = Config(f'{CONFIG_PATH}/config_{vexas_dataset}.ini')
    config.get_model_config('MODEL_1')

    spectra_getter = SpectroscopicSurvey(vexas_dataset)
    spectroscopic_data = spectra_getter.data(bands=config.bands,
                                             additional_datasets=[],
                                             do_imputation=True)
    training_set = TrainingSet(spectroscopic_data)
    train_idx, valid_idx, test_idx = training_set._split_indexes()

    spectroscopic_data.to_csv(f'{TRAIN_PATH}/{vexas_dataset}_SPEC_ALL.csv', index=None)
    spectroscopic_data.iloc[test_idx].to_csv(f'{TRAIN_PATH}/{vexas_dataset}_SPEC_TEST.csv', index=None)
    spectroscopic_data.iloc[train_idx].to_csv(f'{TRAIN_PATH}/{vexas_dataset}_SPEC_TRAIN.csv', index=None)
    spectroscopic_data.iloc[valid_idx].to_csv(f'{TRAIN_PATH}/{vexas_dataset}_SPEC_VALID.csv', index=None)
