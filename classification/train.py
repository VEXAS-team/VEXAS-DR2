import os
import argparse

import numpy as np

from tqdm import tqdm

from models.factory import Ensemble
from config.read_config import Config
from dataset.transform import Transform
from dataset.dataset import SpectroscopicSurvey, TrainingSet
from settings import MAIN_SURVEY, AUX_SURVEYS, SEED, CONFIG_PATH, META_MODEL_NAME, MODELS_PATH



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
    ensemble = Ensemble()
    models = config.get_all_models()
    models_save_path = f"{MODELS_PATH}/{vexas_dataset}"
    if META_MODEL_NAME in models:
        models.remove(META_MODEL_NAME)

    for model_id in models:
        print(model_id)
        config.get_model_config(model_id)
        if 'CatBoost' in config.model_name:

            spectra_getter = SpectroscopicSurvey(vexas_dataset)
            spectroscopic_data = spectra_getter.data(bands=config.bands,
                                                    additional_datasets=AUX_SURVEYS if config.add_labels else [],
                                                    do_imputation=config.do_imputation)

            transformer = Transform(spectroscopic_data, config.mag_transform)
            transformer.transform(config.bands, config.aux_features)
            transformed_data = transformer.dataset
            features = transformer.features
            features_short = transformer.features_short

            training_set = TrainingSet(transformed_data)
            X, y = training_set.get_train_test_val(features, config.classes)
            ensemble.iteration(config.model_name, model_id, X, y,
                            config.classes, features, models_save_path,
                            features_short,
                            additinal_labels=config.add_labels,
                            **config.aux_params)

    # ensemble.stacked_iteration(META_MODEL_NAME, y, models_save_path)
