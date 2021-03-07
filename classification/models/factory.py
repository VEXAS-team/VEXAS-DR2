import os
import numpy as np

from models.models import CatBoost, kNN, ANNClassifier, LogRegression
from log.plot import (plot_confusion_matrix, probability_threshold,
                      regression_coefficients)
from settings import MODELS_PATH, CLASSES



class ModelFactory:
    _single_models = {
        'kNN': kNN,
        'ANN': ANNClassifier,
        'CatBoost': CatBoost,
    }
    @classmethod
    def get(cls, model_name):
        return cls._single_models[model_name]




class Ensemble:
    def __init__(self):
        self.out_of_fold_predictions = {'valid': [], 'test': []}
        self.single_models = []
        self.single_models_classes = []

    def single_model(self, model_name, model_id, num_input_features, num_output_classes, model_save_path, **aux_params):
        model = ModelFactory.get(model_name)
        return model(model_id, num_input_features, num_output_classes, model_save_path, **aux_params)

    def meta_model(self, model_id, model_save_path):
        return LogRegression(model_id, model_save_path)


    def iteration(self, model_name, model_id, X, y, classes, features,
                  model_save_path, features_short=None, additinal_labels=False, **aux_params):
        num_input_features = len(features)
        num_output_classes = len(classes)

        model = self.single_model(model_name, model_id, num_input_features, num_output_classes, model_save_path, **aux_params)
        model.fit(X['train'], y['train'], X['valid'], y['valid'])
        if additinal_labels:
            preds_add = np.squeeze(model.predict(X['add']))
            agreed_filter = y['add'].to_numpy() == np.argmax(preds_add, axis=-1)
            X_add_agreed = X['add'][agreed_filter]
            y_add_agreed = y['add'][agreed_filter]

            X_train = X['train'].append(X_add_agreed, ignore_index=True)
            y_train = y['train'].append(y_add_agreed, ignore_index=True)
            model.fit(X_train, y_train, X['valid'], y['valid'])
        model.load()
        if features_short is not None:
            model.explain(X['train'], y['train'], features_short, classes)
        else:
            model.explain(X['test'], y['train'], features, classes)

        self.out_of_fold_predictions['valid'].append(model.predict(X['valid']))
        self.out_of_fold_predictions['test'].append(model.predict(X['test']))

        self.log_iteration_results(model, y['test'], model.predict(X['test']), classes)
        self.single_models.append(model_id)
        self.single_models_classes.append(classes)


    def stacked_iteration(self, model_id, y, model_save_path):
        metamodel = self.meta_model(model_id, model_save_path)
        stacked_val_predictions = np.hstack(self.out_of_fold_predictions['valid'])
        stacked_test_predictions = np.hstack(self.out_of_fold_predictions['test'])
        metamodel.fit(stacked_val_predictions, y['valid'])

        self.log_stacking_results(metamodel, y['test'],
                                  metamodel.predict(stacked_test_predictions))

    def log_iteration_results(self, model, labels, predictions, classes=CLASSES):
        save_path = model.model_path
        plot_confusion_matrix(labels, predictions, classes, model.model_id, save_path=save_path)
        probability_threshold(labels, predictions, save_path=save_path)

    def log_stacking_results(self, model, labels, predictions, classes=CLASSES):
        self.log_iteration_results(model, labels, predictions, classes)
        regression_coefficients(model, self.single_models,
                                self.single_models_classes,
                                model.model_path, classes=CLASSES)
