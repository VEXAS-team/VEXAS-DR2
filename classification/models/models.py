import warnings; warnings.filterwarnings('ignore')
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging; logging.getLogger('tensorflow').setLevel(logging.ERROR)

import torch
import pickle

from torch.optim import Adam
from torch.nn import BCELoss
from poutyne.framework import Model
from catalyst.dl.utils import UtilsFactory
from catboost import CatBoostClassifier, Pool
from catboost.utils import get_gpu_device_count
from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from poutyne.framework.callbacks import EarlyStopping, ModelCheckpoint

from log.plot import plot_importance
from models.ann import NeuralNetworkClassifier
from settings import (ANN_ENCODING_DIM, ANN_PATIENCE, ADAM_LEARNING_RATE,
                     STANDARD_MODEL_NAME, SEED, DATA_CACHE_DIR)




class CatBoost:
    _verbose = 200
    _train_dir = DATA_CACHE_DIR
    _is_gpu_available = get_gpu_device_count()
    _task_type = "GPU" if _is_gpu_available > 0 else None
    _devices = "GPU" if _is_gpu_available > 0 else None

    def __init__(self, model_id, num_input_features, num_output_classes, model_save_path, **aux_params):
        self.model = CatBoostClassifier(loss_function="MultiClass",
                                        task_type=self._task_type,
                                        devices=self._devices,
                                        train_dir=self._train_dir,
                                        random_seed=SEED)
        self.model.set_params(**aux_params)
        self.model_id = model_id

        path = f"{model_save_path}/{model_id}"
        os.makedirs(path, exist_ok=True)
        self.model_path = path
        self.modelfile_save_path = os.path.join(path, STANDARD_MODEL_NAME)

    def load(self):
        self.model.load_model(self.modelfile_save_path)

    def save(self):
        self.model.save_model(self.modelfile_save_path)

    def fit(self, X_train, y_train, X_valid, y_valid):
        self.model.fit(Pool(X_train, y_train),
                       eval_set=(X_valid, y_valid),
                       use_best_model=True,
                       verbose=self._verbose)
        self.save()

    def predict(self, X, load=False):
        if load:
            self.load()
        return self.model.predict_proba(X)

    def explain(self, X_train, y_train, features, classes):
        importances = self.model.get_feature_importance(data=Pool(X_train, y_train))
        plot_importance(importances, features, self.model_path, self.model_id)




class kNN:
    def __init__(self, model_id, num_input_features, num_output_classes, model_save_path, **aux_params):
        self.model = KNeighborsClassifier(**aux_params, n_jobs=-1)
        self.model_id = model_id

        path = f"{model_save_path}/{model_id}"
        os.makedirs(path, exist_ok=True)
        self.model_path = path
        self.modelfile_save_path = os.path.join(path, STANDARD_MODEL_NAME)


    def load(self):
        self.model = pickle.load(open(self.modelfile_save_path , 'rb'))

    def save(self):
        pickle.dump(self.model, open(self.modelfile_save_path, 'wb'))

    def fit(self, X_train, y_train, X_valid, y_valid):
        self.model.fit(X_train, y_train)
        self.save()

    def predict(self, X, load=False):
        if load:
            self.load()
        return self.model.predict_proba(X)

    def explain(self, X_train, y_train, features, classes):
        pass



class ANNClassifier:
    _lr = ADAM_LEARNING_RATE
    _encoding_dim = ANN_ENCODING_DIM
    _dtype = 'float32'

    def __init__(self, model_id, num_input_features, num_output_classes, model_save_path, **aux_params):
        self.ann_cls = NeuralNetworkClassifier(input_shape = num_input_features,
                                             encoding_dim = self._encoding_dim,
                                             classes = num_output_classes)

        self.model, device = UtilsFactory.prepare_model(self.ann_cls)
        self.model = Model(self.model,
                           Adam(self.model.parameters(),
                                lr=self._lr),
                           BCELoss(),
                           batch_metrics=None)
        self.model = self.model.to(device)

        self.model_id = model_id

        path = f"{model_save_path}/{model_id}"
        os.makedirs(path, exist_ok=True)
        self.model_path = path
        self.modelfile_save_path = os.path.join(path, STANDARD_MODEL_NAME)

        self.num_output_classes = num_output_classes
        self.learning_parameters = {}
        for key, value in aux_params.items():
            self.learning_parameters[key] = value

    def load(self):
        self.model.load_weights(self.modelfile_save_path)

    def fit(self, X_train, y_train, X_valid, y_valid):
        X_trn = X_train.to_numpy().astype(self._dtype)
        X_val = X_valid.to_numpy().astype(self._dtype)
        y_trn = self._y_cat(y_train, self.num_output_classes).astype(self._dtype)
        y_val = self._y_cat(y_valid, self.num_output_classes).astype(self._dtype)

        self.model.fit(X_trn, y_trn,
                  validation_data=(X_val, y_val),
                  callbacks = self._callbacks(),
                  **self.learning_parameters)

    def save(self):
        torch.save(self.model, self.modelfile_save_path)

    def _callbacks(self):
        return [EarlyStopping(patience=ANN_PATIENCE),
                ModelCheckpoint(filename=self.modelfile_save_path,
                                save_best_only=True,
                                restore_best=True)]

    def _y_cat(self, y, num_classes):
        return to_categorical(y, num_classes=num_classes)

    def predict(self, X, load=False):
        if load:
            self.load()
        return self.model.predict(X.to_numpy().astype(self._dtype))

    def explain(self, X_train, y_train, features, classes):
        pass




class LogRegression:
    _penalty = 'l1'
    _solver = 'liblinear'
    _max_iter = 1000

    def __init__(self, model_id, model_save_path):
        self.model = LogisticRegression(n_jobs=-1, random_state=SEED,
                                        max_iter=self._max_iter,
                                        penalty=self._penalty,
                                        solver=self._solver)

        self.model_id = model_id

        path = f"{model_save_path}/{model_id}"
        os.makedirs(path, exist_ok=True)
        self.model_path = path
        self.modelfile_save_path = os.path.join(path, STANDARD_MODEL_NAME)


    def load(self):
        self.model = pickle.load(open(self.modelfile_save_path , 'rb'))

    def save(self):
        pickle.dump(self.model, open(self.modelfile_save_path, 'wb'))

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.save()

    def predict(self, X, load=False):
        if load:
            self.load()
        return self.model.predict_proba(X)
