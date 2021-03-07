import os

import pandas as pd
import numpy as np

from settings import (TRAIN_PATH, MAG_MAGERR, MAIN_SURVEY,
                      CORRECTED_SPEC_OBJECTS_PATH, SEED,
                      TRAIN_SIZE, TEST_SIZE, VALID_SIZE,
                      NUMBER_OF_CLASSES, CLASSES_CODES)



class SpectroscopicSurvey:
    _main_survey = MAIN_SURVEY
    _duplicated_column = 'ID_AllWISE'
    _cols_dtypes = {'SOURCEID_VISTA': str, 'SLAVEOBJID_WISE': str,
                    'ID_AllWISE': str}

    def __init__(self, vexas_table_name: str):
        self.vexas_table_name = vexas_table_name

    def read(self, survey_name: str):
        spec_table_name = f"VEXAS_{self.vexas_table_name}_{survey_name}.csv"
        spec_table_path = os.path.join(TRAIN_PATH, spec_table_name)
        table = pd.DataFrame()
        if os.path.exists(spec_table_path):
            table = pd.read_csv(spec_table_path, converters=self._cols_dtypes)
            table['survey'] = survey_name
        return table

    def _nan_check(self, x):
        return x == x

    def _insert_nan_instead_imputation(self, source: pd.core.series.Series, bands: list) -> list:
        reimputed_source = []
        for mag in bands:
            err = source[MAG_MAGERR.get(mag)]
            mag_to_insert = source[mag] if self._nan_check(err) else np.nan
            reimputed_source.append(mag_to_insert)
        return reimputed_source

    def deimputation(self, dataset, bands):
        dataset[bands] = dataset.apply(lambda source:
                    self._insert_nan_instead_imputation(source, bands),
                    axis=1, result_type='expand')
        return dataset

    def data(self,
             bands: list = [],
             additional_datasets: list = [],
             do_imputation: bool = False
             ) -> pd.core.frame.DataFrame:

        dataset = self.read(self._main_survey)
        for additional_dataset in additional_datasets:
            dataset = dataset.append(self.read(additional_dataset),
                                     ignore_index=True)
        dataset = dataset.drop_duplicates(subset=[self._duplicated_column],
                                          keep='first')
        if not do_imputation:
            dataset = self.deimputation(dataset, bands)
        return dataset



class TrainingSet:
    _duplicated_column = 'ID_AllWISE'
    _spec_objtype_col = 'objtype'
    _label_column = 'objclass'
    _main_survey = MAIN_SURVEY
    _holdout_total_frac = TEST_SIZE + VALID_SIZE

    def __init__(self, dataset):
        np.random.seed(seed=SEED)
        self.dataset = dataset
        self.dataset = self.dataset.reset_index(drop=True)
        self._num_sources = len(self.dataset)
        self.main_survey_filter = self.dataset['survey'] == MAIN_SURVEY

    def _split_indexes(self):
        # Split dataset on train, test, and valid, returning corresponding IDs
        _proba_for_each_index = np.random.random(size=self._num_sources)
        main_survey_idxs = list(self.dataset[self.main_survey_filter].index.values)

        test_idx, valid_idx = [], []
        for idx in main_survey_idxs:
            proba = _proba_for_each_index[idx]
            if proba < TEST_SIZE:
                test_idx.append(idx)
            if proba >= TEST_SIZE and proba < self._holdout_total_frac:
                valid_idx.append(idx)

        train_idx = set(main_survey_idxs) - set(test_idx) - set(valid_idx)
        train_idx = list(train_idx)
        test_idx = self._correct_ids(test_idx)
        valid_idx = self._correct_ids(valid_idx)
        return train_idx, valid_idx, test_idx

    def _correct_dataset_labels(self):
        # Infer IDs of badly labeled sources
        if os.path.exists(CORRECTED_SPEC_OBJECTS_PATH):
            corrected_objects = pd.read_csv(CORRECTED_SPEC_OBJECTS_PATH)
            corrected_objects = corrected_objects[corrected_objects['accept'] == 0]
            merged = pd.merge(self.dataset,
                            corrected_objects[self._duplicated_column],
                            on=[self._duplicated_column],
                            how='left',
                            indicator='Exist')
            merged['Exist'] = np.where(merged.Exist == 'both', True, False)
            return merged[merged['Exist']].index.values
        return []

    def _correct_ids(self, indexes_to_correct):
        # Returns the list of IDs without badly labeled sources
        ids_to_remove = self._correct_dataset_labels()
        return list(set(indexes_to_correct) - set(ids_to_remove))

    def _decode_classes(self, row, classes):
        """
        Decodes object types codes according to the
        classification problem under solve.
         - Three-class:
              -> 0, 1, 2
              <- 0, 1, 2
         - One-vs-Rest:
              -> 0, 1, 2
              <- 0, 1 (where 0 for One, 1 for Rest classes)

        :param row: [description]
        :type row: [type]
        :param classes: [description]
        :type classes: [type]
        :return: [description]
        :rtype: [type]
        """
        if len(classes) == NUMBER_OF_CLASSES:
            return row[self._spec_objtype_col]
        else:
            one_which_is_vs_rest = classes[0]
            code_for_one = CLASSES_CODES.get(one_which_is_vs_rest)
            if code_for_one is not None:
                # `1` for One, `0` for Rest:
                return (row[self._spec_objtype_col] == code_for_one) * 1
            else:
                # If One name is not known, return default labeling `0,1,2`:
                return row[self._spec_objtype_col]

    def get_train_test_val(self, features, classes):
        train_idx, valid_idx, test_idx = self._split_indexes()
        self.dataset[self._label_column] = self.dataset.apply(lambda row:
                                self._decode_classes(row, classes), axis=1)

        X = {'train': self.dataset[features].iloc[train_idx],
             'valid': self.dataset[features].iloc[valid_idx],
             'test' : self.dataset[features].iloc[test_idx],
             'add': self.dataset[~self.main_survey_filter][features]}
        y = {'train': self.dataset[self._label_column].iloc[train_idx],
             'valid': self.dataset[self._label_column].iloc[valid_idx],
             'test' : self.dataset[self._label_column].iloc[test_idx],
             'add': self.dataset[~self.main_survey_filter][self._label_column]}
        return X, y
