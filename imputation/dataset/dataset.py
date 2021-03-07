import os
import configparser
import pandas as pd
import numpy as np
import astropy.io.fits as fits

from tqdm import tqdm
from astropy.table import Table
from sklearn.model_selection import train_test_split

from settings import MISSED_MAGS, CHUNK_SIZE, SCALING_CONSTANT, TEST_SIZE


class VexasDataset:
    def __init__(self, survey_name):
        self.survey_name = survey_name
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.config.read('config.ini')

        self.filepath = self.config[f"{self.survey_name}.FILE"]['PATH']
        self.magnitude_columns = self.config[f"{self.survey_name}.MAGNITUDES"]['MAGS'].split(' ')
        self.magnitude_mask = {}
        for mag in self.magnitude_columns:
            unmasked_rate = self.config[f"{self.survey_name}.MASK"][mag]
            self.magnitude_mask[mag] = float(unmasked_rate)
        self.magnitude_error_constraints = {}
        section = f"{self.survey_name}.CONSTRAINTS"
        if self.config.has_section(section):
            for (key, limit) in self.config.items(section):
                self.magnitude_error_constraints[key] = float(limit)
        self.aux_columns = self.config[f"{self.survey_name}.AUX"]['AUX_FEATURES'].split(' ')
        self.feature_columns = self.magnitude_columns + self.aux_columns
        self.X = {}
        for split in ['train', 'valid']:
            self.X[split] = pd.DataFrame()
            self.X[f'{split}_mags'] = None
            self.X[f'{split}_aux'] = None

    def read_fits(self):
        filefits = fits.open(self.filepath, memmap=True)
        return filefits

    def get_star(self, filefits, idx, batch_size, features_only=True):
        index_low = idx * batch_size
        index_high = (idx + 1) * batch_size
        stars = pd.DataFrame(np.array(filefits[1].data[index_low:index_high]).byteswap().newbyteorder())
        if self.config.has_section(f"{self.survey_name}.TRANSFORMS"):
            keys = self.config[f"{self.survey_name}.TRANSFORMS"]['COLOURS']
            keys = keys.split(' ')
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    stars[f"{keys[i]}-{keys[j]}"] = stars[keys[i]]-stars[keys[j]]
        if self.magnitude_error_constraints:
            for key, limit in self.magnitude_error_constraints.items():
                stars = stars[(stars[key] < limit) | (stars[key].isnull())]
        if features_only:
            return stars[self.feature_columns]
        else:
            return stars

    def construct_mask(self, x):
        """
        Constructs random mask for each feature vector, returning
        0 if feature is considered as a missing one, and 1 otherwise.
        """
        mask = np.zeros(x.shape)
        mask = pd.DataFrame(mask, columns=x.columns)
        x.reset_index(inplace=True, drop=True)
        for column in mask.columns:
            is_masked = self.magnitude_mask.get(column) is not None
            if is_masked and self.magnitude_mask[column] == 1:
                mask[column] = 1
            elif is_masked and self.magnitude_mask[column] == 0:
                mask[column] = 0
            elif not is_masked:
                mask[column] = 1
            else:
                mask[column] = np.random.uniform(size=len(mask[column])) < self.magnitude_mask[column]
            mask[column] *= x[column]
        return mask

    def select_with_all_magnitudes(self):
        filefits = self.read_fits()
        number_of_sources = filefits[1].header['NAXIS2']
        chunk_size = CHUNK_SIZE
        if number_of_sources < chunk_size:
            chunk_size = number_of_sources
        for source_index in tqdm(range(number_of_sources // chunk_size)):
            stars = self.get_star(filefits, source_index, chunk_size)
            stars = stars.replace(MISSED_MAGS, np.nan)
            stars = stars[stars < SCALING_CONSTANT]
            stars[self.magnitude_columns] = stars[self.magnitude_columns] / SCALING_CONSTANT
            self.X['train'] = self.X['train'].append(stars[~stars.isna().any(axis=1)],
                                                     ignore_index=True)

        self.X['train'], self.X['valid'] = train_test_split(self.X['train'], test_size=TEST_SIZE)
        self.X['train_mags'] = self.construct_mask(self.X['train'][self.magnitude_columns])
        self.X['valid_mags'] = self.construct_mask(self.X['valid'][self.magnitude_columns])

        self.X['train_aux'] = self.X['train'][self.aux_columns]
        self.X['valid_aux'] = self.X['valid'][self.aux_columns]

        self.X['train'] = self.X['train'][self.magnitude_columns]
        self.X['valid'] = self.X['valid'][self.magnitude_columns]
        return self.X, len(self.magnitude_columns)

    def inference(self, model, save_path: str):
        filefits = self.read_fits()
        number_of_sources = filefits[1].header['NAXIS2']
        chunk_size = CHUNK_SIZE
        if number_of_sources < chunk_size:
            chunk_size = number_of_sources
        for source_index in tqdm(range(number_of_sources // chunk_size)):
            stars = self.get_star(filefits, source_index, chunk_size, features_only=False)
            stars = stars.replace(MISSED_MAGS, 0)
            stars[self.feature_columns] = stars[self.feature_columns].replace(np.nan, 0)
            for feature in self.magnitude_columns:
                stars = stars[stars[feature] < SCALING_CONSTANT]
            stars[self.magnitude_columns] = stars[self.magnitude_columns] / SCALING_CONSTANT
            prediction = model.predict((stars[self.magnitude_columns],
                                        stars[self.aux_columns]))
            prediction = prediction * SCALING_CONSTANT

            for col_id, col in enumerate(self.magnitude_columns):
                stars[f"{col}"] = prediction[:, col_id]
            if source_index == 0:
                stars.to_csv(f"{save_path}/{self.survey_name}.csv", index=False)
            else:
                stars.to_csv(f"{save_path}/{self.survey_name}.csv", index=False, header=False, mode='a')
