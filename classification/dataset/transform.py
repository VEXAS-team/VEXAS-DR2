import pandas as pd

from settings import CLASSES_CODES, NUMBER_OF_CLASSES, MAG_SHORTN


class Transform:
    _available_transforms = ['color', 'magnitude', 'magnitude_normalized']
    _empty_list = ['None']

    def __init__(self, dataset: pd.core.frame.DataFrame,
                 transform_pipeline: list):
        """
        Class for manipulating photometric information.

        :param dataset: dataframe with photometric info for the sources
        :type dataset: pd.core.frame.DataFrame
        :param transform_pipeline: series of transforms
        :type transform_pipeline: list
        """
        self.features = []
        self.features_short = []
        self.dataset = dataset
        self.transform_pipeline = transform_pipeline
        assert all(transformation in self._available_transforms
                   for transformation in self.transform_pipeline),\
                'Transformation pipeline include NotImplemented transforms.'\
                f' List of available transforms: {self._available_transforms}'

    def transform(self, list_of_magnitudes, list_of_auxiliary_features):
        for transform in self.transform_pipeline:
            if transform == 'color':
                self.make_colors(list_of_magnitudes)
            if transform == 'magnitude':
                self.make_magnitudes(list_of_magnitudes)
            if transform == 'magnitude_normalized':
                self.make_magnitudes(list_of_magnitudes, normalize=True)

        if list_of_auxiliary_features != self._empty_list:
            self.features += list_of_auxiliary_features
            self.features_short += self.short_names(list_of_auxiliary_features)

    def make_magnitudes(self, mag_list, normalize=False):
        if normalize:
            mag_list_norm = [f'{mag_name}_norm' for mag_name in mag_list]
            max_mag_for_each_source = self.dataset[mag_list].max(axis=1)
            self.dataset[mag_list_norm] = self.dataset[mag_list].div(max_mag_for_each_source,
                                                                     axis=0)
            self.features += mag_list_norm
        else:
            self.features += mag_list
        self.features_short += self.short_names(mag_list)

    def make_colors(self, mag_list):
        num_mags = len(mag_list)
        for i in range(num_mags):
            for j in range(i+1, num_mags):
                mag_first = mag_list[i]
                mag_second = mag_list[j]
                color_name = f"{mag_first}_{mag_second}"
                color_short_name = f"{self.short_names(mag_first)}-{self.short_names(mag_second)}"
                self.dataset[color_name] = self.dataset[mag_first] - self.dataset[mag_second]
                self.features.append(color_name)
                self.features_short.append(color_short_name)

    def short_names(self, input_features):
        if type(input_features) == list:
            return [MAG_SHORTN.get(feature) for feature in input_features]
        if type(input_features) == str:
            return MAG_SHORTN.get(input_features)
        return input_features
