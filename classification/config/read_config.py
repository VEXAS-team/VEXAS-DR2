import re
import os
import configparser


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]



class Config:
    def __init__(self, config_filename):
        self.config = configparser.ConfigParser()
        self.config.read(config_filename)

    def get_all_models(self):
        sections = self.config.sections()
        sections = set([section.split('.')[0] for section in sections])
        return sorted(list(sections), key=natural_keys)

    def get_model_config(self, model_id):
        self.classes = self.config[model_id]['CLASSES'].split(',')
        self.bands = self.config[model_id]['BANDS'].split(',')
        self.aux_features = self.config[model_id]['AUX_FEATURES'].split(',')
        self.mag_transform = self.config[model_id]['FEATURE_TRANSFORM'].split(',')
        self.do_imputation = self.config[model_id].getboolean('IMPUTATION')
        self.add_labels = self.config[model_id].getboolean('ADD_LABELING')
        self.model_name = self.config[model_id]['MODEL']
        # self.corrected_path = self.config['LOGS']['CORR_OBJS']
        # self.model_save_path = os.path.join(self.config['LOGS']['SAVE_PATH'], model_id)
        self.aux_params = self.get_auxiliarly_parameters(model_id)

    def get_auxiliarly_parameters(self, model_id):
        aux_params = {}
        section = self.config[f'{model_id}.AUX_PARAMS']
        for key in section:
            key_value = section[key]
            aux_params[key] = self._read_in_correct_type(key_value)
        return aux_params

    def _read_in_correct_type(self, x):
        if x.isdigit():
            return int(x)
        elif x.replace('.', '', 1).isdigit():
            return float(x)
        elif x == 'None':
            return None
        else:
            return x
