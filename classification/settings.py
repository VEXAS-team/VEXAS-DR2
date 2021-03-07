SEED = 100
NAN_TO_NUM = 0.0
CLASSES = ['STAR', 'QSO', 'GALAXY']
CLASSES_CODES = {'STAR': 0, 'QSO': 1, 'GALAXY': 2}

NUMBER_OF_CLASSES = 3

TRAIN_PATH = './data/train'
MODELS_PATH = './data/models'
CONFIG_PATH = './config'
DATA_CACHE_DIR = './data/cache/'

MAIN_SURVEY = 'SDSSDR16'
AUX_SURVEYS = ['GAMADR3', 'WigglezFinal', '2QZ', '6df', 'OzDESDR1']
CORRECTED_SPEC_OBJECTS_PATH = './data/train/corr.csv'

META_MODEL_NAME = 'META_MODEL'

TRAIN_SIZE, TEST_SIZE, VALID_SIZE = 0.6, 0.2, 0.2

ANN_PATIENCE = 5
ANN_ENCODING_DIM = 5
ADAM_LEARNING_RATE = 1e-3
STANDARD_MODEL_NAME = 'model'

MAX_SHAP_DISPLAY = 8

MAG_MAGERR = {'J_VISTA':'EJ_VISTA',
              'KS_VISTA':'EKS_VISTA',
              'mag_auto_g_DES': 'magerr_auto_g_DES',
              'mag_auto_r_DES': 'magerr_auto_r_DES',
              'mag_auto_i_DES': 'magerr_auto_i_DES',
              'mag_auto_z_DES': 'magerr_auto_z_DES',
              'mag_auto_y_DES': 'magerr_auto_y_DES',
              'gpetMag_PS':'gpetMagErr_PS',
              'rpetMag_PS':'rpetMagErr_PS',
              'ipetMag_PS':'ipetMagErr_PS',
              'zpetMag_PS':'zpetMagErr_PS',
              'ypetMag_PS':'ypetMagErr_PS',
              'u_petro_SM': 'e_u_petro_SM',
              'g_petro_SM': 'e_g_petro_SM',
              'r_petro_SM': 'e_r_petro_SM',
              'i_petro_SM': 'e_i_petro_SM',
              'z_petro_SM': 'e_z_petro_SM',
              'W1mag':'e_W1mag',
              'W2mag':'e_W2mag',
              }

MAG_SHORTN = {'J_VISTA':'J',
              'KS_VISTA':'Ks',
              'mag_auto_g_DES': 'g',
              'mag_auto_r_DES': 'r',
              'mag_auto_i_DES': 'i',
              'mag_auto_z_DES': 'z',
              'mag_auto_y_DES': 'y',
              'gpetMag_PS':'g',
              'rpetMag_PS':'r',
              'ipetMag_PS':'i',
              'zpetMag_PS':'z',
              'ypetMag_PS':'y',
              'u_petro_SM': 'u',
              'g_petro_SM': 'g',
              'r_petro_SM': 'r',
              'i_petro_SM': 'i',
              'z_petro_SM': 'z',
              'W1mag':'W1',
              'W2mag':'W2',
              'PSTAR_VISTA': 'PSTAR'
              }

COLOR2CLASS = {'STAR': (255/255, 59/255, 94/255),
               'QSO': (17/255, 153/255, 85/255),
               'GALAXY': (67/255, 92/255, 251/255),
               'QSO+GAL': (128/255, 128/255, 128/255),
               'STAR+GAL': (128/255, 128/255, 128/255),
               'STAR+QSO': (128/255, 128/255, 128/255)}

