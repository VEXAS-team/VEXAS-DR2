import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import linregress
from sklearn.metrics import r2_score, mean_squared_error



def plt_config():
    fontsize=35
    plt.rc('font', family='times new roman', size=fontsize)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['agg.path.chunksize'] = 10000


MAG_COLS = {'J_VISTA':'J',
            'KS_VISTA': 'KS',
            'gpetMag_PS':'g',
            'rpetMag_PS':'r',
            'ipetMag_PS':'i',
            'zpetMag_PS':'z',
            'ypetMag_PS':'y',
            'mag_auto_g_DES': 'g',
            'mag_auto_r_DES': 'r',
            'mag_auto_i_DES': 'i',
            'mag_auto_z_DES': 'z',
            'mag_auto_y_DES': 'y',
            'u_petro_SM': 'u',
            'g_petro_SM': 'g',
            'r_petro_SM': 'r',
            'i_petro_SM': 'i',
            'z_petro_SM': 'z',
            'W1mag':'W1',
            'W2mag':'W2'}

MAG_TYPE = {'J': 'Vega',
            'KS': 'Vega',
            'u': 'AB',
            'g': 'AB',
            'r': 'AB',
            'i': 'AB',
            'z': 'AB',
            'y': 'AB',
            'W1': 'Vega',
            'W2': 'Vega'}


def plot_scatter(x, y, magnitude_name, save_path) -> float:
    plt_config()

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    print(magnitude_name, "R^2="+str(round(r_value**2, 3)))

    plt.figure(figsize=(14,14))
    plt.scatter(x, y, s=1, alpha=0.6, c='k')
    plt.plot([x.min(), x.max()], [x.min(), x.max()], c='r', linewidth=4)

    if MAG_COLS.get(magnitude_name) is not None:
        mag = MAG_COLS[magnitude_name]
    else:
        mag = magnitude_name

    plt.xlabel(f'{mag} true ', fontsize=fontsize)
    plt.ylabel(f'{mag} predicted', fontsize=fontsize)
    plt.xlim([np.quantile(x, 0.05), np.quantile(x, 0.998)])
    plt.ylim([np.quantile(y, 0.05), np.quantile(y, 0.998)])
    plt.savefig(save_path)
    plt.close()
    return round(r_value**2, 3))
