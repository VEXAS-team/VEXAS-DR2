import os
import json
import itertools

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import *
from matplotlib import colors as plt_colors

from settings import COLOR2CLASS, NUMBER_OF_CLASSES, CLASSES_CODES, MAX_SHAP_DISPLAY

COLORS = ['#DB4549', '#2E3853', '#A3C9D3']



def _plot_confusion_matrix(cm, classes, title=None, normalize=True, cmap='Reds'):
    fontsize=35
    label_fontsize = fontsize - 5

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.rc('font', family='times new roman', size=fontsize)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if title is not None:
        plt.title(title, fontsize=label_fontsize)
    plt.grid(False)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=label_fontsize)
    plt.yticks(tick_marks, classes, fontsize=label_fontsize)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt)+'%' if normalize else format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=fontsize,
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True class', fontsize=label_fontsize)
    plt.xlabel('Predicted class', fontsize=label_fontsize)
    plt.tight_layout()


def plot_confusion_matrix(y, preds, classes, title, normalize=True, save_path='./'):
    cnf_matrix = confusion_matrix(y, np.argmax(preds, axis=-1))
    plt.figure(figsize=(11,11))
    _plot_confusion_matrix(cnf_matrix, classes=classes, title=title,
                          normalize=normalize)
    plt.savefig(f'{save_path}/confusion_matrix.png')
    plt.close()



def probability_threshold(y_true_multi, y_pred_proba, save_path='./', classes=['STAR','QSO','GALAXY']):
    for clas in range(len(classes)):
        y_true = (y_true_multi==clas)*1

        thresholds = np.arange(0.0,1,0.01)
        metrics = {'MCC':{'value': [],       'color': '#DB4549'},
                   'Precision':{'value': [], 'color': '#2E3853'},
                   'Recall':{'value': [],    'color': '#3E6A91'},
                   'F1-score':{'value': [],  'color': '#F2CE63'}
                   }

        for threshold in thresholds:
            y_pred = (np.argmax((y_pred_proba>threshold)*1, axis=-1)==clas)*1
            metrics['MCC']['value'].append(matthews_corrcoef(y_true, y_pred))
            metrics['Precision']['value'].append(precision_score(y_true, y_pred, average='macro'))
            metrics['Recall']['value'].append(recall_score(y_true, y_pred, average='macro'))
            metrics['F1-score']['value'].append(f1_score(y_true, y_pred, average='macro'))

        plt.figure(figsize=(15,11))
        for metric in metrics.keys():
            plt.plot(thresholds,
                     metrics[metric]['value'],
                     linewidth=5,
                     color=metrics[metric]['color'],
                     label=metric)
        plt.legend()
        plt.xlim([-0.001,1.001])
        plt.ylim([0.849, 1.0001])
        plt.xticks(np.arange(0, 1+0.1, 0.1))
        plt.grid(True)
        plt.xlabel('Probability threshold')
        plt.title(f'{classes[clas]} vs rest')
        plt.savefig(f'{save_path}/threshold_{classes[clas]}.png')
        plt.close()

def regression_coefficients(meta_model, single_models, single_models_classes, save_path, classes=['STAR','QSO','GALAXY']):
    weights = meta_model.model.coef_
    coef = pd.DataFrame(columns=['class', 'weight', 'model_class'])
    print('weights', weights)
    for cl in range(len(classes)):
        w_cl = weights[cl]
        m_cl = []
        for model_name, model_classes in zip(single_models, single_models_classes):
            for model_class in model_classes:
                m_cl.append(r"$P_{%s, %s}$" % (model_name.split('_')[-1], model_class))
        for i in range(len(w_cl)):
            coef = coef.append({'class': cl,
                        'weight': w_cl[i],
                        'model_class': m_cl[i]},
                        ignore_index=True)

    coef.to_csv(f"{save_path}/coef.csv", index=None)
    df = coef.sort_values(['class', 'weight'], ascending=False)
    df['weight'] = df['weight'] / np.max(df['weight'])
    df['absolute_weight'] = df['weight'].abs()
    # df = df[(df['weight'] != 0.0)]
    df = df.nlargest(5, 'absolute_weight')

    plt.figure(figsize=(25, 7))
    for cl in df['class'].unique():
        plt.subplot(1, 3, cl+1)
        plt.plot(df[df['class'] == cl]['model_class'],
                 df[df['class'] == cl]['weight'],
                 color='k',
                 linewidth=7)
        plt.title(classes[cl])
        plt.grid(True)
        plt.xticks(df[df['class'] == cl]['model_class'],
                   df[df['class'] == cl]['model_class'],
                   rotation='vertical')
        plt.ylim([-1, 1])

    plt.yticks(np.arange(-1, 1.2, step=0.2))
    if cl == 0:
        plt.ylabel('Regression coefficients')
    plt.savefig(f"{save_path}/coef.png", bbox_inches='tight')
    plt.close()


def plot_importance(importances, features, save_path, title=''):
    result = pd.DataFrame({'features': features, 'importances': importances})
    result = result.sort_values(by='importances', ascending=False).reset_index(drop=True).iloc[:8]
    features = result['features'].tolist()
    importances = result['importances'].tolist()

    fig, ax = plt.subplots()
    ax.barh(features, importances, align='center')
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Feature importance')
    if len(title) > 0:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{title}_feature_importance.png", bbox_inches='tight')
    plt.close()
