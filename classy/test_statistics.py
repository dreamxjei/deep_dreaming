# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import matplotlib
matplotlib.use('Agg') # Added for display errors
from matplotlib import pyplot as plt
import numpy as np

from sklearn import metrics


def roc_auc_metrics(y_true, y_score, n_classes, weightfile, network, results_dir):
    # one-hot encode truth
    y_true2 = np.zeros((y_true.shape[0], n_classes))
    for column in range(y_true2.shape[1]):
        y_true2[:, column] = (y_true == column)
    y_true = y_true2

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresholds = dict()

    j_stats = [None]*n_classes
    opt_jstats = [None]*n_classes
    opt_thresholds = [None]*n_classes
    opt_tprs = [None]*n_classes
    opt_fprs = [None]*n_classes

    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        # Youden's J Statistic
        j_stats[i] = tpr[i] - fpr[i]
        max_j_idx = np.argmax(j_stats[i])
        opt_thresholds[i] = thresholds[i][max_j_idx]
        opt_jstats[i] = j_stats[i][max_j_idx]
        opt_tprs[i] = tpr[i][max_j_idx]
        opt_fprs[i] = fpr[i][max_j_idx]

    # # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_score.ravel())
    # roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    fig, ax = plt.subplots()
    lw = 2
    ax.plot(fpr[1], tpr[1], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('False Positive Rate: 1 - Specificity')
    ax.set_ylabel('True Positive Rate: Sensitivity')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(results_dir, network + '_roc_' + str(weightfile) + '.png'))

    auc_score = metrics.roc_auc_score(y_true[:, 1], y_score[:, 1])
    # print('auc_score: ', auc_score)
    return auc_score, opt_jstats, opt_thresholds, opt_tprs, opt_fprs
