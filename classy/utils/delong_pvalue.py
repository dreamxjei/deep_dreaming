import os, sys
import scipy

import numpy as np
import pandas as pd

sys.path.append('..')
import compare_auc_delong_xu


def main():
    rdir = os.path.join('..', 'results')
    raw_m1 = pd.read_csv(os.path.join(rdir, 'resnet18_all_inclusive_stats_000.csv'))
    raw_m2 = pd.read_csv(os.path.join(rdir, 'resnet18_all_inclusive_stats_005.csv'))

    gt = raw_m1['label']
    ppred_m1 = raw_m1['raw_output_1']
    ppred_m2 = raw_m2['raw_output_1']

    pvalue = compare_auc_delong_xu.delong_roc_test(gt, ppred_m1, ppred_m2)
    print('pvalue is:', pvalue)


if __name__ == '__main__':
    main()
