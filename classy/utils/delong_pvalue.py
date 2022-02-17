import os, sys
import scipy

import numpy as np
import pandas as pd

sys.path.append('..')
import compare_auc_delong_xu


def main():
    rdir = os.path.join('internet_delong')
    raw_m1 = pd.read_csv(os.path.join(rdir, 'shoulder_internal.csv'))
    raw_m2 = pd.read_csv(os.path.join(rdir, 'shoulder_external.csv'))

    gt = raw_m1['label'].tolist()
    ppred_m1 = raw_m1['softmax_prob_1'].tolist()
    ppred_m2 = raw_m2['softmax_prob_1'].tolist()

    gt = np.array(gt)
    ppred_m1 = np.array(ppred_m1)
    ppred_m2 = np.array(ppred_m2)

    print(gt)
    print(ppred_m1)
    print(ppred_m2)

    log_pvalue = compare_auc_delong_xu.delong_roc_test(gt, ppred_m1, ppred_m2)
    pvalue = 10**log_pvalue
    print('pvalue is:', pvalue)


if __name__ == '__main__':
    main()
