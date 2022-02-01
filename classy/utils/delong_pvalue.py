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

    gt = raw_m1['label'].tolist()
    ppred_m1_pdseries = raw_m1['softmax_prob']
    ppred_m1_rawstr = ppred_m1_pdseries.tolist()
    ppred_m1 = []
    for i in range(len(ppred_m1_rawstr)):
        ppred_m1_str_to_list = ppred_m1_rawstr[i].strip('][').split()
        ppred_m1_list_to_float = list(map(float, ppred_m1_str_to_list))
        ppred_m1.append(ppred_m1_list_to_float)

    ppred_m2_pdseries = raw_m2['softmax_prob']
    ppred_m2_rawstr = ppred_m2_pdseries.tolist()
    ppred_m2 = []
    for i in range(len(ppred_m2_rawstr)):
        ppred_m2_str_to_list = ppred_m2_rawstr[i].strip('][').split()
        ppred_m2_list_to_float = list(map(float, ppred_m2_str_to_list))
        ppred_m2.append(ppred_m2_list_to_float)

    gt = np.array(gt)
    ppred_m1 = np.array(ppred_m1)
    ppred_m2 = np.array(ppred_m2)

    ppred_m1_class1 = np.array(ppred_m1[:, 1])
    ppred_m2_class2 = np.array(ppred_m2[:, 1])

    # print(gt)
    # print(ppred_m1_class1)
    # print(ppred_m2_class2)

    log_pvalue = compare_auc_delong_xu.delong_roc_test(gt, ppred_m1_class1, ppred_m2_class2)
    pvalue = 10**log_pvalue
    print('pvalue is:', pvalue)


if __name__ == '__main__':
    main()
