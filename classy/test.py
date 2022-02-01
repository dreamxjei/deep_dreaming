# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import matplotlib
matplotlib.use('Agg') # Added for display errors
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms, models
import torch.optim.lr_scheduler as sch
import torch.nn.functional as F

import os, sys
from skimage import io

from PIL import Image
import time

from models import ResNet18_pretrained, ResNet50_pretrained, ResNet152_pretrained, ResNeXt101_pretrained, wide_resnet101_pretrained, inception_v3_pretrained, AlexNet_pretrained, SqueezeNet_pretrained, VGGNet_pretrained, DenseNet_pretrained, efficientnet_b7_pretrained, regnet_x_32gf_pretrained
from dataset import read_dataset
from test_statistics import roc_auc_metrics
import compare_auc_delong_xu

import argparse
from sklearn import metrics

import pandas as pd
import seaborn as sns
import scipy.stats as st


# name classes - for confusion matrix etc, can be different from dataloader classes
result_classes = {
        0: 'no_THA',
        1: 'yes_THA',
        # 2: 'yes_HRA'
    }

thresh_methods = ['youden', 'exclusive']
thresh_method = thresh_methods[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network',
        choices=['resnet18', 'resnet50', 'resnet152', 'resnext101', 'wide_resnet101', 'inception_v3', 'alexnet', 'squeezenet', 'vggnet', 'densenet', 'efficientnet_b7', 'regnet_x_32gf'], default='resnet18',
        help='Choose which neural network to use')
    args = parser.parse_args()
    network = args.network

    n_classes = len(result_classes)

    # model
    if args.network == 'resnet18':
        model = ResNet18_pretrained(n_classes, freeze=False)
        print('model is resnet18')
    elif args.network == 'resnet50':
        model = ResNet50_pretrained(n_classes, freeze=False)
        print('model is resnet50')
    elif args.network == 'resnet152':
        model = ResNet152_pretrained(n_classes, freeze=False)
        print('model is resnet152')
    elif args.network == 'resnext101':
        model = ResNeXt101_pretrained(n_classes, freeze=False)
        print('model is resnext101')
    elif args.network == 'wide_resnet101':
        model = wide_resnet101_pretrained(n_classes, freeze=False)
        print('model is wide_resnet101')
    elif args.network == 'inception_v3':
        model = inception_v3_pretrained(n_classes, freeze=False)
        print('model is inception_v3')
    elif args.network == 'alexnet':
        model = AlexNet_pretrained(n_classes, freeze=False)
        print('model is alexnet')
    elif args.network == 'squeezenet':
        model = SqueezeNet_pretrained(n_classes, freeze=False)
        print('model is squeezenet')
    elif args.network == 'vggnet':
        model = VGGNet_pretrained(n_classes, freeze=False)
        print('model is vggnet')
    elif args.network == 'densenet':
        model = DenseNet_pretrained(n_classes, freeze=False)
        print('model is densenet')
    elif args.network == 'efficientnet_b7':
        model = efficientnet_b7_pretrained(n_classes, freeze=False)
        print('model is efficientnet_b7')
    elif args.network == 'regnet_x_32gf':
        model = regnet_x_32gf_pretrained(n_classes, freeze=False)
        print('model is regnet_x_32gf')

    ############ testing ############
    use_gpu = torch.cuda.is_available()
    if args.network == 'resnet18' or args.network == 'resnet50' or args.network == 'resnet152' or args.network == 'resnext101' or args.network == 'wide_resnet101' or args.network == 'alexnet' or args.network == 'squeezenet' or args.network == 'vggnet' or args.network == 'densenet' or args.network == 'efficientnet_b7' or args.network == 'regnet_x_32gf':
        weightslist = os.listdir('weights/' + network + '_weights/')
        weightslist.sort()
        weightsnum = len(weightslist)
        for weightfileidx in range(weightsnum):
            weightfile = weightslist[weightfileidx]
            if not weightfile.startswith('LOG'):  # avoid LOG.txt
                load_file = 'weights/' + network + '_weights/' + weightfile
                print('testing weight:',weightfile)
                val_data_transform = transforms.Compose([
                  transforms.ToPILImage(),
                  transforms.Resize((256, 256)),
                  transforms.CenterCrop(224),
                  transforms.ToTensor(),
                  # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                  #                      std=[0.229, 0.224, 0.225]),
                  # transforms.Normalize(mean=[0.4059296, 0.40955055, 0.412535],
                  #                      std=[0.21329397, 0.215493, 0.21677108]),
                ])
                # pass weightfile as filename, not index as above
                test(use_gpu, n_classes, load_file, val_data_transform, model, weightfile, network)
    elif args.network == 'inception_v3':
        weightslist = os.listdir('weights/inception_v3_weights/')
        weightslist.sort()
        weightsnum = len(weightslist)
        for weightfileidx in range(weightsnum):
            weightfile = weightslist[weightfileidx]
            if not weightfile.startswith('LOG'):  # avoid LOG.txt
                load_file = 'weights/inception_v3_weights/' + weightfile
                val_data_transform = transforms.Compose([
                  transforms.ToPILImage(),
                  transforms.Resize((300, 300)),
                  transforms.CenterCrop(299),
                  transforms.ToTensor(),
                  # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                  #                      std=[0.229, 0.224, 0.225]),
                  # transforms.Normalize(mean=[0.4059296, 0.40955055, 0.412535],
                  #                      std=[0.21329397, 0.215493, 0.21677108]),
                ])
                test(use_gpu, n_classes, load_file, val_data_transform, model, weightfile, network)


def test(use_gpu, n_classes, load_file, val_data_transform, model, weightfile, network):
    batch_size=10
    # truncate weightfile for result filenames
    wf_only = weightfile.split('_')[0]

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if use_gpu:
        print('Using ' + str(torch.cuda.device_count()) + ' GPU(s)')
        if torch.cuda.device_count() > 1:
            gpu_ids = list(range(torch.cuda.device_count()))
            model = nn.DataParallel(model, device_ids=gpu_ids).cuda()
        else:
            model = model.cuda()

    model.load_state_dict(torch.load(os.path.join('./', load_file)))
    radio_val = read_dataset(mode='test', transform=val_data_transform)
    radio_data_loader = DataLoader(radio_val, batch_size=batch_size, shuffle=True, num_workers=2)

    model.train(False)

    running_corrects = 0
    total = len(radio_val.sample_paths)
    print(total)

    def do_gpu(x):
        return x.cuda() if use_gpu else x

    TP = 0  # pred true, label true
    TN = 0  # pred false, label false
    FP = 0  # pred true, label false
    FN = 0  # pred false, label true

    y_true = []
    y_score = []
    total_labels = []  # for confusion matrix - ground truth
    total_preds = []  # for confusion matrix - preds
    wrong_filenames = []
    all_inclusive_stats = {}
    all_inclusive_stats['filename'] = []
    all_inclusive_stats['label'] = []
    all_inclusive_stats['class_pred'] = []
    for idx in range(len(result_classes)):
        all_inclusive_stats['raw_output_' + str(idx)] = []
    all_inclusive_stats['wrong'] = []
    all_inclusive_stats['softmax_prob'] = []
    
    for data in radio_data_loader:
        inputs, labels, filenames = data

        """
        # show first images of the batch
        plt.imshow(np.transpose(inputs.numpy()[0], (1,2,0)))
        plt.show()
        """

        original = inputs
        inputs = Variable(do_gpu(inputs)).float()
        labels = Variable(do_gpu(labels)).long()

        # forward
        outputs = model(inputs)

        local_y_score = F.softmax(outputs, 1)

        if use_gpu:
            # y_score.append(local_y_score.data.cpu().numpy())
            # y_true.append(labels.data.cpu().numpy())
            local_y_score = local_y_score.clone().detach().cpu().numpy()
            y_true.append(labels.clone().detach().cpu().numpy())
            labels_np = labels.clone().detach().cpu().numpy()
            raw_outputs_np = outputs.clone().detach().cpu().numpy()
            
        else:
            # y_score.append(local_y_score.data.numpy())
            # y_true.append(labels.data.numpy())
            local_y_score = local_y_score.clone().detach().numpy()
            y_true.append(labels.clone().detach().numpy())
            labels_np = labels.clone().detach().numpy()
            raw_outputs_np = outputs.clone().detach().numpy()

        y_score.append(local_y_score)
        total_labels.append(labels_np)

        if thresh_method == 'exclusive':
            # for mutually exclusive tasks only. obsolete?
            _, preds = torch.max(outputs.data, 1)
            total_preds.append(preds.clone().detach().cpu().numpy())
            running_corrects += torch.sum(preds == labels.data)
            preds = preds.float().cpu().numpy()
            labels = labels.data.float().cpu().numpy()

        if n_classes < 3 and thresh_method == 'youden':  # roc/auc for binary output
            # call statistics file for roc/auc
            auc_score, opt_jstats, opt_thresholds = roc_auc_metrics(y_true, y_score, n_classes, wf_only, network, results_dir)
            thresh = opt_thresholds[1]

            # classify based on whether or not class 1 (yes) beats threshold
            preds = local_y_score >= thresh  # one hot predictions
            preds = np.argmax(preds, axis=1)  # change one hot preds to class numbers
            total_preds.append(preds)
            if use_gpu:
                labels = labels.clone().detach().cpu().numpy()
            else:
                labels = labels.clone().detach().numpy()

            running_corrects += np.sum(preds == labels)

        # statistics
        # ROC curve analysis
        TP += np.sum(np.logical_and(preds == 1.0, labels == 1.0))
        TN += np.sum(np.logical_and(preds == 0.0, labels == 0.0))
        FP += np.sum(np.logical_and(preds == 1.0, labels == 0.0))
        FN += np.sum(np.logical_and(preds == 0.0, labels == 1.0))

        """
        # show incorrectly classified images
        for idx in range(len(original)):
          if (preds != labels.data)[idx]:
            plt.imshow(np.transpose(original.numpy()[idx], (1,2,0)))
            plt.show()
            # print('here', idx)
        """

        # append wrong files
        for idx in range(len(original)):  # first dimension of tensor, usually = batch_size
            if (preds != labels)[idx]:
                wrong_filenames.append(filenames[idx])

        # append raw pred, softmax pred, and true labels for delong calculation
        for idx in range(len(original)):  # first dimension of tensor, usually = batch_size
            all_inclusive_stats['filename'].append(filenames[idx])
            all_inclusive_stats['label'].append(labels[idx])
            all_inclusive_stats['class_pred'].append(preds[idx])
            for j in range(n_classes):
                all_inclusive_stats['raw_output_' + str(j)].append(raw_outputs_np[idx][j])
            if preds[idx] != labels[idx]:
                all_inclusive_stats['wrong'].append(1)
            else:
                all_inclusive_stats['wrong'].append(0)
            all_inclusive_stats['softmax_prob'].append(local_y_score[idx])

    # save wrong files to csv
    df_wrongs = pd.DataFrame(list(zip(wrong_filenames)),
                             columns=['wrong_filenames'])
    df_wrongs_output_path = os.path.join(results_dir, network + '_wrong_filenames_' + str(wf_only) + '.csv')
    df_wrongs.to_csv(df_wrongs_output_path, index=False)

    # save all inclusive stats to csv
    df_all_inclusive_stats = pd.DataFrame(
        list(zip(
        all_inclusive_stats['filename'],
        all_inclusive_stats['label'],
        all_inclusive_stats['class_pred'],
        all_inclusive_stats['wrong'],
        all_inclusive_stats['softmax_prob']
    )), columns=['filename', 'label', 'class_pred', 'wrong', 'softmax_prob'])
    for idx in range(n_classes):
        df_all_inclusive_stats['raw_output_' + str(idx)] = all_inclusive_stats['raw_output_' + str(idx)]
    df_all_inclusive_stats = df_all_inclusive_stats.sort_values(by=['filename'])

    df_all_inclusive_stats_output_path = os.path.join(results_dir, network + '_all_inclusive_stats_' + str(wf_only) + '.csv')
    df_all_inclusive_stats.to_csv(df_all_inclusive_stats_output_path, index=False)

    # print results in console/testing result file
    print('---------  correct: {:03d} -----------'.format(running_corrects))
    print('---------  total: {:03d} -----------'.format(total))
    print('---------  accuracy: {:.4f} -----------'.format(float(running_corrects)/total))

    output = open(os.path.join(results_dir, network + '_test_result_' + str(wf_only) + '.txt'), 'w')

    output.write('---------  correct: {:03d} -----------'.format(running_corrects) + "\n")
    output.write('---------  total: {:03d} -----------'.format(total) + "\n")
    output.write('---------  accuracy: {:.4f} -----------'.format(float(running_corrects) / total) + "\n")

    if n_classes < 3:  # roc/auc for binary output
        print('auc_score:', auc_score)
        output.write('auc_score: ' + str(auc_score) + '\n')
        print('Optimal Youden\'s J Statistic:', opt_jstats[1])
        output.write('optimal_youden\'s_jstat : ' + str(opt_jstats[1]) + '\n')
        print('Optimal threshold:', opt_thresholds[1])
        output.write('optimal_threshold :' + str(opt_thresholds[1]) + '\n')

        calc_stats(total_labels, total_preds, result_classes, n_classes, opt_thresholds, output, auc_score, all_inclusive_stats['raw_output_1'])
    
    sensitivity  = TP / (TP + FN)
    specificity  = TN / (TN + FP)
    pos_like_ratio = sensitivity / (1 - specificity)
    neg_like_ratio = (1 - sensitivity) / specificity
    pos_pred_val = TP / (TP + FP)
    neg_pred_val = TN / (TN + FN)

    print('sensitivity: %f\nspecificity: '
          '%f\npositive likelihood value: %f\nnegative likelihood value: '
          '%f\npositive predictive value: %f\nnegative predictive value: '
          '%f\nTP: %f\nTN: %f\nFP: %f\nFN: %f'
          % (sensitivity, specificity, pos_like_ratio, neg_like_ratio, pos_pred_val, neg_pred_val, TP, TN, FP, FN))

    output.write(
        'sensitivity: %f\nspecificity: '
        '%f\npositive likelihood value: %f\nnegative likelihood value: '
        '%f\npositive predictive value: %f\nnegative predictive value: '
        '%f\nTP: %f\nTN: %f\nFP: %f\nFN: %f'
        % (sensitivity, specificity, pos_like_ratio, neg_like_ratio, pos_pred_val, neg_pred_val, TP, TN, FP, FN))
    output.close()

    # confusion matrix: total_labels, total_preds
    print('confusion matrix:')
    confusion_labels = []
    confusion_preds = []
    for i in range(len(total_labels)):
        for j in range(np.shape(total_labels[i])[0]):
            confusion_labels.append(total_labels[i][j])
            confusion_preds.append(total_preds[i][j])

    conf_m = metrics.confusion_matrix(confusion_labels, confusion_preds)
    print(conf_m)

    # save conf_m as seaborn heatmap
    result_classes_names = list(result_classes.values())
    cm_df = pd.DataFrame(conf_m, result_classes_names, result_classes_names)
    plt.figure()
    sns.heatmap(cm_df, annot=True, square=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    cm_savepath = os.path.join(results_dir, network + '_conf_mat_' + str(wf_only) + '.png')
    plt.savefig(cm_savepath)


'''
def calc_threshold(y_true, y_score):
    # Calculate ideal threshold, that gives best f-beta - now use youden's j stat instead
    pre, rec, thresholds = metrics.precision_recall_curve(y_true, y_score)
    
    # ignore 0 p and r records
    f1th = [((2 * (p * r) / (p + r)), t) for p, r, t in zip(pre, rec, thresholds) if r and p]
    
    f1, thresholds = zip(*f1th)
    fm = np.argmax(f1)
    f1, threshold = f1[fm], thresholds[fm]
    return threshold
'''


def calc_stats(labels, preds, result_classes, n_classes, opt_thresholds, output, auc_score, raw_output_1):
    labels = np.concatenate(labels)
    preds = np.concatenate(preds)
    raw_output_1 = np.array(raw_output_1)  # for delong
    # thresh = calc_threshold(y_true, y_score)  # for f1
    thresh = opt_thresholds[1]
    # thresh = 0.5
    accuracy = metrics.accuracy_score(labels, preds)
    precision, recall, f_beta, _ = metrics.precision_recall_fscore_support(labels, preds, average='binary')
    print('Accuracy: {:>2.3f}, Precision: {:>2.2f}, Recall: {:>2.2f}, f-1: {:>2.3f}'.format(accuracy, precision, recall, f_beta))
    output.write('Accuracy: {:>2.3f}, Precision: {:>2.2f}, Recall: {:>2.2f}, f-1: {:>2.3f}'.format(accuracy, precision, recall, f_beta) + '\n')

    
    # confidence intervals (z= {1.64: 90%, 1.96: 95%, 2.33: 98%, 2.58: 99%})
    ci_range = 0.95
    z_choices = {0.9: 1.64, 0.95: 1.96, 0.98: 2.33, 0.99: 2.58}
    z = z_choices[ci_range]

    '''
    # initial method of CI - might be wrong
    n = len(labels)

    # CI radius calculation for accuracy: for error, just replace accuracy with error
    interval = z * np.sqrt( (accuracy * (1 - accuracy)) / n)
    print(str(int(ci_range*100)) + r'% Confidence Interval: ', str(accuracy - interval), 'to', str(accuracy + interval))
    # output.write(str(int(ci_range*100)) + r'% Confidence Interval: ' + str(accuracy - interval) + ' to ' + str(accuracy + interval) + '\n')
    '''

    # confidence intervals
    positive = 1
    auc = auc_score
    n1 = sum(labels == positive)
    n2 = sum(labels != positive)
    q1 = auc / (2 - auc)
    q2 = 2*auc**2 / (1 + auc)
    se_auc = np.sqrt((auc*(1-auc) + (n1 - 1)*(q1 - auc**2) + (n2 - 1)*(q2 - auc**2)) / (n1*n2))

    lower = auc - z*se_auc
    upper = auc + z*se_auc

    if lower < 0:
        lower = 0

    if upper > 1:
        upper = 1

    print(str(int(ci_range*100)) + r'% Confidence Interval: ', lower, 'to', upper)
    output.write(str(int(ci_range*100)) + r'% Confidence Interval: ' + str(lower) + ' to ' + str(upper) + '\n')

    # delong auc and covariance
    auc_delong, variance_delong = compare_auc_delong_xu.delong_roc_variance(
        labels, raw_output_1
    )

    print('DeLong AUC:', auc_delong)
    output.write('DeLong AUC: ' + str(auc_delong) + '\n')

    print('DeLong variance (class 1):', variance_delong)
    output.write('DeLong variance (class 1): ' + str(variance_delong) + '\n')


if __name__ == '__main__':
    main()
