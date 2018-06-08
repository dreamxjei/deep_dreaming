# model track by mxj

# a python file to hold each individual model's code

import torch.nn as nn
from torchvision import models


def ResNet18_pretrained(n_classes, freeze=True):
    model = models.__dict__['resnet18'](pretrained=True)
    # if weights are not frozen, train
    for param in model.parameters():
        if freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # change last layer to output n_classes
    num_filters = model.fc.in_features
    model.fc = nn.Linear(num_filters, n_classes)
    return model


def inception_v3_pretrained(n_classes, freeze=True):
    model = models.__dict__['inception_v3'](pretrained=True)
    # if weights are not frozen, train
    for param in model.parameters():
        if freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # change last layer to output n_classes
    num_filters = model.fc.in_features
    model.fc = nn.Linear(num_filters, n_classes)
    return model


def AlexNet_pretrained(n_classes, freeze=True):
    model = models.__dict__['alexnet'](pretrained=True)
    # if weights are not frozen, train
    for param in model.parameters():
        if freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # change last layer to output n_classes
    lastclasslayer = str(len(model.classifier.modules) - 1)
    num_filters = model.classifier.modules[lastclasslayer].in_features
    model.classifier.modules[lastclasslayer] = nn.Linear(num_filters, n_classes)
    return model


def SqueezeNet_pretrained(n_classes, freeze=True):
    model = models.__dict__['squeezenet1_1'](pretrained=True)
    # if weights are not frozen, train
    for param in model.parameters():
        if freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # change last layer to output n_classes
    model.classifier.modules['1'] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = n_classes
    return model


def VGGNet_pretrained(n_classes, freeze=True):
    model = models.__dict__['vgg16'](pretrained=True)
    # if weights are not frozen, train
    for param in model.parameters():
        if freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # change last layer to output n_classes
    lastclasslayer = str(len(model.classifier.modules) - 1)
    num_filters = model.classifier.modules[lastclasslayer].in_features
    model.classifier.modules[lastclasslayer] = nn.Linear(num_filters, n_classes)
    return model


def DenseNet_pretrained(n_classes, freeze=True):
    model = models.__dict__['densenet161'](pretrained=True)
    # if weights are not frozen, train
    for param in model.parameters():
        if freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # change last layer to output n_classes
    num_filters = model.classifier.in_features
    model.classifier = nn.Linear(num_filters, n_classes)
    return model


if __name__ == '__main__':
    model = ResNet18_pretrained(2)
    print('Main called')
