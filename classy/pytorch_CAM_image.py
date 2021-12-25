# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2

import torch
import os, sys
import models  # custom

import argparse


# input images
############ dataloader ############
result_classes = {
    0: 'no',
    1: 'yes',
}
n_classes = len(result_classes)

###  output directories
output_dir = os.path.join('cam', results)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

###  choose model
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--network',
    choices=['resnet18', 'resnet50', 'resnet152', 'resnext101', 
             'wide_resnet101', 'inception_v3', 'alexnet',
             'squeezenet', 'vggnet', 'densenet',
             'efficientnet_b7', 'regnet_x_32gf'], default='resnet18',
     help='Choose which neural network to use')
args = parser.parse_args()
network = args.network

if network == 'resnet18':
    model = models.ResNet18_pretrained(n_classes, freeze=False)
    finalconv_name = 'layer4'
elif network == 'resnet50':
    model = models.ResNet50_pretrained(n_classes, freeze=False) 
    finalconv_name = 'layer4'
elif network == 'resnet152':
    model = models.ResNet152_pretrained(n_classes, freeze=False)
    finalconv_name = 'layer4'
elif network == 'resnext101':
    model = models.ResNeXt101_pretrained(n_classes, freeze=False)
    finalconv_name = 'layer4'
elif network == 'wide_resnet101':
    model = models.wide_resnet101_pretrained(n_classes, freeze=False)
    finalconv_name = 'layer4'
elif network == 'inception_v3':
    model = models.inception_v3_pretrained(n_classes, freeze=False)
    finalconv_name = 'Mixed_7c'
elif network == 'alexnet':
    model = models.AlexNet_pretrained(n_classes, freeze=False)
    finalconv_name = 'features'
elif network == 'squeezenet':
    model = models.SqueezeNet_pretrained(n_classes, freeze=False)
    finalconv_name = 'features'
elif network == 'vggnet':
    model = models.VGGNet_pretrained(n_classes, freeze=False)
    finalconv_name = 'features'
elif network == 'densenet':
    model = models.DenseNet_pretrained(n_classes, freeze=False)
    finalconv_name = 'denseblock4'
elif network == 'efficientnet_b7':
    model = models.efficientnet_b7_pretrained(n_classes, freeze=False)
    finalconv_name = '3'
elif network == 'regnet_x_32gf':
    model = models.regnet_x_32gf_pretrained(n_classes, freeze=False)
    finalconv_name = 'block4'

# load weight and network
netweight_dir = network + 'weights'
weightslist = os.listdir(os.path.join('weights', netweight_dir))
weightsnum = len(weightslist) - 1
if weightslist[weightsnum].startswith('LOG'):  # avoid LOG.txt
    weightsnum = weightsnum - 1
load_file = os.path.join('weights', netweight_dir, weightslist[weightsnum]
net.load_state_dict(torch.load(os.path.join('./', load_file)))
net.eval()

### preprocessing
preprocess = transforms.Compose([
  # transforms.Resize((256, 256)),
  transforms.Resize((224, 224)),
  # transforms.CenterCrop(224),
  transforms.ToTensor(),
])


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


# ################ SINGLE IMAGE
#
# #image_url = os.path.join(dataset_dir, directories['train_1'])
# #samples = os.listdir(image_url)
#
# # hook the feature extractor
# features_blobs = []
# def hook_feature(module, input, output):
#     features_blobs.append(output.data.cpu().numpy())
#
# net._modules.get(finalconv_name).register_forward_hook(hook_feature)
#
# # get the softmax weight
# params = list(net.parameters())
# weight_softmax = np.squeeze(params[-2].data.numpy())
#
# img_name = os.path.splitext(image_file)[0]
#
# #image_file = os.path.join(image_url, sample)
# image_url = os.path.join('cam/', image_file)
# img_pil = Image.open(image_url).convert('RGB')
# img_pil.save(output_dir + '/' + img_name + '.jpg')
#
# img_tensor = preprocess(img_pil)
# img_variable = Variable(img_tensor.unsqueeze(0))
# logit = net(img_variable)
#
# # imagenet category list
# classes = {int(key):value for (key, value) in result_classes.items()}
#
# h_x = F.softmax(logit).data.squeeze()
# probs, idx = h_x.sort(0, True)
#
# # output the prediction
# #for i in range(0, 2):
# #    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
#
# # generate class activation mapping for the top1 prediction
# CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
#
# # render the CAM and output
# #print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
# img = cv2.imread(output_dir + '/' + img_name + '.jpg')
# height, width, _ = img.shape
# heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
# result = heatmap * 0.3 + img * 0.5
# cv2.imwrite(output_dir + '/' + img_name + '_CAM' + '.jpg', result)
# ################ SINGLE IMAGE


################ REPETITION
# image_url = os.path.join(dataset_dir, directories['train_1'])
image_url = 'cam'
samples = os.listdir(image_url)
for sample in samples:

    if sample.startswith('.'): # avoid .DS_Store
        continue
    
    print('processing',sample)
    
    # hook the feature extractor
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    net._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    img_name = os.path.splitext(sample)[0]

    image_file = os.path.join(image_url, sample)
    img_pil = Image.open(image_file).convert('RGB')
    img_pil.save(os.path.join(output_dir, img_name + '.jpg'))

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = net(img_variable)

    # imagenet category list
    classes = {int(key):value for (key, value) in result_classes.items()}

    h_x = F.softmax(logit).data.squeeze()
    probs, idx = h_x.sort(0, True)

    # # output the prediction
    for i in range(0, 2):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i].item()]))

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # render the CAM and output
    # print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
    img = cv2.imread(os.path.join(output_dir, img_name + '.jpg'))
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite(os.path.join(output_dir, img_name + '_CAM' + '.jpg'), result)
################ REPETITION






