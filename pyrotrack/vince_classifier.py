import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms, models
import os
import time
import copy

data_dir = './data'
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
}
datasets = {phase : datasets.ImageFolder(root=os.path.join(data_dir, phase),
                                         transform=data_transforms[phase])
            for phase in ['train', 'val', 'test']}
dataset_loaders = {phase : torch.utils.data.DataLoader(dataset=datasets[phase], batch_size=10,
                                                       shuffle=True, num_workers=2)
                   for phase in ['train', 'val', 'test']}
dataset_sizes = {phase : len(datasets[phase]) for phase in ['train', 'val', 'test']}
dataset_classes = datasets['train'].classes
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# define train/test methods
def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-'*20)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()               # set model to training mode
            else:
                model.eval()                # set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for samples, labels in dataset_loaders[phase]:
                samples = samples.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(samples)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in train phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * samples.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # end of inner most for loop
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                  phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        # end of 2nd inner most for loop
    # end of outer most for loop

    total_time = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
          total_time // 60, total_time % 60))
    print('Best validation accuracy: {:.4f}'.format(best_acc))
    print('returning from the train_model method')
    return model


# load a pretrained model, reset final fully connected layer
# fine tuning entire model
model_ft = models.resnet152(pretrained=True)
num_features = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_features, 2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

model_ft.to(device)

# train and save the best weight
model_ft = train_model(model_ft, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=30)

'''
# convnet as fixed feature extractor
# only fine tuning the last fc layer
model_conv = torchvision.models.resnet152(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# parameters of newly constructed modules have requires_grad=True by default
num_features = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_features, 2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_conv.fc.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

model_conv.to(device)

model_conv = train_model(model_conv, criterion, optimizer,
                         exp_lr_scheduler, num_epochs=50)
'''
# testing
running_corrects = 0
with torch.no_grad():
    for samples, labels in dataset_loaders['test']:
        samples = samples.to(device)
        labels = labels.to(device)

        outputs = model_ft(samples)
        _, preds = torch.max(outputs.data, 1)
        print('outputs.data:', outputs.data)
        print('preds:', preds)
        print('labels.data:', labels.data)
        running_corrects += torch.sum(preds == labels.data)
    pred_acc = running_corrects.double() / dataset_sizes['test']
    print('prediction acc:', pred_acc)
