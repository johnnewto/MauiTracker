__all__ = ['summary', 'show_image', 'imshow', 'show', 'train_model', 'visualize_model', 'Predict', 'loadModel',
           'images2batch', 'files2batch', 'TilesDataset', 'calc_errors', 'confidence', 'visualize_predict_batch', 'visualize_batch']

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from torchsummary import summary
import torchvision
import utils.show_images as siu
from torch.utils.data import Dataset
import csv, os
import cv2

def opencsv( fileName):
    csv_labels = []
    print(f'Opening {fileName}')
    with open(fileName, mode='r') as infile:
        reader = csv.reader(infile)
        labels = next(reader)[1:]
        print(labels)

        # JN code
        for row in reader:
            try:
                filename = row[0]
                image_n = row[1]
                center_pos = [int(v) for v in row[2:4]]
                labels = [v for v in row[4:]]
                # one_hot = [int(v) for v in row[4:]]
                # label = one_hot_to_label(one_hot, labels)
                itm = [filename, image_n, center_pos, labels]
                if len(itm[3])>0:
                    csv_labels.append(itm)
                # print(itm)
            except Exception as e:
                print(e)
    return csv_labels

class TilesDataset(Dataset):
    label_dict = {'Bird': 0, 'Cloud':1, 'Ground':2, 'Plane':3}
    def __init__(self, csv_file, root_dir, transform=None):
        self.csv = opencsv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [i[0]  for i in self.label_dict.items()]
        self.class_to_idx = self.label_dict

    def get_image_path(self, idx):
        img_name = self.csv[idx][0]
        name = os.path.splitext(img_name)[0]
        path = f'{self.root_dir}{name}-{self.csv[idx][1]}.jpg'
        return path

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.get_image_path(idx)
        image = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
        try:
            label_name = self.csv[idx][3][0]
            label = self.class_to_idx[label_name]
            filename = img_name
        except IndexError:
            print('IndexError')
        # sample = {'image': image}  #, 'landmarks': landmarks}
        if self.transform:
            image = self.transform(image)

        return image, label, filename


def loadModel(device='cpu'):
    """ load a pretrained resnet18 trained model"""
    print('Loading resnet18')
    # model_new = torchvision.models.resnet18(pretrained=True, progress=False).to('cuda')
    model_new = torchvision.models.resnet18(pretrained=True, progress=False).to(device)
    print('Loaded resnet18')

    model_new.eval()
    return model_new

# def loadCustomModel(path='/home/jn/data/large_plane/model.pth', num_classes=4, device='cpu'):
def loadCustomModel(path='data/model.pth', num_classes=4, device='cpu'):
    "/home/jn/data/large_plane/model.pth"
    """ load a custom pretrained resnet18 trained model"""
    model_new = torchvision.models.resnet18(pretrained=False).to(device)
    num_ftrs = model_new.fc.in_features
    model_new.fc = nn.Linear(num_ftrs, num_classes, bias=True)
    model_new.load_state_dict(torch.load(path))
    model_new.eval()
    return model_new


    # model_new = models.resnet18(pretrained=False).to('cuda')
    # num_ftrs = model_new.fc.in_features
    # model_new.fc = nn.Linear(num_ftrs, 4, bias=True)
    #
    # path = '/home/jn/data/large_plane/model.pth'
    # model_new.load_state_dict(torch.load(path))
    # model_new.eval()


import json


def imagenet_classification(predicted_idx):
    """ Return imagenet classification """
    try:
        return imagenet_classification.imagenet_class_index[predicted_idx]
    except AttributeError:
        imagenet_classification.imagenet_class_index = json.load(open('data/imagenet_class_index.json'))
        return imagenet_classification.imagenet_class_index[predicted_idx]

from torchvision import  transforms


class Predict(nn.Module):
    """ Predict classification from model """
    def __init__(self, model):
        super().__init__()
        # self.resnet18 = resnet18(pretrained=True, progress=False).eval()
        self.resnet18 = model.eval()
        self.transforms = nn.Sequential(
            # torchvision.transforms.Resize([36]),  # We use single int value inside a list due to torchscript type restrictions
            # torchvision.transforms.CenterCrop(32),
            torchvision.transforms.ConvertImageDtype(torch.float),
            # torchvision.transforms.ToPILImage(),
            # transforms.Resize(40),
            # transforms.Resize(256),
            # transforms.CenterCrop(40),
            # transforms.CenterCrop(224),
            # torchvision.transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )
        self.imagenet_class_index = json.load(open('data/imagenet_class_index.json'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)   # todo only if not already done
            y_pred = self.resnet18(x)
            # sm = nn.Softmax(dim = 1)
            return y_pred

    def conf(self, y_pred):
        sm = nn.Softmax(dim=1)
        y_pred = sm(y_pred)
        y_hat = y_pred.argmax(dim=1)
        y_max = y_pred.amax(dim=1)
        return y_hat, y_max


def confidence(y_pred):
    """ return confidence of prediction """
    sm = nn.Softmax(dim=1)
    y_pred = sm(y_pred)
    cat = y_pred.argmax(dim=1)
    conf = y_pred.amax(dim=1)
    return cat, conf


def files2batch(images, resize=None):
    """ Convert a list of image paths to  tensor """
    if resize is not None:
        resize_trans = torch.nn.Sequential(
            torchvision.transforms.Resize((resize,resize)),
        )
    images = [torchvision.io.read_image(str(d)) for d in images]
    if resize is not None:
        images = [resize_trans(d) for d in images]
    batch = torch.stack(images)
    return batch

def images2batch(images, crop=40):
    """ Convert a list of numpy images to  tensor """
    if crop is not None:
        crop_trans = torch.nn.Sequential(
            # torchvision.transforms.Resize((resize,resize)),
            torchvision.transforms.CenterCrop(40)
        )
    images = [np.transpose(d,(2,0,1)) for d in images]
    images = [torch.from_numpy(d) for d in images]
    if crop is not None:
        images = [crop_trans(d) for d in images]
    batch = torch.stack(images)
    return batch

def show(imgs):
    """ show non normalised tensor """
    assert 'ByteTensor' in imgs.type() , "Type must be torch.uint8"
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = torchvision.transforms.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def imshow(inp, title=None):
    """Imshow for Tensor."""
    assert 'FloatTensor' in inp.type() , "Type must be torch.float32"
    inp = inp.to('cpu').numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def show_image(img, ax: plt.Axes = None, figsize: tuple = (3, 3), hide_axis: bool = True, cmap: str = 'binary',
               alpha: float = None, title: str = None, **kwargs) -> plt.Axes:
    "Display `Image` in notebook."
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    xtr = dict(cmap=cmap, alpha=alpha, **kwargs)
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    ax.imshow(img, **xtr)
    if hide_axis: ax.axis('off')
    if title is not None:
        ax.set_title(title)
    return ax


def train_model(model, dataloaders, device, criterion, optimizer, scheduler, num_epochs=25):
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}', end=' ')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # for inputs, labels, _ in dataloaders[phase]:   todo no sure why num returned changed from 3 to 2 in  Jan 2022
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} || ', end=' ')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60 :.0f}s')
    print(f'Best val Acc: {best_acc :4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def calc_errors(model, dataloaders, dataset='val', _device='cpu'):
    dataset_size = len(dataloaders[dataset].dataset)

    was_training = model.training
    is_cuda = next(model.parameters()).is_cuda
    predictor = Predict(model).to(_device)
    print(f"Model set to {'cuda' if next(model.parameters()).is_cuda else 'cpu'}")
    model.eval()
    image_cnt = -1
    num_errors = 0
    running_corrects = 0
    # running_corrects1 = 0

    with torch.no_grad():
        for i, (inputs, labels, filenames) in enumerate(dataloaders[dataset]):
            inputs = inputs.to(_device)
            labels = labels.to(_device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # _pred = predictor(inputs)
            # _, preds1 = torch.max(_pred, 1)
            cat, conf = predictor.conf(outputs)
            running_corrects += torch.sum(preds == labels.data)
            # running_corrects1 += torch.sum(preds1 == labels.data)
            for j in range(inputs.size()[0]):
                if cat[j] != labels[j]:
                    filename = filenames[j]
                    print(f'Prediction: {cat[j]}, Label: {labels[j]}  {filename}')

        model.train(mode=was_training)
        model.to('cuda' if is_cuda else 'cpu')
        epoch_acc = running_corrects / dataset_size
        print(f"Accuracy = {running_corrects / dataset_size}")

def visualize_model(model, dataloaders, dataset='val', errors_only=False, cols=4, _device='cpu', num_images=6):
    try:
        class_names = dataloaders['train'].dataset.classes
    except AttributeError:
        class_names = dataloaders['train'].dataset.dataset.classes

    was_training = model.training
    is_cuda = next(model.parameters()).is_cuda
    # model.to(_device)
    # predictor = Predict(model).to(_device)
    print(f"Model set to {'cuda' if next(model.parameters()).is_cuda else 'cpu'}")
    model.eval()
    image_cnt = -1
    num_errors = 0

    with torch.no_grad():
        rows = (num_images + cols-1) // cols
        cols = num_images if rows == 1 else cols
        fig, axs = plt.subplots(rows, cols, figsize=(12, 12))
        axs = axs.flatten()
        exit_loops = False

        # for i, (inputs, labels, filenames) in enumerate(dataloaders[dataset]):   todo not sure why changes jan 2022
        for i, (inputs, labels) in enumerate(dataloaders[dataset]):
            inputs = inputs.to(_device)
            labels = labels.to(_device)

            start = time.perf_counter()
            outputs = model(inputs)
            # _, preds = torch.max(outputs, 1)
            # _pred = predictor(inputs)
            # _, preds = torch.max(_pred, 1)
            cat, conf = confidence(outputs)
            elapsed = time.perf_counter() - start
            # print(f" Inference time on {_device} is {elapsed:.6f} secs for {inputs.shape} images")

            for j in range(inputs.size()[0]):
                # if errors_only:
                #     if cat[j] == labels[j] :
                #         continue
                #     else:
                #         filename = filenames[j]
                #         print(f'Prediction: {cat[j]}, Label: {labels[j]}  {filename}')
                #         num_errors += 1

                image_cnt += 1
                if errors_only:
                    title = (f'{class_names[labels[j]]}')
                else:
                    title = (f'{class_names[cat[j]]}')
                # title = (f'{class_names[preds[j]]}')
                show_image(inputs.cpu().data[j], ax=axs[image_cnt], title=title)

                if image_cnt == num_images - 1:
                    exit_loops = True
                    for ax in axs:
                        ax.axis('off')
                    break

            if exit_loops == True: break

        model.train(mode=was_training)
        model.to('cuda' if is_cuda else 'cpu')
        print(f"Model set to {'cuda' if next(model.parameters()).is_cuda else 'cpu'}")

def visualize_predict_batch(model, batch, _device='cpu', num_images=6, cols=4):
    image_cnt = -1
    with torch.no_grad():
        rows = (num_images + cols-1) // cols
        cols = num_images // rows if rows == 1 else cols
        fig, axs = plt.subplots(rows, cols, figsize=(12, 8))
        axs = axs.flatten()

        predictor = Predict(model).to(_device)
        y_pred = predictor(batch)
        cat, conf = predictor.conf(y_pred)
        for j in range(list(cat.shape)[0]):
            image_cnt += 1
            title = (f'{cat[j].item()} : {conf[j].item():.2f}')
            img = batch.cpu().data[j].numpy().transpose((1, 2, 0))
            siu.show_img(img, ax=axs[image_cnt], title=title, mode='RGB')
            # show_image(batch.cpu().data[j], ax=axs[image_cnt], title=title)

            if image_cnt == num_images - 1:
                exit_loops = True
                for ax in axs:
                    ax.axis('off')
                break

def visualize_batch(batch, class_names, _device='cpu', num_images=6, cols=4):

    image_cnt = -1
    with torch.no_grad():
        rows = (num_images + cols-1) // cols
        cols = num_images // rows if rows == 1 else cols
        fig, axs = plt.subplots(rows, cols, figsize=(12, 8))
        axs = axs.flatten()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        for j in range(list(batch.shape)[0]):
            image_cnt += 1
            # title = (f'{cat[j].item()} : {conf[j].item():.2f}')
            title = (f'{class_names[j]}')
            img = batch.cpu().data[j].numpy().transpose((1, 2, 0))
            img = std * img + mean
            img = np.clip(img, 0, 1)
            siu.show_img(img, ax=axs[image_cnt], title=title, mode='RGB')
            # show_image(batch.cpu().data[j], ax=axs[image_cnt], title=title)

            if image_cnt == num_images - 1:
                exit_loops = True
                for ax in axs:
                    ax.axis('off')
                break
