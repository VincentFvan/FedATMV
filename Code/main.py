# atmv.py

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
import random
import torch.nn.functional as F
import torch.nn.functional as func
import collections
from sklearn.model_selection import train_test_split
from collections import Counter
from utils.language_utils import word_to_indices, letter_to_vec
from utils.ShakeSpeare_reduce import ShakeSpeare
import math
import os

# Import models and configuration
from models.lstm import *
from models.vgg import *
import config

# Solve OpenMP runtime error
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Print environment info
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())


# A simple CNN model for CIFAR datasets
class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, config.NUM_CLASSES)

    def forward(self, x, start_layer_idx=0, logit=False):
        if start_layer_idx < 0:
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        result = {'activation' : x}
        x = x.view(-1, 16 * 5 * 5)
        result['hint'] = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        result['representation'] = x
        x = self.fc3(x)
        result['output'] = x
        return result

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = z_input
        z = self.fc3(z)
        result = {'output': z}
        if logit:
            result['logit'] = z
        return result

def cnncifar():
    return CNNCifar()

# --- ResNet Implementation for CIFAR ---
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNetCifar10(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetCifar10, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, config.NUM_CLASSES)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        result = {}
        x = self.layer1(x)
        result['activation1'] = x
        x = self.layer2(x)
        result['activation2'] = x
        x = self.layer3(x)
        result['activation3'] = x
        x = self.layer4(x)
        result['activation4'] = x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        result['representation'] = x
        x = self.fc(x)
        result['output'] = x
        return result

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = z_input
        z = self.fc(z)
        result = {'output': z}
        if logit:
            result['logit'] = z
        return result

    def forward(self, x, start_layer_idx=0, logit=False):
        if start_layer_idx < 0:
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
        return self._forward_impl(x)

def ResNet18_cifar10(**kwargs):
    return ResNetCifar10(BasicBlock, [2, 2, 2, 2], **kwargs)

def ResNet50_cifar10(**kwargs):
    return ResNetCifar10(Bottleneck, [3, 4, 6, 3], **kwargs)


def test_inference(net_glob, dataset_test):
    """
    Evaluates the global model on the test dataset.
    """
    acc_test, loss_test = test_img(net_glob, dataset_test)
    return acc_test.item()

def test_img(net_g, datatest):
    """
    Helper function for testing the model.
    """
    net_g.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=config.TEST_BC_SIZE)
    l = len(data_loader)
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            log_probs = net_g(data)['output']
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if config.VERBOSE:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

# --- Data Loading and Partitioning ---
def sparse2coarse(targets):
    """
    Convert Pytorch CIFAR-100 sparse targets to coarse targets.
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]

def CIFAR100():
    """
    Load and preprocess the CIFAR-100 dataset.
    """
    trans_cifar100 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR100(root='./data/CIFAR-100', train=True, transform=trans_cifar100, download=True)
    test_dataset = torchvision.datasets.CIFAR100(root='./data/CIFAR-100', train=False, transform=trans_cifar100, download=True)

    total_img,total_label = [],[]
    for imgs,labels in train_dataset:
        total_img.append(imgs.numpy())
        total_label.append(labels)
    total_img = np.array(total_img)
    total_label = np.array(sparse2coarse(total_label))
    cifar = [total_img, total_label]
    return cifar, test_dataset

def get_prob(non_iid_alpha, client_num, class_num=20, iid_mode=False):
    """
    Generate client data distribution probabilities using a Dirichlet distribution.
    """
    if iid_mode:
        return np.ones((client_num, class_num)) / class_num
    else:
        return np.random.dirichlet(np.repeat(non_iid_alpha, class_num), size=client_num)

def create_data_all_train(prob, size_per_client, dataset, N=20):
    """
    Distribute the dataset among clients according to the probability matrix.
    This version allocates all data for training.
    """
    total_each_class = size_per_client * np.sum(prob, 0)
    data, label = dataset


    all_class_set = []
    for i in range(N):
        size = total_each_class[i]
        sub_data = data[label == i]
        sub_label = label[label == i]
        num_samples = int(size)
        if num_samples > len(sub_data):
            print(f"Warning: Not enough samples for class {i}. Adjusting from {num_samples} to {len(sub_data)}")
            num_samples = len(sub_data)
        rand_indx = np.random.choice(len(sub_data), size=num_samples, replace=False).astype(int)
        sub2_data, sub2_label = sub_data[rand_indx], sub_label[rand_indx]
        all_class_set.append((sub2_data, sub2_label))

    index = [0] * N
    clients = []
    for m in range(prob.shape[0]):
        labels, images = [], []
        for n in range(N):
            start, end = index[n], index[n] + int(prob[m][n] * size_per_client)
            image, lbl = all_class_set[n][0][start:end], all_class_set[n][1][start:end]
            index[n] += int(prob[m][n] * size_per_client)
            labels.extend(lbl)
            images.extend(image)
        clients.append((np.array(images), np.array(labels)))
    return clients

def select_server_subset(cifar, percentage=0.1, mode='iid', dirichlet_alpha=1.0):
    """
    Selects a subset of data for the server based on IID or non-IID distribution.
    """
    images, labels = cifar
    unique_classes = np.unique(labels)
    total_num = len(labels)
    server_total = int(total_num * percentage)
    selected_indices = []

    if mode == 'iid':
        for cls in unique_classes:
            cls_indices = np.where(labels == cls)[0]
            num_cls = int(len(cls_indices) * percentage)
            if num_cls > len(cls_indices):
                num_cls = len(cls_indices)
            sampled = np.random.choice(cls_indices, size=num_cls, replace=False)
            selected_indices.extend(sampled)
    elif mode == 'non-iid':
        classes = len(unique_classes)
        prob = np.random.dirichlet(np.repeat(dirichlet_alpha, classes))
        cls_sample_numbers = {cls: int(prob[i] * server_total) for i, cls in enumerate(unique_classes)}
        total_assigned = sum(cls_sample_numbers.values())
        diff = server_total - total_assigned
        if diff > 0:
            for cls in np.random.choice(unique_classes, size=diff, replace=True):
                cls_sample_numbers[cls] += 1
        for cls in unique_classes:
            cls_indices = np.where(labels == cls)[0]
            n_sample = cls_sample_numbers[cls]
            if n_sample > len(cls_indices):
                n_sample = len(cls_indices)
            sampled = np.random.choice(cls_indices, size=n_sample, replace=False)
            selected_indices.extend(sampled)
    else:
        raise ValueError("Mode must be 'iid' or 'non-iid'")

    if config.SERVER_FILL:
        shortfall = server_total - len(selected_indices)
        if shortfall > 0:
            remaining_pool = np.setdiff1d(np.arange(total_num), selected_indices, assume_unique=True)
            extra = np.random.choice(remaining_pool, shortfall, replace=False)
            selected_indices = np.concatenate([selected_indices, extra])

    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices)
    subset_images = images[selected_indices]
    subset_labels = labels[selected_indices]
    return subset_images, subset_labels

# --- Core Training and Aggregation Functions ---
def update_weights(model_weight, dataset, learning_rate, local_epoch):
    """
    Performs local training on a client or server.
    Returns the updated model weights, loss, and the gradient of the first iteration.
    """
    if config.ORIGIN_MODEL == 'resnet':
        model = ResNet18_cifar10().to(device)
    elif config.ORIGIN_MODEL == "lstm":
        model = CharLSTM().to(device)
    elif config.ORIGIN_MODEL == "cnn":
        model = cnncifar().to(device)
    elif config.ORIGIN_MODEL == 'vgg':
        model = VGG16(config.NUM_CLASSES, 3).to(device)
    model.load_state_dict(model_weight)
    model.train()
    epoch_loss = []
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    if config.ORIGIN_MODEL in ['resnet', 'cnn', 'vgg']:
        Tensor_set = TensorDataset(torch.Tensor(dataset[0]).to(device), torch.Tensor(dataset[1]).to(device))
    elif config.ORIGIN_MODEL == 'lstm':
        Tensor_set = TensorDataset(torch.LongTensor(dataset[0]).to(device), torch.Tensor(dataset[1]).to(device))
    
    data_loader = DataLoader(Tensor_set, batch_size=config.BC_SIZE, shuffle=True)
    first_iter_gradient = None

    for iter_idx in range(local_epoch):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(data_loader):
            model.zero_grad()
            outputs = model(images)
            loss = criterion(outputs['output'], labels.long())
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item()/images.shape[0])

            if iter_idx == 0 and batch_idx == 0:
                first_iter_gradient = {name: param.grad.clone() for name, param in model.named_parameters()}
                for name, module in model.named_modules():
                    if isinstance(module, nn.BatchNorm2d):
                        first_iter_gradient[name + '.running_mean'] = module.running_mean.clone()
                        first_iter_gradient[name + '.running_var'] = module.running_var.clone()
        epoch_loss.append(sum(batch_loss)/len(batch_loss))

    return model.state_dict(), sum(epoch_loss) / len(epoch_loss), first_iter_gradient

def weight_differences(w1, w2, factor):
    """
    Calculates the scaled difference between two model weight dictionaries: (w2 - w1) * factor.
    """
    w_diff = copy.deepcopy(w1)
    for key in w_diff.keys():
        if 'num_batches_tracked' in key:
            continue
        w_diff[key] = (w2[key] - w1[key]) * factor
    return w_diff

def update_weights_correction(model_weight, dataset, learning_rate, local_epoch, c_i, c_s):
    """
    Performs local training with server-side gradient correction (for FedCLG-C).
    """
    if config.ORIGIN_MODEL == 'resnet':
        model = ResNet18_cifar10().to(device)
    elif config.ORIGIN_MODEL == "lstm":
        model = CharLSTM().to(device)
    elif config.ORIGIN_MODEL == "cnn":
        model = cnncifar().to(device)
    elif config.ORIGIN_MODEL == 'vgg':
        model = VGG16(config.NUM_CLASSES, 3).to(device)
    model.load_state_dict(model_weight)
    model.train()
    epoch_loss = []
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    if config.ORIGIN_MODEL in ['resnet', 'cnn', 'vgg']:
        Tensor_set = TensorDataset(torch.Tensor(dataset[0]).to(device), torch.Tensor(dataset[1]).to(device))
    elif config.ORIGIN_MODEL == 'lstm':
        Tensor_set = TensorDataset(torch.LongTensor(dataset[0]).to(device), torch.Tensor(dataset[1]).to(device))
    data_loader = DataLoader(Tensor_set, batch_size=config.BC_SIZE, shuffle=True)

    for iter_idx in range(local_epoch):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(data_loader):
            model.zero_grad()
            outputs = model(images)
            loss = criterion(outputs['output'], labels.long())
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.sum().item()/images.shape[0])
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        corrected_gradient = weight_differences(c_i, c_s, learning_rate)
        original_model_weight = model.state_dict()
        corrected_model_weight = weight_differences(corrected_gradient, original_model_weight, 1)
        model.load_state_dict(corrected_model_weight)

    return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        if 'num_batches_tracked' in key:
            continue
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

# --- Federated Learning Algorithm Implementations ---
def server_only(initial_w, global_round, learning_rate, local_epoch):
    """
    Baseline: Centralized training on server data only.
    """
    if config.ORIGIN_MODEL == 'resnet':
        test_model = ResNet18_cifar10().to(device)
    elif config.ORIGIN_MODEL == "lstm":
        test_model = CharLSTM().to(device)
    elif config.ORIGIN_MODEL == "cnn":
        test_model = cnncifar().to(device)
    elif config.ORIGIN_MODEL == 'vgg':
        test_model = VGG16(config.NUM_CLASSES, 3).to(device)
        
    train_w = copy.deepcopy(initial_w)
    test_acc, train_loss = [], []
    
    for round_idx in tqdm(range(global_round)):
        update_server_w, round_loss, _ = update_weights(train_w, server_data, learning_rate, local_epoch)
        train_w = update_server_w
        test_model.load_state_dict(train_w)
        train_loss.append(round_loss)
        test_acc.append(test_inference(test_model, test_dataset))
    
    return test_acc, train_loss

def fedavg(initial_w, global_round, client_lr, client_epochs, num_sampled_clients):
    """
    Implementation of the FedAvg algorithm.
    """
    if config.ORIGIN_MODEL == 'resnet':
        test_model = ResNet18_cifar10().to(device)
    elif config.ORIGIN_MODEL == "lstm":
        test_model = CharLSTM().to(device)
    elif config.ORIGIN_MODEL == "cnn":
        test_model = cnncifar().to(device)
    elif config.ORIGIN_MODEL == 'vgg':
        test_model = VGG16(config.NUM_CLASSES, 3).to(device)
        
    train_w = copy.deepcopy(initial_w)
    test_acc, train_loss = [], []
    
    for round_idx in tqdm(range(global_round)):
        local_weights, local_loss = [], []
        sampled_client = random.sample(range(config.N_CLIENTS), num_sampled_clients)
        for i in sampled_client:
            update_client_w, client_round_loss, _ = update_weights(train_w, client_data[i], client_lr, client_epochs)
            local_weights.append(update_client_w)
            local_loss.append(client_round_loss)

        train_w = average_weights(local_weights)
        test_model.load_state_dict(train_w)
        loss_avg = sum(local_loss) / len(local_loss)
        train_loss.append(loss_avg)
        test_acc.append(test_inference(test_model, test_dataset))
            
    return test_acc, train_loss

def hybridFL(initial_w, global_round, lr, client_epochs, server_epochs, num_sampled_clients):
    """
    Implementation of the Hybrid-FL (advance-then-aggregate) algorithm.
    """
    if config.ORIGIN_MODEL == 'resnet':
        test_model = ResNet18_cifar10().to(device)
    elif config.ORIGIN_MODEL == "lstm":
        test_model = CharLSTM().to(device)
    elif config.ORIGIN_MODEL == "cnn":
        test_model = cnncifar().to(device)
    elif config.ORIGIN_MODEL == 'vgg':
        test_model = VGG16(config.NUM_CLASSES, 3).to(device)
        
    train_w = copy.deepcopy(initial_w)
    test_acc, train_loss = [], []
    
    for round_idx in tqdm(range(global_round)):
        local_weights, local_loss = [], []
        sampled_client = random.sample(range(config.N_CLIENTS), num_sampled_clients)

        for i in sampled_client:
            update_client_w, client_round_loss, _ = update_weights(train_w, client_data[i], lr, client_epochs)
            local_weights.append(update_client_w)
            local_loss.append(client_round_loss)

        update_server_w, server_round_loss, _ = update_weights(train_w, server_data, lr, server_epochs)
        local_weights.append(update_server_w)
        local_loss.append(server_round_loss)

        train_w = average_weights(local_weights)
        test_model.load_state_dict(train_w)
        loss_avg = sum(local_loss) / len(local_loss)
        train_loss.append(loss_avg)
        test_acc.append(test_inference(test_model, test_dataset))
    
    return test_acc, train_loss

def CLG_SGD(initial_w, global_round, client_lr, server_lr, client_epochs, server_epochs, num_sampled_clients):
    """
    Implementation of the CLG-SGD (aggregate-then-advance) algorithm.
    """
    if config.ORIGIN_MODEL == 'resnet':
        test_model = ResNet18_cifar10().to(device)
    elif config.ORIGIN_MODEL == "lstm":
        test_model = CharLSTM().to(device)
    elif config.ORIGIN_MODEL == "cnn":
        test_model = cnncifar().to(device)
    elif config.ORIGIN_MODEL == 'vgg':
        test_model = VGG16(config.NUM_CLASSES, 3).to(device)
        
    train_w = copy.deepcopy(initial_w)
    test_acc, train_loss = [], []
    
    for round_idx in tqdm(range(global_round)):
        local_weights, local_loss = [], []
        sampled_client = random.sample(range(config.N_CLIENTS), num_sampled_clients)
        for i in sampled_client:
            update_client_w, client_round_loss, _ = update_weights(train_w, client_data[i], client_lr, client_epochs)
            local_weights.append(update_client_w)
            local_loss.append(client_round_loss)
        
        train_w = average_weights(local_weights)
        
        update_server_w, round_loss, _ = update_weights(train_w, server_data, server_lr, server_epochs)
        train_w = update_server_w
        local_loss.append(round_loss)

        test_model.load_state_dict(train_w)
        loss_avg = sum(local_loss) / len(local_loss)
        train_loss.append(loss_avg)
        test_acc.append(test_inference(test_model, test_dataset))
    
    return test_acc, train_loss

def Fed_C(initial_w, global_round, client_lr, server_lr, client_epochs, server_epochs, num_sampled_clients):
    """
    Implementation of the FedCLG-C algorithm.
    """
    if config.ORIGIN_MODEL == 'resnet':
        test_model = ResNet18_cifar10().to(device)
    elif config.ORIGIN_MODEL == "lstm":
        test_model = CharLSTM().to(device)
    elif config.ORIGIN_MODEL == "cnn":
        test_model = cnncifar().to(device)
    elif config.ORIGIN_MODEL == 'vgg':
        test_model = VGG16(config.NUM_CLASSES, 3).to(device)

    train_w = copy.deepcopy(initial_w)
    test_acc, train_loss = [], []
    
    for round_idx in tqdm(range(global_round)):
        local_weights, local_loss = [], []
        g_i_list = []
        
        _, _, g_s = update_weights(train_w, server_data, server_lr, 1)
        sampled_client = random.sample(range(config.N_CLIENTS), num_sampled_clients)
        for i in sampled_client:
            _, _, g_i = update_weights(train_w, client_data[i], client_lr, 1)
            g_i_list.append(g_i)

        for i in range(len(sampled_client)):
            update_client_w, client_round_loss = update_weights_correction(train_w, client_data[sampled_client[i]], client_lr, client_epochs, g_i_list[i], g_s)
            local_weights.append(update_client_w)
            local_loss.append(client_round_loss)
        
        train_w = average_weights(local_weights)
        update_server_w, round_loss, _ = update_weights(train_w, server_data, server_lr, server_epochs)
        train_w = update_server_w
        local_loss.append(round_loss)

        test_model.load_state_dict(train_w)
        loss_avg = sum(local_loss) / len(local_loss)
        train_loss.append(loss_avg)
        test_acc.append(test_inference(test_model, test_dataset))

    return test_acc, train_loss

def Fed_S(initial_w, global_round, client_lr, server_lr, client_epochs, server_epochs, num_sampled_clients):
    """
    Implementation of the FedCLG-S algorithm.
    """
    if config.ORIGIN_MODEL == 'resnet':
        test_model = ResNet18_cifar10().to(device)
    elif config.ORIGIN_MODEL == "lstm":
        test_model = CharLSTM().to(device)
    elif config.ORIGIN_MODEL == "cnn":
        test_model = cnncifar().to(device)
    elif config.ORIGIN_MODEL == 'vgg':
        test_model = VGG16(config.NUM_CLASSES, 3).to(device)
    
    train_w = copy.deepcopy(initial_w)
    test_acc, train_loss = [], []
    
    for round_idx in tqdm(range(global_round)):
        local_weights, local_loss = [], []
        g_i_list = []
        
        _, _, g_s = update_weights(train_w, server_data, server_lr, 1)
        sampled_client = random.sample(range(config.N_CLIENTS), num_sampled_clients)
        for i in sampled_client:
            _, _, g_i = update_weights(train_w, client_data[i], client_lr, 1)
            g_i_list.append(g_i)

        for i in range(len(sampled_client)):
            update_client_w, client_round_loss, _ = update_weights(train_w, client_data[sampled_client[i]], client_lr, client_epochs)
            local_weights.append(update_client_w)
            local_loss.append(client_round_loss)
        train_w = average_weights(local_weights)

        g_i_average = average_weights(g_i_list)
        correction_g = weight_differences(g_i_average, g_s, client_epochs * client_lr)
        train_w = weight_differences(correction_g, copy.deepcopy(train_w), 1)

        update_server_w, round_loss, _ = update_weights(train_w, server_data, server_lr, server_epochs)
        train_w = update_server_w
        local_loss.append(round_loss)

        test_model.load_state_dict(train_w)
        loss_avg = sum(local_loss) / len(local_loss)
        train_loss.append(loss_avg)
        test_acc.append(test_inference(test_model, test_dataset))
        
    return test_acc, train_loss

# --- Helper functions for FedDU and FedATMV ---
def KL_divergence(p1, p2):
    """Calculates KL divergence."""
    d = 0
    for i in range(len(p1)):
        if p2[i] == 0 or p1[i] == 0:
            continue
        d += p1[i] * math.log(p1[i]/p2[i], 2)
    return d

def calculate_js_divergence(p1, p2):
    """Calculates Jensen-Shannon divergence."""
    p3 = [(p1[i] + p2[i]) / 2 for i in range(len(p1))]
    return (KL_divergence(p1, p3) / 2) + (KL_divergence(p2, p3) / 2)

def ratio_combine(w1, w2, ratio=0):
    """Combines two weights with a given ratio: w1 * (1-ratio) + w2 * ratio."""
    w = copy.deepcopy(w1)
    for key in w.keys():
        if 'num_batches_tracked' in key:
            continue
        w[key] = (w2[key] - w1[key]) * ratio + w1[key]
    return w

def FedDU_modify(initial_w, global_round, client_lr, server_lr, client_epochs, server_epochs, num_sampled_clients):
    """
    Implementation of the FedDU algorithm.
    """
    if config.ORIGIN_MODEL == 'resnet':
        test_model = ResNet18_cifar10().to(device)
    elif config.ORIGIN_MODEL == "lstm":
        test_model = CharLSTM().to(device)
    elif config.ORIGIN_MODEL == "cnn":
        test_model = cnncifar().to(device)
    elif config.ORIGIN_MODEL == 'vgg':
        test_model = VGG16(config.NUM_CLASSES, 3).to(device)

    train_w = copy.deepcopy(initial_w)
    test_model.load_state_dict(train_w)
    test_acc, train_loss = [], []
    server_min = 0
    
    all_client_labels = [lbl for i in range(config.N_CLIENTS) for lbl in client_data[i][1]]
    unique_classes = np.unique(all_client_labels)
    classes = len(unique_classes)
    
    P = [np.sum(np.array(all_client_labels) == cls) / len(all_client_labels) for i, cls in enumerate(unique_classes)]
    
    server_labels = np.array(server_data[1])
    n_0 = len(server_labels)
    P_0 = [np.sum(server_labels == cls) / n_0 if n_0 > 0 else 0 for i, cls in enumerate(unique_classes)]
    D_P_0 = calculate_js_divergence(P_0, P)

    print("FedDU Initializing:")
    print(f"  Server data size: {n_0}")
    print(f"  Server data non-IID degree: {D_P_0:.6f}")
    
    for round_idx in tqdm(range(global_round)):
        local_weights, local_losses = [], []
        sampled_clients = random.sample(range(config.N_CLIENTS), num_sampled_clients)
        
        num_current = sum(len(client_data[i][0]) for i in sampled_clients)
        
        for i in sampled_clients:
            update_client_w, client_round_loss, _ = update_weights(train_w, client_data[i], client_lr, client_epochs)
            local_weights.append(update_client_w)
            local_losses.append(client_round_loss)
        
        w_t_half = average_weights(local_weights)
        
        selected_client_labels = [lbl for i in sampled_clients for lbl in client_data[i][1]]
        P_t_prime = [np.sum(np.array(selected_client_labels) == cls) / len(selected_client_labels) if len(selected_client_labels) > 0 else 0 for i, cls in enumerate(unique_classes)]
        D_P_t_prime = calculate_js_divergence(P_t_prime, P)
        
        test_model.load_state_dict(w_t_half)
        acc_t = test_inference(test_model, test_dataset) / 100.0
        
        avg_iter = (num_current * client_epochs) / (num_sampled_clients * config.BC_SIZE)
        epsilon = 1e-10
        alpha = (1 - acc_t) * (n_0 * D_P_t_prime) / (n_0 * D_P_t_prime + num_current * D_P_0 + epsilon)
        alpha = alpha * (config.DECAY_RATE ** round_idx) * config.MU
        
        server_iter = max(server_min, int(alpha * avg_iter))
        
        if alpha > 0.001:
            actual_iter = math.ceil(n_0 / config.BC_SIZE) * server_epochs
            server_iter = min(actual_iter, server_iter)
            update_server_w, round_loss, _ = update_weights(copy.deepcopy(w_t_half), server_data, server_lr, server_epochs)
            local_losses.append(round_loss)
            train_w = ratio_combine(w_t_half, update_server_w, alpha)
        else:
            train_w = copy.deepcopy(w_t_half)
            _, round_loss, _ = update_weights(copy.deepcopy(w_t_half), server_data, server_lr, server_epochs)
            local_losses.append(round_loss)
        
        test_model.load_state_dict(train_w)
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        test_acc.append(test_inference(test_model, test_dataset))
    
    return test_acc, train_loss

# --- FedMut and FedATMV specific functions ---
def mutation_spread(iter, w_glob, m, w_delta, alpha):
    """
    Generates mutated model variants for clients (used in FedMut and FedATMV).
    """
    w_locals_new = []
    ctrl_cmd_list = []
    ctrl_rate = config.MUT_ACC_RATE * (1.0 - min(iter * 1.0 / config.MUT_BOUND, 1.0))

    for k in w_glob.keys():
        ctrl_list = []
        for i in range(0, int(m / 2)):
            ctrl = random.random()
            if ctrl > 0.5:
                ctrl_list.extend([1.0, 1.0 * (-1.0 + ctrl_rate)])
            else:
                ctrl_list.extend([1.0 * (-1.0 + ctrl_rate), 1.0])
        random.shuffle(ctrl_list)
        ctrl_cmd_list.append(ctrl_list)
    
    for j in range(m):
        w_sub = copy.deepcopy(w_glob)
        if not (j == m - 1 and m % 2 == 1):
            for ind, k in enumerate(w_sub.keys()):
                w_sub[k] += w_delta[k] * ctrl_cmd_list[ind][j] * alpha
        w_locals_new.append(w_sub)
    return w_locals_new

# In atmv.py

def Aggregation(w, lens):
    """
    Performs weighted or unweighted aggregation of model weights.
    If `lens` is None, it performs a simple average.
    """
    w_avg = None
    if lens == None:
        # If no weights are provided, use equal weighting (simple average)
        total_count = len(w)
        lens = []
        for i in range(len(w)):
            lens.append(1.0)
    else:
        # Use provided weights (e.g., based on dataset size)
        total_count = sum(lens)

    # Perform weighted summation of the model parameters
    for i in range(0, len(w)):
        if i == 0:
            # Initialize with the first weighted model
            w_avg = copy.deepcopy(w[0])
            for k in w_avg.keys():
                w_avg[k] = w[i][k] * lens[i]
        else:
            # Add subsequent weighted models
            for k in w_avg.keys():
                w_avg[k] += w[i][k] * lens[i]

    # Divide by the total weight to get the average
    for k in w_avg.keys():
        w_avg[k] = torch.div(w_avg[k], total_count)

    return w_avg

def FedSub(w, w_old, weight):
    """Calculates the weighted difference between two models: (w - w_old) * weight."""
    w_sub = copy.deepcopy(w)
    for k in w_sub.keys():
        w_sub[k] = (w[k] - w_old[k]) * weight
    return w_sub

def delta_rank(delta_dict):
    """Calculates the L2 norm of the model update dictionary."""
    dict_a = torch.cat([delta_dict[p].view(-1) for p in delta_dict.keys()])
    return torch.norm(dict_a, dim=0)

def FedMut(net_glob, global_round, client_lr, client_epochs, num_sampled_clients):
    """
    Implementation of the FedMut algorithm.
    """
    net_glob.train()
    if config.ORIGIN_MODEL == 'resnet':
        test_model = ResNet18_cifar10().to(device)
    elif config.ORIGIN_MODEL == "lstm":
        test_model = CharLSTM().to(device)
    elif config.ORIGIN_MODEL == "cnn":
        test_model = cnncifar().to(device)
    elif config.ORIGIN_MODEL == 'vgg':
        test_model = VGG16(config.NUM_CLASSES, 3).to(device)
        
    test_acc, train_loss = [], []
    w_locals = [copy.deepcopy(net_glob.state_dict()) for _ in range(num_sampled_clients)]
    max_rank = 0
    
    for round_idx in tqdm(range(global_round)):
        w_old = copy.deepcopy(net_glob.state_dict())
        local_loss = []
        idxs_users = np.random.choice(range(config.N_CLIENTS), num_sampled_clients, replace=False)
        
        for i, idx in enumerate(idxs_users):
            net_glob.load_state_dict(w_locals[i])
            update_client_w, client_round_loss, _ = update_weights(copy.deepcopy(net_glob.state_dict()), client_data[idx], client_lr, client_epochs)
            w_locals[i] = copy.deepcopy(update_client_w)
            local_loss.append(client_round_loss)

        w_agg = Aggregation(w_locals, None)
        net_glob.load_state_dict(w_agg)
        
        test_model.load_state_dict(w_agg)
        loss_avg = sum(local_loss) / len(local_loss)
        train_loss.append(loss_avg)
        test_acc.append(test_inference(test_model, test_dataset))

        w_delta = FedSub(w_agg, w_old, 1.0)
        rank = delta_rank(w_delta)
        if rank > max_rank:
            max_rank = rank
        
        w_locals = mutation_spread(round_idx, w_agg, num_sampled_clients, w_delta, config.RHO)
    return test_acc, train_loss

def FedATMV(net_glob, global_round, client_lr, server_lr, client_epochs, server_epochs, num_sampled_clients, lambda_val=1):
    """
    Implementation of the proposed FedATMV algorithm.
    """
    net_glob.train()
    if config.ORIGIN_MODEL == 'resnet':
        test_model = ResNet18_cifar10().to(device)
    elif config.ORIGIN_MODEL == "lstm":
        test_model = CharLSTM().to(device)
    elif config.ORIGIN_MODEL == "cnn":
        test_model = cnncifar().to(device)
    elif config.ORIGIN_MODEL == 'vgg':
        test_model = VGG16(config.NUM_CLASSES, 3).to(device)
    
    test_acc, train_loss = [], []
    w_locals = [copy.deepcopy(net_glob.state_dict()) for _ in range(num_sampled_clients)]
    max_rank = 0
    
    all_client_labels = [lbl for i in range(config.N_CLIENTS) for lbl in client_data[i][1]]
    unique_classes = np.unique(all_client_labels)
    classes = len(unique_classes)
    
    P = [np.sum(np.array(all_client_labels) == cls) / len(all_client_labels) for i, cls in enumerate(unique_classes)]
    
    server_labels = np.array(server_data[1])
    n_0 = len(server_labels)
    P_0 = [np.sum(server_labels == cls) / n_0 if n_0 > 0 else 0 for i, cls in enumerate(unique_classes)]
    D_P_0 = calculate_js_divergence(P_0, P)

    print("FedATMV Initializing:")
    print(f"  Server data size: {n_0}")
    print(f"  Server data non-IID degree: {D_P_0:.6f}")
    
    alpha_history, improvement_history = [], []
    acc_prev = 0.0

    for round_idx in tqdm(range(global_round)):
        w_old = copy.deepcopy(net_glob.state_dict())
        local_weights, local_loss = [], []
        idxs_users = np.random.choice(range(config.N_CLIENTS), num_sampled_clients, replace=False)
        
        selected_client_labels = []
        num_current = 0
        
        for i, idx in enumerate(idxs_users):
            net_glob.load_state_dict(w_locals[i])
            update_client_w, client_round_loss, _ = update_weights(copy.deepcopy(net_glob.state_dict()), client_data[idx], client_lr, client_epochs)
            w_locals[i] = copy.deepcopy(update_client_w)
            local_loss.append(client_round_loss)
            selected_client_labels.extend(client_data[idx][1])
            num_current += len(client_data[idx][0])

        w_agg = Aggregation(w_locals, None)
        net_glob.load_state_dict(w_agg)
        
        P_t_prime = [np.sum(np.array(selected_client_labels) == cls) / len(selected_client_labels) if len(selected_client_labels) > 0 else 0 for i, cls in enumerate(unique_classes)]
        D_P_t_prime = calculate_js_divergence(P_t_prime, P)
        
        test_model.load_state_dict(w_agg)
        acc_t = test_inference(test_model, test_dataset) / 100.0
        
        epsilon = 1e-10
        r_data = n_0 / (n_0 + num_current + epsilon)
        r_noniid = D_P_t_prime / (D_P_t_prime + D_P_0 + epsilon)
        
        if round_idx == 0:
            improvement = 0.0
        else:
            improvement = max(0.0, acc_prev - acc_t) / (acc_prev + epsilon)
        
        min_alpha, max_alpha = 0.001, 1.0
        alpha_new = config.MU * (1 - acc_t) * r_data * r_noniid + lambda_val * improvement
        alpha_new = max(min_alpha, min(max_alpha, alpha_new))
        
        alpha_history.append(alpha_new)
        improvement_history.append(improvement)
        acc_prev = acc_t

        if alpha_new > 0.001:
            update_server_w, round_loss, _ = update_weights(copy.deepcopy(w_agg), server_data, server_lr, server_epochs)
            local_loss.append(round_loss)
            final_model = ratio_combine(w_agg, update_server_w, alpha_new)
        else:
            final_model = copy.deepcopy(w_agg)
            _, round_loss, _ = update_weights(copy.deepcopy(w_agg), server_data, server_lr, server_epochs)
            local_loss.append(round_loss)
        
        net_glob.load_state_dict(final_model)
        
        test_model.load_state_dict(final_model)
        loss_avg = sum(local_loss) / len(local_loss)
        train_loss.append(loss_avg)
        test_acc.append(test_inference(test_model, test_dataset))
        
        w_delta = FedSub(final_model, w_old, 1.0)
        rank = delta_rank(w_delta)
        if rank > max_rank:
            max_rank = rank
            
        tmp_radius = config.RHO * (1 + config.THETA * alpha_new)
        w_locals = mutation_spread(round_idx, final_model, num_sampled_clients, w_delta, tmp_radius)
        
    return test_acc, train_loss

# --- Main Execution Block ---
def set_random_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = torch.device(f"cuda:{config.GPU}" if torch.cuda.is_available() else 'cpu')

if config.GLOBAL_RANDOM_FIX:
    set_random_seed(config.SEED)

# --- Dataset Preparation ---
if config.DATASET == 'cifar100':
    cifar, test_dataset = CIFAR100()
    prob = get_prob(config.ALPHA, config.N_CLIENTS, class_num=20, iid_mode=config.IS_IID)
    client_data = create_data_all_train(prob, config.SAMPLES_PER_CLIENT, cifar, N=20)
    test_dataset.targets = sparse2coarse(test_dataset.targets).astype(int)
    server_images, server_labels = select_server_subset(cifar, percentage=config.SERVER_DATA_RATIO, mode='non-iid' if not config.SERVER_IID else 'iid', dirichlet_alpha=config.BETA)
    init_model = VGG16(config.NUM_CLASSES, 3).to(device)
    initial_w = copy.deepcopy(init_model.state_dict())
elif config.DATASET == 'shake':
    train_dataset = ShakeSpeare(True)
    test_dataset = ShakeSpeare(False)
    total_shake = np.array([item.numpy() for item, _ in train_dataset])
    total_label = np.array([labels for _, labels in train_dataset])
    shake = [total_shake, total_label]
    dict_users = train_dataset.get_client_dic()
    client_data = []
    for client in sorted(dict_users.keys()):
        indices = np.array(list(dict_users[client]), dtype=np.int64)
        client_data.append((total_shake[indices], total_label[indices]))
    server_images, server_labels = select_server_subset(shake, percentage=config.SERVER_DATA_RATIO, mode='non-iid' if not config.SERVER_IID else 'iid', dirichlet_alpha=config.BETA)
    init_model = CharLSTM().to(device)
    initial_w = copy.deepcopy(init_model.state_dict())
elif config.DATASET == "cifar10":
    trans_cifar10_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trans_cifar10_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10("./data/cifar10", train=True, download=True, transform=trans_cifar10_train)
    test_dataset = torchvision.datasets.CIFAR10("./data/cifar10", train=False, download=True, transform=trans_cifar10_val)
    
    total_img = np.array([np.array(img) for img, _ in train_dataset])
    total_label = np.array([label for _, label in train_dataset])
    cifar = [total_img, total_label]

    prob = get_prob(config.ALPHA, config.N_CLIENTS, class_num=10, iid_mode=config.IS_IID)
    client_data = create_data_all_train(prob, config.SAMPLES_PER_CLIENT, cifar, N=10)
    server_images, server_labels = select_server_subset(cifar, percentage=config.SERVER_DATA_RATIO, mode='non-iid' if not config.SERVER_IID else 'iid', dirichlet_alpha=config.BETA)
    
    if config.ORIGIN_MODEL == 'cnn':
        init_model = cnncifar().to(device)
    elif config.ORIGIN_MODEL == 'resnet':
        init_model = ResNet18_cifar10().to(device)
    initial_w = copy.deepcopy(init_model.state_dict())

# Print data distribution statistics
all_labels = [lbl for data in client_data for lbl in data[1]]
total_count = len(all_labels)
unique_classes, counts = np.unique(all_labels, return_counts=True)
class_counts = [0] * config.NUM_CLASSES
for cls, cnt in zip(unique_classes, counts):
    class_counts[cls] = cnt
print(f"Training Client Total: {total_count} {' '.join(map(str, class_counts))}")

for i, (_, lbls) in enumerate(client_data[:10]):
    total_count = len(lbls)
    unique_classes, counts = np.unique(lbls, return_counts=True)
    class_counts = [0] * config.NUM_CLASSES
    for cls, cnt in zip(unique_classes, counts):
        if cls < len(class_counts):
            class_counts[cls] = cnt
    print(f"Client {i}: {total_count} {' '.join(map(str, class_counts))}")

server_data = [server_images, server_labels]
s_lbls = np.array(server_data[1])
total_count = len(s_lbls)
unique_classes, counts = np.unique(s_lbls, return_counts=True)
class_counts = [0] * config.NUM_CLASSES
for cls, cnt in zip(unique_classes, counts):
    if cls < len(class_counts):
        class_counts[cls] = cnt
print(f"Server: {total_count} {' '.join(map(str, class_counts))}")

def run_once():
    """
    Runs a single experiment trial with all baseline algorithms and FedATMV.
    """
    results_test_acc = {}
    results_train_loss = {}

    print("\n--- Training FedMut ---")
    test_acc_FedMut, train_loss_FedMut = FedMut(copy.deepcopy(init_model), config.T_ROUNDS, config.ETA, config.K_EPOCHS, config.M_CLIENTS)
    results_test_acc['FedMut'] = test_acc_FedMut
    results_train_loss['FedMut'] = train_loss_FedMut

    print("\n--- Training Server-Only ---")
    test_acc_server_only, train_loss_server_only = server_only(initial_w, config.T_ROUNDS, config.ETA_0, config.E_EPOCHS)
    results_test_acc['Server-Only'] = test_acc_server_only
    results_train_loss['Server-Only'] = train_loss_server_only

    print("\n--- Training FedAvg ---")
    test_acc_fedavg, train_loss_fedavg = fedavg(initial_w, config.T_ROUNDS, config.ETA, config.K_EPOCHS, config.M_CLIENTS)
    results_test_acc['FedAvg'] = test_acc_fedavg
    results_train_loss['FedAvg'] = train_loss_fedavg

    print("\n--- Training Hybrid-FL ---")
    test_acc_hybridFL, train_loss_hybridFL = hybridFL(initial_w, config.T_ROUNDS, config.ETA, config.K_EPOCHS, config.E_EPOCHS, config.M_CLIENTS)
    results_test_acc['Hybrid-FL'] = test_acc_hybridFL
    results_train_loss['Hybrid-FL'] = train_loss_hybridFL

    print("\n--- Training CLG-SGD ---")
    test_acc_CLG_SGD, train_loss_CLG_SGD = CLG_SGD(initial_w, config.T_ROUNDS, config.ETA, config.ETA_0, config.K_EPOCHS, config.E_EPOCHS, config.M_CLIENTS)
    results_test_acc['CLG-SGD'] = test_acc_CLG_SGD
    results_train_loss['CLG-SGD'] = train_loss_CLG_SGD

    print("\n--- Training FedCLG-C ---")
    test_acc_Fed_C, train_loss_Fed_C = Fed_C(initial_w, config.T_ROUNDS, config.ETA, config.ETA_0, config.K_EPOCHS, config.E_EPOCHS, config.M_CLIENTS)
    results_test_acc['FedCLG-C'] = test_acc_Fed_C
    results_train_loss['FedCLG-C'] = train_loss_Fed_C

    print("\n--- Training FedCLG-S ---")
    test_acc_Fed_S, train_loss_Fed_S = Fed_S(initial_w, config.T_ROUNDS, config.ETA, config.ETA_0, config.K_EPOCHS, config.E_EPOCHS, config.M_CLIENTS)
    results_test_acc['FedCLG-S'] = test_acc_Fed_S
    results_train_loss['FedCLG-S'] = train_loss_Fed_S

    print("\n--- Training FedDU ---")
    test_acc_FedDU, train_loss_FedDU = FedDU_modify(initial_w, config.T_ROUNDS, config.ETA, config.ETA_0, config.K_EPOCHS, config.E_EPOCHS, config.M_CLIENTS)
    results_test_acc['FedDU'] = test_acc_FedDU
    results_train_loss['FedDU'] = train_loss_FedDU

    print("\n--- Training FedATMV ---")
    test_acc_FedATMV, train_loss_FedATMV = FedATMV(copy.deepcopy(init_model), config.T_ROUNDS, config.ETA, config.ETA_0, config.K_EPOCHS, config.E_EPOCHS, config.M_CLIENTS)
    results_test_acc['FedATMV'] = test_acc_FedATMV
    results_train_loss['FedATMV'] = train_loss_FedATMV

    print("\n--- Final Results ---")
    for algo in results_test_acc:
        if len(results_test_acc[algo]) >= 20:
            print(f"{algo} - Round 20 Test Accuracy: {results_test_acc[algo][19]:.2f}%, Round 20 Train Loss: {results_train_loss[algo][19]:.4f}")
    print("\n")
    for algo in results_test_acc:
        print(f"{algo} - Final Test Accuracy: {results_test_acc[algo][-1]:.2f}%, Final Train Loss: {results_train_loss[algo][-1]:.4f}")
        

    return results_test_acc, results_train_loss

if __name__ == '__main__':
    results_test_acc, results_train_loss = run_once()