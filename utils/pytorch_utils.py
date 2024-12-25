import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Optional
from torch.utils.data import Dataset, TensorDataset
import argparse
import torch
import random
import os
import math


def print_args(args: argparse.ArgumentParser):
    """
    print the hyperparameters
    :param args: hyperparameters
    :return: None
    """
    s = "=========================================================\n"
    for arg, concent in args.__dict__.items():
        s += "{}:{}\n".format(arg, concent)
    return s


def init_weights(model: nn.Module):
    """
    Network Parameters Initialization Function
    :param model: the model to initialize
    :return: None
    """
    classname = model.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight, 1.0, 0.02)
        nn.init.zeros_(model.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(model.weight)
        nn.init.zeros_(model.bias)
    elif classname.find('Conv2d') != -1:
        nn.init.kaiming_uniform_(model.weight)

def init_mask(model: nn.Module):
    classname = model.__class__.__name__
    if classname.find('Mask') != -1:
        nn.init.kaiming_uniform_(model.neuron_mask, a=math.sqrt(5))


def bca_score(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred)
    numb = m.shape[0]
    acc_each_label = 0
    for i in range(numb):
        acc = m[i, i] / np.sum(m[i, :], keepdims=False).astype(np.float32)
        acc_each_label += acc
    return acc_each_label / numb


def standard_normalize(x_train, x_test, clip_range=None):
    mean, std = np.mean(x_train), np.std(x_train)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    if clip_range is not None:
        x = np.clip(x_train, a_min=clip_range[0], a_max=clip_range[1])
        x = np.clip(x_test, a_min=clip_range[0], a_max=clip_range[1])
    return x_train, x_test


def seed(seed: Optional[int] = 0):
    """
    fix all the random seed
    :param seed: random seed
    :return: None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # CUDA随机种子
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 禁止自动优化算法选择，确保实验可复现
    # torch.use_deterministic_algorithms(True)

def adjust_learning_rate(optimizer: nn.Module, epoch: int,
                         learning_rate: float):
    """decrease the learning rate"""
    lr = learning_rate
    if epoch >= 50:
        lr = learning_rate * 0.1
    if epoch >= 100:
        lr = learning_rate * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class CustomTensorDataset(Dataset):
    """ TnsorDataset with support of transforms. """
    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def weight_for_balanced_classes(y: torch.Tensor):
    count = [0.0] * len(np.unique(y.numpy()))
    for label in y:
        count[label] += 1.0
    count = [len(y) / x for x in count]
    weight = [0.0] * len(y)
    for idx, label in enumerate(y):
        weight[idx] = count[label]

    return weight


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, device=None):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        #print(x.shape)
        #print(self.centers.shape)
        #distmat = distmat1 - torch.matmul(2*x, self.centers)
        #distmat.addmm_(1, -2, x, self.centers.t())
        #test = torch.transpose(self.centers, 0, 1)
        distmat = torch.addmm(distmat, x, self.centers.t(), beta=1, alpha=-2)   
        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class GeneralTorchModel(nn.Module):
    def __init__(self, model, n_class=10, im_mean=None, im_std=None):
        super(GeneralTorchModel, self).__init__()
        self.model = model
        self.model.eval()
        self.num_queries = 0
        self.im_mean = im_mean
        self.im_std = im_std
        self.n_class = n_class

    def forward(self, image):
        if len(image.size()) != 4:
            image = image.unsqueeze(0)
        image = self.preprocess(image)
        logits = self.model(image)
        return logits

    def preprocess(self, image):
        if isinstance(image, np.ndarray):
            processed = torch.from_numpy(image).type(torch.FloatTensor)
        else:
            processed = image

        if self.im_mean is not None and self.im_std is not None:
            im_mean = torch.tensor(self.im_mean).cuda().view(1, processed.shape[1], 1, 1).repeat(
                processed.shape[0], 1, 1, 1)
            im_std = torch.tensor(self.im_std).cuda().view(1, processed.shape[1], 1, 1).repeat(
                processed.shape[0], 1, 1, 1)
            processed = (processed - im_mean) / im_std
        return processed

    def predict_prob(self, image):
        with torch.no_grad():
            if len(image.size()) != 4:
                image = image.unsqueeze(0)
            image = self.preprocess(image)
            logits = self.model(image)
            self.num_queries += image.size(0)
        return logits

    def predict_label(self, image):
        logits = self.predict_prob(image)
        _, predict = torch.max(logits, 1)
        return predict