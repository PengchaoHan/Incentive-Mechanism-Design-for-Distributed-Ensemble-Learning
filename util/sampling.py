#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import math
from tqdm import trange
import copy

def dirichlet(indices_by_classes, n_sample, num_classes, num_nodes, alpha=0.5):
    # codes adapted from https://github.com/HeinaZ/FedAVG
    shard_size = n_sample // num_nodes
    dirichlet_pdf = np.random.dirichlet([alpha / 10] * 10, num_nodes)
    dict_users = {}
    for i in range(num_nodes):
        local_pdf = np.floor(dirichlet_pdf[i] * shard_size).astype('int64')
        a = sum(local_pdf)
        indices_per_user = []
        for label_num, label_value in zip(local_pdf, range(num_classes)):
            indices = copy.deepcopy(indices_by_classes[label_value])
            np.random.shuffle(indices)
            if label_num > 1:
                indices_per_user.extend(indices[0:label_num].tolist())
        dict_users[i] = indices_per_user
    return dict_users

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # num_shards, num_imgs = 2*num_users, int(dataset.data.size()[0]/2/num_users)  # choose two number from a set with num_shards, each client has 2*num_imgs images
    # idx_shard = [i for i in range(num_shards)]
    # dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    # idxs = np.arange(dataset.data.size()[0])
    # labels = dataset.train_labels.numpy()
    #
    # # sort labels
    # idxs_labels = np.vstack((idxs, labels))
    # idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    # idxs = idxs_labels[0,:]
    #
    # # divide and assign
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, 2, replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # return dict_users

    label_list = dataset.targets.numpy()
    minLabel = min(label_list)
    numLabels = len(dataset.classes)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    for i in range(0, len(label_list)):
        tmp_target_node = int((label_list[i] - minLabel) % num_users)
        if num_users>numLabels:
            tmpMinIndex=0
            tmpMinVal=math.inf
            for n in range(0, num_users):
                if (n)%numLabels==tmp_target_node and len(dict_users[n])<tmpMinVal:
                    tmpMinVal=len(dict_users[n])
                    tmpMinIndex=n
            tmp_target_node=tmpMinIndex
        dict_users[tmp_target_node] = np.concatenate((dict_users[tmp_target_node], [i]), axis=0)
    return dict_users

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # num_shards, num_imgs = 2 * num_users, int(len(dataset.data) / 2 / num_users)  # choose two number from a set with num_shards, each client has 2*num_imgs images
    # idx_shard = [i for i in range(num_shards)]
    # dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    # idxs = np.arange(len(dataset.data))
    # labels = np.array(dataset.targets)
    #
    # # sort labels
    # idxs_labels = np.vstack((idxs, labels))
    # idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    # idxs = idxs_labels[0,:]
    #
    # # divide and assign
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, 2, replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # return dict_users

    label_list = np.array(dataset.targets)
    minLabel = min(label_list)
    numLabels = len(dataset.classes)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    for i in range(0, len(label_list)):
        tmp_target_node = int((label_list[i] - minLabel) % num_users)
        if num_users > numLabels:
            tmpMinIndex = 0
            tmpMinVal = math.inf
            for n in range(0, num_users):
                if (n) % numLabels == tmp_target_node and len(dict_users[n]) < tmpMinVal:
                    tmpMinVal = len(dict_users[n])
                    tmpMinIndex = n
            tmp_target_node = tmpMinIndex
        dict_users[tmp_target_node] = np.concatenate((dict_users[tmp_target_node], [i]), axis=0)
    return dict_users


def svhn_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def svhn_noniid(dataset, num_users):
    num_shards, num_imgs = 2 * num_users, int(len(dataset.data) / 2 / num_users)  # choose two number from a set with num_shards, each client has 2*num_imgs images
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset.data))
    labels = dataset.labels

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def rearrange_data_by_class(data, targets, n_class):
    new_data = []
    indices = []
    for i in range(n_class):  # trange(n_class):
        idx = targets == i
        new_data.append(data[idx])
        indices.extend(np.where(idx==True))
    return new_data, indices

def split_data(dataset, data_train, n_nodes, iid=False, num_sample=60000, num_classes=10, alpha=0.5):
    if dataset == 'MNIST':
        if n_nodes is None:
            raise Exception('Unknown n_nodes for MNIST.')
        if iid == 'dirichlet':
            # print("Rearrange data by class...")
            data_by_class, indices = rearrange_data_by_class(
                data_train.data.cpu().detach().numpy(),
                data_train.targets.cpu().detach().numpy(),
                num_classes
            )
            dict_users = dirichlet(indices, num_sample, num_classes, n_nodes, alpha=alpha)
        elif iid:
            dict_users = mnist_iid(data_train, n_nodes)
        else:
            dict_users = mnist_noniid(data_train, n_nodes)
    elif dataset == 'cifar10' or dataset == 'cifar100':
        if n_nodes is None:
            raise Exception('Unknown n_nodes for CIFAR*.')
        if iid == 'dirichlet':
            # print("Rearrange data by class...")
            data_by_class, indices = rearrange_data_by_class(
                data_train.data,
                np.array(data_train.targets),
                num_classes
            )
            dict_users = dirichlet(indices, num_sample, num_classes, n_nodes, alpha=alpha)
        elif iid:
            dict_users = cifar_iid(data_train, n_nodes)
        else:
            dict_users = cifar_noniid(data_train, n_nodes)
    elif dataset == 'FEMNIST':
        if iid:
            raise Exception('Only consider NON-IID setting in FEMNIST')
        else:
            dict_users = data_train.get_dict_clients()

    elif dataset == 'celeba':
        if iid:
            raise Exception('Only consider NON-IID setting in FEMNIST')
        else:
            dict_users = data_train.get_dict_clients()
    elif dataset == 'SVHN':
        if n_nodes is None:
            raise Exception('Unknown n_nodes for SVHN.')
        if iid == 'dirichlet':
            # print("Rearrange data by class...")
            data_by_class, indices = rearrange_data_by_class(
                data_train.data,
                data_train.labels,
                num_classes
            )
            dict_users = dirichlet(indices, num_sample, num_classes, n_nodes, alpha=alpha)
        elif iid:
            dict_users = svhn_iid(data_train, n_nodes)
        else:
            dict_users = svhn_noniid(data_train, n_nodes)
    elif dataset == 'shakespeare':
        if iid:
            raise Exception('Only consider NON-IID setting in SHAKESPEARE')
        else:
            dict_users = data_train.get_dict_clients()
    else:
        raise Exception('Unknown dataset name.')
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
