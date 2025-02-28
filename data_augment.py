# built on:
# -*- coding: utf-8 -*-
# @Time    : 2021/6/25 21:01
# @Author  : wenzhang
# @File    : data_augment.py

import numpy as np
from scipy.signal import hilbert


def resamples(x,y,ratio):
    if round(ratio)-ratio == 0.0:
        xs = []
        ys = []
        for r in range(round(ratio)):
            xs.append(x)
            ys.append(y)
        x = np.concatenate(xs)
        y = np.concatenate(ys)
    else:
        xs = []
        ys = []
        for r in range(np.floor(ratio)):
            xs.append(x)
            ys.append(y)
        for i in range(np.unique(y)):
            ind = np.where(y==i)[0]
            ind = np.random.choice(ind,round((ratio-np.floor(ratio))*len(ind)),replace=False)
            xs.append(x[ind])
            ys.append(y[ind])
        x = np.concatenate(xs)
        y = np.concatenate(ys)
    return x, y


def data_aug(data, labels, size, flag_aug):
    # augments data based on boolean inputs reuse_data, noise_data, neg_data, freq_mod data.
    # data: samples * size * n_channels
    # size: int(freq * window_size)
    # Returns: entire training dataset after data augmentation, and the corresponding labels

    # noise_flag, neg_flag, mult_flag, freq_mod_flag test 75.154
    # mult_flag, noise_flag, neg_flag, freq_mod_flag test 76.235
    # noise_flag, neg_flag, freq_mod_flag test 76.157

    mult_flag, noise_flag, neg_flag, freq_mod_flag = flag_aug[0], flag_aug[1], flag_aug[2], flag_aug[3]

    n_channels = data.shape[2]
    data_out = data  # 1 raw features
    labels_out = labels

    if mult_flag:  # 2 features
        mult_data_add, labels_mult = data_mult_f(data, labels, size, n_channels=n_channels)
        data_out = np.concatenate([data_out, mult_data_add], axis=0)
        labels_out = np.append(labels_out, labels_mult)
    if noise_flag:  # 1 features
        noise_data_add, labels_noise = data_noise_f(data, labels, size, n_channels=n_channels)
        data_out = np.concatenate([data_out, noise_data_add], axis=0)
        labels_out = np.append(labels_out, labels_noise)
    if neg_flag:  # 1 features
        neg_data_add, labels_neg = data_neg_f(data, labels, size, n_channels=n_channels)
        data_out = np.concatenate([data_out, neg_data_add], axis=0)
        labels_out = np.append(labels_out, labels_neg)
    if freq_mod_flag:  # 2 features
        freq_data_add, labels_freq = freq_mod_f(data, labels, size, n_channels=n_channels)
        data_out = np.concatenate([data_out, freq_data_add], axis=0)
        labels_out = np.append(labels_out, labels_freq)

    # 最终输出data格式为
    # raw 144, mult_add 144, mult_reduce 144, noise 144, neg 144, freq1 144, freq2 144
    return data_out, labels_out


def data_noise_f(data_ori, labels_ori, noise_mod_val=2, ratio=1.0):
    
    data, labels = resamples(data_ori, labels_ori, ratio)
    new_data = []
    new_labels = []
    # noise_mod_val = 2
    # print("noise mod: {}".format(noise_mod_val))
    for i in range(len(labels)):
        # if labels[i] >= 0:
            stddev_t = np.std(data[i])
            rand_t = np.random.rand(*data[i].shape)
            rand_t = rand_t - 0.5
            to_add_t = rand_t * stddev_t / noise_mod_val
            data_t = data[i] + to_add_t
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.concatenate(new_data)[:,None,:,:]
    new_labels = np.array(new_labels)

    return np.concatenate((data_ori, new_data_ar)), np.concatenate((labels_ori, new_labels))


def data_noise_f(data_ori, labels_ori, noise_mod_val=2, ratio=1.0):
    
    data, labels = resamples(data_ori, labels_ori, ratio)
    new_data = []
    new_labels = []
    # noise_mod_val = 2
    # print("noise mod: {}".format(noise_mod_val))
    for i in range(len(labels)):
        # if labels[i] >= 0:
            stddev_t = np.std(data[i])
            rand_t = np.random.rand(*data[i].shape)
            rand_t = rand_t - 0.5
            to_add_t = rand_t * stddev_t / noise_mod_val
            data_t = data[i] + to_add_t
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.concatenate(new_data)[:,None,:,:]
    new_labels = np.array(new_labels)

    return np.concatenate((data_ori, new_data_ar)), np.concatenate((labels_ori, new_labels))


def data_add(data_ori, labels_ori, ratio=1.0):
    
    data, labels = resamples(data_ori, labels_ori, ratio)

    new_data = []
    new_labels = []
    for i in range(len(np.unique(labels_ori))):
        ind = np.where(labels_ori==i)[0]
        p_ind = np.random.permutation(ind)
        new_data.append((data[ind]+data[p_ind])/2)
        new_labels.append(labels[ind])
        
    new_data_ar = np.concatenate(new_data)
    new_labels = np.concatenate(new_labels)
        
    return np.concatenate((data_ori, new_data_ar)), np.concatenate((labels_ori, new_labels))


def data_mult_f(data_ori, labels_ori, mult_mod=0.05, ratio=1.0):
    data, labels = resamples(data_ori, labels_ori, ratio)
    new_data = []
    new_labels = []
    # mult_mod = 0.05
    # print("mult mod: {}".format(mult_mod))
    for i in range(len(labels)):
        # if labels[i] >= 0:
            # print(data[i])
            data_t = data[i] * (1 + mult_mod)
            new_data.append(data_t)
            new_labels.append(labels[i])

    for i in range(len(labels)):
        # if labels[i] >= 0:
            data_t = data[i] * (1 - mult_mod)
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.concatenate(new_data)[:,None,:,:]
    new_labels = np.array(new_labels)

    return np.concatenate((data_ori, new_data_ar)), np.concatenate((labels_ori, new_labels))


def data_neg_f(data_ori, labels_ori, ratio=1.0):
    # Returns: data double the size of the input over time, with new data
    # being a reflection along the amplitude
    data, labels = resamples(data_ori, labels_ori, ratio)
    new_data = []
    new_labels = []
    for i in range(len(labels)):
        # if labels[i] >= 0:
            data_t = -1 * data[i]
            data_t = data_t - np.min(data_t)
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.concatenate(new_data)[:,None,:,:]
    new_labels = np.array(new_labels)

    return np.concatenate((data_ori, new_data_ar)), np.concatenate((labels_ori, new_labels))



def freq_mod_f(data_ori, labels_ori, dt=1 / 250, freq_mod=0.2, ratio=1.0):
    
    data, labels = resamples(data_ori, labels_ori, ratio)
    new_data = []
    new_labels = []
    # print(data.shape)
    # freq_mod = 0.2
    # print("freq mod: {}".format(freq_mod))
    for i in range(len(labels)):
        # if labels[i] >= 0:
            low_shift = freq_shift(data[i], -freq_mod, dt=dt, num_channels=data.shape[2])
            new_data.append(low_shift)
            new_labels.append(labels[i])

    for i in range(len(labels)):
        # if labels[i] >= 0:
            high_shift = freq_shift(data[i], freq_mod, dt=dt, num_channels=data.shape[2])
            new_data.append(high_shift)
            new_labels.append(labels[i])

    new_data_ar = np.concatenate(new_data)[:,None,:,:]
    new_labels = np.array(new_labels)

    return np.concatenate((data_ori, new_data_ar)), np.concatenate((labels_ori, new_labels))


def freq_shift(x, f_shift, dt=1 / 250, num_channels=22):
    shifted_sig = np.zeros((x.shape))
    len_x = x.shape[-1]
    padding_len = 2 ** nextpow2(len_x)
    padding = np.zeros((1,num_channels,padding_len - len_x))
    with_padding = np.concatenate((x, padding),axis=-1)
    hilb_T = hilbert(with_padding, axis=-1)
    t = np.arange(0, padding_len)
    shift_func = np.exp(2j * np.pi * f_shift * dt * t)
    for i in range(num_channels):
        shifted_sig[0, i, :] = (hilb_T[0, i, :] * shift_func)[:len_x].real

    return shifted_sig


def nextpow2(x):
    return int(np.ceil(np.log2(np.abs(x))))




def rand(x_ori, y_ori, eps, ratio=1.0):
    
    # if ratio != 1.0:
    #     xs = []
    #     ys = []
    #     for i in range(np.unique(y)):
    #         ind = np.where(y==i)[0]
    #         ind = np.random.choice(ind,round(ratio*len(ind)),replace=True)
    #         xs.append(x[ind])
    #         ys.append(y[ind])
    #     x = np.concatenate(xs)
    #     y = np.concatenate(ys)
    
    x, y = resamples(x_ori, y_ori, ratio)
    
    cha_std = x.std(axis=-1)[:,:,:,None]
    delta = np.random.uniform(-eps, eps, x.shape) * cha_std
    
    return np.concatenate((x_ori,x+delta)), np.concatenate((y_ori,y))
    