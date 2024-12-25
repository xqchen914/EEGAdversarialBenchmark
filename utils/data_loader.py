from re import T
from typing import Optional
from scipy import signal
from scipy.signal import resample
import scipy.io as scio
import numpy as np
import os

current_dir = os.path.dirname(os.path.realpath(__file__))

os.chdir(current_dir)

from utils.data_align import centroid_align


def split(x, y, ratio=0.8, shuffle=True):
    idx = np.arange(len(x))
    if shuffle:
        idx = np.random.permutation(idx)
    train_size = int(len(x) * ratio)

    return x[idx[:train_size]], y[idx[:train_size]], x[idx[train_size:]], y[
        idx[train_size:]]




def MI2014001Load(id: int, setup: Optional[str] = 'cross', p: Optional[float] = 0.8, online=0, noea=0):
    # if  ea == 'sess':
    if not online:
        data_path = '/mnt/data1/cxq/processedMI2014001_4s_sea/'
    else:
        data_path = '/mnt/data1/cxq/processedMI2014001/'
        
    if noea:
        data_path = '/mnt/data1/cxq/processedMI2014001/'
        
    x_train, y_train, x_test, y_test = [], [], [], []
    
    if setup == 'cross_sess':
        
        data = scio.loadmat(data_path + f's{id}E.mat')
        x, y = data['x'], data['y']
        y = np.squeeze(np.array(y).flatten())
        
        x_train = x
        y_train = y
        
        data = scio.loadmat(data_path + f's{id}T.mat')
        x2, y2 = data['x'], data['y']
        y2 = np.squeeze(np.array(y2).flatten())
        
        x_test = x2
        y_test = y2
        
    elif setup == 'within_sess':
        
        data = scio.loadmat(data_path + f's{id}E.mat')
        x, y = data['x'], data['y']
        y = np.squeeze(np.array(y).flatten())
        
        x_class = []
        y_class = []
        for c in range(len(np.unique(y))):
            x_class.append(x[y==c])
            y_class.append(y[y==c])
        
        # num = 5
        for c in range(len(np.unique(y))):
            num = round(p*len(x_class[c]))
            x_train.append(x_class[c][:num])
            x_test.append(x_class[c][num:])
            y_train.append(y_class[c][:num])
            y_test.append(y_class[c][num:])

        x_train = np.concatenate(x_train)
        x_test = np.concatenate(x_test)
        
        y_train = np.concatenate(y_train)
        y_test = np.concatenate(y_test)
    
    if online:
        # perform online EA
        ca = centroid_align(center_type='euclid', cov_type='lwf')
        ca.fit(np.squeeze(x_train))
        
        cov_new, x_train = ca.transform(np.squeeze(x_train))
        x_train = x_train[:,None,:,:]
        
        cov_new, x_test = ca.transform(np.squeeze(x_test))
        x_test = x_test[:,None,:,:]
        

    return x_train, y_train, x_test, y_test

ca = centroid_align(center_type='euclid', cov_type='lwf')

def ssvep(id: int, setup: Optional[str] = 'within_sess', p: Optional[int] = 0.66, online=0, noea=0):
    
    x_train, y_train, x_test, y_test = [], [], [], []
    
    # data_path = '/mnt/data1/cxq/Dial/'
    
    if not online:
        data_path = '/mnt/data1/cxq/processedssvepea/'
    else:
        data_path = '/mnt/data1/cxq/processedssvep/'
        
    if noea:
        data_path = '/mnt/data1/cxq/processedssvep/'
    
    if setup == 'within_sess':
        data = scio.loadmat(data_path + f's{id}.mat')
        x, y = data['x'], data['y']
        y = np.squeeze(np.array(y).flatten())
        
        x_class = []
        y_class = []
        for c in range(len(np.unique(y))):
            x_class.append(x[y==c])
            y_class.append(y[y==c])
        
        # num = 5
        for c in range(len(np.unique(y))):
            num = round(p*len(x_class[c]))
            x_train.append(x_class[c][:num])
            x_test.append(x_class[c][num:])
            y_train.append(y_class[c][:num])
            y_test.append(y_class[c][num:])


        x_train = np.concatenate(x_train)
        x_test = np.concatenate(x_test)
        
        y_train = np.concatenate(y_train)
        y_test = np.concatenate(y_test)

    if online:
        ca = centroid_align(center_type='euclid', cov_type='lwf')
        ca.fit(np.squeeze(x_train))
        cov_new, x_train = ca.transform(np.squeeze(x_train))
        x_train = x_train[:,None,:,:]
        
        cov_new, x_test = ca.transform(np.squeeze(x_test))
        x_test = x_test[:,None,:,:]
        
    return x_train, y_train, x_test, y_test



def MI2014004Load(id: int, setup: Optional[str] = 'cross', p: Optional[int] = 0.8, online=0, noea=0):
    # if  ea == 'sess':
    if not online:
        data_path = '/mnt/data1/cxq/processedMI2014004sea/'
    else:
        data_path = '/mnt/data1/cxq/processedMI2014004/'
    
    if noea:
        data_path = '/mnt/data1/cxq/processedMI2014004/'
        
    x_train, y_train, x_test, y_test = [], [], [], []
    
    if setup == 'cross_sess':
        
        data = scio.loadmat(data_path + f's{id}e_0.mat')
        x, y = data['x'], data['y']
        y = np.squeeze(np.array(y).flatten())
        
        x_train = x
        y_train = y
        
        data = scio.loadmat(data_path + f's{id}e_1.mat')
        x2, y2 = data['x'], data['y']
        y2 = np.squeeze(np.array(y2).flatten())

        data = scio.loadmat(data_path + f's{id}t_0.mat')
        x3, y3 = data['x'], data['y']
        y3 = np.squeeze(np.array(y3).flatten())
        
        data = scio.loadmat(data_path + f's{id}t_1.mat')
        x4, y4 = data['x'], data['y']
        y4 = np.squeeze(np.array(y4).flatten())

        data = scio.loadmat(data_path + f's{id}t_2.mat')
        x5, y5 = data['x'], data['y']
        y5 = np.squeeze(np.array(y5).flatten())

        x_test = np.concatenate((x2,x3,x4,x5))
        y_test = np.concatenate((y2,y3,y4,y5))
        
    elif setup == 'within_sess':
        data = scio.loadmat(data_path + f's{id}e_0.mat')
        x, y = data['x'], data['y']
        y = np.squeeze(np.array(y).flatten())
        
        x_class = []
        y_class = []
        for c in range(len(np.unique(y))):
            x_class.append(x[y==c])
            y_class.append(y[y==c])
        
        # num = 5
        for c in range(len(np.unique(y))):
            num = round(p*len(x_class[c]))
            x_train.append(x_class[c][:num])
            x_test.append(x_class[c][num:])
            y_train.append(y_class[c][:num])
            y_test.append(y_class[c][num:])


        x_train = np.concatenate(x_train)
        x_test = np.concatenate(x_test)
        
        y_train = np.concatenate(y_train)
        y_test = np.concatenate(y_test)

    
    if online:
        ca = centroid_align(center_type='euclid', cov_type='lwf')
        ca.fit(np.squeeze(x_train))
        cov_new, x_train = ca.transform(np.squeeze(x_train))
        x_train = x_train[:,None,:,:]
        
        cov_new, x_test = ca.transform(np.squeeze(x_test))
        x_test = x_test[:,None,:,:]
        

    return x_train, y_train, x_test, y_test

def epflLoad(id: int, setup: Optional[str] = 'within', p: Optional[float] = 0.8, online=0, noea=0):
    if not online:
        data_path = '/mnt/data1/cxq/processedepflsea/'
    else:
        data_path = '/mnt/data1/cxq/processedepfl/'
    
    if noea:
        data_path = '/mnt/data1/cxq/processedepfl/'

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    if setup == 'within_sess':

        data = scio.loadmat(data_path + f's{id}_0.mat')
        x, y = data['x'], data['y']
        y = np.squeeze(np.array(y).flatten())
        
        x_class = []
        y_class = []
        for c in range(len(np.unique(y))):
            x_class.append(x[y==c])
            y_class.append(y[y==c])
        
        for c in range(len(np.unique(y))):
            num = round(p*len(x_class[c]))
            x_train.append(x_class[c][:num])
            y_train.append(y_class[c][:num])  
            x_test.append(x_class[c][num:])
            y_test.append(y_class[c][num:])

        x_train = np.concatenate(x_train)
        x_test = np.concatenate(x_test)
        
        y_train = np.concatenate(y_train)
        y_test = np.concatenate(y_test)

        
    elif setup == 'cross_sess':

        data = scio.loadmat(data_path + f's{id}_0.mat')
        x1, y1 = data['x'], data['y']
        y1 = np.squeeze(np.array(y1).flatten())
 
        x_train = x1
        y_train = y1
               
        data = scio.loadmat(data_path + f's{id}_1.mat')
        x2, y2 = data['x'], data['y']
        y2 = np.squeeze(np.array(y2).flatten())

        data = scio.loadmat(data_path + f's{id}_2.mat')
        x3, y3 = data['x'], data['y']
        y3 = np.squeeze(np.array(y3).flatten())

        data = scio.loadmat(data_path + f's{id}_3.mat')
        x4, y4 = data['x'], data['y']
        y4 = np.squeeze(np.array(y4).flatten())

        x_test = np.concatenate((x2,x3,x4))
        y_test = np.concatenate((y2,y3,y4))
        

    if online:
        ca = centroid_align(center_type='euclid', cov_type='lwf')
        ca.fit(np.squeeze(x_train))
        cov_new, x_train = ca.transform(np.squeeze(x_train))
        x_train = x_train[:,None,:,:]
        
        cov_new, x_test = ca.transform(np.squeeze(x_test))
        x_test = x_test[:,None,:,:]
        
    return x_train, y_train, x_test, y_test



def P3002014009Load(id: int, setup: Optional[str] = 'within', p: Optional[float] = 0.8, online=0, noea=0):
    if not online:
        data_path = '/mnt/data1/cxq/data/processed2014009sea/'
    else:
        data_path = '/mnt/data1/cxq/data/processed2014009/'
    
    if noea:
        data_path = '/mnt/data1/cxq/data/processed2014009/'

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    if setup == 'within_sess':

        data = scio.loadmat(data_path + f's{id}_0.mat')
        x, y = data['x'], data['y']
        y = np.squeeze(np.array(y).flatten())
        
        x_class = []
        y_class = []
        for c in range(len(np.unique(y))):
            x_class.append(x[y==c])
            y_class.append(y[y==c])
        
        for c in range(len(np.unique(y))):
            num = round(p*len(x_class[c]))
            x_train.append(x_class[c][:num])
            y_train.append(y_class[c][:num])  
            x_test.append(x_class[c][num:])
            y_test.append(y_class[c][num:])

        x_train = np.concatenate(x_train)
        x_test = np.concatenate(x_test)
        
        y_train = np.concatenate(y_train)
        y_test = np.concatenate(y_test)

        
    elif setup == 'cross_sess':

        data = scio.loadmat(data_path + f's{id}_0.mat')
        x1, y1 = data['x'], data['y']
        y1 = np.squeeze(np.array(y1).flatten())
 
        x_train = x1
        y_train = y1
               
        data = scio.loadmat(data_path + f's{id}_1.mat')
        x2, y2 = data['x'], data['y']
        y2 = np.squeeze(np.array(y2).flatten())

        data = scio.loadmat(data_path + f's{id}_2.mat')
        x3, y3 = data['x'], data['y']
        y3 = np.squeeze(np.array(y3).flatten())


        x_test = np.concatenate((x2,x3))
        y_test = np.concatenate((y2,y3))
        

    else:
        raise Exception('No such Experiment setup.')

    if online:
        ca = centroid_align(center_type='euclid', cov_type='lwf')
        ca.fit(np.squeeze(x_train))
        cov_new, x_train = ca.transform(np.squeeze(x_train))
        x_train = x_train[:,None,:,:]
        
        cov_new, x_test = ca.transform(np.squeeze(x_test))
        x_test = x_test[:,None,:,:]
        
    return x_train, y_train, x_test, y_test





if __name__ == '__main__':
    x_train, y_train, x_test, y_test = ssvepbench(id=1,setup='within_sess', online=0, p=0.5)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print(np.unique(y_test))