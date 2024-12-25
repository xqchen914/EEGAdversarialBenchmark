import copy
import os
import logging
import torch
import argparse
import attack_lib
import defense_lib
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from model import EEGNet, DeepConvNet, ShallowConvNet
from utils.data_loader import MI2014001Load, epflLoad, split, MI2014004Load, P3002014009Load, ssvep
from utils.pytorch_utils import bca_score, init_weights, print_args, seed, CustomTensorDataset, adjust_learning_rate, weight_for_balanced_classes
from autoattack import AutoAttack
from pandas import Series,DataFrame
import pandas as pd

def train(x: torch.Tensor, y: torch.Tensor, x_test: torch.Tensor,
          y_test: torch.Tensor, model_save_path: str, args):
    # initialize the model
    if args.model == 'EEGNet':
        model = EEGNet(n_classes=len(np.unique(y.numpy())),
                       Chans=x.shape[2],
                       Samples=x.shape[3],
                       kernLenght=64,
                       F1=4,
                       D=2,
                       F2=8,
                       dropoutRate=0.25,
                       norm_rate=0.25).to(args.device)
    elif args.model == 'DeepCNN':
        model = DeepConvNet(n_classes=len(np.unique(y.numpy())),
                            Chans=x.shape[2],
                            Samples=x.shape[3],
                            dropoutRate=0.5).to(args.device)
    elif args.model == 'ShadowCNN':
        model = ShallowConvNet(n_classes=len(np.unique(y.numpy())),
                               Chans=x.shape[2],
                               Samples=x.shape[3],
                               dropoutRate=0.5).to(args.device)
    else:
        raise 'No such model!'
    model.apply(init_weights)

    # trainable parameters
    params = []
    for _, v in model.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    optimizer = optim.Adam(params, weight_decay=5e-4)
    criterion_cal = nn.CrossEntropyLoss().to(args.device)

    # input transform
    transform = defense_lib.get_transform(transform_name=args.transform,
                                          gs_am=0.1,
                                          scale_factor=0.5,
                                          shift_scale=0.5,
                                          shuffle_rate=0.2,
                                          max_ratio=1.5,
                                          n_transform=2)

    for epoch in range(args.epochs):
        # data loader
        # for unbalanced dataset, create a weighted sampler
        sample_weights = weight_for_balanced_classes(y)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            sample_weights, len(sample_weights))
        train_loader = DataLoader(dataset=CustomTensorDataset((x, y),
                                                              transform),
                                  batch_size=args.batch_size,
                                  sampler=sampler,
                                  num_workers=1,
                                  drop_last=False)
        # model training
        adjust_learning_rate(optimizer=optimizer,
                             epoch=epoch + 1,
                             learning_rate=args.lr)
        model.train()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)


            model.train()
            optimizer.zero_grad()

            out = model(batch_x)
            loss = criterion_cal(out, batch_y)

            loss.backward()
            optimizer.step()

            model.MaxNormConstraint()

        # if (epoch + 1) % 20 == 0 or (epoch + 1) == args.epochs:
        #     model.eval()
        #     train_loss, train_acc, train_bca = eval(model, criterion_cal, train_loader)
        #     test_loss, test_acc, test_bca = eval_transform(model, criterion_cal,
        #                                             x_test, y_test, transform)

        #     logging.info(
        #             'Epoch {}/{}: train_loss: {:.4f} train acc: {:.4f} train bca: {:.2f} | test_loss: {:.4f} test acc: {:.4f} test bca: {:.2f}'
        #             .format(epoch + 1, args.epochs, train_loss, train_acc, train_bca, test_loss,
        #                     test_acc, test_bca))

    torch.save(model.state_dict(), model_save_path + '/model.pt')

    model.eval()
    _, test_acc, test_bca = eval_transform(model, criterion_cal, x_test,
                                           y_test, transform)
    logging.info(f'test acc: {test_acc}, test bca: {test_bca}')
    recorder_bca[round(r*(len(['AA'])*len(args.epss)+1)),t] = test_bca
    recorder_acc[round(r*(len(['AA'])*len(args.epss)+1)),t] = test_acc


    adv_accs = []
    adv_bcas = []
    for i in range(3):
        adversary = AutoAttack(model, norm='Linf', seed=r, eps=args.epss[i],
            version='',attacks_to_run=['apgd-ce'],device=args.device,n_iter=10)
        adv_x = adversary.run_standard_evaluation_individual(x_test,
            y_test, bs=args.batch_size)
        adv_x = adv_x['apgd-ce']    
        _, adv_acc, adv_bca = eval_transform(model, criterion_cal, adv_x.cpu(),
                                       y_test, transform)
        recorder_bca[round(r*(len(['AA'])*len(args.epss)+1))+i+1,t] = adv_bca
        recorder_acc[round(r*(len(['AA'])*len(args.epss)+1))+i+1,t] = adv_acc

        adv_accs.append(adv_acc)
        adv_bcas.append(adv_bca)
        logging.info(f'AA eps: {args.epss[i]}, acc: {adv_acc}, bca: {adv_bca}')
        

    return test_acc, test_bca, adv_accs, adv_bcas



def eval(model: nn.Module, criterion: nn.Module, data_loader: DataLoader):
    loss, correct = 0., 0
    labels, preds = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(args.device), y.to(args.device)
            out = model(x)
            pred = nn.Softmax(dim=1)(out).cpu().argmax(dim=1)
            loss += criterion(out, y).item()
            correct += pred.eq(y.cpu().view_as(pred)).sum().item()
            labels.extend(y.cpu().tolist())
            preds.extend(pred.tolist())
    loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)
    bca = bca_score(labels, preds)

    return loss, acc, bca


def eval_transform(model: nn.Module, criterion: nn.Module, x: torch.Tensor,
                   y: torch.Tensor, transform):
    loss, correct = 0., 0
    with torch.no_grad():
        probs = None
        for repeat in range(10):
            if probs is None:
                probs = torch.zeros(10, len(x), len(np.unique(y.numpy())))
            dataloader = DataLoader(dataset=CustomTensorDataset((x, y),
                                                                transform),
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=1,
                                    drop_last=False)
            preds = None
            for batch_x, batch_y in dataloader:
                # plot_raw(x[0].numpy().squeeze(), batch_x[0].numpy().squeeze(), file_name='fig/shift', is_norm=True)
                # exit()
                batch_x, batch_y = batch_x.to(args.device), batch_y.to(
                    args.device)
                out = model(batch_x)
                pred = nn.Softmax(dim=1)(out).cpu()
                loss += criterion(out, batch_y).item()

                preds = pred if preds is None else torch.cat(
                    (preds, pred), dim=0)

            probs[repeat, :, :] = preds
    probs = probs.mean(dim=0).argmax(dim=1)
    correct = probs.eq(y.cpu().view_as(probs)).sum().item()
    loss /= (len(x) * 10)
    acc = correct / len(x)
    bca = bca_score(y.cpu().tolist(), probs.tolist())

    return loss, acc, bca


# 'guassian', 'sampling', 'shifting', 'shuffling', 'amchange', 'random'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input Transform')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--model', type=str, default='ShadowCNN')
    parser.add_argument('--dataset', type=str, default='MI4C')
    parser.add_argument('--setup', type=str, default='cross')
    
    parser.add_argument('--transform', type=str, default='guassian')

    parser.add_argument('--epss', nargs='+',type=float,default=[0.01,0.03,0.05])#[0.001,0.003,0.005,0.01,0.03,0.05]
    parser.add_argument('--online', type=int, default=1)
    parser.add_argument('--percent', type=float, default=0.33)
   
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0.01)
    

    args = parser.parse_args()

    if args.dataset == 'EPFL':
        args.batch_size = 128
    else:
        args.batch_size = 32
    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    subject_num_dict = {'MI4C': 9, 'MI2C': 9, 'ERP': 10, 'EPFL': 8,'ssvep': 10}

    # ========================savings=======================
    model_path = f'/mnt/data1/cxq/advbenchmark/bci_adv_defense_model/input_transform/{args.dataset}/{args.model}/{args.setup}'
    if args.online:
        model_path = f'/mnt/data1/cxq/advbenchmark/bci_adv_defense_model/online_input_transform/{args.dataset}/{args.model}/{args.setup}'

    log_path = f'/home/xqchen/bci_adv_defense-main_copy/log/input_transform'
    if args.online:
        log_path = f'/home/xqchen/bci_adv_defense-main_copy/log/online_input_transform'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, f'{args.setup}_{args.dataset}_{args.model}.log')
    
    
    excel_path = f'/home/xqchen/bci_adv_defense-main_copy/excel/input_transform'
    if args.online:
        excel_path = f'/home/xqchen/bci_adv_defense-main_copy/excel/online_input_transform'
    if not os.path.exists(excel_path):
        os.makedirs(excel_path)
    excel_name = os.path.join(excel_path,f'{args.setup}_{args.dataset}_{args.model}.xlsx')


    recorder_bca = np.zeros(((len(['AA'])*len(args.epss)+1)*args.repeat,subject_num_dict[args.dataset]))
    recorder_acc = np.zeros(((len(['AA'])*len(args.epss)+1)*args.repeat,subject_num_dict[args.dataset]))
      
    # ========================logging========================
    # logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # print logging
    print_log = logging.StreamHandler()
    logger.addHandler(print_log)
    # save logging
    save_log = logging.FileHandler(log_name, mode='w', encoding='utf8')
    logger.addHandler(save_log)

    logging.info(print_args(args) + '\n')

    r_acc, r_adv_acc = [], []
    r_bca, r_adv_bca = [], []
    for r in range(args.repeat):
        seed(r)
        # model train
        acc_list = []
        adv_acc_list = []
        bca_list = []
        adv_bca_list = []
        for t in range(subject_num_dict[args.dataset]):
            # build model path
            model_save_path = os.path.join(model_path, f'{r}/{t}')
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            logging.info(f'subject id: {t}')
            # load data
            if args.dataset == 'MI4C':
                x_train, y_train, x_test, y_test = MI2014001Load(id=t,
                                                            setup=args.setup, online=args.online, p=args.percent, noea=args.noea)
            elif args.dataset == 'EPFL':
                x_train, y_train, x_test, y_test = epflLoad(id=t,
                                                            setup=args.setup, online=args.online, p=args.percent, noea=args.noea)
            elif args.dataset == '2014004':
                x_train, y_train, x_test, y_test = MI2014004Load(id=t,
                                                            setup=args.setup, online=args.online, p=args.percent, noea=args.noea)
            elif args.dataset == '2014009':
                x_train, y_train, x_test, y_test = P3002014009Load(id=t,
                                                            setup=args.setup, online=args.online, p=args.percent, noea=args.noea)
            elif args.dataset == 'ssvep':
                x_train, y_train, x_test, y_test = ssvep(id=t,
                                                            setup=args.setup, online=args.online, p=args.percent, noea=args.noea)

                
            logging.info(f'train: {x_train.shape},{x_train.mean()},{x_train.std()},{np.bincount(y_train.astype(int))}, test: {x_test.shape},{x_test.mean()},{x_test.std()},{np.bincount(y_test.astype(int))}')
            x_train = Variable(
                torch.from_numpy(x_train).type(torch.FloatTensor))
            y_train = Variable(
                torch.from_numpy(y_train).type(torch.LongTensor))
            x_test = Variable(torch.from_numpy(x_test).type(torch.FloatTensor))
            y_test = Variable(torch.from_numpy(y_test).type(torch.LongTensor))

            test_acc, test_bca, adv_acc, adv_bca = train(x_train, y_train, x_test, y_test,
                                      model_save_path, args)
            acc_list.append(test_acc)
            bca_list.append(test_bca)
            adv_acc_list.append(adv_acc)
            adv_bca_list.append(adv_bca)

        r_acc.append(acc_list)
        r_bca.append(bca_list)
        r_adv_acc.append(adv_acc_list)
        r_adv_bca.append(adv_bca_list)

        logging.info(f'Repeat {r + 1}')
        logging.info(
            f'Mean acc: {np.mean(acc_list)} | Mean bca: {np.mean(bca_list)}')
        logging.info(
            f'Mean adv acc: {np.mean(adv_acc_list, axis=0)} | Mean adv bca: {np.mean(adv_bca_list, axis=0)}'
        )

    recorder_bca = DataFrame(recorder_bca,
               columns = [f's{g}' for g in range(subject_num_dict[args.dataset])],
               index = pd.MultiIndex.from_product([[f'repeat_{g}' for g in range(args.repeat)],
                                                   ['normal']+[i+str(j) for i in ['AA'] for j in args.epss]]))
    recorder_acc = DataFrame(recorder_acc,
            columns = [f's{g}' for g in range(subject_num_dict[args.dataset])],
            index = pd.MultiIndex.from_product([[f'repeat_{g}' for g in range(args.repeat)],
                                                ['normal']+[i+str(j) for i in ['AA'] for j in args.epss]]))

    with pd.ExcelWriter(excel_name) as writer:
        recorder_bca.to_excel(writer, sheet_name='bca')
        recorder_acc.to_excel(writer, sheet_name='acc')


    r_acc, r_adv_acc = np.mean(r_acc, 1), np.mean(r_adv_acc, axis=1)
    r_bca, r_adv_bca = np.mean(r_bca, 1), np.mean(r_adv_bca, axis=1)
    r_adv_acc, r_adv_bca = np.array(r_adv_acc), np.array(r_adv_bca)
    logging.info('*' * 50)
    logging.info(
        f'Repeat mean acc | bca: {round(np.mean(r_acc), 4)}-{round(np.std(r_acc), 4)} | {round(np.mean(r_bca), 4)}-{round(np.std(r_bca), 4)}'
    )
    
    for i in range(3):
        logging.info(
            f'Repeat mean adv acc | bca: (AA {args.epss[i]}): {round(np.mean(r_adv_acc[:, i]), 4)}-{round(np.std(r_adv_acc[:, i]), 4)} | {round(np.mean(r_adv_bca[:, i]), 4)}-{round(np.std(r_adv_bca[:, i]), 4)}'
        )
