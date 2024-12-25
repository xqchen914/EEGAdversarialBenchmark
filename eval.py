import os
import logging
import torch
import argparse
import attack_lib
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from model import EEGNet, DeepConvNet, ShallowConvNet
from utils.data_loader import MI2014001Load, epflLoad, split, MI2014004Load, P3002014009Load, ssvep
from utils.pytorch_utils import bca_score, CustomTensorDataset, init_weights, print_args, seed, adjust_learning_rate, weight_for_balanced_classes
from autoattack import AutoAttack
from pandas import Series,DataFrame
import pandas as pd
from utils.pytorch_utils import GeneralTorchModel
import defense_lib
import gc
from fab import Square
import ast

def str_to_list(arg):
    return ast.literal_eval(arg)

def generate_new_labels(labels, num_classes):

    rand_shift = torch.randint(1, num_classes, labels.size())
    new_labels = (labels + rand_shift) % num_classes
    
    return new_labels

def train(x: torch.Tensor, y: torch.Tensor, x_test: torch.Tensor,
          y_test: torch.Tensor, model_save_path: str, args):
    n_class = len(np.unique(y_test.numpy()))
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
                             noise_std=args.noise_std,
                             SAP_frac=args.SAP_frac).to(args.device)
    elif args.model == 'DeepCNN':
        model = DeepConvNet(n_classes=len(np.unique(y.numpy())),
                                  Chans=x.shape[2],
                                  Samples=x.shape[3],
                                  dropoutRate=0.5,
                                  noise_std=args.noise_std,
                                  SAP_frac=args.SAP_frac).to(args.device)
    elif args.model == 'ShadowCNN':
        model = ShallowConvNet(n_classes=len(np.unique(y.numpy())),
                                     Chans=x.shape[2],
                                     Samples=x.shape[3],
                                     dropoutRate=0.5,
                                     noise_std=args.noise_std,
                                     SAP_frac=args.SAP_frac).to(args.device)
    else:
        raise 'No such model!'
    
    criterion = nn.CrossEntropyLoss().to(args.device)

    test_loader = DataLoader(dataset=TensorDataset(x_test, y_test),
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=1,
                             drop_last=False)


    model.load_state_dict(
        torch.load(model_save_path + '/model.pt',
                   map_location=lambda storage, loc: storage))
    model.to(device=args.device)

    model.eval()
    _, test_acc, test_bca = eval(model, criterion, test_loader)
    logging.info(f'test acc: {test_acc}, test bca: {test_bca}')
    recorder_bca[round(r*(len([args.attack])*len(args.epss)+1)),t] = test_bca
    recorder_acc[round(r*(len([args.attack])*len(args.epss)+1)),t] = test_acc

    if args.attack == 'sub':
         sub_model = attack_lib.TrainSub(model,
                                    x_sub=x,
                                    y_sub=y,
                                    aug_repeat=2)
         
    # attack using PGD and FGSM [0.01, 0.03, 0.05]
    adv_accs = []
    adv_bcas = []
    if hasattr(model, 'enhanced_block'):
        model.train()
    if args.target: adv_label = generate_new_labels(y_test, n_class)
    else: adv_label = y_test 
    for i in range(len(args.epss)):
        
        if args.attack == 'AA':
            adversary = AutoAttack(model, norm='Linf', seed=r, eps=args.epss[i],
                version='',attacks_to_run=['apgd-ce'],device=args.device,n_iter=10)
            adv_x = adversary.run_standard_evaluation_individual(x_test,
                adv_label, bs=test_loader.batch_size)
            adv_x = adv_x['apgd-ce']
        elif args.attack == 'FGSM':
            adv_x = attack_lib.FGSM_cha(model,
                                    x_test,
                                    adv_label,
                                    eps=args.epss[i],
                                    distance=args.distance,
                                    target=args.target)
        elif args.attack == 'PGD':
            adv_x = attack_lib.PGD_cha(model,
                                   x_test,
                                   adv_label,
                                   eps=args.epss[i],
                                   alpha=args.epss[i] / 10,
                                   steps=20,
                                   target=args.target)
        elif args.attack == 'Rays':
            torch_model = GeneralTorchModel(model, n_class=len(np.unique(y.numpy())), im_mean=None, im_std=None)
            adversary = attack_lib.RayS(torch_model, epsilon=args.epss[i], order=np.inf if args.distance == 'inf' else 2)
            adv_x = adversary.attack_batch(x_test, y_test, target=adv_label if args.target else None, query_limit=2000)
        elif args.attack == 'sub':
            adv_x = attack_lib.FGSM_cha(sub_model,
                                    x_test,
                                    adv_label,
                                    eps=args.epss[i],
                                    distance=args.distance,
                                    target=args.target)
        elif args.attack == 'square':
            adversary = Square(model,norm=args.distance,eps=args.epss[i],seed=r)
            adversary.targeted = args.target
            adv_x = adversary.attack_batch(x_test, adv_label)

        
        adv_loader = DataLoader(dataset=TensorDataset(adv_x.cpu(), adv_label),
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=1,
                                drop_last=False)
        
        if args.ensemble_eval:
            _, adv_acc, adv_bca = ensemble_eval(model, criterion, adv_loader)
        elif args.ensemble_transform:
            _, adv_acc, adv_bca = eval_transform(model, criterion, adv_x.cpu(), adv_label)
        else:
            _, adv_acc, adv_bca = eval(model, criterion, adv_loader)
        
        
        recorder_bca[round(r*(len([args.attack])*len(args.epss)+1))+i+1,t] = adv_bca
        recorder_acc[round(r*(len([args.attack])*len(args.epss)+1))+i+1,t] = adv_acc

        adv_accs.append(adv_acc)
        adv_bcas.append(adv_bca)
        logging.info(f'{args.attack} eps: {args.epss[i]}, acc: {adv_acc}, bca: {adv_bca}')
        

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

def ensemble_eval(model: nn.Module, criterion: nn.Module,
                  data_loader: DataLoader):
    loss, correct = 0., 0
    labels, preds = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(args.device), y.to(args.device)
            probs = torch.zeros(20, len(x), len(np.unique(data_loader.dataset.tensors[1].numpy())))
            for repeat in range(20):
                out = model(x)
                pred = nn.Softmax(dim=1)(out).cpu()
                loss += criterion(out, y).item()
                probs[repeat, :, :] = pred
            probs = probs.mean(dim=0).argmax(dim=1)
            correct += probs.eq(y.cpu().view_as(probs)).sum().item()
            labels.extend(y.cpu().tolist())
            preds.extend(probs.tolist())
    loss /= (20 * len(data_loader.dataset))
    acc = correct / len(data_loader.dataset)
    bca = bca_score(labels, preds)

    return loss, acc, bca

def eval_transform(model: nn.Module, criterion: nn.Module, x: torch.Tensor,
                   y: torch.Tensor):
    transform = defense_lib.get_transform(transform_name='guassian',
                                        gs_am=0.1,
                                        scale_factor=0.5,
                                        shift_scale=0.5,
                                        shuffle_rate=0.2,
                                        max_ratio=1.5,
                                        n_transform=2)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial training')
    parser.add_argument('--gpu_id', type=str, default='3')
    parser.add_argument('--model', type=str, default='DeepCNN')
    parser.add_argument('--dataset', type=str, default='EPFL')
    parser.add_argument('--setup', type=str, default='cross_sess')

    parser.add_argument('--batch_size', type=int, default=128)
    
    parser.add_argument('--epss', type=str_to_list, default="[0.01, 0.05, 0.1]")# evaluated attack perturbation amplitudes
    parser.add_argument('--online', type=int, default=0)
    parser.add_argument('--percent', type=float, default=0.33)
    
    parser.add_argument('--attack', type=str, default='FGSM')
    parser.add_argument('--distance', type=str, default='inf')# inf l2
    parser.add_argument('--defense', type=str, default='SEAT_0.03_mixup')
    
    parser.add_argument('--ensemble_eval', type=int, default=0)
    parser.add_argument('--ensemble_transform', type=int, default=0)
    parser.add_argument('--target', type=int, default=1)
    
    parser.add_argument('--repeat', type=int, default=5)
    
    parser.add_argument('--noise_std', type=float, default=None)
    parser.add_argument('--SAP_frac', type=float, default=None)
    parser.add_argument('--pruning_rate', type=float, default=None)

    args = parser.parse_args()
    
    args.noise_std = 0.2 if 'RSE' in args.defense else None
    args.SAP_frac = float(args.defense[-3:]) if 'SAP' in args.defense else None
    
    args.ensemble_eval = 1 if 'RSE' in args.defense else 0
    args.ensemble_transform = 1 if 'input_transform' in args.defense else 0
    
    if args.attack == 'FGSM':
        args.epss = [0.01,0.03,0.05]
    else:
        args.epss = [0.01,0.05,0.1]
    
    if args.dataset == 'EPFL':
        args.batch_size = 128
    else:
        args.batch_size = 32

    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    subject_num_dict = {'MI4C': 9, 'MI2C': 9, 'ERP': 10, 'EPFL': 8,'ssvep': 10}

    # ========================savings=======================
    load = args.defense
    if 'SAP' in args.defense:
        load=args.defense[:-8]
    model_path = f'/mnt/data1/cxq/bci_adv_defense_model/{load}/{args.dataset}/{args.model}/{args.setup}'
    if args.online:
        model_path = f'/mnt/data1/cxq/bci_adv_defense_model/online_{load}/{args.dataset}/{args.model}/{args.setup}'


    eps_str = ''.join([str(i) for i in args.epss])
    log_path = f'/mnt/data4/cxq/bci_adv_defense-main_copy/bci_adv_defense-main/result_eval_{args.distance}_{args.target}/log/{args.defense}_{args.attack}_{eps_str}'
    if args.online:
        log_path = f'/mnt/data4/cxq/bci_adv_defense-main_copy/bci_adv_defense-main/result_eval_{args.distance}_{args.target}/log/online_{args.defense}_{args.attack}_{eps_str}'

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, f'{args.setup}_{args.dataset}_{args.model}.log')
    
    excel_path = f'/mnt/data4/cxq/bci_adv_defense-main_copy/bci_adv_defense-main/result_eval_{args.distance}_{args.target}/excel/{args.defense}_{args.attack}_{eps_str}'
    if args.online:
        excel_path = f'/mnt/data4/cxq/bci_adv_defense-main_copy/bci_adv_defense-main/result_eval_{args.distance}_{args.target}/excel/online_{args.defense}_{args.attack}_{eps_str}'
    if not os.path.exists(excel_path):
        os.makedirs(excel_path)
    excel_name = os.path.join(excel_path,f'{args.setup}_{args.dataset}_{args.model}.xlsx')


    recorder_bca = np.zeros(((len([args.attack])*len(args.epss)+1)*args.repeat,subject_num_dict[args.dataset]))
    recorder_acc = np.zeros(((len([args.attack])*len(args.epss)+1)*args.repeat,subject_num_dict[args.dataset]))
    
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
            # build model save path
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
   
            if args.dataset == 'ssvep':
                x_class = []
                y_class = []
                x_val = []
                y_val = []
                for c in range(len(np.unique(y_train))):
                    x_class.append(x_train[y_train==c])
                    y_class.append(y_train[y_train==c])
                for c in range(len(np.unique(y_train))):
                    num = 1
                    x_val.append(x_class[c][:num])
                    y_val.append(y_class[c][:num])  
                x_val = np.concatenate(x_val)
                y_val = np.concatenate(y_val)
            else:
                x_train, y_train, x_val, y_val = split(x_train,
                                                    y_train,
                                                    ratio=0.75)
                
            logging.info(f'train: {x_train.shape},{x_train.mean()},{x_train.std()},{np.bincount(y_train.astype(int))}, test: {x_test.shape},{x_test.mean()},{x_test.std()},{np.bincount(y_test.astype(int))}')
            x_val = Variable(
                torch.from_numpy(x_val).type(torch.FloatTensor))
            y_val = Variable(
                torch.from_numpy(y_val).type(torch.LongTensor))
            x_test = Variable(torch.from_numpy(x_test).type(torch.FloatTensor))
            y_test = Variable(torch.from_numpy(y_test).type(torch.LongTensor))

            test_acc, test_bca, adv_acc, adv_bca = train(x_val, y_val, x_test, y_test,
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

    gc.collect()
    torch.cuda.empty_cache()
    
    recorder_bca = DataFrame(recorder_bca,
               columns = [f's{g}' for g in range(subject_num_dict[args.dataset])],
               index = pd.MultiIndex.from_product([[f'repeat_{g}' for g in range(args.repeat)],
                                                   ['normal']+[i+'_'+str(j) for i in [args.defense] for j in args.epss]]))
    recorder_acc = DataFrame(recorder_acc,
            columns = [f's{g}' for g in range(subject_num_dict[args.dataset])],
            index = pd.MultiIndex.from_product([[f'repeat_{g}' for g in range(args.repeat)],
                                                ['normal']+[i+'_'+str(j) for i in [args.defense] for j in args.epss]]))

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
    
    for i in range(len(args.epss)):
        logging.info(
            f'Repeat mean adv acc | bca: ({args.defense} {args.epss[i]}): {round(np.mean(r_adv_acc[:, i]), 4)}-{round(np.std(r_adv_acc[:, i]), 4)} | {round(np.mean(r_adv_bca[:, i]), 4)}-{round(np.std(r_adv_bca[:, i]), 4)}'
        )
