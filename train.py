import argparse
import os
import time
import h5py

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from load_data import ORGDataset
from model import ResPointNetCls
from utils.logger import create_logger
from utils.metrics import classify_report, per_class_metric, \
    calculate_prec_recall_f1, best_swap, save_best_weights, calculate_average_metric
from utils.funcs import unify_path, makepath, fix_seed

# GPU check
device = torch.device("cuda:0")

def load_data():
    train_dataset = ORGDataset(
        logger=logger,
        num_fold=num_fold,
        k=args.k_fold,
        split='train')
    val_dataset = ORGDataset(
        logger=logger,
        num_fold=num_fold,
        k=args.k_fold,
        split='val')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size,
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size,
        shuffle=False)

    train_data_size = len(train_dataset)
    val_data_size = len(val_dataset)
    logger.info('The training data size is:{}'.format(train_data_size))
    logger.info('The validation data size is:{}'.format(val_data_size))
    num_classes = len(train_dataset.label_names)
    logger.info('The number of classes is:{}'.format(num_classes))

    # load label names
    train_label_names = train_dataset.obtain_label_names()
    val_label_names = val_dataset.obtain_label_names()
    assert train_label_names == val_label_names
    label_names = train_label_names
    label_names_h5 = h5py.File(os.path.join(args.out_path, 'label_names.h5'), 'w')
    label_names_h5['y_names'] = label_names
    logger.info('The label names are: {}'.format(str(label_names)))

    return train_loader, val_loader, label_names, num_classes, train_data_size, val_data_size


def train_val_net(net):
    """train and validation of the network"""
    time_start = time.time()
    train_num_batch = train_data_size / args.train_batch_size
    val_num_batch = val_data_size / args.val_batch_size
    # save training and validating process data
    train_loss_lst, val_loss_lst, train_acc_lst, val_acc_lst, \
    train_precision_lst, val_precision_lst, train_recall_lst, val_recall_lst, \
    train_f1_lst, val_f1_lst = [], [], [], [], [], [], [], [], [], []
    # save weights with best metrics
    best_acc, best_f1_mac = 0, 0
    best_acc_epoch, best_f1_epoch = 1, 1
    best_acc_wts, best_f1_wts = None, None
    best_acc_val_labels_lst, best_f1_val_labels_lst = [], []
    best_acc_val_pred_lst, best_f1_val_pred_lst = [], []
  
    for epoch in range(args.epoch):
        train_start_time = time.time()
        epoch += 1
        total_train_loss, total_val_loss = 0, 0
        train_labels_lst, train_predicted_lst = [], []
        total_train_correct, total_val_correct = 0, 0
        val_labels_lst, val_predicted_lst = [], []
        # 5-fold training
        for i, data in enumerate(train_loader, 0):
            points, label,region,ind_region = data  # points [B, N, 3]
            # points,label,ind_region = data
            label = label[:, 0]  # [B,1] rank2 to (, B) rank1
            points = points.transpose(2, 1)  # points [B, 3, N]
            #ind _region[B,105]
            points, label,region,ind_region = points.to(device), label.to(device), region.to(device),ind_region.to(device)
            # points, label,ind_region= points.to(device), label.to(device),ind_region.to(device)
            optimizer.zero_grad()
            net = net.train()
            # pred = net(points,ind_region)
            pred=net(points,region,ind_region)
            loss = F.nll_loss(pred, label)
            loss.backward()
            optimizer.step()
            if args.scheduler == 'wucd':
                scheduler.step(epoch-1 + i/train_num_batch)
            _, pred_idx = torch.max(pred, dim=1)
            correct = pred_idx.eq(label.data).cpu().sum()
            # for calculating training accuracy and loss
            total_train_correct += correct.item()
            total_train_loss += loss.item()
            # for calculating training weighted and macro metrics
            label = label.cpu().detach().numpy().tolist()
            train_labels_lst.extend(label)
            pred_idx = pred_idx.cpu().detach().numpy().tolist()
            train_predicted_lst.extend(pred_idx)
        if args.scheduler == 'step':
            scheduler.step()
        # train accuracy loss
        avg_train_acc = total_train_correct / float(train_data_size)
        avg_train_loss = total_train_loss / float(train_num_batch)
        train_acc_lst.append(avg_train_acc)
        train_loss_lst.append(avg_train_loss)
        # train macro p, r, f1
        mac_train_precision, mac_train_recall, mac_train_f1 = calculate_prec_recall_f1(train_labels_lst, train_predicted_lst)
        train_precision_lst.append(mac_train_precision)
        train_recall_lst.append(mac_train_recall)
        train_f1_lst.append(mac_train_f1)
        train_end_time = time.time()
        train_time = round(train_end_time-train_start_time, 2)
        logger.info('epoch [{}/{}] time: {}s train loss: {} accuracy: {} f1: {}'.format(
            epoch, args.epoch, train_time, round(avg_train_loss, 4), round(avg_train_acc, 4), round(mac_train_f1, 4)))
        if avg_train_acc > best_acc:
            best_acc, best_acc_epoch, best_acc_wts, best_acc_train_labels_lst, best_acc_train_pred_lst = \
                best_swap(avg_train_acc, epoch, net, train_labels_lst, train_predicted_lst)
        if mac_train_f1 > best_f1_mac:
            best_f1_mac, best_f1_epoch, best_f1_wts, best_f1_train_labels_lst, best_f1_train_pred_lst = \
                best_swap(mac_train_f1, epoch, net, train_labels_lst, train_predicted_lst)
    
        # validation
        with torch.no_grad():
            val_start_time = time.time()
            for j, data in (enumerate(val_loader, 0)):
                points, label,region,ind_region = data
                # points,label,ind_region = data
                label = label[:, 0]
                points = points.transpose(2, 1)
            
                points, label,region,ind_region = points.to(device), label.to(device), region.to(device),ind_region.to(device)
                # points, label,ind_region= points.to(device), label.to(device),ind_region.to(device)
                net = net.eval()
                # pred = net(points,ind_region)
                pred = net(points,region,ind_region)
                loss = F.nll_loss(pred, label)
                _, pred_idx = torch.max(pred, dim=1)
                correct = pred_idx.eq(label.data).cpu().sum()
                # for calculating validation accuracy and loss
                total_val_correct += correct.item()
                total_val_loss += loss.item()
                # for calculating validation weighted and macro metrics
                label = label.cpu().detach().numpy().tolist()
                val_labels_lst.extend(label)
                pred_idx = pred_idx.cpu().detach().numpy().tolist()
                val_predicted_lst.extend(pred_idx)
        # calculate the validation accuracy and loss for the epoch
        avg_val_acc = total_val_correct / float(val_data_size)
        avg_val_loss = total_val_loss / float(val_num_batch)
        val_acc_lst.append(avg_val_acc)
        val_loss_lst.append(avg_val_loss)
        # calculate the validation macro metrics
        mac_val_precision, mac_val_recall, mac_val_f1 = calculate_prec_recall_f1(val_labels_lst, val_predicted_lst)
        val_precision_lst.append(mac_val_precision)
        val_recall_lst.append(mac_val_recall)
        val_f1_lst.append(mac_val_f1)
        val_end_time = time.time()
        val_time = round(val_end_time-val_start_time, 2)
        logger.info('epoch [{}/{}] time: {}s val loss: {} accuracy: {} f1: {}'.format(
            epoch, args.epoch, val_time, round(avg_val_loss, 4), round(avg_val_acc, 4), round(mac_val_f1, 4)))
        # swap and save the best metric
        if avg_val_acc > best_acc:
            best_acc, best_acc_epoch, best_acc_wts, best_acc_val_labels_lst, best_acc_val_pred_lst = \
                best_swap(avg_val_acc, epoch, net, val_labels_lst, val_predicted_lst)
        if mac_val_f1 > best_f1_mac:
            best_f1_mac, best_f1_epoch, best_f1_wts, best_f1_val_labels_lst, best_f1_val_pred_lst = \
                best_swap(mac_val_f1, epoch, net, val_labels_lst, val_predicted_lst)
        
    # save best weights
    save_best_weights(net, best_acc_wts, args.out_path, 'acc', best_acc_epoch, best_acc, logger)
    save_best_weights(net, best_f1_wts, args.out_path, 'f1', best_f1_epoch, best_f1_mac, logger)
    # calculate classification report and plot class analysis curves for different metrics
    label_names_str = [label_name.decode() for label_name in label_names]
    # accuracy
    classify_report(best_acc_val_labels_lst, best_acc_val_pred_lst, label_names_str, logger, args.out_path, 'acc')
    per_class_metric(best_acc_val_labels_lst, best_acc_val_pred_lst, label_names_str, val_data_size, logger,
                     args.out_path, 'acc')
    # macro f1
    classify_report(best_f1_val_labels_lst, best_f1_val_pred_lst, label_names_str, logger, args.out_path, 'f1')
    per_class_metric(best_f1_val_labels_lst, best_f1_val_pred_lst, label_names_str, val_data_size, logger,
                     args.out_path, 'f1')
    #train dataset
    
    # accuracy
    classify_report(best_acc_train_labels_lst, best_acc_train_pred_lst, label_names_str, logger, args.out_path, 'acc')
    per_class_metric(best_acc_train_labels_lst, best_acc_train_pred_lst, label_names_str, train_data_size, logger,
                     args.out_path, 'acc')
    # macro f1
    classify_report(best_f1_train_labels_lst, best_f1_train_pred_lst, label_names_str, logger, args.out_path, 'f1')
    per_class_metric(best_f1_train_labels_lst, best_f1_train_pred_lst, label_names_str, train_data_size, logger,
                     args.out_path, 'f1')
    # total processing time
    time_end = time.time()
    total_time = round(time_end-time_start, 2)
    logger.info('Total processing time is {}s'.format(total_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='data/ORG/swm/train/',help='Input graph data and labels')
    parser.add_argument('--out_path_base', type=str, default='./ModelWeights', help='Save trained models')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--train_batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--val_batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epoch', type=int, default=40, help='the number of epochs')
    parser.add_argument('--best_metric', type=str, default='f1', help='evaluation metric')
    args = parser.parse_args()

    args.manualSeed = 0  # fix seed
    print("Random Seed: ", args.manualSeed)
    fix_seed(args.manualSeed)

    script_name = '<train>'

    args.input_path = unify_path(args.input_path)
    args.out_path_base = unify_path(args.out_path_base)

    if args.eval_fold_zero:
        fold_lst = [0]
    else:
        fold_lst = [i for i in range(args.k_fold)]

    for num_fold in fold_lst:
        num_fold = num_fold + 1
        args.out_path = os.path.join(args.out_path_base, str(num_fold))
        makepath(args.out_path)

        # Record the training process and values
        logger = create_logger(args.out_path)
        logger.info('=' * 55)
        logger.info(args)
        logger.info('=' * 55)
        logger.info('Implement {} fold experiment'.format(num_fold))
        # load data
        train_loader, val_loader, label_names, \
        num_classes, train_data_size, val_data_size = load_data()

        # model setting
        classifier = ResPointNetCls(k=num_classes)  # Remove transformation nets
        optimizer = optim.Adam(classifier.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        classifier.to(device)
        train_val_net(classifier)

    # clean the logger
    logger.handlers.clear()

    # average metric
    num_files = len(fold_lst)
    calculate_average_metric(args.out_path_base, num_files, args.best_metric)