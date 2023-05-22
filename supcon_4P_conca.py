"""
Adapted from: https://github.com/HobbitLong/SupContrast
"""
from early_stopping import EarlyStopping
import copy
import os
import argparse
import time
import tensorboard_logger as tb_logger
import torch
from util import AverageMeter
from util import load_matrices_and_labels, load_indices
from util import set_optimizer, keep_model, save_model_from_state
from util import load_list, set_models_4, get_embedding_from_4_models
from util import  get_ckpts_4_models, get_transformations_from_models, set_model_supcon
from models import Model_supcon
from losses import SupConLoss
import scipy.sparse
from sklearn.feature_selection import VarianceThreshold
from os import listdir
from os.path import isfile, join

torch.manual_seed(0)
import random
random.seed(0)

import numpy as np
np.random.seed(0)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
 

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument(
        '--batch_size', type=int, default=10, help='batch_size')
    parser.add_argument(
        '--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument(
        '--epochs', type=int, default=2000,
        help='number of training epochs')
    parser.add_argument(
        '--batch_transfo_size', type=int, default=1000,
        help='batch size for generating the embeddings')

    parser.add_argument(
        '--learning_rate', type=float, default=0.001,
        help='learning rate')
    parser.add_argument(
        '--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument(
        '--approach', type=str, default="", help='approach')
    parser.add_argument(
        '--keyword_approach', type=str, default="",
        help='keyword_approach: either dr, rev, mf, mp, ma, or mco')
    
    #set es_brk to true to have the early stoping constraint
    parser.add_argument(
        '--es_brk', type=lambda x: (str(x).lower() == 'true'), 
        default=True, help='es_brk')
    
    #to use models trainined without early stopping
    parser.add_argument(
        '--long', type=str, default="no", help='long')
    parser.add_argument(
        '--split_id', type=int,
        help='the id of the dataset split to use: 1, 2, 3, 4, 5 or time', 
        required=True)   
    parser.add_argument(
        '--threshold', type=str, default="5", help='threshold')

    parser.add_argument(
        '--model', type=str, default='Model_supcon')
    parser.add_argument(
        '--path_indices', type=str,
        help='the path to the "indices" folder. '\
        'you can refer to split_data scripts', required=True)

    parser.add_argument(
        '--temp', type=float, default=0.07,
        help='temperature for loss function')
    #set mem parameter to True if you have huge matrices 
    #(ex: drebin, reveal, mama_p and malscan_co)
    parser.add_argument(
        '--mem', type=lambda x: (str(x).lower() == 'true'), 
        default=False, help='mem_for_huge_matrices') 

    opt = parser.parse_args()
    if (opt.keyword_approach == "dr" or 
        opt.keyword_approach == "rev"):
        opt.fs = True
    else:
        opt.fs = False
        
    opt.dataset = "NEW_{}_diff_4P_conca".format(opt.approach)

    opt.model_path = './save/SupCon_{}/{}_models'.\
                     format(opt.threshold, opt.dataset)
    opt.tb_path = './save/SupCon_{}/{}_tensorboard'.\
                  format(opt.threshold, opt.dataset)

    opt.model_name = 'SUPCON_{}_{}_lr_{}_bsz_{}_temp_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate,
               opt.batch_size, opt.temp)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    
    return opt

def set_loader(opt):
    (indexes_diff_tr, indexes_diff_tr_tp, 
     indexes_diff_tr_tn, indexes_diff_tr_fp, 
     indexes_diff_tr_fn, indexes_diff_va, 
     indexes_diff_te) =  load_indices(opt)
    (x_train1, x_valid1, x_test1, 
     y_train, y_valid, y_test) = load_matrices_and_labels(opt, opt.approach)

    opt, data_loaders = get_embedding_from_4_models(opt, x_train1, x_valid1, 
                                                    x_test1, y_train, y_valid, 
                                                    y_test, indexes_diff_tr, 
                                                    indexes_diff_tr_tp, 
                                                    indexes_diff_tr_tn, 
                                                    indexes_diff_tr_fp, 
                                                    indexes_diff_tr_fn, 
                                                    indexes_diff_va, 
                                                    indexes_diff_te)
    
    return data_loaders

def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    
    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        
        data_time.update(time.time() - end)
        
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        features = model(images)
        
        
        features = features.unsqueeze(1)
        
        loss = criterion(features, labels)

        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
 
    torch.cuda.empty_cache()
    return losses.avg



def main():
    opt = parse_option()
    opt = get_ckpts_4_models(opt)

    # build data loader
    data_loaders = set_loader(opt)

    model1, model2, model3, model4, _ = set_models_4(opt)
    
    
    if opt.mem == True:
        my_train = [i[0][0] for i in data_loaders]
        x_train1 = [i[0][1] for i in data_loaders]
    else:
        my_train = [i[0] for i in data_loaders]
        x_train1 = ""

    train_loader = get_transformations_from_models(
        my_train, x_train1, [model1, model2, model3, model4], 
        opt, shuffle = True, drop_last = True)
        
    model1, model2, model3, model4, data_loaders = "", "", "", "", ""
    my_train, x_train1 = "", ""


    # build model and criterion
    model, criterion = set_model_supcon(opt, 512)
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    es = EarlyStopping(patience=100, mode='min')
    state_best = None
    best_loss = 1000
    es_reached = False
    
    # training routine
    for epoch in range(1, opt.epochs + 1):
        
        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}, loss {:.4f}'.\
              format(epoch, time2 - time1, loss))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        
        if es.step(loss):
            print("early stop criterion is met")
            if opt.es_brk == True:
                break  # early stop criterion is met, we can stop now
            else:
                es_reached = True
                save_model_from_state(state_best)
                print('best loss is : {:.4f}, from epoch {}'.\
                      format(best_loss, state_best["epoch"]))
                #set early stoping to the number of epochs
                es = EarlyStopping(patience=opt.epochs, mode='min')
            
            
        ctime = str(time.ctime(time.time())).replace(" ", "_")
        
        if loss < best_loss:
            best_loss = loss
            model = model.to('cpu')
            if es_reached == True:
                save_file = os.path.join(
                    opt.save_folder, 
                    'LONG_ckpt_supcon_epoch_{epoch}_{ctime}_loss_{loss}.pth'.\
                    format(epoch=epoch, ctime=ctime, loss=best_loss))
            else:
                save_file = os.path.join(
                    opt.save_folder, 
                    'ckpt_supcon_epoch_{epoch}_{ctime}_loss_{loss}.pth'.\
                    format(epoch=epoch, ctime=ctime, loss=best_loss))
            best_model = copy.deepcopy(model)
            best_opt = copy.deepcopy(opt)
            state_best = keep_model(best_model, optimizer, best_opt, epoch, save_file)
            model = model.cuda()
 

    save_model_from_state(state_best)
    print('best loss is : {:.4f}, from epoch {}'.format(best_loss, state_best["epoch"]))


if __name__ == '__main__':
    main()
