"""
Adapted from: https://github.com/HobbitLong/SupContrast
"""
from early_stopping import EarlyStopping
import argparse
import time
import os
from util import set_optimizer, keep_model, save_model_from_state
import torch
import copy
from util import load_matrices_and_labels, load_indices
from util import AverageMeter
from util import load_list, set_models_4, get_embedding_from_4_models
from util import  get_ckpts_4_models, get_transformations_from_models, set_model_supcon
from util import set_optimizer
from models import Model_linear, Model_supcon
import tensorboard_logger as tb_logger
from os import listdir
from os.path import isfile, join
import metrics as metrics

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
        '--num_workers', type=int, default=4,
        help='num of workers to use')
    parser.add_argument(
        '--epochs', type=int, default=2000, 
        help='number of training epochs')
    parser.add_argument(
        '--batch_transfo_size', type=int, default=1000,
        help='batch size for generating the embeddings')

    parser.add_argument(
        '--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument(
        '--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument(
        '--path_indices', type=str,
        help='the path to the "indices" folder. '\
        'you can refer to split_data scripts', required=True)

    parser.add_argument(
        '--model', type=str, default='Model_linear')


    parser.add_argument(
        '--threshold', type=str, default="5", help='threshold')
    parser.add_argument(
        '--split_id', type=int,
        help='the id of the dataset split to use: 1, 2, 3, 4, 5 or time', 
        required=True)
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
    #set mem parameter to True if you have huge matrices 
    #(ex: drebin, reveal, mama_p and malscan_co)
    parser.add_argument(
        '--mem', type=lambda x: (str(x).lower() == 'true'), default=False,
        help='mem_for_huge_matrices') 

    parser.add_argument(
        '--temp', type=float, default=0.07,
        help='temperature for loss function')

    opt = parser.parse_args()
    if (opt.keyword_approach == "dr" or 
        opt.keyword_approach == "rev"):
        opt.fs = True
    else:
        opt.fs = False
        
    opt.dataset = "NEW_{}_diff_4P".format(opt.approach)
    if opt.long == "yes":
        opt.dataset = opt.dataset + "_LONG"
        
    opt.model_path = './save/SupConLinear_{}/{}_models'.\
                     format(opt.threshold, opt.dataset)
    opt.tb_path = './save/SupConLinear_{}/{}_tensorboard'.\
                  format(opt.threshold, opt.dataset)

    opt.model_name = 'SupConLinear{}_{}_lr_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate,
               opt.batch_size)
  
    
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    
    return opt
    
    
def get_ckpts(opt):
    ckpts = []
    my_learning_rate = 0.001
    my_temp = 0.07
    my_model = 'Model_supcon'
    opt = get_ckpts_4_models(opt)


    #model concatenation
    dataset_4p = "NEW_{}_diff_4P_conca".format(opt.approach)
    model_path_4p = './save/SupCon_{}/{}_models'.\
                    format(opt.threshold, dataset_4p)
    model_name_4p = 'SUPCON_{}_{}_lr_{}_bsz_{}_temp_{}'.\
        format(dataset_4p, my_model, my_learning_rate,
               opt.batch_size, my_temp)
    save_folder_4p = os.path.join(model_path_4p, model_name_4p)
    
    if opt.long == "yes":
        ckpt_ = [f 
                 for f 
                 in listdir(save_folder_4p) 
                 if (isfile(join(save_folder_4p, f)) and 
                     f.startswith("LONG"))]
    else:
        ckpt_ = [f 
                 for f 
                 in listdir(save_folder_4p) 
                 if (isfile(join(save_folder_4p, f)) and not 
                     f.startswith("LONG"))]
    print(ckpt_)
    assert(len(ckpt_)==1)
    opt.ckpt = os.path.join(save_folder_4p, ckpt_[0])
    

def set_model(opt):
    model = Model_supcon(dim_in=512)
    criterion = torch.nn.BCELoss()
    
    classifier = Model_linear()

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()

        model.load_state_dict(state_dict)

    return model, classifier, criterion



    
    
def set_loader(opt):
    (indexes_diff_tr, indexes_diff_tr_tp, 
     indexes_diff_tr_tn, indexes_diff_tr_fp, 
     indexes_diff_tr_fn, indexes_diff_va, 
     indexes_diff_te) =  load_indices(opt)
    (x_train1, x_valid1, x_test1, 
     y_train, y_valid, y_test) = load_matrices_and_labels(opt, opt.approach)

    opt, data_loaders = get_embedding_from_4_models(
        opt, x_train1, x_valid1, x_test1, y_train, y_valid, y_test,
        indexes_diff_tr, indexes_diff_tr_tp, indexes_diff_tr_tn, 
        indexes_diff_tr_fp, indexes_diff_tr_fn, indexes_diff_va, 
        indexes_diff_te)
    return data_loaders





def train(train_loader, model, classifier, 
          criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    my_metrics = metrics.Metric(2)

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)
        bsz = labels.shape[0]
        
        one_hot_label = torch.nn.functional.one_hot(
            labels.to(torch.int64), 2)
        one_hot_label = one_hot_label.float().cuda()

        output = classifier(images)
        output_argmax = torch.argmax(output, dim=1)
        output_argmax = output_argmax.cpu().detach().tolist()
        
        my_metrics.update(output_argmax, labels)
        
        loss = criterion(output, one_hot_label)

        # update metric
        losses.update(loss.item(), bsz)
        
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
    (acc, pre, rec, f1, 
     tn, fp, fn, tp) = my_metrics.get_metrics(reduction='none')
    print("Tr lo, ac, pr, re, f1, tn, fp, fn, tp", 
          np.round(losses.avg, 4), np.round(acc, 4), 
          np.round(pre, 4), np.round(rec, 4), 
          np.round(f1, 4), tn, fp, fn, tp)
    return np.round(losses.avg, 4), np.round(f1, 4)


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    my_metrics = metrics.Metric(2)

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            bsz = labels.shape[0]
            one_hot_label = torch.nn.functional.one_hot(
                labels.to(torch.int64), 2)
            one_hot_label = one_hot_label.float().cuda()

            output = classifier(images)
            output_argmax = torch.argmax(output, dim=1)
            output_argmax = output_argmax.cpu().detach().tolist()

            my_metrics.update(output_argmax, labels)

            loss = criterion(output, one_hot_label)
            
            # update metric
            losses.update(loss.item(), bsz)

            
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
    acc, pre, rec, f1, tn, fp, fn, tp = my_metrics.get_metrics(reduction='none')
    print("\tVa lo, ac, pr, re, f1, tn, fp, fn, tp", 
          np.round(losses.avg, 4), np.round(acc, 4), 
          np.round(pre, 4), 
          np.round(rec, 4), np.round(f1, 4), tn, fp, fn, tp)
    return np.round(losses.avg, 4), np.round(f1, 4)


def test(data_loader, classifier, criterion, opt, keyword):
    #model.eval()    
    classifier.eval()
    
    my_metrics = metrics.Metric(2)
    losses = AverageMeter()
    
    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(data_loader):
            images = images.float().cuda()
            bsz = labels.shape[0]
            #labels = labels.cuda()
            one_hot_label = torch.nn.functional.one_hot(
                labels.to(torch.int64), 2)
            one_hot_label = one_hot_label.float().cuda()

            output = classifier(images)
            output_argmax = torch.argmax(output, dim=1)
            output_argmax = output_argmax.cpu().detach().tolist()

            my_metrics.update(output_argmax, labels)
            
            loss = criterion(output, one_hot_label)
            
            # update metric
            losses.update(loss.item(), bsz)
    (acc, pre, rec, f1, 
     tn, fp, fn, tp) = my_metrics.get_metrics(reduction='none')

    print("\t\t{keyword} loss, acc, pre, rec, f1".format(keyword=keyword), 
          np.round(losses.avg, 4), np.round(acc, 4), 
          np.round(pre, 4), np.round(rec, 4), 
          np.round(f1, 4), tn, fp, fn, tp)


def get_transformations_from_model_conca(data_loader, model, opt, 
                                         shuffle, drop_last=False):
    model.eval()
    list_features = []
    list_labels = []
    for idx, (images, labels) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        
        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
            
        list_features.append(features.cpu().detach().numpy())
        list_labels.append(labels.cpu().detach().numpy())
        
    all_features = np.concatenate(list_features)
    all_labels = np.concatenate(list_labels)
    
    new_data = torch.utils.data.TensorDataset(
        torch.from_numpy(all_features), torch.from_numpy(all_labels))
    
    new_dataloader = torch.utils.data.DataLoader(
        new_data, batch_size=opt.batch_size, shuffle=shuffle, 
        drop_last=drop_last, num_workers=opt.num_workers, 
        pin_memory=True, sampler=None)
 
    
    return new_dataloader

def main():
    
    opt = parse_option()
    get_ckpts(opt)
    
    data_loaders = set_loader(opt)

    # get models 
    model1, model2, model3, model4, _ = set_models_4(opt)
    
    if opt.mem == True:
        my_train_loader = [i[0][0] for i in data_loaders]
        my_valid_loader = [i[1][0] for i in data_loaders]
        my_test_loader = [i[2][0] for i in data_loaders]

        x_train1 = [i[0][1] for i in data_loaders]
        x_valid1 = [i[1][1] for i in data_loaders]
        x_test1 = [i[2][1] for i in data_loaders]
    else:
        my_train_loader = [i[0] for i in data_loaders]
        my_valid_loader = [i[1] for i in data_loaders]
        my_test_loader = [i[2] for i in data_loaders]
        
        x_train1, x_valid1, x_test1 = "", "", ""

    train_loader = get_transformations_from_models(
        my_train_loader, x_train1, [model1, model2, model3, model4], 
        opt, shuffle = False, drop_last = False)
    val_loader = get_transformations_from_models(
        my_valid_loader, x_valid1, [model1, model2, model3, model4], 
        opt, shuffle = False, drop_last = False)
    test_loader = get_transformations_from_models(
        my_test_loader, x_test1, [model1, model2, model3, model4], 
        opt, shuffle = False, drop_last = False)
        
    my_train_loader, my_valid_loader, my_test_loader = "", "", ""    
    model1, model2, model3, model4, data_loaders = "", "", "", "", ""
    x_train1, x_valid1, x_test1 = "", "", ""    
    
    model, classifier, criterion = set_model(opt)
    train_loader = get_transformations_from_model_conca(
        train_loader, model, opt, shuffle=True, 
        drop_last=opt.drop_last_train)
    val_loader = get_transformations_from_model_conca(
        val_loader, model, opt, shuffle=False)
    test_loader = get_transformations_from_model_conca(
        test_loader, model, opt, shuffle=False)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)
    
    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    state_best = None
    es_reached = False
    F1None = True
    best_f1 = 0
    
    # training routine
    for epoch in range(1, opt.epochs + 1):
        # train for one epoch
        time1 = time.time()
        loss, train_f1 = train(train_loader, model, classifier, 
                               criterion, optimizer, epoch, opt)
        if np.isnan(train_f1):
            train_f1 = 0
            
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, f1:{:.2f}'.format(
            epoch, time2 - time1, train_f1))
        
        # tensorboard logger
        logger.log_value('train_loss', loss, epoch)
        logger.log_value('train_f1', train_f1, epoch)
        logger.log_value('learning_rate', 
                         optimizer.param_groups[0]['lr'], epoch)

        # eval for one epoch
        loss, val_f1 = validate(val_loader, model, 
                                classifier, criterion, opt)
        logger.log_value('val_loss', loss, epoch)
        logger.log_value('val_f1', val_f1, epoch)
        
        if np.isnan(val_f1):
            val_f1 = 0
         
        #during trainig, some models stuck at the 6th epoch
        #we apply early stopping after epoch 6 so the models will converge
        #we apply early stopping after f1 on validation is not nan
        if ((val_f1 > best_f1) and (F1None==True) and epoch>6):
            F1None=False
            es = EarlyStopping(patience=100, mode='max')

        if F1None==False:
            if es.step(val_f1):
                print("early stopping criterion is met")
                if opt.es_brk == True:
                    break
                else:
                    es_reached = True
                    save_model_from_state(state_best)
                    print('best f1: {:.4f}'.format(best_f1))
                    _, my_best_classifier, _ = set_model(opt)
                    my_best_classifier.load_state_dict(state_best["model"])
                    test(train_loader, my_best_classifier, 
                         criterion, state_best["opt"], "Train")
                    test(val_loader, my_best_classifier, 
                         criterion, state_best["opt"], "Valid")
                    test(test_loader, my_best_classifier, 
                         criterion, state_best["opt"], "Test")
                    my_best_classifier = ""
                    save_model_from_state(state_best)   
                    #set early stoping to the number of epochs
                    es = EarlyStopping(patience=opt.epochs, mode='max') 

            if val_f1 > best_f1:
                best_f1 = val_f1
                classifier = classifier.to('cpu')
                ctime = str(time.ctime(time.time())).replace(" ", "_")
                if es_reached == True:
                    save_file = os.path.join(
                        opt.save_folder, 
                        'LONG_ckpt_linear_epoch_{epoch}_{ctime}_f1_{f1}.pth'.\
                        format(epoch=epoch, ctime=ctime, f1=best_f1))
                else:
                    save_file = os.path.join(
                        opt.save_folder, 
                        'ckpt_linear_epoch_{epoch}_{ctime}_f1_{f1}.pth'.\
                        format(epoch=epoch, ctime=ctime, f1=best_f1))
                best_classifier = copy.deepcopy(classifier)
                best_opt = copy.deepcopy(opt)
                state_best = keep_model(best_classifier, optimizer, 
                                        best_opt, epoch, save_file)
                classifier = classifier.cuda()

    save_model_from_state(state_best)
    opt = state_best["opt"]
    _, best_classifier, _ = set_model(opt)
    best_classifier.load_state_dict(state_best["model"])
    
    test(train_loader, best_classifier, criterion, state_best["opt"], "Train")
    test(val_loader, best_classifier, criterion, state_best["opt"], "Valid")
    test(test_loader, best_classifier, criterion, state_best["opt"], "Test")

    print('best f1: {:.4f}'.format(best_f1))
    
if __name__ == '__main__':
    main()
