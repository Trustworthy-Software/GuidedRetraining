"""
Some functions are adapted from: https://github.com/HobbitLong/SupContrast
"""
import math
import numpy as np
import torch
import torch.optim as optim
from losses import SupConLoss
import scipy.sparse
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from models import Model_supcon, Model_linear
import os
import pickle
from os import listdir
from os.path import isfile, join
from sklearn.metrics import recall_score, accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, f1_score

def file_to_list(path):
    with open(path) as f:
        return f.read().splitlines()
    
def load_list(path):
    with open(path, "rb") as fp:   # Unpickling
        b = pickle.load(fp)
    return b
    
def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
    
def save_obj(obj, path ):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
    
def get_X_Y(y_true, y_pred, percent):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    number_fp = int(fp*(percent/100))
    number_fn = int(fn*(percent/100))
    print("number_fp", number_fp, "number_fn", number_fn)
    return number_fp, number_fn

def get_diff_indices(y_true, y_pred, y_prob, number_fp, number_fn):
    indices_diff = []
    
    positives_indices = {i: y_prob[i][1] 
                         for i 
                         in range(len(y_prob)) 
                         if y_prob[i][1]>=0.5}
    negatives_indices = {i: y_prob[i][1] 
                         for i 
                         in range(len(y_prob)) 
                         if y_prob[i][1]<0.5}
    k_p, v_p = zip(*positives_indices.items())
    v_p, k_p = zip(*sorted(zip(v_p, k_p), reverse=True))

    k_n, v_n = zip(*negatives_indices.items())
    v_n, k_n = zip(*sorted(zip(v_n, k_n)))
    
    positives_done, negatives_done = 0, 0
    easy_pos, easy_neg = 0, 0
    
    for j in range(len(k_p)):
        if positives_done == number_fp:
            threshold_pos = v_p[j]
            indices_diff+=k_p[j:]
            break            

        if y_true[k_p[j]] == 0:
            positives_done+=1
        else:
            easy_pos+=1
            
    for j in range(len(k_n)):
        if negatives_done == number_fn:
            threshold_neg = v_n[j]
            indices_diff+=k_n[j:]
            break
            
        if y_true[k_n[j]] == 1:
            negatives_done+=1
        else:
            easy_neg+=1
    return indices_diff, threshold_pos, threshold_neg


def get_diff_indices_train_test(y_true, y_pred, y_prob, 
                                threshold_pos, threshold_neg):
    indices_diff = []

    for i in range(len(y_true)):
        if y_prob[i][1]<threshold_pos and y_prob[i][1]>threshold_neg:
            indices_diff.append(i)
    
    print("len all", len(y_true), "len_diff", len(indices_diff))
    return indices_diff

def get_scores_oneline(y_test, y_predict, keyword=""):
    recall = recall_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict)
    accuracy = accuracy_score(y_test, y_predict)
    cnf_mtrx = confusion_matrix(y_test, y_predict, 
                                labels=np.unique(np.array(y_test)))
    print(keyword, recall, precision, f1, accuracy, cnf_mtrx.ravel())
    
    
def get_scores_diff_easy(y_true, y_pred, indices_diff):
    get_scores_oneline(y_true, y_pred, "all")

    y_pred_easy = [y_pred[i] 
                   for i 
                   in range(len(y_true)) 
                   if i not in indices_diff] 
    y_true_easy = [y_true[i] 
                   for i 
                   in range(len(y_true)) 
                   if i not in indices_diff]
    get_scores_oneline(y_true_easy, y_pred_easy, "easy")

    y_pred_diff = [y_pred[i] for i in indices_diff] 
    y_true_diff = [y_true[i] for i in indices_diff]
    get_scores_oneline(y_true_diff, y_pred_diff, "diff")
    
    
def get_scores(clf, x_test, y_test, keyword, proba=1):
    y_predict = clf.predict(x_test)
    if proba == 1:
        y_predict_proba = clf.predict_proba(x_test)
    elif proba == 0:
        y_predict_proba = clf.decision_function(x_test)
    recall = recall_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict)
    accuracy = accuracy_score(y_test, y_predict)
    cnf_mtrx = confusion_matrix(y_test, y_predict, 
                                labels=np.unique(np.array(y_test)))
    print(keyword, recall, precision, f1, accuracy, cnf_mtrx.ravel())
    return (y_predict, y_predict_proba, recall, 
            precision, f1, accuracy, cnf_mtrx)

def load_matrices_and_labels(opt, approach):    
    x_train1 = scipy.sparse.load_npz('dataset/matrices/x_train_{}_{}.npz'.\
                                     format(approach, opt.split_id))
    x_valid1 = scipy.sparse.load_npz('dataset/matrices/x_valid_{}_{}.npz'.\
                                     format(approach, opt.split_id))
    x_test1 = scipy.sparse.load_npz('dataset/matrices/x_test_{}_{}.npz'.\
                                    format(approach, opt.split_id))
    
    y_train = file_to_list("dataset/data_splits_files/y_train_{}".\
                           format(opt.split_id))
    y_valid = file_to_list("dataset/data_splits_files/y_valid_{}".\
                           format(opt.split_id))
    y_test = file_to_list("dataset/data_splits_files/y_test_{}".\
                          format(opt.split_id))
    
    y_train = [int(i) for i in y_train]
    y_valid = [int(i) for i in y_valid]
    y_test = [int(i) for i in y_test]
    
    return x_train1, x_valid1, x_test1, y_train, y_valid, y_test


def save_y_pred_prob(y_predict_test, y_predict_valid, 
                     y_predict_train, y_proba_train,
                     y_proba_valid, y_proba_test, opt, approach):
    
    save_obj(
        y_predict_test, 
        os.path.join(opt.path_save, 
                     "y_predict_{}_test_split_{}".format(approach, 
                                                         opt.split_id)))
    save_obj(
        y_predict_valid, 
        os.path.join(opt.path_save, 
                     "y_predict_{}_valid_split_{}".format(approach, 
                                                          opt.split_id)))
    save_obj(
        y_predict_train, 
        os.path.join(opt.path_save, 
                     "y_predict_{}_train_split_{}".format(approach, 
                                                          opt.split_id)))
    save_obj(
        y_proba_train, 
        os.path.join(opt.path_save, 
                     "y_proba_{}_train_split_{}".format(approach, 
                                                        opt.split_id)))
    save_obj(
        y_proba_valid, 
        os.path.join(opt.path_save, 
                     "y_proba_{}_valid_split_{}".format(approach, 
                                                        opt.split_id)))
    save_obj(
        y_proba_test, 
        os.path.join(opt.path_save, 
                     "y_proba_{}_test_split_{}".format(approach, 
                                                       opt.split_id)))
    
def load_y_pred_prob(opt, approach):
    
    y_predict_test = load_obj(
        os.path.join(opt.path_files, 
                     "y_predict_{}_test_split_{}".format(approach, 
                                                         opt.split_id)))
    y_predict_valid = load_obj(
        os.path.join(opt.path_files, 
                     "y_predict_{}_valid_split_{}".format(approach, 
                                                          opt.split_id)))
    y_predict_train = load_obj(
        os.path.join(opt.path_files, 
                     "y_predict_{}_train_split_{}".format(approach, 
                                                          opt.split_id)))
    y_proba_train = load_obj(
        os.path.join(opt.path_files, 
                     "y_proba_{}_train_split_{}".format(approach, 
                                                        opt.split_id)))
    y_proba_valid = load_obj(
        os.path.join(opt.path_files, 
                     "y_proba_{}_valid_split_{}".format(approach, 
                                                        opt.split_id)))
    y_proba_test = load_obj(
        os.path.join(opt.path_files, 
                     "y_proba_{}_test_split_{}".format(approach, 
                                                       opt.split_id)))
    
    return (y_predict_test, y_predict_valid, y_predict_train, 
            y_proba_train, y_proba_valid, y_proba_test)

def decision_function_to_proba(InputLow_pos, InputHigh_pos, InputLow_neg, 
                               InputHigh_neg, y_train, y_valid, y_test, 
                            y_proba_drebin_train, y_proba_drebin_valid, 
                               y_proba_drebin_test):
    #Decision_func_to_predict_proba = ((Input - InputLow) / 
    #                                  (InputHigh - InputLow)) * 
    #                                  (OutputHigh - OutputLow) + OutputLow
    
    #InputLow_pos = min([i for i in y_proba_drebin_train if i>=0])
    #InputHigh_pos = max([i for i in y_proba_drebin_train if i>=0])
    #max because they are negative
    #InputLow_neg = abs(max([i for i in y_proba_drebin_train if i<=0]))
    #same for min
    #InputHigh_neg = abs(min([i for i in y_proba_drebin_train if i<=0])) 
    
    y_proba_drebin_train2 = []
    for i in range(len(y_train)):
        if y_proba_drebin_train[i] < 0:
            scalled_value = (((-y_proba_drebin_train[i] - InputLow_neg) / \
                              (InputHigh_neg - InputLow_neg)) * 0.5)
            y_proba_drebin_train2.append([scalled_value + 0.5, 
                                          0.5 - scalled_value])
        elif y_proba_drebin_train[i]>0:
            scalled_value = (((y_proba_drebin_train[i] - InputLow_pos) / \
                              (InputHigh_pos - InputLow_pos)) * 0.5)
            y_proba_drebin_train2.append([0.5 - scalled_value, 
                                          scalled_value + 0.5])
        else:
            y_proba_drebin_train2.append([0.5, 0.5])

    y_proba_drebin_valid2 = []
    for i in range(len(y_valid)):
        if y_proba_drebin_valid[i] < 0:
            if abs(y_proba_drebin_valid[i])>InputHigh_neg:
            #because they are negative so the higher value 
            #should be that of train
                scalled_value = (((InputHigh_neg - InputLow_neg) / \
                                  (InputHigh_neg - InputLow_neg)) * 0.5)
            else:
                scalled_value = (((-y_proba_drebin_valid[i] - InputLow_neg) / \
                                  (InputHigh_neg - InputLow_neg)) * 0.5)
            y_proba_drebin_valid2.append([scalled_value + 0.5, 
                                          0.5 - scalled_value])
        elif y_proba_drebin_valid[i] > 0:
            #in this case Input = InputHight
            if y_proba_drebin_valid[i] > InputHigh_pos:
                scalled_value = (1 * 0.5)
            else:
                scalled_value = (((y_proba_drebin_valid[i] - InputLow_pos) / \
                                  (InputHigh_pos - InputLow_pos)) * 0.5)
            y_proba_drebin_valid2.append([0.5 - scalled_value, 
                                          scalled_value + 0.5])
        else:
            y_proba_drebin_valid2.append([0.5, 0.5])

    y_proba_drebin_test2 = []
    for i in range(len(y_test)):
        if y_proba_drebin_test[i] < 0:
            #because they are negative so the higher value 
            #should be that of train
            if abs(y_proba_drebin_test[i]) > InputHigh_neg:
                scalled_value = (((InputHigh_neg - InputLow_neg) / \
                                  (InputHigh_neg - InputLow_neg)) * 0.5)
            else:
                scalled_value = (((-y_proba_drebin_test[i] - InputLow_neg) / \
                                  (InputHigh_neg - InputLow_neg)) * 0.5)
            y_proba_drebin_test2.append([scalled_value + 0.5, 
                                         0.5 - scalled_value])

        elif y_proba_drebin_test[i] > 0:
            #in this case Input = InputHight
            if y_proba_drebin_test[i] > InputHigh_pos:
                scalled_value = (1 * 0.5)
            else:
                scalled_value = (((y_proba_drebin_test[i] - InputLow_pos) / \
                                  (InputHigh_pos - InputLow_pos)) * 0.5)
            y_proba_drebin_test2.append([0.5 - scalled_value, 
                                         scalled_value + 0.5])
        else:
            y_proba_drebin_test2.append([0.5, 0.5])
    return (y_proba_drebin_train2, y_proba_drebin_valid2, 
            y_proba_drebin_test2)


def get_ckpts_4_models(opt):
    ckpts = []
    my_learning_rate = 0.001
    my_temp = 0.07
    my_model = 'Model_supcon'

    for j in range(1, 5):
        dataset_4p = "NEW_{}_diff_4P_{}".format(opt.approach, j)
        model_path_4p = './save/SupCon_{}/{}_models'.format(opt.threshold, 
                                                            dataset_4p)
        try: #some models use batch size 20
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
            ckpts.append(os.path.join(save_folder_4p, ckpt_[0]))
        except:
            model_name_4p = 'SUPCON_{}_{}_lr_{}_bsz_{}_temp_{}'.\
                format(dataset_4p, my_model, my_learning_rate,
                       20, my_temp)
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
            ckpts.append(os.path.join(save_folder_4p, ckpt_[0]))
    opt.ckpt1, opt.ckpt2 = ckpts[0], ckpts[1]
    opt.ckpt3, opt.ckpt4 = ckpts[2], ckpts[3]
    return opt
    

def load_indices(opt): 
    indexes_diff_tr = load_list(
        os.path.join(opt.path_indices, 
                     "indices_{}".format(opt.split_id), 
                     "indices_diff_tr_{}".format(opt.approach)))
    indexes_diff_va = load_list(
        os.path.join(opt.path_indices, 
                     "indices_{}".format(opt.split_id), 
                     "indices_diff_va_{}".format(opt.approach)))
    indexes_diff_te = load_list(
        os.path.join(opt.path_indices, 
                     "indices_{}".format(opt.split_id), 
                     "indices_diff_te_{}".format(opt.approach)))

    indexes_diff_tr_tp = load_list(
        os.path.join(opt.path_indices, 
                     "indices_{}".format(opt.split_id), 
                     "indices_diff_tr_{}_tp".format(opt.approach)))
    indexes_diff_tr_tn = load_list(
        os.path.join(opt.path_indices, 
                     "indices_{}".format(opt.split_id), 
                     "indices_diff_tr_{}_tn".format(opt.approach)))
    indexes_diff_tr_fp = load_list(
        os.path.join(opt.path_indices, 
                     "indices_{}".format(opt.split_id), 
                     "indices_diff_tr_{}_fp".format(opt.approach)))
    indexes_diff_tr_fn = load_list(
        os.path.join(opt.path_indices, 
                     "indices_{}".format(opt.split_id), 
                     "indices_diff_tr_{}_fn".format(opt.approach)))
    
    return (indexes_diff_tr, indexes_diff_tr_tp, indexes_diff_tr_tn, 
            indexes_diff_tr_fp, indexes_diff_tr_fn, indexes_diff_va, 
            indexes_diff_te)



def get_embedding_from_4_models(opt, x_train1, x_valid1, x_test1, 
                                y_train, y_valid, y_test, 
                                indexes_diff_tr, indexes_diff_tr_tp, 
                                indexes_diff_tr_tn, indexes_diff_tr_fp, 
                                indexes_diff_tr_fn, indexes_diff_va, 
                                indexes_diff_te):
    
    x_train1_all = x_train1[indexes_diff_tr]
    x_valid1 = x_valid1[indexes_diff_va]
    x_test1 = x_test1[indexes_diff_te]
    
    y_train = [y_train[i] for i in indexes_diff_tr]
    y_valid = [y_valid[i] for i in indexes_diff_va]
    y_test = [y_test[i] for i in indexes_diff_te]
    
    opt.batch_size = len(y_train)//opt.batch_size
    
    if (len(y_train) % opt.batch_size) <= 1:
        opt.drop_last_train = True
    else:
        opt.drop_last_train = False  
        
    ps = [indexes_diff_tr_tp + indexes_diff_tr_fp, 
          indexes_diff_tr_tp + indexes_diff_tr_tn, 
          indexes_diff_tr_tn + indexes_diff_tr_fn, 
          indexes_diff_tr_fp + indexes_diff_tr_fn]

    data_loaders = []
    shapes = []     
    count_p = 1
    
    if opt.mem == True:
        
        indices_train = [j for j in range(len(y_train))]
        indices_valid = [j for j in range(len(y_valid))]
        indices_test = [j for j in range(len(y_test))]

        train_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(indices_train), torch.Tensor(y_train).to(torch.int8))
        valid_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(indices_valid), torch.Tensor(y_valid).to(torch.int8))
        test_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(indices_test), torch.Tensor(y_test).to(torch.int8))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_transfo_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=False, sampler=None, 
            drop_last=False)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=opt.batch_transfo_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=False, sampler=None, 
            drop_last=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=opt.batch_transfo_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=False, sampler=None, 
            drop_last=False)
              
        if opt.keyword_approach == "dr" or opt.keyword_approach == "rev":
            for p in ps:
                x_train1_ = x_train1[p]

                selector = VarianceThreshold()
                selector.fit(x_train1_)
                x_train1_1 = selector.transform(x_train1_all)
                x_valid1_1 = selector.transform(x_valid1)
                x_test1_1 = selector.transform(x_test1)

                scores_indices = load_list(
                    os.path.join(opt.path_indices, 
                                 "indices_{}".format(opt.split_id), 
                                 "scores_indices_mutual_info_{}_p{}".\
                                 format(opt.approach, opt.part)))
                count_p+=1
                scores, indices =  zip(*scores_indices)
                x_train1_1 = x_train1_1[:, indices[:200000]]
                x_valid1_1 = x_valid1_1[:, indices[:200000]]
                x_test1_1 = x_test1_1[:, indices[:200000]]
                
                shapes.append(x_train1_1.shape[1])

                data_loaders.append([[train_loader, x_train1_1], 
                                     [valid_loader, x_valid1_1], 
                                     [test_loader, x_test1_1]])
        else:
             #for mama_p, malscan_co
            data_loaders.append([[train_loader, x_train1_all], 
                                 [valid_loader, x_valid1], 
                                 [test_loader, x_test1]])
            shapes = [x_train1_all.shape[1] for m in range(4)]
    else:
        x_train1_1 = x_train1_all.toarray()
        x_valid1_1 = x_valid1.toarray()
        x_test1_1 = x_test1.toarray()
        shapes = [x_train1_all.shape[1] for m in range(4)]
        
        train_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(x_train1_1), 
            torch.Tensor(y_train).to(torch.int8))
        valid_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(x_valid1_1), 
            torch.Tensor(y_valid).to(torch.int8))
        test_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(x_test1_1), 
            torch.Tensor(y_test).to(torch.int8))
         
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_transfo_size, 
            shuffle=False, num_workers=opt.num_workers, pin_memory=False, 
            sampler=None, drop_last=False)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=opt.batch_transfo_size, 
            shuffle=False, num_workers=opt.num_workers, pin_memory=False, 
            sampler=None, drop_last=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=opt.batch_transfo_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=False, 
            sampler=None, drop_last=False)

        data_loaders.append([train_loader, valid_loader, test_loader])
        
    opt.shapes = shapes
    return opt, data_loaders
        
        
        
        
def set_models_4(opt):
    model1 = Model_supcon(dim_in=opt.shapes[0])
    model2 = Model_supcon(dim_in=opt.shapes[1])
    model3 = Model_supcon(dim_in=opt.shapes[2])
    model4 = Model_supcon(dim_in=opt.shapes[3])
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = SupConLoss(temperature=opt.temp)

    ckpt1 = torch.load(opt.ckpt1, map_location='cpu')
    ckpt2 = torch.load(opt.ckpt2, map_location='cpu')
    ckpt3 = torch.load(opt.ckpt3, map_location='cpu')
    ckpt4 = torch.load(opt.ckpt4, map_location='cpu')
    
    
    state_dict1 = ckpt1['model']
    state_dict2 = ckpt2['model']
    state_dict3 = ckpt3['model']
    state_dict4 = ckpt4['model']

    if torch.cuda.is_available():
        new_state_dict1 = {}
        for k, v in state_dict1.items():
            k = k.replace("module.", "")
            new_state_dict1[k] = v
        state_dict1 = new_state_dict1

        new_state_dict2 = {}
        for k, v in state_dict2.items():
            k = k.replace("module.", "")
            new_state_dict2[k] = v
        state_dict2 = new_state_dict2

        new_state_dict3 = {}
        for k, v in state_dict3.items():
            k = k.replace("module.", "")
            new_state_dict3[k] = v
        state_dict3 = new_state_dict3

        new_state_dict4 = {}
        for k, v in state_dict4.items():
            k = k.replace("module.", "")
            new_state_dict4[k] = v
        state_dict4 = new_state_dict4
            
            
            
        model1 = model1.cuda()
        model2 = model2.cuda()
        model3 = model3.cuda()
        model4 = model4.cuda()
        criterion = criterion.cuda()

        model1.load_state_dict(state_dict1)
        model2.load_state_dict(state_dict2)
        model3.load_state_dict(state_dict3)
        model4.load_state_dict(state_dict4)

    return model1, model2, model3, model4, criterion        

    
def get_transformations_from_models(data_loaders, list_x_train, 
                                    models, opt, shuffle, drop_last):
    models[0].eval()
    models[1].eval()
    models[2].eval()
    models[3].eval()
    
    list_features = []
    list_labels = []
    
    for j in range(4):
        my_list_features = []
        model = models[j]
        
        
        if (opt.keyword_approach == "dr" or 
            opt.keyword_approach == "rev"):
            my_data_loader = data_loaders[j]
            my_x = list_x_train[j]
        elif opt.mem == True:
            my_data_loader = data_loaders[0]
            my_x = list_x_train[0]
        else:
            my_data_loader = data_loaders[0]

        for idx, (i, labels) in enumerate(my_data_loader):
            if opt.mem == True:
                indices = i
                my_x_ = my_x[indices.tolist()]
                my_x_ = my_x_.toarray()
                images = torch.Tensor(my_x_)
            else:
                images = i

            images = images.cuda(non_blocking=True)
            with torch.no_grad():
                features1 = model.encoder(images)
            my_list_features.append(features1.cpu().detach().numpy())
            if j==0:
                list_labels.append(labels.cpu().detach().numpy())
        list_features.append(my_list_features)  
        
    all_features1 = np.concatenate(list_features[0])
    all_features2 = np.concatenate(list_features[1])
    all_features3 = np.concatenate(list_features[2])
    all_features4 = np.concatenate(list_features[3])

    all_features = np.hstack((all_features1, all_features2, 
                              all_features3, all_features4))

    all_labels = np.concatenate(list_labels)


    new_data = torch.utils.data.TensorDataset(torch.from_numpy(all_features), 
                                              torch.from_numpy(all_labels))

    new_dataloader = torch.utils.data.DataLoader(
        new_data, batch_size=opt.batch_size, shuffle=shuffle,
        num_workers=opt.num_workers, pin_memory=True, sampler=None, 
        drop_last=drop_last)

    return new_dataloader



def set_model_supcon(opt, dim_in):
    model = Model_supcon(dim_in)
    criterion = SupConLoss(temperature=opt.temp)
  
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    return model, criterion
  


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions 
    #for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().\
                        view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, 
                         total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum)
    return optimizer

    
def keep_model(model, optimizer, opt, epoch, save_file):
    print('==> Keeping...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'save_file': save_file,
    }
    return state

def save_model_from_state(state):
    if state==None:
        return
    print('==> Saving...')
    save_file = state["save_file"]
    torch.save(state, save_file)
    del state
