import sys
from util import load_y_pred_prob, load_matrices_and_labels
from util import get_diff_indices, get_X_Y, get_diff_indices_train_test, get_scores_diff_easy
import os
import argparse
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
import pickle
import numpy as np

    
def save_obj(obj, path ):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
 

def parse_option():
    parser = argparse.ArgumentParser(
                'Split data for DREBIN, RevealDroid and MaMaDroid')
    parser.add_argument(
        '--path_save', type=str,
        help='path where to save outputs', required=True)
    parser.add_argument(
        '--path_files', type=str,
        help='path where predictions and probabilities are located', 
        required=True)
    parser.add_argument(
        '--split_id', type=int,
        help='the id of the dataset split to use: 1, 2, 3, 4, 5 or time', 
        required=True)
    parser.add_argument(
        '--approach', type=str,
        help='either drebin, revealdroid, mama_f or mama_p', required=True)
    
    opt = parser.parse_args()
    return opt
                                                                                
def main():
    
    opt = parse_option()
    
    print("---------------------{}----------------------".format(opt.approach))
    path_indices = os.path.join(opt.path_save, "indices", 
                                "indices_{}".format(opt.split_id))
    if not os.path.exists(path_indices):
        os.makedirs(path_indices)
    
    (x_train1, x_valid1, x_test1, 
     y_train, y_valid, y_test) = load_matrices_and_labels(opt, opt.approach)
    (y_predict_test, y_predict_valid, 
     y_predict_train, y_proba_train, 
     y_proba_valid, y_proba_test) = load_y_pred_prob(opt, opt.approach)
    
    x_train1, x_valid1, x_test1 = x_train1[:2000], x_valid1[:2000], x_test1[:2000]
    y_train, y_valid, y_test = y_train[:2000], y_valid[:2000], y_test[:2000]
    y_predict_test, y_predict_valid, y_predict_train, y_proba_train, y_proba_valid, y_proba_test = y_predict_test[:2000], y_predict_valid[:2000], y_predict_train[:2000], y_proba_train[:2000], y_proba_valid[:2000], y_proba_test[:2000]
    
    #number of tolerated FPs and FNs
    number_fp, number_fn = get_X_Y(y_valid, y_predict_valid, 5) 
    indices_diff_va, thr_pos, thr_neg = get_diff_indices(y_valid, 
                                                         y_predict_valid, 
                                                         y_proba_valid, 
                                                         number_fp, 
                                                         number_fn)
    print("threshold positives", thr_pos, "threshold negatives", thr_neg)

    indices_diff_tr = get_diff_indices_train_test(y_train, y_predict_train, 
                                                  y_proba_train, thr_pos, 
                                                  thr_neg)
    indices_diff_te = get_diff_indices_train_test(y_test, y_predict_test, 
                                                  y_proba_test, thr_pos, 
                                                  thr_neg)
    indices_diff_va = get_diff_indices_train_test(y_valid, y_predict_valid, 
                                                  y_proba_valid, thr_pos, 
                                                  thr_neg)


    print("---------valid---------")
    get_scores_diff_easy(y_valid, y_predict_valid, indices_diff_va)
    print("---------train--------")
    get_scores_diff_easy(y_train, y_predict_train, indices_diff_tr)
    print("---------test----------")
    get_scores_diff_easy(y_test, y_predict_test, indices_diff_te)

    save_obj(indices_diff_tr, 
             os.path.join(path_indices, 
                          "indices_diff_tr_{}".format(opt.approach)))
    save_obj(indices_diff_va, 
             os.path.join(path_indices, 
                          "indices_diff_va_{}".format(opt.approach)))
    save_obj(indices_diff_te, 
             os.path.join(path_indices, 
                          "indices_diff_te_{}".format(opt.approach)))


    #Get difficult TP, FP, TN and FN
    y_train_pred_easy = []
    y_train_easy = []
    y_train_pred_diff = []
    y_train_diff = []
    indexes_diff_tr_fn, indexes_diff_tr_fp = [], []
    indexes_diff_tr_tn, indexes_diff_tr_tp = [], []
    for i in range(len(y_train)):
        if i not in indices_diff_tr:
            y_train_pred_easy.append(y_predict_train[i])
            y_train_easy.append(y_train[i])
        else:

            y_train_pred_diff.append(y_predict_train[i])
            y_train_diff.append(y_train[i])
            if y_train[i]==1 and y_predict_train[i]==0:
                indexes_diff_tr_fn.append(i)
            elif y_train[i]==1 and y_predict_train[i]==1:
                indexes_diff_tr_tp.append(i)
            elif y_train[i]==0 and y_predict_train[i]==0:
                indexes_diff_tr_tn.append(i)
            elif y_train[i]==0 and y_predict_train[i]==1:
                indexes_diff_tr_fp.append(i)
    
    save_obj(indexes_diff_tr_tp, 
             os.path.join(path_indices, 
                          "indices_diff_tr_{}_tp".format(opt.approach)))
    save_obj(indexes_diff_tr_tn, 
             os.path.join(path_indices, 
                          "indices_diff_tr_{}_tn".format(opt.approach)))
    save_obj(indexes_diff_tr_fp, 
             os.path.join(path_indices, 
                          "indices_diff_tr_{}_fp".format(opt.approach)))
    save_obj(indexes_diff_tr_fn, 
             os.path.join(path_indices, 
                          "indices_diff_tr_{}_fn".format(opt.approach)))


    if opt.approach == "drebin" or opt.approach == "reveal":
        ###Feature selection
        print("-----feature selection-----")
        #ALL the training samples
        selector = VarianceThreshold()
        x_train1_ = selector.fit_transform(x_train1)
        x_train1_ = x_train1_.astype(np.int8, copy=False)

        print(x_train1.shape, x_train1_.shape)
        scores = mutual_info_classif(x_train1_, y_train, 
                                     discrete_features=True, 
                                     random_state=0)

        indices_new_array = [i for i in range(len(scores))]
        scores_indices = sorted(zip(scores, indices_new_array), 
                                reverse=True)
        scores_, indices_ =  zip(*scores_indices)

        save_obj(scores_indices, 
                 os.path.join(path_indices, 
                              "scores_indices_mutual_info_{}_all"\
                              .format(opt.approach)))

        #diff subset
        selector = VarianceThreshold()
        x_train1_ = selector.fit_transform(x_train1[indices_diff_tr])
        print(x_train1.shape, x_train1_.shape)
        y_train_ = [y_train[i] for i in indices_diff_tr]
        x_train1_ = x_train1_.astype(np.int8, copy=False)
        scores = mutual_info_classif(x_train1_, y_train_, 
                                     discrete_features=True, 
                                     random_state=0)
        indices_new_array = [i for i in range(len(scores))]
        scores_indices = sorted(zip(scores, indices_new_array), 
                                reverse=True)
        scores_, indices_ =  zip(*scores_indices)
        save_obj(scores_indices, 
                 os.path.join(path_indices, 
                              "scores_indices_mutual_info_{}_diff"\
                              .format(opt.approach)))

        #4 models for Guided Retraining

        indexes = [indexes_diff_tr_tp, indexes_diff_tr_tn, 
                   indexes_diff_tr_fp, indexes_diff_tr_fn]
        indexes_diff_tr = indexes[0] + indexes[2]

        selector = VarianceThreshold()
        x_train1_ = selector.fit_transform(x_train1[indexes_diff_tr])
        x_train1_ = x_train1_.astype(np.int8, copy=False)
        print(x_train1.shape, x_train1_.shape)
        y_train_ = [y_train[i] for i in indexes_diff_tr]
        scores = mutual_info_classif(x_train1_, y_train_, 
                                     discrete_features=True, 
                                     random_state=0)
        indices_new_array = [i for i in range(len(scores))]
        scores_indices = sorted(zip(scores, indices_new_array), 
                                reverse=True)
        scores_, indices_ =  zip(*scores_indices)
        save_obj(scores_indices, 
                 os.path.join(path_indices, 
                              "scores_indices_mutual_info_{}_p1"\
                              .format(opt.approach)))


        indexes_diff_tr = indexes[0] + indexes[1]
        selector = VarianceThreshold()
        x_train1_ = selector.fit_transform(x_train1[indexes_diff_tr])
        x_train1_ = x_train1_.astype(np.int8, copy=False)
        print(x_train1.shape, x_train1_.shape)
        y_train_ = [y_train[i] for i in indexes_diff_tr]
        scores = mutual_info_classif(x_train1_, y_train_, 
                                     discrete_features=True, 
                                     random_state=0)
        indices_new_array = [i for i in range(len(scores))]
        scores_indices = sorted(zip(scores, indices_new_array), 
                                reverse=True)
        scores_, indices_ =  zip(*scores_indices)
        save_obj(scores_indices, 
                 os.path.join(path_indices, 
                              "scores_indices_mutual_info_{}_p2"\
                              .format(opt.approach)))


        indexes_diff_tr = indexes[1] + indexes[3]
        selector = VarianceThreshold()
        x_train1_ = selector.fit_transform(x_train1[indexes_diff_tr])
        x_train1_ = x_train1_.astype(np.int8, copy=False)
        print(x_train1.shape, x_train1_.shape)
        y_train_ = [y_train[i] for i in indexes_diff_tr]
        scores = mutual_info_classif(x_train1_, y_train_, 
                                     discrete_features=True, 
                                     random_state=0)
        indices_new_array = [i for i in range(len(scores))]
        scores_indices = sorted(zip(scores, indices_new_array), 
                                reverse=True)
        scores_, indices_ =  zip(*scores_indices)
        save_obj(scores_indices, 
                 os.path.join(path_indices, 
                              "scores_indices_mutual_info_{}_p3"\
                               .format(opt.approach)))


        indexes_diff_tr = indexes[2] + indexes[3]
        selector = VarianceThreshold()
        x_train1_ = selector.fit_transform(x_train1[indexes_diff_tr])
        x_train1_ = x_train1_.astype(np.int8, copy=False)
        print(x_train1.shape, x_train1_.shape)  
        y_train_ = [y_train[i] for i in indexes_diff_tr]
        scores = mutual_info_classif(x_train1_, y_train_, 
                                     discrete_features=True, 
                                     random_state=0)
        indices_new_array = [i for i in range(len(scores))]
        scores_indices = sorted(zip(scores, indices_new_array), 
                                reverse=True)
        scores_, indices_ =  zip(*scores_indices)
        save_obj(scores_indices, 
                 os.path.join(path_indices, 
                              "scores_indices_mutual_{}_info_p4"\
                              .format(opt.approach)))


if __name__ == '__main__':
    main()