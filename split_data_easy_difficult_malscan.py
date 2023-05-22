import sys
from util import load_y_pred_prob, load_matrices_and_labels
from util import get_scores_oneline, get_scores
import os
import argparse
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

    
def save_obj(obj, path ):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
 
def get_indices(y, y_predict, y_predict2, y_proba2, y_RF, keyword):
    y_RF_predicted_easy = []
    y_RF_easy = []
    y_RF_predicted_diff = []
    y_RF_diff = []
    indices_diff = []
    for i in range(len(y_proba2)):
        if y_proba2[i][1]==0:
            y_RF_predicted_easy.append(y_predict2[i])
            y_RF_easy.append(y_RF[i])
        else:
            y_RF_predicted_diff.append(y_predict2[i])
            y_RF_diff.append(y_RF[i])
            indices_diff.append(i)


    print("-----MalScan, {}-------".format(keyword))
    y_MalScan_predicted_easy = []
    y_MalScan_easy = []
    y_MalScan_predicted_diff = []
    y_MalScan_diff = []
    for i in range(len(y_proba2)):
        if i not in indices_diff:
            y_MalScan_predicted_easy.append(y_predict[i])
            y_MalScan_easy.append(y[i])
        else:
            y_MalScan_predicted_diff.append(y_predict[i])
            y_MalScan_diff.append(y[i])

    get_scores_oneline(y, y_predict, "All")
    get_scores_oneline(y_MalScan_easy, 
                       y_MalScan_predicted_easy, "Easy")       
    get_scores_oneline(y_MalScan_diff, 
                       y_MalScan_predicted_diff, "Difficult")
    return indices_diff


def parse_option():
    parser = argparse.ArgumentParser('Split data for MalScan')
    parser.add_argument(
        '--path_save', type=str,
        help='path where to save outputs', required=True)
    parser.add_argument(
        '--path_files', type=str,
        help='path where matrices, preds and probabs are located', 
        required=True)
    parser.add_argument(
        '--split_id', type=int,
        help='the id of the data split to use: 1, 2, 3, 4, 5 or time', 
        required=True)
    parser.add_argument(
        '--approach', type=str,
        help='either malscan_a or malscan_co', required=True)
    
    opt = parser.parse_args()
    return opt


def main():    
    opt = parse_option()
    
    print("-------------------{}----------------------".format(opt.approach))
    path_indices = os.path.join(opt.path_save, "indices", 
                                "indices_{}".format(opt.split_id))
    if not os.path.exists(path_indices):
        os.makedirs(path_indices)
    
    (x_train1, x_valid1, x_test1,
     y_train, y_valid, y_test) = load_matrices_and_labels(opt, opt.approach)
    (y_predict_test, y_predict_valid, y_predict_train,
     y_proba_train, y_proba_valid, y_proba_test) = load_y_pred_prob(opt, 
                                                                  opt.approach)

    y_train_RF = [1 
                  if y_train[i]!=y_predict_train[i] 
                  else 0 
                  for i in range(len(y_train))]
    y_valid_RF = [1 
                  if y_valid[i]!=y_predict_valid[i] 
                  else 0 
                  for i in range(len(y_valid))]
    y_test_RF = [1 
                 if y_test[i]!=y_predict_test[i] 
                 else 0 
                 for i in range(len(y_test))]

    clf = RandomForestClassifier(n_estimators=1000, random_state=0)
    clf.fit(x_train1, y_train_RF)
    print("RF")
    y_predict_test2, y_proba_test2, _, _, _, _, _ = get_scores(clf, x_test1, 
                                                               y_test_RF, 
                                                               "Test", 1)
    y_predict_valid2, y_proba_valid2, _, _, _, _, _ = get_scores(clf, x_valid1, 
                                                                 y_valid_RF, 
                                                                 "Valid", 1)
    y_predict_train2, y_proba_train2, _, _, _, _, _  = get_scores(clf, x_train1, 
                                                                  y_train_RF, 
                                                                  "Train", 1)
    
    indices_diff_te = get_indices(y_test, y_predict_test, y_predict_test2, 
                                  y_proba_test2, y_test_RF, "Test")
    indices_diff_va = get_indices(y_valid, y_predict_valid, y_predict_valid2, 
                                  y_proba_valid2, y_valid_RF, "Valid")
    indices_diff_tr = get_indices(y_train, y_predict_train, y_predict_train2, 
                                  y_proba_train2, y_train_RF, "Train")
    
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
    
if __name__ == '__main__':
    main()