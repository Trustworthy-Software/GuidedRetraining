import sys
from util import file_to_list, get_scores, load_matrices_and_labels
from util import decision_function_to_proba, save_y_pred_prob, load_indices
import os
from sklearn.feature_extraction.text import CountVectorizer as CountVectorizer
from sklearn.svm import LinearSVC
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def parse_option():
    parser = argparse.ArgumentParser('Train RBC')
    parser.add_argument(
        '--path_save', type=str,
        help='path where to save outputs', required=True)
    parser.add_argument(
        '--split_id', type=int,
        help='the id of the dataset split to use: 1, 2, 3, 4, 5 or time', 
        required=True)
    parser.add_argument(
        '--path_indices', type=str,
        help='the path to the "indices" folder. '\
        'you can refer to split_data scripts', required=True)
    
    opt = parser.parse_args()
    return opt

def get_difficult_matrices_and_labels(
        indexes_diff_tr, indexes_diff_va, indexes_diff_te, x_train1, 
        x_valid1, x_test1, y_train, y_valid, y_test):
    
    x_train1 = x_train1[indexes_diff_tr]
    x_valid1 = x_valid1[indexes_diff_va]
    x_test1 = x_test1[indexes_diff_te]
    y_train = [y_train[i] for i in indexes_diff_tr]
    y_valid = [y_valid[i] for i in indexes_diff_va]
    y_test = [y_test[i] for i in indexes_diff_te]
    
    return x_train1, x_valid1, x_test1, y_train, y_valid, y_test


def main():
    opt = parse_option()
    
    if not os.path.exists(opt.path_save):
        os.makedirs(opt.path_save)
        
    ################################# DREBIN #################################
    x_train1, x_valid1, x_test1, y_train, y_valid, y_test = load_matrices_and_labels(opt, 
                                                                                     "drebin")
    opt.approach = "drebin"

    (indexes_diff_tr, _, _, _, _, indexes_diff_va, indexes_diff_te) =  load_indices(opt)

    
    (x_train1, x_valid1, x_test1, y_train, y_valid, y_test) = get_difficult_matrices_and_labels(
        indexes_diff_tr, indexes_diff_va, indexes_diff_te, x_train1, 
        x_valid1, x_test1, y_train, y_valid, y_test)
    
    clf = LinearSVC(random_state=0)
    clf.fit(x_train1, y_train)
    y_predict_test, y_proba_test, rec, pre, f1, acc, _ = get_scores(clf, x_test1, 
                                                                    y_test, "Test", 0)   
    y_predict_valid, y_proba_valid, rec, pre, f1, acc, _ = get_scores(clf, x_valid1, 
                                                                      y_valid, "Valid", 0)
    y_predict_train, y_proba_train, rec, pre, f1, acc, _  = get_scores(clf, x_train1, 
                                                                       y_train, "Train", 0)


    #Decision_func_to_predict_proba = ((Input - InputLow) / 
    # (InputHigh - InputLow)) * (OutputHigh - OutputLow) + OutputLow
    InputLow_pos = min([i for i in y_proba_train if i>=0])
    InputHigh_pos = max([i for i in y_proba_train if i>=0])
    #max because they are negative
    InputLow_neg = abs(max([i for i in y_proba_train if i<=0]))
    InputHigh_neg = abs(min([i for i in y_proba_train if i<=0])) #same for min

    y_proba_train, y_proba_valid, y_proba_test = decision_function_to_proba(
        InputLow_pos, InputHigh_pos, InputLow_neg, InputHigh_neg, y_train, y_valid, 
        y_test, y_proba_train, y_proba_valid, y_proba_test)


    save_y_pred_prob(y_predict_test, y_predict_valid, y_predict_train, 
                     y_proba_train, y_proba_valid, y_proba_test, opt, "drebin")

    
    #######################"# REVEAL #####################################
    x_train1, x_valid1, x_test1, y_train, y_valid, y_test = load_matrices_and_labels(opt, 
                                                                                     "reveal")
    opt.approach = "reveal"

    (indexes_diff_tr, _, _, _, _, indexes_diff_va, indexes_diff_te) =  load_indices(opt)
    
    (x_train1, x_valid1, x_test1, y_train, y_valid, y_test) = get_difficult_matrices_and_labels(
        indexes_diff_tr, indexes_diff_va, indexes_diff_te, x_train1, 
        x_valid1, x_test1, y_train, y_valid, y_test)
    
    clf = LinearSVC(C=0.01, penalty="l1", dual=False, random_state=0)
    clf.fit(x_train1, y_train)
    y_predict_test, y_proba_test, rec, pre, f1, acc, _ = get_scores(clf, x_test1, 
                                                                    y_test, "Test", 0)   
    y_predict_valid, y_proba_valid, rec, pre, f1, acc, _ = get_scores(clf, x_valid1, 
                                                                      y_valid, "Valid", 0)
    y_predict_train, y_proba_train, rec, pre, f1, acc, _  = get_scores(clf, x_train1, 
                                                                       y_train, "Train", 0)


    #Decision_func_to_predict_proba = ((Input - InputLow) / 
    # (InputHigh - InputLow)) * (OutputHigh - OutputLow) + OutputLow
    InputLow_pos = min([i for i in y_proba_train if i>=0])
    InputHigh_pos = max([i for i in y_proba_train if i>=0])
    #max because they are negative
    InputLow_neg = abs(max([i for i in y_proba_train if i<=0])) 
    #same for min
    InputHigh_neg = abs(min([i for i in y_proba_train if i<=0])) 

    y_proba_train, y_proba_valid, y_proba_test = decision_function_to_proba(
        InputLow_pos, InputHigh_pos, InputLow_neg, InputHigh_neg, y_train, y_valid, 
        y_test, y_proba_train, y_proba_valid, y_proba_test)


    save_y_pred_prob(y_predict_test, y_predict_valid, y_predict_train, 
                     y_proba_train, y_proba_valid, y_proba_test, opt, "reveal")
    
    
    ############################### MaMaF ###################################
    
    x_train1, x_valid1, x_test1, y_train, y_valid, y_test = load_matrices_and_labels(opt, 
                                                                                     "mama_f")
    opt.approach = "mama_f"
    (indexes_diff_tr, _, _, _, _, indexes_diff_va, indexes_diff_te) =  load_indices(opt)
    
    (x_train1, x_valid1, x_test1, y_train, y_valid, y_test) = get_difficult_matrices_and_labels(
        indexes_diff_tr, indexes_diff_va, indexes_diff_te, x_train1, 
        x_valid1, x_test1, y_train, y_valid, y_test)
    
    clf = RandomForestClassifier(max_depth=8, n_estimators=51, random_state=0)
    clf.fit(x_train1, y_train)
    y_predict_test, y_proba_test, rec, pre, f1, acc, _ = get_scores(clf, x_test1, 
                                                                    y_test, "Test", 1)   
    y_predict_valid, y_proba_valid, rec, pre, f1, acc, _ = get_scores(clf, x_valid1, 
                                                                      y_valid, "Valid", 1)
    y_predict_train, y_proba_train, rec, pre, f1, acc, _  = get_scores(clf, x_train1, 
                                                                       y_train, "Train", 1)

    save_y_pred_prob(y_predict_test, y_predict_valid, y_predict_train, y_proba_train, 
                     y_proba_valid, y_proba_test, opt, "mama_f")
    
    ############################# MaMaP ###################################
    
    x_train1, x_valid1, x_test1, y_train, y_valid, y_test = load_matrices_and_labels(opt, 
                                                                                     "mama_p")
    opt.approach = "mama_p"
    (indexes_diff_tr, _, _, _, _, indexes_diff_va, indexes_diff_te) =  load_indices(opt)
    
    (x_train1, x_valid1, x_test1, y_train, y_valid, y_test) = get_difficult_matrices_and_labels(
        indexes_diff_tr, indexes_diff_va, indexes_diff_te, x_train1, 
        x_valid1, x_test1, y_train, y_valid, y_test)
    
    clf = RandomForestClassifier(max_depth=64, n_estimators=101, random_state=0)
    clf.fit(x_train1, y_train)
    y_predict_test, y_proba_test, rec, pre, f1, acc, _ = get_scores(clf, x_test1, 
                                                                    y_test, "Test", 1)   
    y_predict_valid, y_proba_valid, rec, pre, f1, acc, _ = get_scores(clf, x_valid1, 
                                                                      y_valid, "Valid", 1)
    y_predict_train, y_proba_train, rec, pre, f1, acc, _  = get_scores(clf, x_train1, 
                                                                       y_train, "Train", 1)

    save_y_pred_prob(y_predict_test, y_predict_valid, y_predict_train, y_proba_train, 
                     y_proba_valid, y_proba_test, opt, "mama_p")
    
    ############################### MalScanA #############################
    
    x_train1, x_valid1, x_test1, y_train, y_valid, y_test = load_matrices_and_labels(opt, 
                                                                                     "malscan_a")
    opt.approach = "malscan_a"

    (indexes_diff_tr, _, _, _, _, indexes_diff_va, indexes_diff_te) =  load_indices(opt)
    
    (x_train1, x_valid1, x_test1, y_train, y_valid, y_test) = get_difficult_matrices_and_labels(
        indexes_diff_tr, indexes_diff_va, indexes_diff_te, x_train1, 
        x_valid1, x_test1, y_train, y_valid, y_test)
    
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(x_train1, y_train)
    y_predict_test, y_proba_test, rec, pre, f1, acc, _ = get_scores(clf, x_test1, 
                                                                    y_test, "Test", 1)
    y_predict_valid, y_proba_valid, rec, pre, f1, acc, _ = get_scores(clf, x_valid1, 
                                                                      y_valid, "Valid", 1)
    y_predict_train, y_proba_train, rec, pre, f1, acc, _  = get_scores(clf, x_train1, 
                                                                       y_train, "Train", 1)

    save_y_pred_prob(y_predict_test, y_predict_valid, y_predict_train, y_proba_train, 
                     y_proba_valid, y_proba_test, opt, "malscan_a")
    
    
    ######################## MalScanCO ###################################
    

    x_train1, x_valid1, x_test1, y_train, y_valid, y_test = load_matrices_and_labels(opt, 
                                                                                     "malscan_co")
    opt.approach = "malscan_co"

    (indexes_diff_tr, _, _, _, _, indexes_diff_va, indexes_diff_te) =  load_indices(opt)
    
    (x_train1, x_valid1, x_test1, y_train, y_valid, y_test) = get_difficult_matrices_and_labels(
        indexes_diff_tr, indexes_diff_va, indexes_diff_te, x_train1, 
        x_valid1, x_test1, y_train, y_valid, y_test)
    
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(x_train1, y_train)
    y_predict_test, y_proba_test, rec, pre, f1, acc, _ = get_scores(clf, x_test1, 
                                                                    y_test, "Test", 1) 
    y_predict_valid, y_proba_valid, rec, pre, f1, acc, _ = get_scores(clf, x_valid1, 
                                                                      y_valid, "Valid", 1)
    y_predict_train, y_proba_train, rec, pre, f1, acc, _  = get_scores(clf, x_train1, 
                                                                       y_train, "Train", 1)

    save_y_pred_prob(y_predict_test, y_predict_valid, y_predict_train, y_proba_train, 
                     y_proba_valid, y_proba_test, opt, "malscan_co")
    
    
    
if __name__ == '__main__':
    main()