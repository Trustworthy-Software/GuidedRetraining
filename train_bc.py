import sys
from util import file_to_list, get_scores, load_matrices_and_labels
from util import decision_function_to_proba, save_y_pred_prob
import os
from sklearn.feature_extraction.text import CountVectorizer as CountVectorizer
from sklearn.svm import LinearSVC
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def parse_option():
    parser = argparse.ArgumentParser('Train BC')
    parser.add_argument(
        '--path_save', type=str,
        help='path where to save outputs', required=True)
    parser.add_argument(
        '--split_id', type=int,
        help='the id of the dataset split to use: 1, 2, 3, 4, 5 or time', 
        required=True)
    
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_option()
    
    if not os.path.exists(opt.path_save):
        os.makedirs(opt.path_save)
        
    ################################# DREBIN #################################
    x_train1, x_valid1, x_test1, y_train, y_valid, y_test = load_matrices_and_labels(opt, 
                                                                                     "drebin")
    
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