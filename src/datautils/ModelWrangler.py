#Author: Ramine Tinati

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


'''
A simple class providing methods for performing model analysis
'''
class ModelWrangler:

    def __init__(self):
        print('Model Wrangler')

    def prepare_test_train_data(self, df, target_col_str='Class',
                                train_split=0.7, isStratify=True,
                                isShuffle=False, isStringTarget = True,
                                features = None, isScaler = False):

        if isStringTarget:
            df[target_col_str] = df[target_col_str].replace({0: 'False', 1: 'True'})

        y = df[target_col_str]
        #     X = df['V21'].values
        if features is not None:
            X = df[features].values
        else:
            X = df.drop(target_col_str, axis=1).values

        if isScaler:
            scaler = StandardScaler()
            X  = scaler.fit_transform(X)

        if isStratify:
            isStratify = y
        else:
            isStratify = None
            #print(np.unique(y))

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_split,
                                                            shuffle=isShuffle,
                                                            stratify=isStratify)

        print('Generating Training / Test Split')
        print('Training X: {}, y: {}'.format(len(X_train), len(y_train)))
        print('Test X: {}, y: {}'.format(len(X_test), len(y_test)))

        return X_train, X_test, y_train, y_test

    def prepare_auto_enconder_test_train_data(self, df, target_col_str='Class',
                                test_size=0.2, isStratify=True,
                                isShuffle=False, isStringTarget = True,
                                features = None, isScaler = False):

        RANDOM_SEED = 42

        X_train, X_test = train_test_split(df, test_size=test_size, random_state=RANDOM_SEED)

        X_train = X_train[X_train[target_col_str] == 0]
        X_train = X_train.drop(target_col_str, axis=1)
        y_test = X_test[target_col_str]
        X_test = X_test.drop(target_col_str, axis=1)

        if features:
            X_train, X_test = X_train[features], X_test[features]

        X_train = X_train.values
        X_test = X_test.values

        print('Generating Training / Test Split')
        print('Training X: {}'.format(len(X_train)))
        print('Test X: {}, y: {}'.format(len(X_test), len(y_test)))

        return X_train, X_test, y_test

    def prepare_kfold_test_train_data(self, df, target_col_str='Class', train_split=0.7, isStratify=True, isShuffle=False):

        k_folds = []
        df[target_col_str] = df[target_col_str].replace({0: 'False', 1: 'True'})

        y = df[target_col_str]
        X = df.drop(target_col_str, axis=1).values

        if isStratify:
            skf = StratifiedKFold(n_splits=5, shuffle=isShuffle)
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                k_folds.append((X_train, X_test, y_train, y_test))
        else:
            kf = KFold(n_splits=5)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                k_folds.append((X_train, X_test, y_train, y_test))

        print('Generating {} KFold Training / Test Splits'.format(len(k_folds)))
        for split in k_folds:
            print('Training X: {}, y: {}'.format(len(split[0]), len(split[2])))
            print('Test X: {}, y: {}'.format(len(split[1]), len(split[3])))

        return k_folds

    def calculate_binary_model_performance(self, df,
                                    target_col_name,
                                    target_class_a,
                                    target_class_b,
                                    predictions,
                                    printoutput = False):

        if df.shape[0] == 0:
            print('DataFrame cannot be Empty ')
            return None

        elif df.shape[0] != predictions.shape[0]:
            print('DataFrame length and Predictions different Lengths')
            return None
        else:
            df['preds'] = predictions
            tp, tn, fp, fn = 0, 0, 0, 0
            for idx, row in df.iterrows():
                if row[target_col_name] == target_class_a:  # assumes 1 is
                    if row['preds'] == target_class_a:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if row['preds'] == target_class_b:
                        fp += 1
                    else:
                        tn += 1

            positive_classes = df[target_col_name][df[target_col_name] == target_class_a].count()
            negative_classes = df[target_col_name][df[target_col_name] == target_class_b].count()
            class_report = classification_report(df[target_col_name], df['preds'])

            if printoutput:
                print('Positive Records {}, Negative Records {}'.format(positive_classes, negative_classes))
                print('Detection Results (Absolute):')
                print('True Positive {}'.format(tp))
                print('True Negative {}'.format(tn))
                print('False Positive {}'.format(fp))
                print('False Negative {}'.format(fn))

                # And use Classification Report from Skit-Learn
                print(class_report)

            return class_report, tp, tn, fp, fn


    def roc_plot(self, model_data, model):

        X_train, X_test, y_train, y_test = model_data
        ns_probs = [0 for _ in range(len(y_test))]
        # fit a model
        model = LogisticRegression(solver='lbfgs')
        model.fit(X_train, y_train)
        # predict probabilities
        lr_probs = model.predict_proba(X_test)
        # keep probabilities for the positive outcome only
        lr_probs = lr_probs[:, 1]
        # calculate scores
        ns_auc = roc_auc_score(y_test, ns_probs)
        lr_auc = roc_auc_score(y_test, lr_probs)
        # summarize scores
        print('No Skill: ROC AUC=%.3f' % (ns_auc))
        print('Logistic: ROC AUC=%.3f' % (lr_auc))
        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs, pos_label='True')
        lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs, pos_label='True')
        # plot the roc curve for the model
        pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()


    def logR_model(self, model_data):
        X_train, X_test, y_train, y_test = model_data
        clf = LogisticRegression()

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
        print(classification_report(y_test, y_pred))
        return clf


    def xgboost_model(self, model_data, lr = 0.01):
        X_train, X_test, y_train, y_test = model_data
        model = XGBClassifier(learning_rate=lr)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # evaluate predictions
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        print(classification_report(y_test, y_pred))
        return model
