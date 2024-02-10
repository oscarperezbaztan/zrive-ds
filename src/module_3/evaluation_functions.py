
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
import category_encoders as ce
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.model_selection import PredefinedSplit
import pyarrow.parquet as pq
from sklearn import metrics
from sklearn.metrics import roc_auc_score, confusion_matrix


def plot_precision_recall_vs_threshold(clf, X, y):
    if hasattr(clf, 'predict_proba'):
        model_probs = clf.predict_proba(X)
        model_probs = model_probs[:, 1]
    else:
        model_probs = clf.decision_function(X)

    model_precision, model_recall, thresholds = metrics.precision_recall_curve(y, model_probs)
    thresholds = np.insert(thresholds, thresholds.size, 1., axis=0)

    plt.figure()
    plt.plot(thresholds, model_precision, 'r', label='Precision')
    plt.plot(thresholds, model_recall, 'b', label='Recall')
    plt.xlabel('Threshold Value')
    plt.title('Precision and Recall vs Threshold')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def threshold_classification(positive_class_probs, threshold=0.5):
    return (positive_class_probs >= threshold).astype('int')

def find_best_threshold(model, X_train, y_train, X_val, y_val, metric='fscore'):
    if hasattr(model, 'predict_proba'):
        model_probs = model.predict_proba(X_train)
        model_probs = model_probs[:, 1]
        model_probs_val = model.predict_proba(X_val)
        model_probs_val = model_probs_val[:, 1]
    else:
        model_probs = model.decision_function(X_train)
        model_probs_val = model.decision_function(X_val)
        
    if (metric == 'fscore'):
        model_precision, model_recall, thresholds = metrics.precision_recall_curve(y_train, model_probs)  
        performance_thresholds = [metrics.f1_score(y_val, threshold_classification(model_probs_val, t)) * 100.0 for t in thresholds]
        best_threshold_index = np.argmax(performance_thresholds)
    elif (metric == 'auc'):
        model_fpr, model_tpr, thresholds = metrics.roc_curve(y_train, model_probs)
        performance_thresholds = [metrics.roc_auc_score(y_val, threshold_classification(model_probs_val, t)) * 100.0 for t in thresholds]
        best_threshold_index = np.argmax(performance_thresholds)
    elif (metric == 'fpr'):
        model_fpr, model_tpr, thresholds = metrics.roc_curve(y_train, model_probs)
        performance_thresholds = np.array([100 - metrics.recall_score(y_val, threshold_classification(model_probs_val, t), pos_label=1) * 100.0 for t in thresholds])
        valid_thresholds = np.where(performance_thresholds <= 6.0)
        best_threshold_index = valid_thresholds[0][0]
    
    best_threshold = thresholds[best_threshold_index]
    print('Best threshold={:.3f}, {} on validation={:.2f}'.format(best_threshold, metric, performance_thresholds[best_threshold_index]))
    return best_threshold

def evaluate_best_threshold_test(model, threshold, X_test, y_test, metric='fscore'):
    if hasattr(model, 'predict_proba'):
        model_probs_test = model.predict_proba(X_test)
        model_probs_test = model_probs_test[:, 1]
    else:
        model_probs_test = model.decision_function(X_test)
        
    if (metric == 'fscore'):
        print('Fscore of the best model on test after optimizing threshold {:.2f}'.format(metrics.f1_score(y_test, threshold_classification(model_probs_test, threshold)) * 100.0))
    elif (metric == 'recall_mean'):
        print('Mean recall by classes of the best model on test after optimizing threshold {:.2f}'.format(metrics.roc_auc_score(y_test, threshold_classification(model_probs_test, threshold)) * 100.0))
    elif (metric == 'fpr'):
        print('FPR of the best model on test after optimizing threshold {:.2f}'.format(100 - metrics.recall_score(y_test, threshold_classification(model_probs_test, threshold), pos_label=1) * 100.0))
    predictions_test_best_threshold = threshold_classification(model_probs_test, threshold)
    print(metrics.confusion_matrix(y_test, predictions_test_best_threshold, labels=[1, 0]))

def plot_validation_curve(train_scores, test_scores, param_range, ax=None, param='Param', score='Score', scale=None):
    train_scores_mean = np.mean(-train_scores, axis=1)
    train_scores_std = np.std(-train_scores, axis=1)
    test_scores_mean = np.mean(-test_scores, axis=1)
    test_scores_std = np.std(-test_scores, axis=1)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.set_title("Validation Curve")
    ax.set_xlabel(param)
    ax.set_ylabel(score)
        
    lw = 2
    if scale is None:
        plot_fun = ax.plot
    else:
        plot_fun = ax.semilogx
    plot_fun(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    ax.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plot_fun(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    ax.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    ax.legend(loc="best")
    
    return ax

def evaluate_classifier(clf, X, y):
    prediction = clf.predict(X)
    recall_mean = metrics.roc_auc_score(y, prediction) * 100
    cm = metrics.confusion_matrix(y, prediction, labels=[1, 0])
    recall = cm[0, 0] / cm[0, :].sum() * 100
    tnr = cm[1, 1] / cm[1, :].sum() * 100
    GM = (recall * tnr) ** 0.5
    fscore = metrics.f1_score(y, prediction) * 100
    print('Mean recall by classes is {:.2f}'.format(recall_mean))
    print('Geometric mean is {:.2f}'.format(GM))
    print('Fscore is {:.2f}'.format(fscore))
    print('Confusion Matrix:')
    print(cm)

def plot_ROC_curve(clfs_list, X, y, labels):
    ns_probs = [0 for _ in range(len(y))]
    ns_auc = metrics.roc_auc_score(y, ns_probs) * 100.0
    ns_fpr, ns_tpr, _ = metrics.roc_curve(y, ns_probs)
    
    for i, clf in enumerate(clfs_list):
        if hasattr(clf, 'predict_proba'):
            model_probs = clf.predict_proba(X)
            model_probs = model_probs[:, 1]
        else:
            model_probs = clf.decision_function(X)

        model_auc = metrics.roc_auc_score(y, model_probs) * 100.0
        model_fpr, model_tpr, _ = metrics.roc_curve(y, model_probs)
        print('Area under the ROC curve for {}: {:.2f}'.format(labels[i], metrics.auc(model_fpr, model_tpr) * 100))
        plt.plot(model_fpr, model_tpr, label=labels[i])
        
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('ROC Curves')
    plt.show()

def plot_PR_curve(clfs_list, X, y, labels):
    no_skill = len(y[y == 1]) / len(y)
    
    for i, clf in enumerate(clfs_list):
        predictions = clf.predict(X)

        if hasattr(clf, 'predict_proba'):
            model_probs = clf.predict_proba(X)
            model_probs = model_probs[:, 1]
        else:
            model_probs = clf.decision_function(X)

        model_precision, model_recall, thresholds = metrics.precision_recall_curve(y, model_probs)
        model_f1 = metrics.f1_score(y, predictions) * 100.0
        model_auc_PR = metrics.auc(model_recall, model_precision)
        print('Area under the PR curve for {}: {:.2f}'.format(labels[i], model_auc_PR))

        plt.plot(model_recall, model_precision, label=labels[i])
    
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('PR Curves')
    plt.show()
