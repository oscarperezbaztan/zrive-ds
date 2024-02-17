
import pandas as pd
import numpy as np
import logging
import os
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve

logger = logging.getLogger(__name__)
logger.level = logging.INFO

consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)

STORAGE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..", "data")
)

def load_dataset(dataset_name:str) -> pd.DataFrame:
    loading_file = os.path.join(STORAGE_PATH, dataset_name)
    logger.info(f"Loading dataset from {loading_file}")
    return pd.read_csv(loading_file)


def sample_dataset_by_orders(data:pd.DataFrame, sample_percentage:int = 0.5) -> pd.DataFrame:
    order_ids = data['order_id'].unique()
    sample_orders_amount = int(len(data['order_id'].unique()) * sample_percentage)
    sampled_orders = np.random.choice(order_ids, size=sample_orders_amount, replace=False)
    sampled_data = data[data['order_id'].isin(sampled_orders)]

    print("Original dataset size:", data.shape)
    print("Sampled dataset size:", sampled_data.shape)

    return sampled_data


def relevant_orders(df:pd.DataFrame, min_products: int = 5) -> pd.DataFrame:
    num_products_ordered = df.groupby('order_id')['outcome'].sum()
    orders_with_5_or_more_unique_products = num_products_ordered[num_products_ordered>=min_products].index
    filtered_data = df[df['order_id'].isin(orders_with_5_or_more_unique_products)]
    return filtered_data


def preprocess_data_manually() -> pd.DataFrame:
    logging.info("Building feature frame")
    return(
        load_dataset("feature_frame.csv")
        .pipe(sample_dataset_by_orders)
        .pipe(relevant_orders)
        .assign(created_at=lambda x: pd.to_datetime(x.created_at))
        .assign(order_date=lambda x:pd.to_datetime(x.order_date).dt.date)
    )


def feature_label_split(feature_cols: list, target_col: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return df[feature_cols], df[target_col]


def plot_metrics(
        model_name:str, y_pred: pd.Series, y_test: pd.Series, figure:Tuple[matplotlib.figure.Figure, np.array]=None
    ):
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    if figure is None:
        fig, ax = plt.subplots(1, 2, figsize=(14,7))

    else:
        fig, ax = figure

    ax[0].plot(recall, precision, label= f"{model_name}; AUC: {pr_auc:.2f}")
    ax[0].set_xlabel('Recall')
    ax[0].set_ylabel('Precision')
    ax[0].set_title('Precision-Recall Curve')
    ax[0].legend(loc="lower right")

    ax[1].plot(fpr, tpr, label= f"{model_name}; AUC: {roc_auc:.2f}")
    ax[1].set_xlabel('FPR')
    ax[1].set_ylabel('TPR')
    ax[1].set_title('ROC Curve')
    ax[1].legend(loc="lower right")
    

def evaluate_model(model_name: str, y_test: pd.Series, y_pred: pd.Series) -> Tuple[float, float]:

    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_test, y_pred)
    logger.info(
        f"{model_name} results:{{PR AUC: {pr_auc:.2f}, 'ROC AUC':{roc_auc:.2f}}}"
    )
    return pr_auc, roc_auc


def displayVariableImportance(tree, attributeNames):
    importances = list(tree.feature_importances_ * 100.0)
    variable_importance = [(varName, round(importance, 5)) for varName, importance in zip(attributeNames, importances)]
    variable_importance = sorted(variable_importance, key=lambda x: x[1], reverse=True)
    for varName, imp in variable_importance:
        print("The importance of variable {} is: {}".format(varName, imp))

    plt.figure(figsize=(15, 15))
    yValues = list(range(len(importances)))
    plt.barh(yValues, importances)
    plt.yticks(yValues, attributeNames, rotation='horizontal')
    plt.ylabel('Variable')
    plt.xlabel('Importance')
    plt.title('Variable Importance')
    plt.show()
