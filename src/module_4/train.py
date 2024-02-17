import pandas as pd
import numpy as np
import logging
import os
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve
from utils import *
from sklearn.base import BaseEstimator
import datetime
import joblib
import xgboost as xgb 

logger = logging.getLogger(__name__)
logger.level = logging.INFO

consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)

TARGET = ['outcome']
FEATURE_COLS = ['user_order_seq', 'ordered_before', 'abandoned_before',
       'active_snoozed', 'set_as_regular', 'normalised_price', 'discount_pct',
       'global_popularity', 'count_adults', 'count_children', 'count_babies',
       'count_pets', 'people_ex_baby', 'days_since_purchase_variant_id',
       'avg_days_to_buy_variant_id', 'std_days_to_buy_variant_id',
       'days_since_purchase_product_type', 'avg_days_to_buy_product_type',
       'std_days_to_buy_product_type']

TRAIN_SIZE = 0.7

OUTPUT_PATH = os.path.join(STORAGE_PATH, "module_4_models")

PARAM_GRID = {'learning_rate': [0.001, 0.01, 0.1, 0.25, 1, 10]}


def train_test_split(df:pd.DataFrame, train_size:float) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

    final_data_sorted = df.sort_values(by='order_date')

    train_end = final_data_sorted['order_date'].quantile(train_size)

    train_data = final_data_sorted[final_data_sorted['order_date'] < train_end]
    val_data = final_data_sorted[(final_data_sorted['order_date'] >= train_end)]

    X_train, y_train = feature_label_split(FEATURE_COLS, TARGET, train_data)
    X_val, y_val = feature_label_split(FEATURE_COLS, TARGET, val_data)

    logger.info(f"Splitting data on {X_train.shape[0]} training samples and {X_val.shape[0]} validation samples")

    return X_train, y_train, X_val, y_val
    

def save_model(model: BaseEstimator, model_name:str) -> None:
    logger.info(f"Saving model {model_name} to {OUTPUT_PATH}")
    if not os.path.exists(OUTPUT_PATH):
        logger.info(f"Creating directory {OUTPUT_PATH}")
        os.makedirs(OUTPUT_PATH)

    model_name = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{model_name}.pkl"
    joblib.dump(model, os.path.join(OUTPUT_PATH, model_name))

def XGBoost_model_selection(df: pd.DataFrame) -> None:
    train_size = TRAIN_SIZE
    param_grid = PARAM_GRID
    X_train, y_train, X_val, y_val = train_test_split(df, train_size)

    best_auc = 0
    best_params = None


    for learning_rate in param_grid['learning_rate']:

        xgboost = xgb.XGBClassifier(
        learning_rate =learning_rate,
        max_depth=5,
        n_estimators = 100,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=1,
        use_label_encoder=False,
        seed=27, reg_alpha = 0.01, reg_lambda = 100)

        xgboost.fit(X_train, y_train)

        y_pred = xgboost.predict_proba(X_val)[:, 1]

        precision, recall, _ = precision_recall_curve(y_val, y_pred)
        pr_auc = auc(recall, precision)

        if pr_auc > best_auc:
            best_auc = pr_auc
            best_params = {'learning_rate': learning_rate}

    logger.info(f"Training best model with learning_rate = {best_params['learning_rate']} over whole dataset")
    best_model = xgboost = xgb.XGBClassifier(
        learning_rate =best_params['learning_rate'],
        max_depth=5,
        n_estimators = 100,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=1,
        use_label_encoder=False,
        seed=27, reg_alpha = 0.01, reg_lambda = 100)
    
    X,y = feature_label_split(FEATURE_COLS, TARGET, df)
    best_model.fit(X, y)

    save_model(best_model, f"XGBoost_{best_params['learning_rate']}")

def main():
    data = preprocess_data_manually()
    XGBoost_model_selection(data)

if __name__ == "__main__":
    main()
