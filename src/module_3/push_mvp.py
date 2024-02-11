

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
from evaluation_functions import *

file_path = '/home/oscar/data/feature_frame.parquet'
save_models_path = 'src/module_3/models'


def read_data(file_path: str) -> pd.DataFrame:
    try:
        data = pq.read_table(file_path).to_pandas()
        
        print("Data shape:", data.shape)
        print("Data columns:", data.columns.tolist())
        print("Data types:\n", data.dtypes)
        
        return data
    except FileNotFoundError:
        print("Error: File not found at the specified path.")
        return None
    except Exception as e:
        print("An error occurred:", e)
        return None

def sample_data_by_orders(data: pd.DataFrame, sample_percentage: float) -> pd.DataFrame:

    order_ids = data['order_id'].unique()
    
    sample_orders_amount = int(len(order_ids) * sample_percentage)
    
    sampled_orders = np.random.choice(order_ids, size=sample_orders_amount, replace=False)
    
    sampled_data = data[data['order_id'].isin(sampled_orders)]
    
    print("Original dataset size:", data.shape)
    print("Sampled dataset size:", sampled_data.shape)
    
    return sampled_data

def preprocess_data(data: pd.DataFrame, num_items_to_filter: int, remove_na: bool, features_to_remove: list) -> pd.DataFrame:

    num_products_ordered = data.groupby('order_id')['outcome'].sum()
    
    orders_with_min_num_products = num_products_ordered[num_products_ordered >= num_items_to_filter].index
    filtered_data = data[data['order_id'].isin(orders_with_min_num_products)]
    
    if remove_na:
        filtered_data = filtered_data.dropna()
    
    filtered_data['order_date'] = pd.to_datetime(filtered_data['order_date'])
    filtered_data['created_at'] = pd.to_datetime(filtered_data['created_at'])
    
    preprocessed_data = filtered_data.drop(features_to_remove, axis=1)
    
    return preprocessed_data

def split_data(data: pd.DataFrame, train_split: float, val_split: float) -> tuple:

    try:
        if data['order_date'].dtype == 'datetime64[ns]':
            pass
        else:
            data['order_date'] = pd.to_datetime(data['order_date'])
    except KeyError:
        print("Error: 'order_date' column not found.")
        return None


    final_data_sorted = data.sort_values(by='order_date')

    train_end = final_data_sorted['order_date'].quantile(train_split)
    val_end = final_data_sorted['order_date'].quantile(val_split)

    train_data = final_data_sorted[final_data_sorted['order_date'] < train_end]
    val_data = final_data_sorted[(final_data_sorted['order_date'] >= train_end) & (final_data_sorted['order_date'] < val_end)]
    test_data = final_data_sorted[final_data_sorted['order_date'] >= val_end]

    print("Training set size:", train_data.shape[0] / final_data_sorted.shape[0])
    print("Validation set size:", val_data.shape[0] / final_data_sorted.shape[0])
    print("Testing set size:", test_data.shape[0] / final_data_sorted.shape[0])

    X_train = train_data.drop(columns=['outcome', 'order_date'])
    y_train = train_data['outcome']

    X_val = val_data.drop(columns=['outcome', 'order_date'])
    y_val = val_data['outcome']

    X_test = test_data.drop(columns=['outcome', 'order_date'])
    y_test = test_data['outcome']

    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    cols_to_remove = ['variant_id', 'order_id', 'user_id', 'created_at', 'count_adults', 'count_children', 'count_babies', 'count_pets', 'people_ex_baby']
    data_loaded = read_data(file_path)
    data_sampled = sample_data_by_orders(data_loaded, 0.2)
    data_processed = preprocess_data(data_sampled, 5, False, cols_to_remove )
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(data_processed, 0.7, 0.85)


if __name__ == "__main__":
    main()


