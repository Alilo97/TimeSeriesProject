import numpy as np
import pandas as pd
# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import random

def remove_random_indexes(df, num):
    size_x, size_y = df.shape
    result = np.empty(num, dtype=tuple)
    for i in range(0, num):
        result[i] = np.random.randint(0, high=size_x), np.random.randint(0, high=size_y)
        df.iloc[result[i]] = np.NAN
    return result

def predict_data(df):
    missing_df = df.copy()

    missing_num = int(missing_df.size * 0.25)
    missing_idx = remove_random_indexes(missing_df, missing_num)

    actual_data = np.empty(missing_num)
    for i in range(0, missing_num):
        actual_data[i] = df.iloc[missing_idx[i]]
    
    imputer = IterativeImputer(LinearRegression())
    impute_data = pd.DataFrame(imputer.fit_transform(missing_df))

    predicted_data = np.empty(missing_num)
    for i in range(0, missing_num):
        predicted_data[i] = impute_data.iloc[missing_idx[i]]

    mae = mean_absolute_error(actual_data, predicted_data)

    return mae, actual_data, predicted_data
