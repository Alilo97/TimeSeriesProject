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
    # Choose 25% of the index positions at random
    idx_miss = np.random.randint(0, high=len(df), size=int(len(df)*.25)) #adjust the value for missing data

    # Create response indicator matrix
    R = ~df.index.isin(idx_miss)

    # create observed and missing data
    observed = df[R]
    missing = df[~R]

    lrg = LinearRegression()
    lrg.fit(observed.index.values.reshape(-1, 1), observed.values)
    imputations = lrg.predict(missing.index.values.reshape(-1, 1))

    actual_data = missing.values

    # Calculate residuals, variance and noise vector from residual distr.
    residuals = observed.values - lrg.predict(observed.index.values.reshape(-1, 1))
    variance = residuals.var()
    rnoise = np.random.normal(0,np.sqrt(variance), len(imputations)).reshape(-1, 1)

    # Add noise vector to predicton vector from regression model.
    simputations = imputations + rnoise

    mae = mean_absolute_error(actual_data, simputations)

    return mae, actual_data, simputations
